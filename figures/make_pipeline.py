"""Pipeline figure for the Methodology section.

    python3 figures/make_pipeline.py                     # picks the example route automatically
    python3 figures/make_pipeline.py --image-id 053674   # force a specific test image

Only reads saved models and results; nothing is trained and no other file is changed.
Writes figures/pipeline.pdf and figures/pipeline.png.

Route choice: the first route in src/rover-1/01-faster-rcnn/results.json where
  - both rovers reached the goal without entering a true crater (saved results),
  - the straight start-goal line crosses at least MIN_BLOCKED pixels of detected crater,
  - both the A* path and the Rover 2 path stray at least MIN_BEND pixels from that straight line,
  - start and goal are at least EDGE pixels from the image border.
"""
import argparse
import heapq
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from PIL import Image
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "figures"
images_path = ROOT / "dataset/images"
data = json.loads((ROOT / "dataset/gt_public.json").read_text())

device = "cpu"
cutoff = 0.8      # Faster R-CNN score cutoff, as in 08-rover-1.ipynb
MIN_BEND = 18     # pixels
MIN_BLOCKED = 15  # pixels of crater the straight start-goal line must cross
EDGE = 8          # pixels: start and goal at least this far from the image border, so markers are not cut off

torch.manual_seed(0)


# ---------- crater map (same as 08-rover-1.ipynb) ----------

def make_mask(boxes):
    mask = np.zeros((256, 256), dtype=np.uint8)
    yy, xx = np.mgrid[0:256, 0:256]
    for x, y, w, h in boxes:
        cx = x + w / 2
        cy = y + h / 2
        r = max(w, h) / 2
        inside = (xx + 0.5 - cx) ** 2 + (yy + 0.5 - cy) ** 2 <= r ** 2
        mask[inside] = 1
    return mask


detector = fasterrcnn_resnet50_fpn(weights=None, weights_backbone=None, min_size=256)
in_features = detector.roi_heads.box_predictor.cls_score.in_features
detector.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=2)
detector.load_state_dict(torch.load(ROOT / "src/faster-rcnn/01-baseline/model.pt", map_location=device))
detector.to(device)
detector.eval()


def load_image(image_id):
    image = Image.open(images_path / f"{image_id}.png").convert("RGB")
    return torch.tensor(np.array(image), dtype=torch.float32).permute(2, 0, 1) / 255.0


def detect_boxes(image_id):
    # Faster R-CNN gives (x1, y1, x2, y2); convert to (x, y, w, h) like the labels
    with torch.no_grad():
        out = detector([load_image(image_id).to(device)])[0]
    keep = out["scores"] > cutoff
    return [[x1, y1, x2 - x1, y2 - y1] for x1, y1, x2, y2 in out["boxes"][keep].cpu().tolist()]


# ---------- Rover 1: A* (same as 08-rover-1.ipynb) ----------

moves = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]


def heuristic(a, b):
    dy, dx = abs(a[0] - b[0]), abs(a[1] - b[1])
    return max(dy, dx) + (2 ** 0.5 - 1) * min(dy, dx)


def astar(blocked, start, goal):
    open_list = [(heuristic(start, goal), 0.0, start)]
    came_from = {start: None}
    cost_so_far = {start: 0.0}

    while open_list:
        _, cost, cell = heapq.heappop(open_list)
        if cell == goal:
            path = []
            while cell is not None:
                path.append(cell)
                cell = came_from[cell]
            return path[::-1]
        if cost > cost_so_far[cell]:
            continue

        for dy, dx in moves:
            ny, nx = cell[0] + dy, cell[1] + dx
            if not (0 <= ny < 256 and 0 <= nx < 256) or blocked[ny, nx]:
                continue
            new_cost = cost + (2 ** 0.5 if dy and dx else 1.0)
            if new_cost < cost_so_far.get((ny, nx), float("inf")):
                cost_so_far[(ny, nx)] = new_cost
                came_from[(ny, nx)] = cell
                heapq.heappush(open_list, (new_cost + heuristic((ny, nx), goal), new_cost, (ny, nx)))
    return None


# ---------- Rover 2: MLP on a 15x15 window (same as 09-rover-2.ipynb) ----------

window = 15
half = window // 2

policy = nn.Sequential(
    nn.Linear(227, 128),
    nn.ReLU(),
    nn.Linear(128, 8),
).to(device)
policy.load_state_dict(torch.load(ROOT / "src/rover-2/01-basic-mlp/model.pt", map_location=device))
policy.eval()


def observe(detected_map, cell, goal):
    padded = np.pad(detected_map, half, constant_values=1)
    y, x = cell
    view = padded[y:y + window, x:x + window]

    dy, dx = goal[0] - y, goal[1] - x
    length = max((dy ** 2 + dx ** 2) ** 0.5, 1e-6)
    direction = [dy / length, dx / length]

    return np.concatenate([view.flatten(), direction]).astype(np.float32)


def drive(detected_map, start, goal):
    cell = start
    path = [cell]
    max_steps = int(3 * heuristic(start, goal))
    for _ in range(max_steps):
        if max(abs(cell[0] - goal[0]), abs(cell[1] - goal[1])) <= 2:
            return path, True
        x = torch.tensor(observe(detected_map, cell, goal)).unsqueeze(0).to(device)
        with torch.no_grad():
            move = moves[policy(x).argmax().item()]
        ny, nx = cell[0] + move[0], cell[1] + move[1]
        if 0 <= ny < 256 and 0 <= nx < 256:
            cell = (ny, nx)
        path.append(cell)
    return path, False


def path_length(path):
    return sum(2 ** 0.5 if (a[0] != b[0] and a[1] != b[1]) else 1.0 for a, b in zip(path, path[1:]))


# ---------- route choice ----------

def bend(path, start, goal):
    # largest distance (pixels) of any path cell from the straight start-goal line
    p = np.array(path, dtype=float)
    a, b = np.array(start, dtype=float), np.array(goal, dtype=float)
    d = b - a
    return np.abs(d[0] * (p[:, 1] - a[1]) - d[1] * (p[:, 0] - a[0])).max() / np.linalg.norm(d)


def line_hits_crater(crater_map, start, goal):
    # True if the straight start-goal line runs through at least MIN_BLOCKED pixels of crater
    t = np.linspace(0, 1, 400)
    ys = np.rint(start[0] + t * (goal[0] - start[0])).astype(int)
    xs = np.rint(start[1] + t * (goal[1] - start[1])).astype(int)
    blocked = crater_map[ys, xs].mean() * np.hypot(goal[0] - start[0], goal[1] - start[1])
    return blocked >= MIN_BLOCKED


def pick_route(image_id=None):
    rover1 = json.loads((ROOT / "src/rover-1/01-faster-rcnn/results.json").read_text())
    rover2 = json.loads((ROOT / "src/rover-2/01-basic-mlp/results.json").read_text())

    for r1, r2 in zip(rover1, rover2):
        assert r1["image_id"] == r2["image_id"] and r1["start"] == r2["start"]
        if image_id is not None and r1["image_id"] != image_id:
            continue
        if image_id is None and not (r1["reached"] and r2["reached"]
                                     and r1["entered"] == 0 and r2["entered"] == 0):
            continue
        if image_id is None and not all(EDGE <= v <= 255 - EDGE for v in r1["start"] + r1["goal"]):
            continue

        start, goal = tuple(r1["start"]), tuple(r1["goal"])
        crater_map = make_mask(detect_boxes(r1["image_id"]))
        path1 = astar(crater_map, start, goal)
        if path1 is None:
            continue
        if image_id is None and not (line_hits_crater(crater_map, start, goal)
                                     and bend(path1, start, goal) >= MIN_BEND):
            continue
        path2, reached2 = drive(crater_map, start, goal)
        if image_id is None and not (reached2 and bend(path2, start, goal) >= MIN_BEND):
            continue
        return r1, r2, crater_map, path1, path2

    raise SystemExit("no route matched")


# ---------- figure ----------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-id", default=None)
    args = parser.parse_args()

    r1, r2, crater_map, path1, path2 = pick_route(args.image_id)
    image_id, start, goal = r1["image_id"], tuple(r1["start"]), tuple(r1["goal"])

    # check that the saved results are reproduced
    print(f"image {image_id}, start {start}, goal {goal}, "
          f"bend: A* {bend(path1, start, goal):.0f} px, Rover 2 {bend(path2, start, goal):.0f} px")
    print(f"Rover 1 length: saved {r1['length']:.4f}, here {path_length(path1):.4f}")
    print(f"Rover 2 length: saved {r2['length']:.4f}, here {path_length(path2):.4f}")

    plt.rcParams.update({
        "font.size": 8.5,
        "pdf.fonttype": 42,
        "savefig.facecolor": "white",
    })

    W, H = 7.0, 2.0          # figure size, inches
    P = 1.2                  # panel size
    y0 = 0.44                # panel bottom (room for two-line labels)
    ymid = y0 + P / 2
    box_w, box_h = 0.9, 0.62
    x_img = 0.10
    x_box = x_img + P + 0.25
    x_map = x_box + box_w + 0.25
    x_r1 = x_map + P + 0.32
    x_r2 = x_r1 + P + 0.25

    gray = "#c8c8c8"         # craters
    color1 = "#0072B2"       # Rover 1: solid blue
    color2 = "#D55E00"       # Rover 2: dashed vermillion

    fig = plt.figure(figsize=(W, H), facecolor="white")
    inch = fig.dpi_scale_trans

    def panel(x):
        ax = fig.add_axes([x / W, y0 / H, P / W, P / H])
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_linewidth(0.5)
        return ax

    def label(x_center, text):
        # every label's first line sits on the same baseline; extra lines go below it
        for i, line in enumerate(text.split("\n")):
            fig.text(x_center / W, (y0 - 0.16 - i * 0.14) / H, line, ha="center", va="baseline")

    def arrow(a, b):
        fig.add_artist(FancyArrowPatch(a, b, transform=inch, arrowstyle="-|>", mutation_scale=7,
                                       lw=0.8, color="black", shrinkA=0, shrinkB=0))

    def crater_panel(x):
        ax = panel(x)
        ax.imshow(crater_map, cmap=matplotlib.colors.ListedColormap(["white", gray]),
                  vmin=0, vmax=1, interpolation="nearest")
        return ax

    def route(ax, path, color, dashes):
        ys, xs = zip(*path)
        line, = ax.plot(xs, ys, color=color, lw=1.3, solid_capstyle="round")
        if dashes:
            line.set_dashes(dashes)
        ax.scatter(start[1], start[0], s=22, facecolor="white", edgecolor="black", lw=0.8, zorder=3)
        ax.scatter(goal[1], goal[0], s=70, marker="*", facecolor="#F0E442", edgecolor="black",
                   lw=0.6, zorder=3)
        ax.set_xlim(-0.5, 255.5)
        ax.set_ylim(255.5, -0.5)

    # 1. orbital image, shown as stored (no contrast change)
    ax = panel(x_img)
    ax.imshow(np.array(Image.open(images_path / f"{image_id}.png").convert("L")),
              cmap="gray", vmin=0, vmax=255, interpolation="nearest")
    label(x_img + P / 2, "Orbital image")

    # 2. detector box
    fig.add_artist(FancyBboxPatch((x_box, ymid - box_h / 2), box_w, box_h, transform=inch,
                                  boxstyle="round,pad=0,rounding_size=0.06",
                                  facecolor="#f2f2f2", edgecolor="black", lw=0.8))
    fig.text((x_box + box_w / 2) / W, ymid / H, "U-Net /\nFaster R-CNN /\nDINOv3",
             ha="center", va="center", linespacing=1.4)
    label(x_box + box_w / 2, "Crater detector")

    # 3. crater map
    crater_panel(x_map)
    label(x_map + P / 2, "Crater map")

    # 4. the two rovers on the same map, same start and goal
    route(crater_panel(x_r1), path1, color1, None)
    label(x_r1 + P / 2, "Rover 1:\nA* on full map")

    route(crater_panel(x_r2), path2, color2, (3, 1.6))
    label(x_r2 + P / 2, "Rover 2:\nMLP, 15×15 window")

    # arrows: image -> detector -> map, then one arrow that forks into both rover panels
    gap = 0.04
    arrow((x_img + P + gap, ymid), (x_box - gap, ymid))
    arrow((x_box + box_w + gap, ymid), (x_map - gap, ymid))
    x_feed = x_map + P + 0.16        # vertical feed line, between crater map and Rover 1
    y_bar = y0 + P + 0.16            # height of the branch bar above the rover panels
    x_c1, x_c2 = x_r1 + P / 2, x_r2 + P / 2
    stem = dict(transform=inch, lw=0.8, color="black", solid_joinstyle="miter", solid_capstyle="butt")
    fig.add_artist(Line2D([x_map + P + gap, x_feed, x_feed, x_c2], [ymid, ymid, y_bar, y_bar], **stem))
    for x_c in (x_c1, x_c2):
        arrow((x_c, y_bar), (x_c, y0 + P + gap))
    fig.add_artist(plt.Circle((x_c1, y_bar), 0.022, transform=inch, color="black"))   # split point

    fig.savefig(OUT_DIR / "pipeline.pdf")
    fig.savefig(OUT_DIR / "pipeline.png", dpi=300)
    print("saved", OUT_DIR / "pipeline.pdf", "and", OUT_DIR / "pipeline.png")


if __name__ == "__main__":
    main()

"""Example routes of both rovers, for the Results section.

    python3 figures/make_rover_routes.py

Only reads saved models and results; nothing is trained and no other file is changed.
Writes figures/rover_routes.pdf and figures/rover_routes.png.

Top row: Rover 1 (A* on the Faster R-CNN crater map, src/rover-1/01-faster-rcnn/results.json).
Bottom row: Rover 2 (MLP on a 15x15 window, src/rover-2/01-basic-mlp/results.json).

Route choice: among all saved test routes where
  - the rover reached the goal without entering a true crater (saved results, and checked again here),
  - the straight start-goal line runs at least MIN_CHORD pixels through one true crater
    of radius at least MIN_RADIUS,
  - the rover's path strays at least MIN_BEND pixels from that straight line,
  - the path keeps at least MIN_CLEAR pixels from every true crater circle,
  - start and goal are at least EDGE pixels from the image border and MARKER_CLEAR pixels from any crater,
  - the path recomputed here from the saved models matches the saved path length,
the N_PANELS with the largest min(crossing, bend) per rover, skipping images in EXCLUDE
(checked by eye: the label crossed there shows no visible crater). Every panel uses a different
route (a different test image).
"""
import heapq
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from matplotlib.patches import Circle
from PIL import Image
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "figures"
images_path = ROOT / "dataset/images"
data = json.loads((ROOT / "dataset/gt_public.json").read_text())

device = "cpu"
cutoff = 0.8       # Faster R-CNN score cutoff, as in 08-rover-1.ipynb
N_PANELS = 3       # routes per rover
MIN_CHORD = 10    # pixels of one true crater the straight start-goal line must cross
MIN_RADIUS = 6     # pixels: the crossed crater must be large enough to see
MIN_BEND = 15      # pixels
MIN_CLEAR = 3      # pixels: every path cell stays this far outside every true crater circle,
                   # so the drawn line does not touch a circle
MARKER_CLEAR = 6   # pixels: start and goal this far outside every crater, so the markers do not cover one
EXCLUDE = {
    "031804",      # the crossed label circle has no visible crater in the image
}
EDGE = 8           # pixels: start and goal at least this far from the image border

torch.manual_seed(0)


# ---------- crater maps (same as 08-rover-1.ipynb) ----------

def make_mask(boxes):
    mask = np.zeros((256, 256), dtype=np.uint8)
    yy, xx = np.mgrid[0:256, 0:256]
    for x, y, w, h in boxes:
        cx, cy, r = x + w / 2, y + h / 2, max(w, h) / 2
        mask[(xx + 0.5 - cx) ** 2 + (yy + 0.5 - cy) ** 2 <= r ** 2] = 1
    return mask


def true_boxes(image_id):
    return [b for b in data[image_id] if b[2] > 0 and b[3] > 0]


detector = fasterrcnn_resnet50_fpn(weights=None, weights_backbone=None, min_size=256)
in_features = detector.roi_heads.box_predictor.cls_score.in_features
detector.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=2)
detector.load_state_dict(torch.load(ROOT / "src/faster-rcnn/01-baseline/model.pt", map_location=device))
detector.to(device)
detector.eval()


def detect_boxes(image_id):
    # Faster R-CNN gives (x1, y1, x2, y2); convert to (x, y, w, h) like the labels
    image = Image.open(images_path / f"{image_id}.png").convert("RGB")
    image = torch.tensor(np.array(image), dtype=torch.float32).permute(2, 0, 1) / 255.0
    with torch.no_grad():
        out = detector([image.to(device)])[0]
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


# ---------- route checks ----------

def craters_entered(path, boxes):
    # indices of true craters whose circle contains at least one path cell (as in 08-rover-1.ipynb)
    ys = np.array([p[0] for p in path]) + 0.5
    xs = np.array([p[1] for p in path]) + 0.5
    return [i for i, (x, y, w, h) in enumerate(boxes)
            if np.any((xs - x - w / 2) ** 2 + (ys - y - h / 2) ** 2 <= (max(w, h) / 2) ** 2)]


def longest_chord(boxes, start, goal):
    # longest piece (pixels) of the straight start-goal segment inside one crater circle of radius >= MIN_RADIUS
    a = np.array([start[1], start[0]], dtype=float) + 0.5      # (x, y), cell centres
    b = np.array([goal[1], goal[0]], dtype=float) + 0.5
    d = b - a
    seg = np.linalg.norm(d)
    u = d / seg
    best = 0.0
    for x, y, w, h in boxes:
        r = max(w, h) / 2
        if r < MIN_RADIUS:
            continue
        c = np.array([x + w / 2, y + h / 2])
        t = np.dot(c - a, u)                                   # foot of the perpendicular
        dist2 = np.sum((c - a - t * u) ** 2)
        if dist2 >= r ** 2:
            continue
        s = (r ** 2 - dist2) ** 0.5
        best = max(best, min(t + s, seg) - max(t - s, 0.0))
    return best


def clearance(path, boxes):
    # smallest distance (pixels) from any path cell centre to the edge of any true crater circle
    p = np.array(path, dtype=float) + 0.5
    return min((np.hypot(p[:, 1] - x - w / 2, p[:, 0] - y - h / 2) - max(w, h) / 2).min()
               for x, y, w, h in boxes)


def bend(path, start, goal):
    # largest distance (pixels) of any path cell from the straight start-goal line
    p = np.array(path, dtype=float)
    a, b = np.array(start, dtype=float), np.array(goal, dtype=float)
    d = b - a
    return np.abs(d[0] * (p[:, 1] - a[1]) - d[1] * (p[:, 0] - a[0])).max() / np.linalg.norm(d)


def pick_routes():
    rover1 = json.loads((ROOT / "src/rover-1/01-faster-rcnn/results.json").read_text())
    rover2 = json.loads((ROOT / "src/rover-2/01-basic-mlp/results.json").read_text())

    # every route that passes all checks, for each rover
    candidates = {1: [], 2: []}
    for r1, r2 in zip(rover1, rover2):
        assert r1["image_id"] == r2["image_id"] and r1["start"] == r2["start"]
        image_id, start, goal = r1["image_id"], tuple(r1["start"]), tuple(r1["goal"])
        if image_id in EXCLUDE or not all(EDGE <= v <= 255 - EDGE for v in start + goal):
            continue
        boxes = true_boxes(image_id)
        if clearance([start, goal], boxes) < MARKER_CLEAR:
            continue
        chord = longest_chord(boxes, start, goal)
        if chord < MIN_CHORD:
            continue

        wanted = [k for k, r in ((1, r1), (2, r2)) if r["reached"] and r["entered"] == 0]
        if not wanted:
            continue
        detected_map = make_mask(detect_boxes(image_id))

        for k in wanted:
            if k == 1:
                path, reached, saved = astar(detected_map, start, goal), True, r1
            else:
                (path, reached), saved = drive(detected_map, start, goal), r2
            if path is None or not reached or craters_entered(path, boxes):
                continue
            if bend(path, start, goal) < MIN_BEND or clearance(path, boxes) < MIN_CLEAR:
                continue
            # draw only paths that reproduce the saved result exactly (a few Rover 2 runs differ
            # slightly on CPU, where the MLP's best move can flip on near ties)
            if abs(path_length(path) - saved["length"]) > 1e-6:
                print(f"skip Rover {k} image {image_id}: path differs from saved result")
                continue
            candidates[k].append(dict(image_id=image_id, start=start, goal=goal, path=path, boxes=boxes,
                                      chord=chord, bend=bend(path, start, goal)))

    # clearest examples first: both the crater crossing and the detour should be large.
    # Rover 2 has fewer candidates, so it chooses first; each image is used in only one panel.
    chosen = {1: [], 2: []}
    used = set()
    for k in (2, 1):
        for c in sorted(candidates[k], key=lambda c: -min(c["chord"], c["bend"])):
            if len(chosen[k]) < N_PANELS and c["image_id"] not in used:
                chosen[k].append(c)
                used.add(c["image_id"])
        if len(chosen[k]) < N_PANELS:
            raise SystemExit(f"only {len(chosen[k])} routes matched for Rover {k}")
    return chosen


# ---------- figure ----------

def main():
    chosen = pick_routes()
    for k in (1, 2):
        for c in chosen[k]:
            print(f"Rover {k}: image {c['image_id']}, start {c['start']}, goal {c['goal']}, "
                  f"line inside crater {c['chord']:.0f} px, bend {c['bend']:.0f} px, "
                  f"length {path_length(c['path']):.1f}")

    plt.rcParams.update({
        "font.size": 8.5,
        "pdf.fonttype": 42,
        "savefig.facecolor": "white",
    })

    W = 7.0                  # figure width, inches
    P = 2.1                  # panel size
    gap = 0.12               # between panels
    x0 = 0.42                # room for the row labels
    y_top = 0.08
    H = y_top + 2 * P + gap + 0.08

    crater_color = "#333333"
    color = {1: "#0072B2", 2: "#D55E00"}     # Rover 1: solid blue, Rover 2: dashed vermillion
    dashes = {1: None, 2: (3, 1.6)}
    row_label = {1: "Rover 1: A*", 2: "Rover 2: MLP"}

    fig = plt.figure(figsize=(W, H), facecolor="white")
    x_left = x0 + (W - x0 - (3 * P + 2 * gap)) / 2

    for row, k in enumerate((1, 2)):
        y = H - y_top - (row + 1) * P - row * gap
        for col, c in enumerate(chosen[k]):
            ax = fig.add_axes([(x_left + col * (P + gap)) / W, y / H, P / W, P / H])
            ax.set_xticks([])
            ax.set_yticks([])
            for s in ax.spines.values():
                s.set_linewidth(0.5)

            # faded orbital image: shown as stored, then washed towards white
            image = np.array(Image.open(images_path / f"{c['image_id']}.png").convert("L"))
            ax.imshow(image, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
            ax.imshow(np.ones_like(image), cmap="gray", vmin=0, vmax=1, alpha=0.55, interpolation="nearest")

            # true craters (pixel (r, c) has its centre at (c, r) in imshow coordinates, so shift by -0.5)
            for bx, by, bw, bh in c["boxes"]:
                ax.add_patch(Circle((bx + bw / 2 - 0.5, by + bh / 2 - 0.5), max(bw, bh) / 2, fill=False,
                                    edgecolor=crater_color, lw=0.6))

            # straight start-goal line, for reference
            (sy, sx), (gy, gx) = c["start"], c["goal"]
            ax.plot([sx, gx], [sy, gy], color="black", lw=0.6, ls=(0, (1, 1.5)))

            ys, xs = zip(*c["path"])
            line, = ax.plot(xs, ys, color=color[k], lw=1.4, solid_capstyle="round")
            if dashes[k]:
                line.set_dashes(dashes[k])
            ax.scatter(sx, sy, s=26, facecolor="white", edgecolor="black", lw=0.8, zorder=3)
            ax.scatter(gx, gy, s=80, marker="*", facecolor="#F0E442", edgecolor="black", lw=0.6, zorder=3)
            ax.set_xlim(-0.5, 255.5)
            ax.set_ylim(255.5, -0.5)

        fig.text((x_left - 0.1) / W, (y + P / 2) / H, row_label[k], rotation=90, ha="right", va="center")

    fig.savefig(OUT_DIR / "rover_routes.pdf")
    fig.savefig(OUT_DIR / "rover_routes.png", dpi=300)
    print("saved", OUT_DIR / "rover_routes.pdf", "and", OUT_DIR / "rover_routes.png")


if __name__ == "__main__":
    main()

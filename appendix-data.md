# Appendix data

Collected from the notebooks, run folders (`src/`) and git history.

## Issues to resolve before use

1. **Faster R-CNN has two different test results.** `src/faster-rcnn/01-baseline/test_scores.json` (same as HEAD, old Python 3.12 run) says P 0.7949 / R 0.8029 / F1 0.7989. The current output in `10-test-scores.ipynb` (new MPS run) says P 0.7892 / R 0.8047 / F1 0.7969, and its final table uses these. Pick one and use it everywhere. U-Net and DINOv3 test results are the same in both runs.
2. **U-Net 1 "best epoch 9" is wrong.** Run 1 overwrote `unet.pt` every epoch, so the saved weights are from **epoch 10**. Epoch 10 has val IoU 0.3742 at cutoff 0.5 and 0.5044 at 0.8. Its crater scores also come from epoch 10. `src/comparison/pixel_iou.csv` reports epoch 9 (0.4495).
3. **Val and test pixel IoU use different cutoffs.** For U-Net 2 and DINOv3, val IoU is at 0.8 and test IoU is at 0.5 and 0.9. Val IoU at 0.5 and test IoU at 0.8 were never saved. MLP and CNN were never scored per crater on validation.

## 1. Training settings

Split for all models: 41,470 train / 5,184 val / 5,184 test (`train_test_split`, `random_state=0`). No data augmentation in any model.

| Model | Optimizer | LR | Momentum | Weight decay | Schedule | Epochs | Batch | Seeds | Input |
|---|---|---|---|---|---|---|---|---|---|
| MLP | Adam | 1e-3 | default betas | 0 | constant | 10 | 16 | torch 0, random 0 | 256×256, 1-channel gray |
| CNN | Adam | 1e-3 | default betas | 0 | constant | 10 | 16 | torch 0, random 0 | 256×256, 1-channel gray |
| U-Net 1 | Adam | 1e-3 | default betas | 0 | constant | 10 | 16 | torch 0, random 0 | 256×256, 1-channel gray |
| U-Net 2 | Adam | 1e-3 | default betas | 0 | cosine, T_max=15 | 15 | 16 | torch 0, random 0 | 256×256, 1-channel gray |
| DINOv3 head | Adam | 1e-3 | default betas | 0 | cosine, T_max=15 | 15 | 64 | no torch seed (split only) | 256×256 RGB, ImageNet normalization → 16×16 patch tokens |
| Faster R-CNN | SGD | 0.005 | 0.9 | 5e-4 | cosine, T_max=15 | 15 | 8 | torch 0, random 0 | RGB, `min_size=256` |
| Rover 2 MLP | Adam | 1e-3 | default betas | 0 | constant | 10 | 256 | torch 0, numpy rng 0 | 227-dim vector |

All mask models use `BCEWithLogitsLoss` with `pos_weight=10`.

## 2. Model selection

| Model | Selected on | Best epoch | Pixel cutoff | Crater cutoff | Cutoffs tried |
|---|---|---|---|---|---|
| MLP | val IoU | 6 (0.1814) | 0.5 | 0.5 (not tuned) | 0.5 |
| CNN | val IoU | 7 (0.2207) | 0.5 | 0.5 (not tuned) | 0.5 |
| U-Net 1 | none (last epoch kept) | 10 (0.3742 @0.5) | 0.8 (0.5044), chosen afterwards | 0.9 | 0.5–0.9 |
| U-Net 2 | val IoU | 10 (0.5188) | 0.8 | 0.9 | 0.5–0.9 |
| DINOv3 | val IoU | 13 (0.4975) | 0.8 | 0.9 | 0.5–0.9 |
| Faster R-CNN | val F1 | 15 (0.7954) | – | 0.8 | 0.3, 0.5, 0.7, 0.8, 0.9 |

Validation crater F1 at each cutoff (from `crater_scores.json`):

| Cutoff | U-Net 1 | U-Net 2 | DINOv3 |
|---|---|---|---|
| 0.5 | 0.1878 | 0.2391 | 0.1043 |
| 0.6 | 0.2781 | 0.3312 | 0.1369 |
| 0.7 | 0.3908 | 0.4190 | 0.1730 |
| 0.8 | 0.4698 | 0.5172 | 0.2048 |
| 0.9 | **0.4896** | **0.5818** | **0.2143** |

Faster R-CNN only saved the best cutoff per epoch (0.8 in 11 of 15 epochs, 0.9 in 3, 0.7 in 1), not the F1 at every cutoff.

## 3. U-Net run 1 vs run 2

Architecture, Adam learning rate, `pos_weight`, batch size and seed are the same in both runs.

| | Run 1 | Run 2 |
|---|---|---|
| LR schedule | constant | cosine to 0 |
| Epochs | 10 | 15 |
| Checkpoint kept | last epoch (overwritten each epoch) | best val IoU |
| Cutoff during training | fixed 0.5 | best of 0.5–0.9, every epoch |
| Final train loss | 0.2243 | 0.1503 |
| Val IoU (kept model) | 0.3742 @0.5 / 0.5044 @0.8 | 0.5188 @0.8 |
| Val crater P / R / F1 @0.9 | 0.4186 / 0.5897 / 0.4896 | 0.4980 / 0.6996 / 0.5818 |
| Test | not run | see section 4 |

## 4. Validation vs test

| Model | Cutoff | Val P / R / F1 | Test P / R / F1 | Val IoU | Test IoU @0.5 | Test IoU @0.9 |
|---|---|---|---|---|---|---|
| MLP | 0.5 | – | 0.0669 / 0.3163 / 0.1105 | 0.1814 @0.5 | 0.1758 | – |
| CNN | 0.5 | – | 0.0312 / 0.6338 / 0.0594 | 0.2207 @0.5 | 0.2153 | – |
| U-Net 2 | 0.9 | 0.4980 / 0.6996 / 0.5818 | 0.4813 / 0.7028 / 0.5713 | 0.5188 @0.8 | 0.4143 | 0.4880 |
| DINOv3 | 0.9 | 0.3112 / 0.1634 / 0.2143 | 0.3061 / 0.1581 / 0.2085 | 0.4975 @0.8 | 0.4142 | 0.4848 |
| Faster R-CNN | 0.8 | 0.7935 / 0.7974 / 0.7954 | 0.7949 / 0.8029 / 0.7989 (JSON), or 0.7892 / 0.8047 / 0.7969 (notebook) | – | – | – |

A predicted crater counts as found when its box IoU with a true crater is at least 0.5; each true crater can be matched once. Mask models turn each connected blob of pixels into one box. Pixel IoU is total overlap divided by total union over all images.

## 5. Rover 2

| Item | Value |
|---|---|
| Model | MLP 227 → 128 (ReLU) → 8, 30,216 parameters |
| Input | 15×15 window of the Faster R-CNN detected map (cutoff 0.8, outside the map counts as crater) + 2-number unit direction to goal |
| Labels | A* moves on the true crater map (behaviour cloning, no DAgger) |
| Training data | 88,252 train examples from 500 train routes; 18,409 val examples from 100 val routes |
| Routes | random free start/goal pairs at least 150 cells apart, 8-way moves |
| Val accuracy | best 0.7528 at epoch 1, which is the saved model (range 0.747–0.753 over 10 epochs) |
| Reached goal | within 2 cells of the goal in both directions (Chebyshev distance ≤ 2) |
| Step limit | `int(3 × octile distance(start, goal))` steps, then the run counts as lost |
| Off-map move | rover stays in place; the step still counts |
| Test | the same 1,000 test routes as Rover 1 |

## Optional additions

### A. Rover 1 vs Rover 2 per-route results

Same 1,000 test routes (`results.json` in each run folder).

| | Rover 1 (A* on detected map) | Rover 2 (MLP) |
|---|---|---|
| Reached goal | 100% | 100% (none lost) |
| Runs that entered a crater | 93 (9.3%) | 112 (11.2%) |
| Craters entered / missed by detector | 100 / 32 | 117 / 23 |
| Path length, mean / median | 209.3 / 205.4 | 207.0 / 203.6 |
| Entered a crater in both / only this rover | 61 / 32 | 61 / 51 |

### B. Rover 2 teacher move counts

| Up | Down | Left | Right | Up-left | Up-right | Down-left | Down-right |
|---|---|---|---|---|---|---|---|
| 13,754 | 11,251 | 10,685 | 12,103 | 9,339 | 9,774 | 9,316 | 12,030 |

Always guessing the most common move would score 15.6%; the model gets 75.3% validation accuracy.

### C. Unfinished Faster R-CNN 800 px run

`src/faster-rcnn/02-size800` stopped after 1 epoch: val P 0.7845 / R 0.8086 / F1 0.7964 at cutoff 0.8. The 256 px baseline after epoch 1 had F1 0.7791. Report it as stopped early, not as a result.

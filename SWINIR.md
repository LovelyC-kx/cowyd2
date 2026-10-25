# SwinIR extracted model, matched capacity and common pipeline

## What matches the current task, and what does not

Upstream: https://github.com/JingyunLiang/SwinIR
Pinned commit: `6545850fbf8df298df73d81f3e8cba638787c8bd`.
Network source blob: `461fb354ce5a7614d9ffbfcad4d32a2811134ae4`.
Source: `models/network_swinir.py`; official task configurations:
`main_test_swinir.py::define_model`.

There is NO official pretrained configuration for this experiment's Gaussian
PSF + downsampling + noise THz/DIV2K protocol. Classical SR, real-world GAN SR,
Gaussian denoising and JPEG artifact reduction are distinct tasks. We extract
the official architecture, then train on our SAME declared data pairs. Do not
call an official denoising or RealSR checkpoint an exact match for our task.

Two choices are available, explicitly separated:

| CLI | Official architectural branch | Input/output boundary | Default parameters |
|---|---|---|---:|
| `swinir` | same-size residual reconstruction (`upsampler=''`) | native LR -> shared bicubic -> SwinIR -> HR | 2,248,083 |
| `swinir_sr` | lightweight SR (`upsampler='pixelshuffledirect'`) | native LR -> SwinIR -> Conv+PixelShuffle -> HR | 2,255,868 (x2) |

Use `swinir` for the closest controlled input-boundary comparison with submission
(both refine preupsampled RGB). Use `swinir_sr` for the official learned native-
LR SR design; it avoids computing the entire Transformer on a bicubic HR image.
This boundary difference must be stated; it is a valid complete-system baseline,
not an isolated attention-vs-neuron ablation. Both see the SAME underlying LR/HR
pairs, crop, noise, optimizer, loss and metric definitions.

## Exactly how capacity was reduced

Comparison of configurations, all RGB, window8 and MLP ratio2:

| Setting | Official classical SR | Official color denoising | Our matched models |
|---|---:|---:|---:|
| embedding dimension C | 180 | 180 | **96** |
| residual Swin groups RSTB | 6 | 6 | **4** |
| Swin blocks per group | 6 | 6 | **6** |
| total Swin blocks | 36 | 36 | **24** |
| heads | 6 | 6 | **6** (head dimension16) |
| MLP ratio | 2 | 2 | **2** |
| window / alternating shift | 8 / 4 | 8 / 4 | **unchanged** |
| RSTB residual convolution | 1conv | 1conv | **unchanged** |
| parameters | 11,752,487 (x2) | 11,504,163 | 2,255,868 SR / 2,248,083 restore |

Submission's measured count is 2,302,901. Our counts are **2.04% lower** (native
SR x2) and **2.38% lower** (restore). Counts include all registered parameters.
The official 60-channel four-group lightweight SR is already only 910,152 (x2):
we increase its width to96 for capacity matching; we do NOT claim we shrank that
already smaller model. We shrink relative to the full 180-channel six-group
models and choose the official lightweight direct head for native SR.

Transformer/conv weights are dominated by O(C^2), so width scaling contributes
about (96/180)^2=.284 and group count contributes 4/6. Head reductions alone do
not generally shrink QKV/projection weights at fixed C, so they were not used as
a misleading parameter-reduction knob. We retain W-MSA/SW-MSA, learned relative
position biases, LayerNorm, GELU MLPs, RSTB convolutions, feature residuals and
upstream RGB mean subtraction/addition. No new FFT/attention/SNN modules.

## Files and runtime-only modifications

- `baseline_suite/_swinir_vendor.py`: extracted official model/support classes;
  original author notices retained. Replaced timm helpers by equivalent Torch
  helpers; explicit meshgrid indexing. Apache-2.0 copy in SWINIR_LICENSE.txt.
- `baseline_suite/swinir.py`: configuration/interface wrapper and shared constant
  mask cache. Default stochastic depth0; dropout0, no activation checkpoint
  recomputation, no self-ensemble or pretrained weight download. The upstream
  official test constructor inherits stochastic depth .1; ours explicitly uses
  0 to make speed/reproducibility settings clear. This does not change parameter
  counts and is recorded in model configuration.
- Immutable shifted-window masks shared between blocks, bounded to two cached
  shape/device keys; dynamic masks are not recomputed separately in every block.
  Derived static masks omitted from our checkpoints; learned relative biases are
  retained and loaded strictly. Cached/uncached outputs are numerically checked.
- Tiny-image replication fallback only when upstream reflection padding would
  be invalid. Normal-sized inputs retain reflection padding to window multiples.
  Native SR lets this wrapper handle padding; the same-size comparison retains
  the common external replication-to-8 boundary used by the other models.
- `runtime.py`: supports both same-size and native-SR outputs. Native tiles are
  mapped from HR output tile units to LR units. For x2, tile192/overlap32 means
  LR tiles96/overlap16; x3 needs divisible overlap, e.g. --overlap 30.

Fewer parameters do NOT guarantee equal memory/latency to SNN. Full-resolution
window attention can still be expensive; native-LR x2 has approximately 1/4 as
many trunk spatial tokens as the preupsampled mode, but no GPU speedup factor is
claimed without measurement. Perform target-GPU batch4 smoke checks first.

## Training commands (shared trainer, existing GPU PyTorch)

No timm, torchvision or new dependency installation is required. Update the
whole package, not just models.py. Existing `co_wyd/submission` and its source /
weights are NOT edited. NOTE: old cowyd2 baseline checkpoints enforce source
hashes. Updating these shared files changes those hashes; keep an old checkout
for resuming earlier baseline runs. Do not bypass the old integrity check.

Inspect counts / verify:

```bash
python -m baseline_suite.models
python check_swinir.py
```

Windows commands using the user's existing DIV2K location, from updated project:

```cmd
python train_baseline.py --model swinir --data-root "D:\bohanzhang\data\DIV2K" --device cuda:0 --epochs 1 --batch-size 4 --patch-size 192 --max-train-steps 2 --max-val-images 1 --output runs/check_swinir_matched_x2

python train_baseline.py --model swinir_sr --data-root "D:\bohanzhang\data\DIV2K" --device cuda:0 --epochs 1 --batch-size 4 --patch-size 192 --scale 2 --max-train-steps 2 --max-val-images 1 --output runs/check_swinir_native_x2
```

Formal same-input-boundary comparison:

```cmd
python train_baseline.py --model swinir --data-root "D:\bohanzhang\data\DIV2K" --device cuda:0 --epochs 200 --batch-size 4 --patch-size 192 --scale 2 --lr 5e-6 --val-every 5 --output runs/swinir_matched_x2
```

Formal native-LR learned super-resolution:

```cmd
python train_baseline.py --model swinir_sr --data-root "D:\bohanzhang\data\DIV2K" --device cuda:0 --epochs 200 --batch-size 4 --patch-size 192 --scale 2 --lr 5e-6 --val-every 5 --output runs/swinir_native_x2
```

These are our common Adam/MSE/LR settings, NOT the original SwinIR optimized
training recipe or original scores. The low LR5e-6 was chosen to align with
current SNN runs, not assumed optimal for SwinIR. Monitor convergence / use a
separately declared equal-budget LR search if needed. No official pretrained
checkpoint fits the changed width/depth. Train new weights; do not strict=False
load a large official checkpoint or claim initialized compatibility.

Linux uses identical flags with the actual Linux --data-root path. Run these
two models sequentially unless separate free GPUs are assigned. Existing
run_baselines.py still defaults to the original SIX models. To run these two
sequentially, set --models swinir swinir_sr explicitly.

## Re-evaluation and key inference

```bash
python test_baseline.py --weights runs/swinir_matched_x2/model_best.pth --device cuda:0 --output results/swinir_matched_valid

python infer_baseline_key.py --weights runs/swinir_matched_x2/model_best.pth --matrix yaoshi.xls --opju yaoshi.opju --scales 1 2 3 4 --device cuda:0 --output results/key_swinir_matched_v1

python infer_baseline_key.py --weights runs/swinir_native_x2/model_best.pth --matrix yaoshi.xls --opju yaoshi.opju --scales 2 --device cuda:0 --output results/key_swinir_native_x2_v1
```

Native-SR heads are SCALE-SPECIFIC: an x2 checkpoint cannot run x1/x3/x4.
Train separate x3/x4 runs and scale-specific heads if required. Restore mode can
run all external interpolation scales, but nontrained scales are exploratory.
Both exports remain compatible with plot_key_results.py / make_key_paper_figures.py.
Direct measured-key inference still has no independent HR reference or valid
reference-based quality metrics. Default key inference is full-frame, not tiled.

## Executed checks

`swinir_validation.json`: CPU forward/backward at default 2.25M capacity, window
partition/reverse, upstream extracted-branch equivalence, cache numerical
equivalence, bounded cache, odd/tiny sizes, batched tiling, native scales2/3/4,
wrong-scale rejection, both shared train/val/strict reload/resume paths. Existing
six-baseline regression checks are rerun. Target-GPU memory/AMP/latency and trained
accuracy remain unmeasured; synthetic smoke outputs are not scientific results.

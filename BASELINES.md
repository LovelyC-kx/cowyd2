# Six explicitly specified RGB reconstruction baselines

Clean-room PyTorch implementations; no external model downloads, no SNN/FFT/
attention/auxiliary heads, no ensembles and no training-time image exports.
Existing cloud/SNN training and inference files are NOT modified.

## Architecture and measured parameter counts (RGB)

| CLI | Implemented variant | Parameters |
|---|---|---:|
| srcnn | 9x9 3->64, ReLU, 1x1 64->32, ReLU, 5x5 32->3; same padding | 20,099 |
| vdsr | 20 convolutions, width64, no convolution biases, global image residual | 667,008 |
| edsr | EDSR-inspired preupsampled adaptation, 16 blocks x64, no BN/post-add ReLU; res_scale=1 | 1,222,147 |
| crnn | **CRNN-Restore custom ConvGRU**, width32, 4 shared iterations at half resolution | 66,403 |
| resnet | ResNet-Restore, 8 blocks x64, post-add ReLU, no BN/classifier, image residual | 594,371 |
| unet | U-Net-Restore, widths32/64/128/256, three pooling stages, bilinear+1x1 upsampling | 1,796,643 |

All are scale-independent SAME-SIZE BCHW RGB networks. Shared bicubic preparation
upsamples native LR before network evaluation, matching the current co_wyd SNN
boundary. x1 cloud restoration bypasses bicubic. All models use replication
padding to multiples of 8 and crop afterward. U-Net has concatenation skips.

### Sources and deviations (do not label these exact original-paper reproductions)

- SRCNN: Dong et al., Image Super-Resolution Using Deep Convolutional Networks,
  https://arxiv.org/abs/1501.00092 . Basic 9-1-5 topology, RGB and same padding here.
- VDSR: Kim et al., CVPR 2016,
  https://cv.snu.ac.kr/research/VDSR/ and
  https://openaccess.thecvf.com/content_cvpr_2016/papers/Kim_Accurate_Image_Super-Resolution_CVPR_2016_paper.pdf .
  Twenty-layer residual mapping, adapted from Y to RGB; shared Adam recipe is not
  original high-LR SGD/adjustable clipping training.
- EDSR: Lim et al., CVPRW 2017; author implementation
  https://github.com/sanghyun-son/EDSR-PyTorch/blob/master/src/model/edsr.py .
  Use the 16x64 BASELINE capacity. To share the existing restoration interface,
  replace native-LR learned PixelShuffle tail with common pre-bicubic input and
  same-size RGB residual output; omit fixed mean shifts. This changes compute and
  architecture and is named an adaptation, not canonical native-LR EDSR or the
  large 32x256 EDSR. No official pretrained weights are compatible.
- ResNet: He et al., CVPR 2016,
  https://openaccess.thecvf.com/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html .
  Restoration-specific constant-width residual blocks, no BN or classification
  stages; NOT ImageNet ResNet-18/34 and NOT the THz-Net residual architecture.
- U-Net: Ronneberger et al., MICCAI 2015,
  https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/ . Adapted for RGB
  regression, width32 and three downsampling stages, same padding, bilinear
  decoder, image residual; not original width64 four-pool valid-convolution
  segmentation U-Net. Removing segmentation heads is required for regression.
- CRNN warning: THz-Net cites Shi/Bai/Yao,
  https://arxiv.org/abs/1507.05717 , an OCR CNN+BiLSTM+CTC network. It does not
  define an image-to-image reconstruction baseline. We cannot reproduce that
  paper's unspecified reconstruction CRNN. `crnn` here is explicitly CUSTOM
  ConvGRU restoration, NOT that OCR architecture and not a reproduction of its
  reported THz-Net table entry. Gates use shared 3x3 convolutions; state resets
  to zero for each independent image; no recurrence across unrelated batches.

## Shared data/loss/metrics

Default DIV2K protocol is ported from LovelyC-kx/co_wyd dataset_load.py,
thz_protocol.py, metrics.py (main at implementation time). Deterministic crop and
noise per (seed, epoch, image ID) ensure switching models cannot change pairs.
Gaussian PSF sigma=1.6, kernel21, antialiased bicubic downsampling, Gaussian
noise std=.01, no LR clipping or augmentation. These are DECLARED choices, not
unpublished THz-Net author settings. Train800/valid100 IDs are checked; partial
sets/step limits are marked as smoke tests. Fingerprints separate data from epoch
budget so extending training does not falsely change data identity.

All six use RGB MSE without clamping during training, Adam betas(.9,.999), eps1e-8,
no decay, fixed LR5e-6 by default. Default batch4, HR patch192, seed1234. This
matches your current DIV2K SNN recipe; it is NOT every original model's optimized
recipe. Low LR can converge slowly; inspect curves or conduct a separate,
declared equally-budgeted LR search rather than calling undertrained baselines
inferior. No automatic early stop, scheduler or claim that 200 epochs converges.

Evaluation uses the SAME studio-range BT.601 Y conversion, PSNR data_range=1,
11x11 sigma1.5 valid-window SSIM, border0 and prediction clamp as co_wyd. Do not
confuse it with full-range grayscale used in key previews. Full validation uses
native HR dimensions, tiled192/overlap32 uniform averaging as before; tile_batch4
batches independent tiles for efficiency (tested against tile_batch1). No center
crop/first-image shortcut in formal runs. Tile batch is not training batch.

Validation is every5 epochs and final epoch; train/val times are separated,
training progress every25 batches and validation progress every10 images. No
metrics on nonvalidation epochs. Best is selected only on validation epochs.
Comparing to SNN validated every epoch entails a different selection cadence;
state it or equalize cadence/re-evaluate matched snapshots. The 100 images select
best checkpoints, so scores are VALIDATION scores, not an independent test set.

## Install and inspect

```bash
cd /root/cowyd2-main
python -m pip install numpy Pillow scipy
python -m baseline_suite.models
python check_baselines.py
```

Reuse existing GPU PyTorch. Do not reinstall it merely to get these baselines.
No spikingjelly/timm/torchvision dependency. CPU tests were performed here; CUDA
speed/memory/AMP must be verified on the target GPU. No invented speedup factor.

## One shared smoke test / formal sequential run

Use your ACTUAL DIV2K location; /root/autodl-tmp/DIV2K below is only an example.
It must contain DIV2K_train_HR and DIV2K_valid_HR directories.

```bash
python train_baseline.py --model srcnn --data-root /root/autodl-tmp/DIV2K --device cuda:0 --epochs 1 --batch-size 4 --patch-size 192 --max-train-steps 2 --max-val-images 1 --output runs/baseline_smoke/srcnn

python run_baselines.py --output-root runs/baselines_div2k_x2 --data-root /root/autodl-tmp/DIV2K --device cuda:0 --epochs 200 --batch-size 4 --patch-size 192 --scale 2 --lr 5e-6 --val-every 5
```

Runner executes the six SEQUENTIALLY, not all on one GPU at once. Stop on any
failure. Specify `--models srcnn vdsr` to restrict. Output names are fixed model
names inside output-root. Use a fresh output-root unless resuming. To resume the
batch of runs, repeat the runner with --resume-existing and identical settings;
completed runs are skipped. Never concurrently write the same output directory.

Windows (existing environment / actual project location):

```cmd
python train_baseline.py --model edsr --data-root "D:\bohanzhang\data\DIV2K" --device cuda:0 --epochs 200 --batch-size 4 --patch-size 192 --lr 5e-6 --val-every 5 --output runs/edsr_div2k_x2
```

These files must first be uploaded to that Windows project, not just AutoDL.

## Checkpoints and resumption

model_last.pth is atomically replaced each epoch; model_best.pth after an improved
validation score; model_epoch_0010.pth etc every10 epochs. Includes model,
optimizer, AMP scaler, Python/NumPy/Torch/selected-GPU RNG, epoch and phase.
Epoch data order/crops are deterministic. No resume within a batch: an unsaved
partial epoch rolls back and repeats. BEFORE expensive validation, save trained
epoch with phase=validation_pending; if interrupted there, resume validates it
without retraining it. Complete checkpoint is saved before JSONL append; resume
can repair the final missing log row. Atomic replacement protects an old file
from interrupted serialization but cannot guarantee survival of all storage/
power failures. Keep historical snapshots and never run two writers together.

```bash
python train_baseline.py --model edsr --data-root /root/autodl-tmp/DIV2K --device cuda:0 --epochs 200 --batch-size 4 --patch-size 192 --output runs/baselines_div2k_x2/edsr --resume runs/baselines_div2k_x2/edsr/model_last.pth
```

Total epochs can extend; validation/save/log frequency, threads, worker count and
device can change, with invocation recorded in events.jsonl. Model, data, LR,
batch, precision, crop, source and tile evaluation settings must match. CPU tests
confirm bitwise resume equivalence; bitwise CUDA equivalence depends on kernels
and environment. New checkpoints are not drop-in resumes of old SNN checkpoints.

FP32 deterministic cuDNN default. Optional --precision fp16/bf16 and --fast-cudnn
are explicit speed experiments, recorded in config, not silently enabled. AMP
training still evaluates in FP32. Test for nonfinite losses and validation drift.
Do not compare elapsed times from mixed precision/settings without disclosing.

## Validation re-evaluation and key inference

```bash
python test_baseline.py --weights runs/baselines_div2k_x2/edsr/model_best.pth --device cuda:0 --output results/edsr_div2k_valid --save-images

python infer_baseline_key.py --weights runs/baselines_div2k_x2/edsr/model_best.pth --matrix yaoshi.xls --opju yaoshi.opju --scales 1 2 3 4 --device cuda:0 --output results/key_edsr_div2k_v1

python plot_key_results.py --root results
```

The key report/file schema is compatible with existing plot_key_results.py and
make_key_paper_figures.py; include new run folders in --runs. No retraining for
plotting. Default key inference is full-frame (like the legacy key script), not
validation tiling. Nontrained scales are exploratory scale transfer. The output
is not measured HR truth; no PSNR/SSIM without an independent reference.

## Optional paired cloud-removal training

Same trainer, MSE, metrics and model boundary; scale1. Supply disjoint folders
explicitly; NO automatic train=test fallback or pair-by-order matching.

```bash
python train_baseline.py --model unet --task paired --train-input /data/CR1/train/input --train-target /data/CR1/train/target --val-input /data/CR1/valid/input --val-target /data/CR1/valid/target --scale 1 --epochs 200 --batch-size 4 --patch-size 192 --lr 5e-6 --output runs/unet_cr1
```

Paths above are placeholders. Split selection must match an intended comparison.
Old centralized A2 training used Charbonnier+SSIM and potentially different
train/test crops/splits; these new MSE baselines cannot be called a controlled
comparison against old A2 numbers without aligning those conditions/retraining.

## Validation evidence

`baseline_validation.json` records executed CPU checks: every model's odd-size
forward/backward, no recurrence leakage, tile-batch equivalence, deterministic
data, all-six smoke train/validation/strict reload, exact resume after extending
epochs, validation-interruption recovery and atomic-save failure preservation.
Weights used for these checks are temporary synthetic-data smoke weights, not
trained scientific results. No model-quality claims are derived from them.

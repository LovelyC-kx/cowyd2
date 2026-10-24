# Key matrix inference (legacy cloud checkpoints)

This entry point does not alter training or model definitions. It loads all model
weights strictly. A mismatch is an error, not silently ignored parameters.

## Input and dependencies

`yaoshi.xls` in this repository is header-free tab-separated text, not a binary
Excel workbook. It is read with NumPy. Real .xls/.xlsx workbooks additionally need
`pandas xlrd openpyxl`. OPJU is optional provenance; it is not parsed for numeric
data and Linux Origin automation is not required. No screenshot is used as input.

Use your existing PyTorch/SpikingJelly environment. Minimal packages:

```bash
python -m pip install numpy Pillow spikingjelly==0.0.0.0.14 einops
python infer_key.py --matrix yaoshi.xls --opju yaoshi.opju --inspect-only --output results/key_input_v1
```

Inspect input_preview.png and report.json before interpreting results. Default
mapping is input minmax; use --polarity negative only if the signal interpretation
requires reversed contrast. --low and --high provide explicit fixed bounds.

## Small model: centralized full SNN A2

```bash
python infer_key.py --model small --weights /root/globecom20260503-main/Outputs_v1/centralized_A2_vlif_cr2_best.pt --matrix yaoshi.xls --opju yaoshi.opju --scales 1 2 3 4 --device cuda:0 --output results/key_small_cr2_v1
```

The centralized checkpoint stores state_dict and config. Width and active block
counts are inferred from tensor keys; T and BN variant use checkpoint config.
ANN/ablation checkpoints are rejected. The final checkpoint is an optional second
comparison, not a reason to select the visually most attractive result.

## Original large model: use its original project

The original origin_decloud-main model is not included in this repository.
Do not substitute Spiking_Models/VLIFNet, which contains a different modified
architecture. Import the ORIGINAL source directory instead:

```bash
python infer_key.py --model external --model-dir /root/autodl-tmp/origin_decloud-main --factory model:model --weights /root/autodl-tmp/origin_decloud-main/checkpoints/VLIFNet/models/CUHK-CR2-exp1/model_best.pth --matrix yaoshi.xls --opju yaoshi.opju --scales 1 2 3 4 --device cuda:0 --output results/key_large_cr2_v1
```

This command assumes the original model:model factory matches its checkpoint.
If it requires constructor options, pass --model-kwargs with a JSON object. A
strict mismatch or import error must be resolved against the original training
source; do not use strict=False. CR1 model_best.pth is another named-domain
comparison, not automatically interchangeable with CR2.

## Outputs and interpretation

For a 101x101 input, x1/x2/x3/x4 produce 101/202/303/404 square outputs.
Scaling is bicubic BEFORE a same-size restoration model. It is zero-shot task
transfer, not learned scale-specific super-resolution and not proof of improved
physical resolution. x1's bicubic baseline is the original normalized input.

Each scale exports restored_preview.png, bicubic_preview.png, restored_rgb.npy,
restored_normalized.npy/.tsv, restored_signal.npy/.tsv, bicubic_signal.npy/.tsv.
Previews clip to [0,1] using the SAME input mapping, with no per-output contrast
stretch. Floating outputs retain overshoot. Inverse mapping is a numerical unit
conversion, not physical calibration. No PSNR/SSIM without independent HR truth.
report.json records hashes, input mapping, model, scales, padding and overshoot.

Inference uses eval + no_grad, resetting SNN state per scale, replication padding
to a multiple of 8 and cropping afterward. Full-frame FFT preserves the intended
frequency context; no tiling is applied. On OOM, use a free GPU or run fewer/lower
scales first. Outputs are protected from overwrite; choose a fresh directory.

Validation here: syntax, matrix parsing and inspect-only path. Real checkpoint
loading/GPU inference require the user's private checkpoint and GPU environment;
the checkpoint files themselves were not available during implementation.

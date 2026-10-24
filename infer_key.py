"""Zero-shot key reconstruction with legacy cloud-removal checkpoints."""
import argparse
import hashlib
import importlib
import inspect
import json
from collections.abc import Mapping
from pathlib import Path
import sys

import numpy as np
from PIL import Image


def read_matrix(path):
    if path.suffix.lower() == '.npy':
        a = np.load(path, allow_pickle=False)
    else:
        magic = path.read_bytes()[:8]
        if magic.startswith(b'PK') or magic.startswith(bytes.fromhex('d0cf11e0')):
            import pandas as pd
            a = pd.read_excel(path, header=None).to_numpy(dtype=float)
        else:
            a = np.loadtxt(path, delimiter=',' if path.suffix.lower() == '.csv' else None)
    a = np.asarray(a, dtype=np.float64)
    if a.ndim != 2 or min(a.shape) < 2 or not np.isfinite(a).all():
        raise ValueError('Expected a finite, header-free 2-D numeric matrix')
    return a


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def export(folder, name, a):
    np.save(folder / (name + '.npy'), a)
    np.savetxt(folder / (name + '.tsv'), a, delimiter='\t', fmt='%.17g')


def preview(path, a):
    Image.fromarray(np.rint(np.clip(a, 0, 1) * 255).astype(np.uint8)).save(path)


def extract_state(checkpoint, torch):
    state = checkpoint
    for _ in range(4):
        if isinstance(state, Mapping) and state and all(torch.is_tensor(v) for v in state.values()):
            break
        if not isinstance(state, Mapping):
            raise ValueError('Checkpoint is not a state dictionary')
        key = next((k for k in ('state_dict', 'model_state_dict', 'model', 'net')
                    if isinstance(state.get(k), Mapping)), None)
        if key is None:
            raise ValueError('Cannot locate weights; checkpoint keys: ' + str(list(state)[:30]))
        state = state[key]
    if not state or not all(torch.is_tensor(v) for v in state.values()):
        raise ValueError('Invalid state dictionary')
    state = dict(state)
    if all(k.startswith('module.') for k in state):
        state = {k[7:]: v for k, v in state.items()}
    return {k: v for k, v in state.items()
            if k.split('.')[-1] not in ('total_ops', 'total_params')}


def make_model(args, checkpoint, state, torch):
    from spikingjelly.activation_based import functional
    if args.model == 'small':
        from cloud_removal_v1.models.vlifnet import build_vlifnet
        cfg = checkpoint.get('config', {})
        if not isinstance(cfg, Mapping):
            raise ValueError('Checkpoint config must be a dictionary')
        if cfg.get('ablation', 'none') != 'none' or cfg.get('backbone', 'vlif') != 'vlif':
            raise ValueError('Use a full SNN checkpoint, not an ANN or ablation checkpoint')
        def blocks(prefix):
            indices = [int(k.split('.')[1]) for k in state if k.startswith(prefix + '.')]
            return max(indices) + 1 if indices else 0
        dim = state['patch_embed.proj.weight'].shape[0]
        en = [2, blocks('encoder_level2'), blocks('encoder_level3'), 4]
        de = [2, blocks('decoder_level2'), blocks('decoder_level3'), 2]
        model = build_vlifnet(dim=dim, en_num_blocks=en, de_num_blocks=de,
                             T=cfg.get('T', 4), backend='torch',
                             bn_variant=cfg.get('bn_variant', 'tdbn'), backbone='snn',
                             use_refinement=any(k.startswith('refinement_blocks.') for k in state))
        spec = dict(model='small', dim=dim, en_blocks=en, de_blocks=de)
    else:
        if args.model_dir is None:
            raise ValueError('--model-dir must point to the ORIGINAL large-model project')
        sys.path.insert(0, str(args.model_dir.resolve()))
        module, name = args.factory.split(':')
        imported = importlib.import_module(module)
        from legacy_key_compat import prepare_original
        prepare_original(imported, state, args.legacy_attention_output)
        factory = getattr(imported, name)
        kwargs = json.loads(args.model_kwargs)
        model = factory(**kwargs)
        spec = dict(model='external', factory=args.factory, kwargs=kwargs,
                    attention_output=args.legacy_attention_output,
                    legacy_repairs=['module registration', 'BN dimension check'],
                    exact_training_environment_verified=False)
    if not isinstance(model, torch.nn.Module):
        raise TypeError('Factory must return torch.nn.Module')
    model.load_state_dict(state, strict=True)
    model.to(args.device).eval()
    functional.set_backend(model, backend='torch')
    functional.set_step_mode(model, step_mode='m')
    spec['source'] = inspect.getfile(type(model))
    spec['source_sha256'] = sha(spec['source'])
    spec['parameters'] = sum(p.numel() for p in model.parameters())
    return model, spec


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--matrix', type=Path, default=Path(__file__).with_name('yaoshi.xls'))
    p.add_argument('--opju', type=Path, help='Optional provenance only; numeric input comes from --matrix')
    p.add_argument('--weights', type=Path)
    p.add_argument('--model', choices=['small', 'external'], default='small')
    p.add_argument('--model-dir', type=Path)
    p.add_argument('--factory', default='model:model')
    p.add_argument('--legacy-attention-output', choices=['relu', 'identity'],
                   help='Explicit original attention final activation; cannot be inferred from weights')
    p.add_argument('--model-kwargs', default='{}')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--scales', nargs='+', type=int, choices=[1, 2, 3, 4], default=[1, 2, 3, 4])
    p.add_argument('--polarity', choices=['positive', 'negative'], default='positive')
    p.add_argument('--low', type=float)
    p.add_argument('--high', type=float)
    p.add_argument('--inspect-only', action='store_true')
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        p.error('Output directory is nonempty; choose a fresh output')
    raw = read_matrix(args.matrix)
    if (args.low is None) != (args.high is None):
        p.error('Specify both --low and --high, or neither for minmax')
    low, high = (raw.min(), raw.max()) if args.low is None else (args.low, args.high)
    if not np.isfinite([low, high]).all() or high <= low:
        p.error('Mapping bounds must be finite and high > low')
    x = (raw - low) / (high - low)
    clipped = int(np.count_nonzero((x < 0) | (x > 1)))
    x = np.clip(x, 0, 1)
    if args.polarity == 'negative':
        x = 1 - x
    report = dict(matrix=str(args.matrix.resolve()), matrix_sha256=sha(args.matrix),
                  input_shape=list(raw.shape), low=float(low), high=float(high),
                  polarity=args.polarity, clipped_input_count=clipped,
                  interpretation='Zero-shot cloud-removal transfer; interpolation does not prove physical super-resolution',
                  reference_available=False, scales={})
    if args.opju:
        report['opju'] = dict(path=str(args.opju.resolve()), sha256=sha(args.opju))
    model = None
    if not args.inspect_only:
        if not args.weights:
            p.error('--weights is required for inference')
        import torch
        import torch.nn.functional as F
        from spikingjelly.activation_based import functional
        checkpoint = torch.load(args.weights, map_location='cpu', weights_only=True)
        state = extract_state(checkpoint, torch)
        model, spec = make_model(args, checkpoint, state, torch)
        report.update(model=spec, weights=str(args.weights.resolve()), weights_sha256=sha(args.weights))
        print('Strict checkpoint load OK:', spec, flush=True)
    args.output.mkdir(parents=True, exist_ok=True)
    export(args.output, 'input_signal', raw)
    export(args.output, 'input_normalized', x)
    preview(args.output / 'input_preview.png', x)
    if model is not None:
        tensor = torch.from_numpy(x.astype(np.float32))[None, None].repeat(1, 3, 1, 1).to(args.device)
        with torch.no_grad():
            for scale in dict.fromkeys(args.scales):
                folder = args.output / f'x{scale}'
                folder.mkdir()
                inp = tensor if scale == 1 else F.interpolate(tensor, scale_factor=scale, mode='bicubic', align_corners=False, antialias=False)
                h, w = inp.shape[-2:]
                padded = F.pad(inp, (0, -w % 8, 0, -h % 8), mode='replicate')
                functional.reset_net(model)
                try:
                    pred = model(padded)
                finally:
                    functional.reset_net(model)
                if pred.shape != padded.shape:
                    raise ValueError(f'Expected same-size RGB output, got {pred.shape}')
                rgb = pred[0, :, :h, :w].cpu().numpy()
                gray = np.einsum('c,chw->hw', np.array([.299, .587, .114]), rgb)
                if not np.isfinite(gray).all():
                    raise ValueError('Nonfinite model output')
                baseline = inp[0, 0].cpu().numpy()
                def inverse(a):
                    return (1 - a if args.polarity == 'negative' else a) * (high - low) + low
                export(folder, 'restored_normalized', gray)
                export(folder, 'restored_signal', inverse(gray))
                export(folder, 'bicubic_signal', inverse(baseline))
                np.save(folder / 'restored_rgb.npy', rgb)
                preview(folder / 'restored_preview.png', gray)
                preview(folder / 'bicubic_preview.png', baseline)
                report['scales'][str(scale)] = dict(output_shape=[h, w], padded_shape=list(padded.shape[-2:]),
                    outside_mapping_range=int(np.count_nonzero((gray < 0) | (gray > 1))))
                (args.output / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
                print(f'x{scale}: {h}x{w} saved to {folder}', flush=True)
    (args.output / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('Done:', args.output, flush=True)


if __name__ == '__main__':
    main()

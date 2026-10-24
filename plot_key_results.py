"""Plot existing key results; no model/GPU or new inference required."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def draw(ax, array, title, cmap='turbo', vmin=0, vmax=1):
    image = ax.imshow(array, cmap=cmap, vmin=vmin, vmax=vmax,
                      origin='upper', interpolation='nearest', extent=(0, 1, 1, 0))
    ax.set_title(title, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    return image


def load_run(root):
    report = json.loads((root / 'report.json').read_text())
    original = np.load(root / 'input_normalized.npy')
    items = []
    for scale in range(1, 5):
        folder = root / f'x{scale}'
        if not (folder / 'restored_normalized.npy').exists():
            continue
        restored = np.load(folder / 'restored_normalized.npy')
        signal = np.load(folder / 'bicubic_signal.npy')
        base = (signal - report['low']) / (report['high'] - report['low'])
        if report['polarity'] == 'negative':
            base = 1 - base
        if base.shape != restored.shape:
            raise ValueError(f'Baseline/output shape mismatch: {folder}')
        items.append((scale, base, restored))
    return original, items


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path('results'))
    p.add_argument('--runs', nargs='+', type=Path, help='Optional explicit run directories')
    p.add_argument('--cmap', default='turbo')
    args = p.parse_args()
    runs = args.runs or sorted({f.parent for f in args.root.rglob('report.json')
                               if (f.parent / 'input_normalized.npy').exists()})
    successful = []
    for root in runs:
        original, items = load_run(root)
        if not items:
            continue
        report = json.loads((root / 'report.json').read_text())
        mapping = (report['low'], report['high'], report['polarity'], report['matrix_sha256'])
        fig, axes = plt.subplots(len(items), 3, figsize=(11, 3.5 * len(items)),
                                 squeeze=False, constrained_layout=True)
        fig.suptitle(root.name + ' | shared input mapping [0, 1]')
        for row, (scale, base, restored) in enumerate(items):
            size = f'{restored.shape[1]}x{restored.shape[0]}'
            titles = [f'Measured input ({original.shape[1]}x{original.shape[0]})',
                      f'Bicubic x{scale} ({size})', f'Model x{scale} ({size})']
            arrays = [original, base, restored]
            for ax, a, title in zip(axes[row], arrays, titles):
                image = draw(ax, a, title, args.cmap)
            single, single_axes = plt.subplots(1, 3, figsize=(11, 3.7), constrained_layout=True)
            single.suptitle(root.name)
            for ax, a, title in zip(single_axes, arrays, titles):
                im = draw(ax, a, title, args.cmap)
            single.colorbar(im, ax=list(single_axes), label='Normalized signal', shrink=.8)
            single.savefig(root / f'x{scale}' / 'comparison_color.png', dpi=200)
            plt.close(single)
            diff = restored - base
            limit = max(float(np.abs(diff).max()), 1e-8)
            dfig, dax = plt.subplots(figsize=(4.8, 4), constrained_layout=True)
            di = draw(dax, diff, f'Model - bicubic x{scale}', 'coolwarm', -limit, limit)
            dfig.colorbar(di, ax=dax, label='Normalized difference')
            dfig.savefig(root / f'x{scale}' / 'difference_from_bicubic.png', dpi=200)
            plt.close(dfig)
        fig.colorbar(image, ax=list(axes.flat), label='Normalized signal', shrink=.6)
        fig.savefig(root / 'comparison_all_scales.png', dpi=200)
        plt.close(fig)
        successful.append((root, original, items, mapping))
        print('Saved comparisons:', root, flush=True)

    # Compare runs directly only if they use identical matrix and mapping.
    groups = {}
    for item in successful:
        groups.setdefault(item[3], []).append(item)
    args.root.mkdir(parents=True, exist_ok=True)
    for group_index, group in enumerate(groups.values(), 1):
        if len(group) < 2:
            continue
        scales = sorted(set.intersection(*[{x[0] for x in item[2]} for item in group]))
        if not scales:
            continue
        fig, axes = plt.subplots(len(group), len(scales) + 1,
                                 figsize=(3 * (len(scales) + 1), 3 * len(group)),
                                 squeeze=False, constrained_layout=True)
        for row, (root, original, items, _) in enumerate(group):
            draw(axes[row, 0], original, root.name + '\nInput', args.cmap)
            lookup = {s: out for s, base, out in items}
            for col, scale in enumerate(scales, 1):
                image = draw(axes[row, col], lookup[scale], f'x{scale}', args.cmap)
        fig.colorbar(image, ax=list(axes.flat), label='Normalized signal', shrink=.6)
        path = args.root / f'comparison_models_group{group_index}.png'
        fig.savefig(path, dpi=200)
        plt.close(fig)
        print('Saved model overview:', path)
    if not successful:
        raise SystemExit('No completed inference results found')


if __name__ == '__main__':
    main()

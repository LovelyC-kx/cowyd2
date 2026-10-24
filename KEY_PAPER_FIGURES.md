# Fig4(a-c), Fig5 and single-key Fig6

Run on existing float matrices; no checkpoint, GPU or inference needed.

```bash
cd /root/cowyd2-main
python -m pip install numpy scipy matplotlib scikit-learn umap-learn
python make_key_paper_figures.py --runs results/key_large_cr1_all_v1 results/key_large_cr2_all_v1 --output paper_figures_cr1_cr2_v1
```

Use a fresh output directory. Explicit --runs avoids accidentally including
duplicate trial runs. Alternatively --root results discovers all completed runs.
Runs are grouped only if their original matrix hash, bounds and polarity match.
All completed x1/x2/x3/x4 results are plotted. Incomplete scales are omitted.

## Output

`group1/x1` through `group1/x4` contain PNG (240 dpi) and vector PDF:

- Fig4a_images_ROI: whole-image and identical ROI comparisons, with horizontal
  cyan and vertical magenta profile lines on the whole-image panels.
- Fig4b_horizontal_profiles: full horizontal strip-average profiles and ROI zoom.
- Fig4c_vertical_profiles: full vertical strip-average profiles and ROI zoom.
- Fig4b_profiles.csv / Fig4c_profiles.csv: exact plotted numbers.
- Fig5_images_surfaces: matched 2-D maps and 3-D signal surfaces, with shared color
  limits, height range and view. These are NOT physical object topography.

`group1/Fig6_patch_embeddings.png/.pdf`: joint t-SNE/UMAP at x1 by default.
Top row is colored by method; bottom row by source patch row (position control).
`Fig6_samples.csv` records patch coordinates, paired IDs and embedding coordinates.
`Fig6_features_embeddings.npz` preserves actual features, PCA and embeddings.
`figure_manifest.json` records input/report hashes, ROI, profiles, parameters,
script hash and dependency versions.

## Fixed conventions and customization

All images use the input normalization and turbo [0,1], no output-wise stretching.
Float values (including overshoot) are retained in profiles and surfaces. Image
colors clip at the fixed limits. Spatial axes use input pixel-center coordinates,
so all scales share a field of view. No physical unit is invented. Profiles use
linear interpolation and a 3-input-pixel strip average; no curve smoothing.

Default ROI is a quarter-width/height region around the strongest gradient of the
ORIGINAL input, away from image borders. It is not chosen from model output and
not guaranteed to select the most scientifically relevant key feature. Review it.
The row/column defaults pass through the ROI center. Freeze the ROI before final
comparisons; manual input-pixel coordinates are supported:

```bash
python make_key_paper_figures.py --runs results/key_large_cr1_all_v1 results/key_large_cr2_all_v1 --output paper_figures_manual_v1 --roi 30 35 25 25 --row 47 --col 42 --band 3
```

The example coordinates are illustrative, not verified anatomical key locations.

Fig6 is an EXPLORATORY single-image matched-patch plot, not the paper's multi-
system dataset experiment. Each method is mapped back to the original input grid
with linear interpolation; matching 16x16 patches, stride=10, maximum 81 positions
per method (seeded selection if needed), are flattened without per-patch
normalization. StandardScaler and PCA<=50 are fit jointly across methods, then
t-SNE (perplexity<=30, learning_rate=200, PCA init, 1000 iterations, seed=1234)
and UMAP (neighbors<=15, min_dist=.1, Euclidean, seed=1234, one thread).
The same patch locations are used for every method. Overlapping patches are
dependent, not independent images. At x1 bicubic equals input; duplicate feature
vectors are expected. Embedding separation does not measure reconstruction
quality or prove generalization. No invented system labels or confidence ellipses.

To inspect another scale use --fig6-scale 2/3/4. Downmapping removes some fine
detail; Fig6 is not a high-resolution frequency analysis. For a single-key map
without model comparison, see the input-colored samples/position controls rather
than making artificial class labels.

Optional parameters: --patch, --stride, --max-patches, --seed, --cmap. Record
changes, do not tune seeds for visually attractive separation. --skip-fig6 makes
Fig4/5 without requiring UMAP. UMAP's first run may spend time compiling Numba.

No PSNR/SSIM is computed without an independent reference. Existing large-model
legacy activation uncertainty remains; visualization does not resolve it.

Validation uses synthetic paired matrices (not claimed as real key results),
multi-scale coordinate checks, end-to-end output generation and image review.
Actual private inference matrices are generated on the user's AutoDL server.

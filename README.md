# Pseudo-Low-Light-RAW-Synthesis

This project builds a self-contained pipeline for synthesizing pseudo low-light RAW data from ordinary RGB PNG/JPEG images. It first reconstructs Canon-style pseudo clean RAW with an InvISP inverse model, converts the demosaiced RAW estimate back to a Bayer plane, and then applies an ELD-style physics-based low-light noise model.

The main goal is to create controlled RAW-like low-light inputs from normal-light image datasets such as Point-Line, while keeping the runtime code independent from the vendored research repositories under `third-party/`.

## Features

- Convert RGB PNG/JPEG images into pseudo clean Bayer RAW.
- Synthesize amplified low-light noisy RAW at user-provided exposure ratios.
- Save training-friendly `.npz` files containing packed clean and low-light RAW arrays.
- Preview generated RAW data with a compact ratio grid.
- Show two views per preview ratio:
  - **RAW demosaic**: direct nearest-neighbor Bayer visualization of the generated noisy Bayer RAW. This view does not apply ISP, gamma, tone mapping, denoising, or normalization beyond clipping for PNG display.
  - **InvISP forward RGB**: the generated noisy RAW view passed through the copied InvISP forward model.
- Default preview ratios are `100`, `200`, and `300`.

## Method

The synthesis pipeline is:

1. Load an RGB PNG/JPEG image as float data in `[0, 1]`.
2. Crop the image to even height and width so Bayer extraction is valid.
3. Run the copied InvISP Canon model in inverse mode to estimate demosaiced pseudo RAW.
4. Extract a single-plane Bayer RAW image using the configured CFA pattern. The default and currently supported pattern is `RGGB`.
5. Apply the ELD baseline Poisson + Gaussian noise model:
   - scale by `saturation_level`
   - divide by the exposure `ratio`
   - apply shot noise and read noise
   - multiply by the same `ratio`
   - divide by `saturation_level`
6. Pack clean and low-light Bayer RAW as `4 x H/2 x W/2` arrays in RGGB order: R, G on the R row, B, G on the B row.
7. Save one `.npz` per input image and ratio.

The `ratio` represents an exposure reduction factor followed by gain compensation. Larger ratios produce stronger low-light noise after amplification.

## Outputs

The project intentionally exposes only two output workflows.

### 1. RAW `.npz` Synthesis

`scripts/synthesize_pseudo_lowlight_raw.py` writes only `.npz` files. It does not create preview images.

Each `.npz` contains:

- `clean_raw`
- `low_light_raw`
- `ratio`
- `noise_params`
- `source_path`
- `cfa`
- `height_before_crop`
- `width_before_crop`
- `height`
- `width`
- `saturation_level`

### 2. Preview PNG Generation

`scripts/preview_pseudo_lowlight_raw.py` writes preview PNGs. It can read either:

- a PNG/JPEG image, then synthesize preview ratios on the fly
- a `.npz` file or directory of `.npz` files, then render existing generated RAW data

The default preview layout is one row with three ratio tiles: `100`, `200`, and `300`. Each tile stacks RAW demosaic above InvISP forward RGB.

## Installation

Use a Python environment with NumPy, Pillow, PyTorch, and pytest. The development environment used for validation is:

```bash
/home/rjyu/miniconda3/envs/normal/bin/python
```

Install pytest if needed:

```bash
/home/rjyu/miniconda3/envs/normal/bin/python -m pip install pytest
```

The required project-owned assets are already expected at:

```text
assets/checkpoints/invisp_canon_eos_5d.pth
assets/camera_params/CanonEOS5D4_params.npy
```

These assets were copied from the reference repositories into project-owned paths so runtime code does not import from `third-party/`.

## Usage

### Synthesize RAW `.npz` Files

```bash
/home/rjyu/miniconda3/envs/normal/bin/python scripts/synthesize_pseudo_lowlight_raw.py \
  --input /home/rjyu/data/Point-Line/wireframe/images \
  --output outputs/point_line_wireframe_raw \
  --ratios 100 200 300 \
  --limit 4 \
  --seed 0 \
  --device cpu
```

### Preview One PNG/JPEG Input

```bash
/home/rjyu/miniconda3/envs/normal/bin/python scripts/preview_pseudo_lowlight_raw.py \
  --input /home/rjyu/data/Point-Line/wireframe/images/00030043.png \
  --output outputs/point_line_preview.png \
  --seed 0 \
  --device cpu
```

This synthesizes ratios `100`, `200`, and `300` and writes one preview PNG.

### Preview Existing `.npz` Outputs

```bash
/home/rjyu/miniconda3/envs/normal/bin/python scripts/preview_pseudo_lowlight_raw.py \
  --input outputs/point_line_wireframe_raw \
  --output outputs/point_line_wireframe_raw_preview.png \
  --device cpu
```

When previewing a directory of `.npz` files, entries are sorted by numeric `ratio` metadata.

## Testing

Run the full test suite:

```bash
/home/rjyu/miniconda3/envs/normal/bin/python -m pytest tests -q
```

Run a small synthesis and preview smoke test:

```bash
/home/rjyu/miniconda3/envs/normal/bin/python scripts/synthesize_pseudo_lowlight_raw.py \
  --input /home/rjyu/data/Point-Line/wireframe/images \
  --output outputs/actual_effect_preview/npz \
  --ratios 100 200 300 \
  --limit 1 \
  --seed 0 \
  --device cpu

/home/rjyu/miniconda3/envs/normal/bin/python scripts/preview_pseudo_lowlight_raw.py \
  --input outputs/actual_effect_preview/npz \
  --output outputs/actual_effect_preview/preview.png \
  --device cpu
```

## Acknowledgements

This repository contains vendored reference projects under `third-party/`. They are not imported by the final runtime package, but they informed the implementation and supplied required assets.

- `third-party/Invertible-ISP`: reference InvISP implementation and Canon checkpoint used for RGB-to-RAW inversion.
- `third-party/ELD`: reference ELD noise model and CanonEOS5D4 calibrated camera noise parameters.
- `third-party/DarkFeat`: reference low-light RAW feature pipeline that combines InvISP-style RGB-to-RAW conversion with low-light noise simulation.

Please cite and follow the licenses of the original projects when using this repository in research or derived work.

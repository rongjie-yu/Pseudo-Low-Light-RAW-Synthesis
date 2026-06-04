# Pseudo-Low-Light-RAW-Synthesis

This project builds a self-contained pipeline for synthesizing pseudo low-light RAW data from ordinary RGB PNG/JPEG images. It first reconstructs Canon-style pseudo clean RAW with an InvISP inverse model, converts the demosaiced RAW estimate back to a Bayer plane, and then applies an ELD-style physics-based low-light noise model.

## Usage

Two modes are supported:

### 1. Synthesis — save `.npz` data files

```bash
python scripts/synthesize_pseudo_lowlight_raw.py \
  --input /path/to/image.png \
  --output outputs/raw_data \
  --ratio 100 \
  --seed 0
```

Each `.npz` contains `clean_raw`, `low_light_raw`, `ratio`, `noise_params`, and metadata.

### 2. Preview — render a visual preview

```bash
python scripts/preview_pseudo_lowlight_raw.py \
  --input /path/to/image.png \
  --output outputs/preview.png \
  --ratio 200 \
  --seed 0
```

The preview is a 2×2 grid:

| Original | Low-light RAW |
|----------|---------------|
| Noisy RAW | ISP RGB |

- **Original**: the input RGB image
- **Low-light RAW**: clean RAW ÷ ratio + ELD noise, Z-score normalized for visibility
- **Noisy RAW**: the gain-amplified noisy RAW demosaiced
- **ISP RGB**: the noisy RAW passed through the InvISP forward model

Ratio information is embedded in the output filename (e.g. `preview_r200.png`).

## Preview Examples

All examples use `--seed 0` for reproducibility.

### Ratio 100 (low-light)

| Input: `1.png` | Input: `P1020171.png` | Input: `P1020177.png` |
|-----------------|------------------------|------------------------|
| ![1_r100](docs/previews/1_r100.png) | ![P1020171_r100](docs/previews/P1020171_r100.png) | ![P1020177_r100](docs/previews/P1020177_r100.png) |

### Ratio 200 (stronger low-light effect)

![P1020171_r200](docs/previews/P1020171_r200.png)

*Each preview grid shows: Original \| Low-light RAW (Z-score normalized) \| Noisy RAW \| ISP RGB.*

## Method

The synthesis pipeline is:

1. Load an RGB PNG/JPEG image as float data in `[0, 1]`.
2. Crop the image to even height and width so Bayer extraction is valid.
3. Run the InvISP Canon model in inverse mode to estimate demosaiced pseudo RAW.
4. Extract a single-plane Bayer RAW image using the configured CFA pattern (RGGB).
5. Apply the ELD baseline Poisson + Gaussian noise model:
   - scale by `saturation_level`
   - divide by the exposure `ratio`
   - apply shot noise and read noise
   - multiply by the same `ratio`
   - divide by `saturation_level`
6. Pack clean and low-light Bayer RAW as `4 × H/2 × W/2` arrays in RGGB order.
7. Save one `.npz` per input image and ratio (synthesis mode) or render a preview PNG.

## Installation

Use a Python environment with NumPy, Pillow, PyTorch, and pytest. The development environment is:

```bash
/home/rjyu/miniconda3/envs/normal/bin/python
```

Install pytest if needed:

```bash
/home/rjyu/miniconda3/envs/normal/bin/python -m pip install pytest
```

Required project-owned assets:

```text
assets/checkpoints/invisp_canon_eos_5d.pth
assets/camera_params/CanonEOS5D4_params.npy
```

## Testing

```bash
/home/rjyu/miniconda3/envs/normal/bin/python -m pytest tests -q
```

## Acknowledgements

This repository contains vendored reference projects under `third-party/`. They are not imported by the final runtime package, but they informed the implementation and supplied required assets.

- `third-party/Invertible-ISP`: reference InvISP implementation and Canon checkpoint used for RGB-to-RAW inversion.
- `third-party/ELD`: reference ELD noise model and CanonEOS5D4 calibrated camera noise parameters.
- `third-party/DarkFeat`: reference low-light RAW feature pipeline.

Please cite and follow the licenses of the original projects when using this repository in research or derived work.

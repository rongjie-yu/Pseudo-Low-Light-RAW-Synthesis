# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This project synthesizes pseudo low-light RAW data from normal-light RGB PNG/JPEG images. It reconstructs Canon-style pseudo clean RAW via an invertible ISP network (InvISP), extracts a Bayer pattern, then applies an ELD physics-based Poisson+Gaussian noise model at user-specified exposure ratios.

The vendored reference repositories under `third-party/` (InvISP, ELD, DarkFeat) are **not imported at runtime**. All runtime code lives in `pllraw_synthesis/` and `scripts/`, using project-owned checkpoints and camera params copied into `assets/`.

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run all tests
python -m pytest tests -q

# Run a single test file
python -m pytest tests/test_eld_noise.py -q

# Synthesis mode: save .npz data files
python scripts/synthesize_pseudo_lowlight_raw.py \
  --input data/1.png --output outputs/raw --ratio 100 --seed 0

# Preview mode: render 2×2 preview grid
python scripts/preview_pseudo_lowlight_raw.py \
  --input data/1.png --output outputs/preview_r100.png --ratio 100 --seed 0
```

## Architecture

### Pipeline (the core data flow)

```
PNG/JPEG (HxWx3, [0,1])
  → crop to even dimensions (cfa.crop_even)
  → InvISP inverse: RGB → demosaiced RAW (invisp_inverse.InvISPInverse.rgb_to_demosaiced_raw)
  → extract single-plane Bayer RGGB (cfa.extract_bayer)
  → pack to 4×H/2×W/2 RGGB order (packing.pack_bayer)
  → [saved as clean_raw in .npz]

  → ELD noise model: apply Poisson+Gaussian at given ratio (eld_noise.ELDNoiseModel.apply)
  → pack noisy Bayer (packing.pack_bayer)
  → [saved as low_light_raw in .npz]
```

### Package: `pllraw_synthesis/`

| Module | Purpose |
|--------|---------|
| `cfa.py` | Crop to even dimensions; extract single-plane Bayer (RGGB only) from 3-channel demosaiced RAW |
| `eld_noise.py` | `ELDNoiseModel` loads CanonEOS5D4 camera params, samples correlated K/g_scale, applies Poisson shot noise + Gaussian read noise. `NoiseParams` is a frozen dataclass |
| `invisp_inverse.py` | `InvISPInverse` wraps `InvISPNet` with checkpoint loading, device resolution, and tensor↔numpy conversion. Provides both `rgb_to_demosaiced_raw` (inverse) and `demosaiced_raw_to_rgb` (forward) |
| `invisp_model/` | Copied InvISP network: `InvISPNet` → stack of `InvBlock` → each block has `DenseBlock` sub-networks (F, G, H) + `InvertibleConv1x1` permutation. Bidirectional via `rev` flag |
| `io.py` | Image discovery, RGB loading (PIL → float32 [0,1] → crop), output-stem naming with SHA1, `.npz` save with all metadata fields |
| `packing.py` | Pack/unpack between 2D Bayer (H×W) and 4×H/2×W/2 RGGB channel stack |
| `preview.py` | Nearest-neighbor Bayer demosaic, Z-score normalization for low-light visibility, 2×2 grid PNG rendering |

### Scripts

- `scripts/synthesize_pseudo_lowlight_raw.py` — batch synthesis of `.npz` files from PNG/JPEG images. Accepts single image or directory via `--input`. Uses `--ratio` for a single exposure ratio.
- `scripts/preview_pseudo_lowlight_raw.py` — renders a 2×2 preview grid from a single PNG/JPEG input: Original | Low-light RAW (Z-score normalized) | Noisy RAW | ISP RGB. Ratio info is embedded in the output filename.

### Data format conventions

- **Bayer CFA:** RGGB only (`SUPPORTED_CFA = {"RGGB"}`). The pattern maps R=top-left, G=top-right/bottom-left, B=bottom-right on the 2D Bayer plane.
- **Packed RAW:** 4-channel array `(4, H/2, W/2)` with channels in order: R, G (R-row), B, G (B-row). This is the format stored in `.npz` files as `clean_raw` and `low_light_raw`.
- **Demosaiced RAW:** 3-channel H×W×3 array where the three channels hold the R, G, B values that will become the Bayer plane.
- **All arrays are float32 in [0, 1]** unless stated otherwise.

### Key constraints

- Input images must have even height and width after crop (enforced by `crop_even`).
- Only `RGGB` CFA is supported; other patterns will raise `ValueError`.
- The InvISP checkpoint (`assets/checkpoints/invisp_canon_eos_5d.pth`) uses `strict=False` loading — some keys may be missing.
- Camera params (`assets/camera_params/CanonEOS5D4_params.npy`) contain `Kmin`, `Kmax`, and `Profile-1` with `g_scale` distribution parameters.
- Device resolution: defaults to `"cuda"`. Use `--device cpu` for CPU-only environments.
- `ELDNoiseModel` is seeded per-model, not per-call. For deterministic results across multiple images, create a new model with the same seed.

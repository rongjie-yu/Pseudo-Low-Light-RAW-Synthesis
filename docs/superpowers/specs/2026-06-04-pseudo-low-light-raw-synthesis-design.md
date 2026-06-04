# Pseudo Low-Light RAW Synthesis Design

## Purpose

Build a self-contained tool that converts ordinary RGB PNG/JPEG images into Canon-style pseudo clean RAW using InvISP Canon_EOS_5D weights, then synthesizes pseudo low-light RAW using ELD CanonEOS5D4 camera noise parameters.

The first target dataset for smoke testing is `/home/rjyu/data/Point-Line`, especially `wireframe/images` and `york/images`.

## Scope

The tool will be implemented from scratch in this repository. `third-party/` and `reference/` are reference material only. The final tool must not modify those directories and must not import or execute code from them at runtime.

Useful reference code may be copied into project-owned modules and adapted. Required assets may also be copied into project-owned asset paths:

- InvISP Canon checkpoint copied from `third-party/Invertible-ISP/pretrained/canon.pth`.
- ELD CanonEOS5D4 camera parameter file copied from `third-party/ELD/camera_params/release/CanonEOS5D4_params.npy`.

When implementation questions arise, the priority order is:

1. Check the papers in `reference/`, especially InvISP and ELD.
2. Check `third-party/DarkFeat` for the similar InvISP + ELD-style synthesis flow.
3. Check other `third-party/` implementation details as secondary reference.

## Architecture

Create a project-owned Python package named `pllraw_synthesis/` plus one CLI script under `scripts/`.

Planned modules:

- `pllraw_synthesis/invisp_model/`: copied and locally adapted InvISP model definitions needed for inference.
- `pllraw_synthesis/invisp_inverse.py`: loads the copied Canon checkpoint and maps RGB images to demosaiced pseudo clean RAW.
- `pllraw_synthesis/cfa.py`: converts demosaiced RAW to single-plane Bayer RAW using a configurable CFA pattern.
- `pllraw_synthesis/eld_noise.py`: implements the ELD baseline Poisson + Gaussian read-noise synthesis using copied CanonEOS5D4 params.
- `pllraw_synthesis/packing.py`: packs and unpacks Bayer RAW into 4-channel training tensors.
- `pllraw_synthesis/preview.py`: creates optional visual previews for quick inspection.
- `pllraw_synthesis/io.py`: image discovery, `.npz` output, and metadata handling.
- `scripts/synthesize_pseudo_lowlight_raw.py`: CLI entry point.

## Data Flow

1. Discover RGB input images from a file or directory. Support `.png`, `.jpg`, and `.jpeg`.
2. Load each image as RGB float data in `[0, 1]`.
3. Crop to even height and width. Point-Line includes odd dimensions such as `500x375`, and Bayer extraction requires even dimensions.
4. Run InvISP inverse inference with the copied Canon_EOS_5D checkpoint.
5. Treat InvISP output as demosaiced RAW, not Bayer RAW.
6. Extract a single-plane Bayer RAW according to CFA pattern. Default is `RGGB`:
   - R from `[0::2, 0::2, 0]`
   - G from `[0::2, 1::2, 1]`
   - G from `[1::2, 0::2, 1]`
   - B from `[1::2, 1::2, 2]`
7. Normalize the clean Bayer RAW into the linear range expected by the ELD baseline noise model.
8. For each user-provided low-light `ratio`, synthesize noisy RAW:
   - scale by `saturation_level`
   - divide by `ratio`
   - apply Poisson shot noise
   - apply Gaussian read noise
   - multiply by `ratio`
   - divide by `saturation_level`
9. Pack clean and noisy Bayer RAW into 4-channel arrays for training-friendly output.
10. Save one `.npz` per source image and ratio.
11. Optionally save preview PNGs for human inspection.

## Ratio Handling

The CLI must support arbitrary user-provided ratios, for example:

```bash
--ratios 25 50 100 150 200
```

Tests and previews should use fixed ratio lists for reproducibility. Random ratio sampling is not required for the first implementation, but the module boundaries should not prevent adding it later.

## Output Format

Each `.npz` should contain:

- `clean_raw`: packed 4-channel clean pseudo RAW.
- `low_light_raw`: packed 4-channel pseudo low-light RAW.
- `ratio`: low-light exposure ratio used for this sample.
- `noise_params`: sampled or explicit noise parameters used by ELD synthesis.
- `source_path`: original image path.
- `cfa`: CFA pattern string, default `RGGB`.
- `height_before_crop` and `width_before_crop`.
- `height` and `width` after even-size crop.
- `saturation_level` used by the noise model.

The default packed layout should be stable and documented by `packing.py`. Preview images are diagnostic artifacts only and should not be treated as training data.

## Noise Model

The first implementation should use the ELD baseline `G+P` behavior:

- Sample `K` from the copied CanonEOS5D4 parameter range.
- Compute `g_scale` from `Profile-1/g_scale`.
- Use `saturation_level = 16383 - 800` by default, matching the ELD reference baseline.
- Apply Poisson shot noise and Gaussian read noise in the exposure-reduced domain, then amplify by `ratio`.

The tool should expose deterministic seeding so the same source image and ratio list can be reproduced.

## CLI Shape

Example smoke-test command:

```bash
python scripts/synthesize_pseudo_lowlight_raw.py \
  --input /home/rjyu/data/Point-Line/wireframe/images \
  --output outputs/point_line_wireframe_raw \
  --ratios 50 100 150 200 \
  --preview \
  --limit 8 \
  --seed 0
```

Expected behavior:

- Recursively or directly discover supported images under `--input`.
- Process at most `--limit` images when provided.
- Write outputs under `--output`, preserving enough source name information to avoid collisions.
- Fail clearly if copied assets are missing, if CUDA is requested but unavailable, or if a source image cannot be read.

## Testing And Validation

Use focused tests for pure functions:

- CFA Bayer extraction from small synthetic RGB arrays.
- Bayer packing and unpacking shape/order.
- ELD noise parameter sampling and deterministic seeding.
- CLI image discovery and output naming.

Use a smoke test with `/home/rjyu/data/Point-Line/wireframe/images`:

- Run with `--limit 1` and ratios such as `50 100`.
- Verify `.npz` files exist.
- Verify clean and low-light arrays have expected shapes and finite values.
- Verify preview PNGs are created when `--preview` is set.

Full numeric quality validation is out of scope for the first tool version because the repository does not include ground-truth RAW for Point-Line.

## Non-Goals

- Do not generate DNG/TIFF camera files in the first version.
- Do not train InvISP or ELD.
- Do not modify `third-party/` or `reference/`.
- Do not require Point-Line annotations for synthesis.
- Do not add random-ratio dataset generation until fixed arbitrary-ratio synthesis is working.

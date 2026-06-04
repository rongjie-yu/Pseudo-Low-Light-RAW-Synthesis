# Pseudo-Low-Light-RAW-Synthesis
使用InvISP + ELD， 将常光Jpeg图像，合成为Pseudo-Low-Light-RAW

## Pseudo Low-Light RAW Synthesis Tool

The repository provides a self-contained tool for converting RGB PNG/JPEG images into Canon-style pseudo clean RAW with InvISP and synthesizing pseudo low-light RAW with ELD CanonEOS5D4 noise parameters.

Required project-owned assets:

- `assets/checkpoints/invisp_canon_eos_5d.pth`
- `assets/camera_params/CanonEOS5D4_params.npy`

Smoke-test command:

```bash
/home/rjyu/miniconda3/envs/normal/bin/python scripts/synthesize_pseudo_lowlight_raw.py \
  --input /home/rjyu/data/Point-Line/wireframe/images \
  --output outputs/point_line_wireframe_smoke \
  --ratios 50 100 \
  --preview \
  --limit 1 \
  --seed 0 \
  --device cpu
```

Each output `.npz` contains `clean_raw`, `low_light_raw`, `ratio`, `noise_params`, `source_path`, `cfa`, crop metadata, and `saturation_level`. RAW arrays are packed as `4 x H/2 x W/2` in RGGB order: R, G on the R row, B, G on the B row.

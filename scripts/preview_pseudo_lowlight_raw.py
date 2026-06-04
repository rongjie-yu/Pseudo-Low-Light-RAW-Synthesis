#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pllraw_synthesis.cfa import extract_bayer
from pllraw_synthesis.eld_noise import ELDNoiseModel
from pllraw_synthesis.invisp_inverse import InvISPInverse
from pllraw_synthesis.io import load_rgb_image
from pllraw_synthesis.preview import PreviewTile, bayer_to_nearest_rgb, normalize_for_display, save_grid_preview


DEFAULT_RATIO = 100.0
DEFAULT_PARAMS = Path("assets/camera_params/CanonEOS5D4_params.npy")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preview pseudo low-light RAW synthesis: Original | RAW demosaic | ISP RGB."
    )
    parser.add_argument("--input", required=True, help="Input PNG/JPEG image path.")
    parser.add_argument("--output", required=True, help="Output preview PNG path (include ratio in filename).")
    parser.add_argument(
        "--ratio",
        type=float,
        default=DEFAULT_RATIO,
        help=f"Low-light exposure ratio (default: {DEFAULT_RATIO:g}).",
    )
    parser.add_argument("--seed", type=int, default=None, help="Seed for deterministic ELD noise sampling.")
    parser.add_argument("--device", default="cuda", help="Torch device: cuda, cpu, or cuda:0.")
    parser.add_argument("--cfa", default="RGGB", help="CFA pattern (default: RGGB).")
    parser.add_argument(
        "--checkpoint",
        default="assets/checkpoints/invisp_canon_eos_5d.pth",
        help="InvISP checkpoint path.",
    )
    parser.add_argument(
        "--camera-params",
        default=str(DEFAULT_PARAMS),
        help="CanonEOS5D4 camera params .npy file.",
    )
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> int:
    input_path = Path(args.input)
    if not input_path.is_file():
        raise FileNotFoundError(f"Input image not found: {input_path}")

    invisp = InvISPInverse(checkpoint_path=args.checkpoint, device=args.device)

    original_rgb, _ = load_rgb_image(input_path)
    demosaiced = invisp.rgb_to_demosaiced_raw(original_rgb)
    clean_bayer = extract_bayer(demosaiced, cfa=args.cfa)

    noise_model = ELDNoiseModel(params_file=args.camera_params, seed=args.seed)
    low_bayer, noise_params = noise_model.apply(clean_bayer, ratio=args.ratio, amplify=False)
    low_demosaic = normalize_for_display(bayer_to_nearest_rgb(low_bayer))

    amp_bayer = np.clip(low_bayer * args.ratio, 0.0, 1.0).astype(np.float32)
    amp_demosaic = bayer_to_nearest_rgb(amp_bayer)
    isp_rgb = invisp.demosaiced_raw_to_rgb(amp_demosaic)
    tile = PreviewTile(ratio=args.ratio, raw_demosaic=amp_demosaic, isp_rgb=isp_rgb)

    save_grid_preview(args.output, original_rgb, low_demosaic, tile)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())

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
from pllraw_synthesis.io import IMAGE_EXTENSIONS, load_rgb_image
from pllraw_synthesis.packing import unpack_bayer
from pllraw_synthesis.preview import PreviewTile, bayer_to_nearest_rgb, save_ratio_grid_preview


DEFAULT_PREVIEW_RATIOS = (100.0, 200.0, 300.0)
DEFAULT_PARAMS = Path("assets/camera_params/CanonEOS5D4_params.npy")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preview pseudo low-light RAW as demosaic RAW and InvISP RGB.")
    parser.add_argument("--input", required=True, help="Input PNG/JPEG image, .npz file, or directory of .npz files.")
    parser.add_argument("--output", required=True, help="Output preview PNG path.")
    parser.add_argument(
        "--ratios",
        nargs="+",
        type=float,
        default=list(DEFAULT_PREVIEW_RATIOS),
        help="Ratios used when input is PNG/JPEG.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Seed for deterministic ELD noise sampling.")
    parser.add_argument("--device", default="auto", help="Torch device: auto, cpu, cuda, or cuda:0.")
    parser.add_argument("--cfa", default="RGGB", help="CFA pattern. First version supports RGGB.")
    parser.add_argument(
        "--checkpoint",
        default="assets/checkpoints/invisp_canon_eos_5d.pth",
        help="Copied InvISP checkpoint.",
    )
    parser.add_argument("--camera-params", default=str(DEFAULT_PARAMS), help="Copied CanonEOS5D4 params .npy file.")
    return parser.parse_args(argv)


def _tile_from_bayer(ratio: float, bayer: np.ndarray, invisp: InvISPInverse) -> PreviewTile:
    raw_demosaic = bayer_to_nearest_rgb(bayer)
    isp_rgb = invisp.demosaiced_raw_to_rgb(raw_demosaic)
    return PreviewTile(ratio=float(ratio), raw_demosaic=raw_demosaic, isp_rgb=isp_rgb)


def _tiles_from_image(path: Path, args: argparse.Namespace, invisp: InvISPInverse) -> list[PreviewTile]:
    rgb, _ = load_rgb_image(path)
    demosaiced = invisp.rgb_to_demosaiced_raw(rgb)
    clean_bayer = extract_bayer(demosaiced, cfa=args.cfa)
    noise_model = ELDNoiseModel(params_file=args.camera_params, seed=args.seed)

    tiles = []
    for ratio in args.ratios:
        low_bayer, _ = noise_model.apply(clean_bayer, ratio=ratio)
        tiles.append(_tile_from_bayer(ratio, low_bayer, invisp))
    return tiles


def _tiles_from_npz_files(paths: Sequence[Path], invisp: InvISPInverse) -> list[PreviewTile]:
    tiles = []
    for path in sort_npz_paths_by_ratio(paths):
        data = np.load(path, allow_pickle=True)
        bayer = unpack_bayer(data["low_light_raw"])
        ratio = float(data["ratio"])
        tiles.append(_tile_from_bayer(ratio, bayer, invisp))
    return tiles


def sort_npz_paths_by_ratio(paths: Sequence[Path]) -> list[Path]:
    def ratio_key(path: Path) -> tuple[float, str]:
        data = np.load(path, allow_pickle=True)
        return float(data["ratio"]), str(path)

    return sorted(paths, key=ratio_key)


def _npz_paths(path: Path) -> list[Path]:
    if path.is_file() and path.suffix.lower() == ".npz":
        return [path]
    if path.is_dir():
        return sorted(p for p in path.glob("*.npz") if p.is_file())
    return []


def run(args: argparse.Namespace) -> int:
    input_path = Path(args.input)
    invisp = InvISPInverse(checkpoint_path=args.checkpoint, device=args.device)

    if input_path.suffix.lower() in IMAGE_EXTENSIONS:
        tiles = _tiles_from_image(input_path, args, invisp)
    else:
        paths = _npz_paths(input_path)
        if not paths:
            raise RuntimeError(f"No previewable PNG/JPEG or .npz files found at {input_path}")
        tiles = _tiles_from_npz_files(paths, invisp)

    save_ratio_grid_preview(args.output, tiles, columns=3)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())

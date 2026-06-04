#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **_kwargs):
        return iterable

from pllraw_synthesis.cfa import extract_bayer
from pllraw_synthesis.eld_noise import ELDNoiseModel
from pllraw_synthesis.invisp_inverse import InvISPInverse
from pllraw_synthesis.io import discover_images, load_rgb_image, make_output_stem, save_npz
from pllraw_synthesis.packing import pack_bayer
from pllraw_synthesis.preview import save_preview


DEFAULT_PARAMS = Path("assets/camera_params/CanonEOS5D4_params.npy")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Synthesize pseudo low-light RAW from RGB PNG/JPEG images.")
    parser.add_argument("--input", required=True, help="Input image file or directory.")
    parser.add_argument("--output", required=True, help="Output directory.")
    parser.add_argument("--ratios", required=True, nargs="+", type=float, help="One or more low-light ratios.")
    parser.add_argument("--preview", action="store_true", help="Write diagnostic preview PNGs.")
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of input images to process.")
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


def run(args: argparse.Namespace) -> int:
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    preview_dir = output_dir / "preview"
    images = discover_images(args.input)
    if args.limit is not None:
        images = images[: args.limit]
    if not images:
        raise RuntimeError(f"No supported images found under {args.input}")

    invisp = InvISPInverse(checkpoint_path=args.checkpoint, device=args.device)
    noise_model = ELDNoiseModel(params_file=args.camera_params, seed=args.seed)

    for image_path in tqdm(images, desc="synthesizing"):
        rgb, crop_meta = load_rgb_image(image_path)
        demosaiced = invisp.rgb_to_demosaiced_raw(rgb)
        clean_bayer = extract_bayer(demosaiced, cfa=args.cfa)
        clean_packed = pack_bayer(clean_bayer)

        for ratio in args.ratios:
            low_bayer, noise_params = noise_model.apply(clean_bayer, ratio=ratio)
            low_packed = pack_bayer(low_bayer)
            stem = make_output_stem(image_path, ratio)
            metadata = {
                **crop_meta,
                "ratio": float(ratio),
                "source_path": str(image_path),
                "cfa": args.cfa.upper(),
                "noise_params": noise_params.to_dict(),
                "saturation_level": noise_params.saturation_level,
            }
            save_npz(output_dir / f"{stem}.npz", clean_raw=clean_packed, low_light_raw=low_packed, metadata=metadata)
            if args.preview:
                save_preview(preview_dir / f"{stem}.png", clean_bayer=clean_bayer, low_light_bayer=low_bayer)

    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())

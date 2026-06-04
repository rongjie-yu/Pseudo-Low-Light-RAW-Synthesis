from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from PIL import Image

from .cfa import crop_even


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}


def discover_images(input_path: str | Path) -> List[Path]:
    path = Path(input_path)
    if path.is_file():
        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            raise ValueError(f"Unsupported image extension: {path}")
        return [path]
    if not path.exists():
        raise FileNotFoundError(f"Input path not found: {path}")
    if not path.is_dir():
        raise ValueError(f"Input path is neither file nor directory: {path}")

    return sorted(p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)


def load_rgb_image(path: str | Path) -> Tuple[np.ndarray, Dict[str, int]]:
    image_path = Path(path)
    with Image.open(image_path) as image:
        rgb = image.convert("RGB")
        array = np.asarray(rgb, dtype=np.float32) / 255.0
    return crop_even(array)


def _format_ratio(ratio: float) -> str:
    if float(ratio).is_integer():
        return str(int(ratio))
    return str(ratio).replace(".", "p")


def make_output_stem(source_path: str | Path, ratio: float) -> str:
    path = Path(source_path)
    digest = hashlib.sha1(str(path).encode("utf-8")).hexdigest()[:10]
    return f"{path.stem}_{digest}_ratio{_format_ratio(ratio)}"


def save_npz(
    output_path: str | Path,
    clean_raw: np.ndarray,
    low_light_raw: np.ndarray,
    metadata: Dict[str, Any],
) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        clean_raw=clean_raw.astype(np.float32, copy=False),
        low_light_raw=low_light_raw.astype(np.float32, copy=False),
        ratio=np.array(metadata["ratio"], dtype=np.float32),
        noise_params=np.array(metadata["noise_params"], dtype=object),
        source_path=np.array(str(metadata["source_path"]), dtype=object),
        cfa=np.array(str(metadata["cfa"]), dtype=object),
        height_before_crop=np.array(metadata.get("height_before_crop", -1), dtype=np.int32),
        width_before_crop=np.array(metadata.get("width_before_crop", -1), dtype=np.int32),
        height=np.array(metadata.get("height", -1), dtype=np.int32),
        width=np.array(metadata.get("width", -1), dtype=np.int32),
        saturation_level=np.array(metadata.get("saturation_level", -1.0), dtype=np.float32),
    )

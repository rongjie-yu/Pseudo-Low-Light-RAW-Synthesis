from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def _to_uint8(raw: np.ndarray) -> np.ndarray:
    clipped = np.clip(raw, 0.0, 1.0)
    return (clipped * 255.0 + 0.5).astype(np.uint8)


def bayer_to_nearest_rgb(bayer: np.ndarray) -> np.ndarray:
    if bayer.ndim != 2:
        raise ValueError(f"Expected 2D Bayer RAW, got shape {bayer.shape}")
    height, width = bayer.shape
    if height % 2 or width % 2:
        raise ValueError(f"Bayer dimensions must be even, got {height}x{width}")

    rgb = np.zeros((height, width, 3), dtype=np.float32)
    r = bayer[0::2, 0::2]
    g_top = bayer[0::2, 1::2]
    g_bottom = bayer[1::2, 0::2]
    b = bayer[1::2, 1::2]

    rgb[0::2, 0::2, 0] = r
    rgb[0::2, 1::2, 0] = r
    rgb[1::2, 0::2, 0] = r
    rgb[1::2, 1::2, 0] = r
    rgb[0::2, 0::2, 1] = g_top
    rgb[0::2, 1::2, 1] = g_top
    rgb[1::2, 0::2, 1] = g_bottom
    rgb[1::2, 1::2, 1] = g_bottom
    rgb[0::2, 0::2, 2] = b
    rgb[0::2, 1::2, 2] = b
    rgb[1::2, 0::2, 2] = b
    rgb[1::2, 1::2, 2] = b
    return rgb


def save_preview(output_path: str | Path, clean_bayer: np.ndarray, low_light_bayer: np.ndarray) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    clean_rgb = bayer_to_nearest_rgb(clean_bayer)
    low_rgb = bayer_to_nearest_rgb(low_light_bayer)
    preview = np.concatenate((_to_uint8(clean_rgb), _to_uint8(low_rgb)), axis=1)
    Image.fromarray(preview, mode="RGB").save(output)

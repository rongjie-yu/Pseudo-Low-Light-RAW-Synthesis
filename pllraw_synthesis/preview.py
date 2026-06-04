from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image, ImageDraw


@dataclass(frozen=True)
class PreviewTile:
    ratio: float
    raw_demosaic: np.ndarray
    isp_rgb: np.ndarray


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


def save_ratio_grid_preview(output_path: str | Path, tiles: Sequence[PreviewTile], columns: int = 3) -> None:
    if not tiles:
        raise ValueError("Expected at least one preview tile")
    if columns <= 0:
        raise ValueError(f"columns must be positive, got {columns}")

    first_shape = tiles[0].raw_demosaic.shape
    if len(first_shape) != 3 or first_shape[2] != 3:
        raise ValueError(f"Expected HxWx3 preview arrays, got {first_shape}")

    tile_h, tile_w = first_shape[:2]
    label_h = 20
    cell_h = label_h + tile_h * 2
    rows = math.ceil(len(tiles) / columns)
    canvas = np.zeros((rows * cell_h, columns * tile_w, 3), dtype=np.uint8)

    for index, tile in enumerate(tiles):
        if tile.raw_demosaic.shape != first_shape or tile.isp_rgb.shape != first_shape:
            raise ValueError("All preview arrays must share the same HxWx3 shape")
        row = index // columns
        col = index % columns
        y0 = row * cell_h
        x0 = col * tile_w

        canvas[y0 : y0 + label_h, x0 : x0 + tile_w] = 24
        canvas[y0 + label_h : y0 + label_h + tile_h, x0 : x0 + tile_w] = _to_uint8(tile.raw_demosaic)
        canvas[y0 + label_h + tile_h : y0 + cell_h, x0 : x0 + tile_w] = _to_uint8(tile.isp_rgb)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    image = Image.fromarray(canvas, mode="RGB")
    draw = ImageDraw.Draw(image)
    for index, tile in enumerate(tiles):
        row = index // columns
        col = index % columns
        y0 = row * cell_h
        x0 = col * tile_w
        label = f"ratio {tile.ratio:g}"
        draw.text((x0 + 4, y0 + 4), label, fill=(235, 235, 235))
    image.save(output)

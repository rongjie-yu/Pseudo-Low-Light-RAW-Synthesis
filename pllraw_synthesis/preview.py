from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

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


def normalize_for_display(raw: np.ndarray) -> np.ndarray:
    """Z-score normalize a grayscale-converted image for visibility.

    Designed for extremely dark low-light RAW demosaics that are nearly
    invisible under direct [0,1] clipping.  Returns a float32 [0,1] array
    with the same spatial shape as *raw*.
    """
    gray = raw.astype(np.float64).mean(axis=-1)
    mean = float(gray.mean())
    std = float(gray.std())
    if std < 1e-10:
        std = 1.0
    z = (gray - mean) / std
    z_min = float(z.min())
    z_max = float(z.max())
    z_norm = (z - z_min) / max(z_max - z_min, 1e-10)
    return np.dstack([z_norm] * raw.shape[-1]).astype(np.float32)


def save_grid_preview(
    output_path: str | Path,
    original_rgb: np.ndarray,
    low_light_demosaic: np.ndarray,
    tile: PreviewTile,
) -> None:
    """Render a 2×2 preview grid.

    Layout::

        row 0:  Original       |  Low-light RAW
        row 1:  Noisy RAW      |  ISP RGB

    Each cell has a 20 px label bar.  No ratio label is drawn on the image;
    embed the ratio in the output filename instead.
    """
    if original_rgb.ndim != 3 or original_rgb.shape[2] != 3:
        raise ValueError(f"Expected HxWx3 original RGB, got shape {original_rgb.shape}")
    if low_light_demosaic.ndim != 3 or low_light_demosaic.shape[2] != 3:
        raise ValueError(f"Expected HxWx3 low-light demosaic, got shape {low_light_demosaic.shape}")
    if tile.raw_demosaic.ndim != 3 or tile.raw_demosaic.shape[2] != 3:
        raise ValueError(f"Expected HxWx3 noisy RAW demosaic, got shape {tile.raw_demosaic.shape}")
    if tile.isp_rgb.ndim != 3 or tile.isp_rgb.shape[2] != 3:
        raise ValueError(f"Expected HxWx3 ISP RGB, got shape {tile.isp_rgb.shape}")

    tile_h, tile_w = original_rgb.shape[:2]
    label_h = 20
    cell_h = label_h + tile_h

    canvas = np.zeros((2 * cell_h, 2 * tile_w, 3), dtype=np.uint8)

    cells = [
        (0, 0, "Original", _to_uint8(original_rgb)),
        (0, 1, "Low-light RAW", _to_uint8(low_light_demosaic)),
        (1, 0, "Noisy RAW", _to_uint8(tile.raw_demosaic)),
        (1, 1, "ISP RGB", _to_uint8(tile.isp_rgb)),
    ]

    for row, col, _label, img_uint8 in cells:
        y0 = row * cell_h
        x0 = col * tile_w

        canvas[y0 : y0 + label_h, x0 : x0 + tile_w] = 24
        canvas[y0 + label_h : y0 + cell_h, x0 : x0 + tile_w] = img_uint8

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    image = Image.fromarray(canvas, mode="RGB")
    draw = ImageDraw.Draw(image)

    for row, col, label, _ in cells:
        y0 = row * cell_h
        x0 = col * tile_w
        draw.text((x0 + 4, y0 + 1), label, fill=(235, 235, 235))

    image.save(output)

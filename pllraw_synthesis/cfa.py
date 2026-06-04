from __future__ import annotations

from typing import Dict, Tuple

import numpy as np


SUPPORTED_CFA = {"RGGB"}


def crop_even(image: np.ndarray) -> Tuple[np.ndarray, Dict[str, int]]:
    if image.ndim < 2:
        raise ValueError(f"Expected image with at least 2 dimensions, got shape {image.shape}")

    height_before, width_before = image.shape[:2]
    height = height_before - (height_before % 2)
    width = width_before - (width_before % 2)
    cropped = image[:height, :width, ...]
    meta = {
        "height_before_crop": int(height_before),
        "width_before_crop": int(width_before),
        "height": int(height),
        "width": int(width),
    }
    return cropped, meta


def extract_bayer(demosaiced_raw: np.ndarray, cfa: str = "RGGB") -> np.ndarray:
    cfa = cfa.upper()
    if cfa not in SUPPORTED_CFA:
        raise ValueError(f"Unsupported CFA '{cfa}'. Supported patterns: {sorted(SUPPORTED_CFA)}")
    if demosaiced_raw.ndim != 3 or demosaiced_raw.shape[2] != 3:
        raise ValueError(f"Expected HxWx3 demosaiced RAW, got shape {demosaiced_raw.shape}")

    raw, _ = crop_even(demosaiced_raw)
    bayer = np.empty(raw.shape[:2], dtype=np.float32)
    bayer[0::2, 0::2] = raw[0::2, 0::2, 0]
    bayer[0::2, 1::2] = raw[0::2, 1::2, 1]
    bayer[1::2, 0::2] = raw[1::2, 0::2, 1]
    bayer[1::2, 1::2] = raw[1::2, 1::2, 2]
    return bayer

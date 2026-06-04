from __future__ import annotations

import numpy as np


def pack_bayer(bayer: np.ndarray) -> np.ndarray:
    if bayer.ndim != 2:
        raise ValueError(f"Expected 2D Bayer RAW, got shape {bayer.shape}")
    height, width = bayer.shape
    if height % 2 or width % 2:
        raise ValueError(f"Bayer RAW dimensions must be even, got {height}x{width}")

    return np.stack(
        (
            bayer[0:height:2, 0:width:2],
            bayer[0:height:2, 1:width:2],
            bayer[1:height:2, 1:width:2],
            bayer[1:height:2, 0:width:2],
        ),
        axis=0,
    ).astype(np.float32, copy=False)


def unpack_bayer(packed: np.ndarray) -> np.ndarray:
    if packed.ndim != 3 or packed.shape[0] != 4:
        raise ValueError(f"Expected packed RAW with shape 4xHxW, got {packed.shape}")

    _, half_h, half_w = packed.shape
    bayer = np.empty((half_h * 2, half_w * 2), dtype=np.float32)
    bayer[0::2, 0::2] = packed[0]
    bayer[0::2, 1::2] = packed[1]
    bayer[1::2, 1::2] = packed[2]
    bayer[1::2, 0::2] = packed[3]
    return bayer

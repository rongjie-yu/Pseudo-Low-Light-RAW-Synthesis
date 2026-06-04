from pathlib import Path

import numpy as np
from PIL import Image

from pllraw_synthesis.preview import PreviewTile, bayer_to_nearest_rgb, save_grid_preview


def test_bayer_to_nearest_rgb_directly_expands_bayer_values():
    bayer = np.array(
        [
            [0.1, 0.2],
            [0.3, 0.4],
        ],
        dtype=np.float32,
    )

    rgb = bayer_to_nearest_rgb(bayer)

    expected = np.array(
        [
            [[0.1, 0.2, 0.4], [0.1, 0.2, 0.4]],
            [[0.1, 0.3, 0.4], [0.1, 0.3, 0.4]],
        ],
        dtype=np.float32,
    )
    np.testing.assert_array_equal(rgb, expected)


def test_save_grid_preview_2x2_layout(tmp_path: Path):
    original = np.full((8, 10, 3), 0.8, dtype=np.float32)
    low_light = np.full((8, 10, 3), 0.5, dtype=np.float32)
    tile = PreviewTile(
        ratio=200.0,
        raw_demosaic=np.full((8, 10, 3), 0.3, dtype=np.float32),
        isp_rgb=np.full((8, 10, 3), 0.7, dtype=np.float32),
    )
    output = tmp_path / "preview_r200.png"

    save_grid_preview(output, original, low_light, tile)

    assert output.exists()
    with Image.open(output) as img:
        assert img.mode == "RGB"
        # 2 cols × 10 = 20 wide, 2 rows × (label 20 + tile 8) = 56 tall
        assert img.size == (20, 56)

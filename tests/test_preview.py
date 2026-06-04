from pathlib import Path

import numpy as np
from PIL import Image

from pllraw_synthesis.preview import PreviewTile, bayer_to_nearest_rgb, save_ratio_grid_preview


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


def test_save_ratio_grid_preview_uses_three_column_layout(tmp_path: Path):
    tiles = [
        PreviewTile(
            ratio=float(ratio),
            raw_demosaic=np.full((8, 10, 3), ratio / 300.0, dtype=np.float32),
            isp_rgb=np.full((8, 10, 3), 1.0 - ratio / 300.0, dtype=np.float32),
        )
        for ratio in (100, 200, 300)
    ]
    output = tmp_path / "preview.png"

    save_ratio_grid_preview(output, tiles)

    assert output.exists()
    with Image.open(output) as img:
        assert img.mode == "RGB"
        assert img.size == (30, 36)

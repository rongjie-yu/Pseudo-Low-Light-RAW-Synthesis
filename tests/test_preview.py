from pathlib import Path

import numpy as np
from PIL import Image

from pllraw_synthesis.preview import PreviewTile, save_ratio_grid_preview


def test_save_ratio_grid_preview_uses_three_by_two_layout(tmp_path: Path):
    tiles = [
        PreviewTile(
            ratio=float(ratio),
            raw_demosaic=np.full((8, 10, 3), ratio / 300.0, dtype=np.float32),
            isp_rgb=np.full((8, 10, 3), 1.0 - ratio / 300.0, dtype=np.float32),
        )
        for ratio in (50, 100, 150, 200, 250, 300)
    ]
    output = tmp_path / "preview.png"

    save_ratio_grid_preview(output, tiles)

    assert output.exists()
    with Image.open(output) as img:
        assert img.mode == "RGB"
        assert img.size == (30, 72)

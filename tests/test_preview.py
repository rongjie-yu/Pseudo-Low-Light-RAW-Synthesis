from pathlib import Path

import numpy as np
from PIL import Image

from pllraw_synthesis.preview import save_preview


def test_save_preview_writes_png(tmp_path: Path):
    clean = np.full((4, 4), 0.25, dtype=np.float32)
    low = np.full((4, 4), 0.75, dtype=np.float32)
    output = tmp_path / "preview.png"

    save_preview(output, clean_bayer=clean, low_light_bayer=low)

    assert output.exists()
    with Image.open(output) as img:
        assert img.mode == "RGB"
        assert img.size == (8, 4)

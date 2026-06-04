from pathlib import Path

import numpy as np
from PIL import Image

from pllraw_synthesis.io import discover_images, load_rgb_image, make_output_stem, save_npz


def test_discover_images_supports_png_jpeg_and_sorting(tmp_path: Path):
    (tmp_path / "b.txt").write_text("ignore", encoding="utf-8")
    Image.new("RGB", (2, 2)).save(tmp_path / "b.jpg")
    Image.new("RGB", (2, 2)).save(tmp_path / "a.png")
    nested = tmp_path / "nested"
    nested.mkdir()
    Image.new("RGB", (2, 2)).save(nested / "c.jpeg")

    paths = discover_images(tmp_path)

    assert [p.name for p in paths] == ["a.png", "b.jpg", "c.jpeg"]


def test_load_rgb_image_returns_float_and_crop_meta(tmp_path: Path):
    image_path = tmp_path / "odd.png"
    Image.new("RGB", (7, 5), color=(255, 128, 0)).save(image_path)

    image, meta = load_rgb_image(image_path)

    assert image.shape == (4, 6, 3)
    assert image.dtype == np.float32
    assert image.max() <= 1.0
    assert meta["height_before_crop"] == 5
    assert meta["width_before_crop"] == 7
    assert meta["height"] == 4
    assert meta["width"] == 6


def test_make_output_stem_is_collision_resistant():
    path = Path("/data/a/b/sample.png")

    stem = make_output_stem(path, ratio=100.0)

    assert stem.endswith("_ratio100")
    assert "sample" in stem


def test_save_npz_writes_expected_arrays_and_metadata(tmp_path: Path):
    output = tmp_path / "sample_ratio50.npz"
    clean = np.zeros((4, 2, 2), dtype=np.float32)
    low = np.ones((4, 2, 2), dtype=np.float32)

    save_npz(
        output,
        clean_raw=clean,
        low_light_raw=low,
        metadata={
            "ratio": 50.0,
            "source_path": "/tmp/sample.png",
            "cfa": "RGGB",
            "noise_params": {"K": 1.0, "g_scale": 2.0, "ratio": 50.0, "saturation_level": 15583.0},
        },
    )

    loaded = np.load(output, allow_pickle=True)
    np.testing.assert_array_equal(loaded["clean_raw"], clean)
    np.testing.assert_array_equal(loaded["low_light_raw"], low)
    assert loaded["ratio"].item() == 50.0
    assert loaded["source_path"].item() == "/tmp/sample.png"
    assert loaded["cfa"].item() == "RGGB"
    assert loaded["noise_params"].item()["K"] == 1.0

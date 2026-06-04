import numpy as np
import pytest

from pllraw_synthesis.cfa import crop_even, extract_bayer


def test_crop_even_removes_last_odd_row_and_column():
    image = np.zeros((5, 7, 3), dtype=np.float32)

    cropped, meta = crop_even(image)

    assert cropped.shape == (4, 6, 3)
    assert meta == {
        "height_before_crop": 5,
        "width_before_crop": 7,
        "height": 4,
        "width": 6,
    }


def test_extract_bayer_rggb_from_demosaiced_raw():
    demosaiced = np.zeros((4, 4, 3), dtype=np.float32)
    demosaiced[..., 0] = 10.0
    demosaiced[..., 1] = 20.0
    demosaiced[..., 2] = 30.0

    bayer = extract_bayer(demosaiced, cfa="RGGB")

    expected = np.array(
        [
            [10.0, 20.0, 10.0, 20.0],
            [20.0, 30.0, 20.0, 30.0],
            [10.0, 20.0, 10.0, 20.0],
            [20.0, 30.0, 20.0, 30.0],
        ],
        dtype=np.float32,
    )
    np.testing.assert_array_equal(bayer, expected)


def test_extract_bayer_rejects_unknown_cfa():
    demosaiced = np.zeros((4, 4, 3), dtype=np.float32)

    with pytest.raises(ValueError, match="Unsupported CFA"):
        extract_bayer(demosaiced, cfa="BADCFA")

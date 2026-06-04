import numpy as np

from pllraw_synthesis.packing import pack_bayer, unpack_bayer


def test_pack_bayer_rggb_channel_order_matches_eld_reference():
    bayer = np.array(
        [
            [1.0, 2.0, 5.0, 6.0],
            [4.0, 3.0, 8.0, 7.0],
            [9.0, 10.0, 13.0, 14.0],
            [12.0, 11.0, 16.0, 15.0],
        ],
        dtype=np.float32,
    )

    packed = pack_bayer(bayer)

    assert packed.shape == (4, 2, 2)
    np.testing.assert_array_equal(packed[0], np.array([[1.0, 5.0], [9.0, 13.0]], dtype=np.float32))
    np.testing.assert_array_equal(packed[1], np.array([[2.0, 6.0], [10.0, 14.0]], dtype=np.float32))
    np.testing.assert_array_equal(packed[2], np.array([[3.0, 7.0], [11.0, 15.0]], dtype=np.float32))
    np.testing.assert_array_equal(packed[3], np.array([[4.0, 8.0], [12.0, 16.0]], dtype=np.float32))


def test_unpack_bayer_reverses_pack_bayer():
    bayer = np.arange(24, dtype=np.float32).reshape(4, 6)

    unpacked = unpack_bayer(pack_bayer(bayer))

    np.testing.assert_array_equal(unpacked, bayer)

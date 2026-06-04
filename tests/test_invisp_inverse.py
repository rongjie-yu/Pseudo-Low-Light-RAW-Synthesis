from pathlib import Path

import numpy as np
import pytest
import torch

from pllraw_synthesis.invisp_inverse import InvISPInverse


def test_invisp_inverse_rejects_missing_checkpoint(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        InvISPInverse(checkpoint_path=tmp_path / "missing.pth", device="cpu")


def test_prepare_rgb_tensor_shape_without_loading_model():
    image = np.zeros((4, 6, 3), dtype=np.float32)

    tensor = InvISPInverse.prepare_rgb_tensor(image, device=torch.device("cpu"))

    assert tensor.shape == (1, 3, 4, 6)
    assert tensor.dtype == torch.float32


def test_postprocess_demosaiced_raw_clips_and_returns_hwc():
    tensor = torch.tensor([[[[-1.0, 0.5]], [[2.0, 0.25]], [[0.75, 0.0]]]], dtype=torch.float32)

    raw = InvISPInverse.postprocess_demosaiced_raw(tensor)

    assert raw.shape == (1, 2, 3)
    assert raw.dtype == np.float32
    assert raw.min() >= 0.0
    assert raw.max() <= 1.0

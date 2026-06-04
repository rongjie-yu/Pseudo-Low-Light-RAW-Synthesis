from pathlib import Path

import numpy as np
import torch

from pllraw_synthesis.invisp_model.model import InvISPNet


def test_copied_camera_params_have_expected_keys():
    params_path = Path("assets/camera_params/CanonEOS5D4_params.npy")
    params = np.load(params_path, allow_pickle=True).item()

    assert {"Kmin", "Kmax", "Profile-1"}.issubset(params.keys())
    assert "g_scale" in params["Profile-1"]


def test_copied_invisp_checkpoint_loads_into_copied_model_on_cpu():
    checkpoint = Path("assets/checkpoints/invisp_canon_eos_5d.pth")
    model = InvISPNet(channel_in=3, channel_out=3, block_num=8)

    state_dict = torch.load(checkpoint, map_location="cpu")
    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    assert missing == []
    assert all(key.endswith((".actnorm.bias", ".actnorm.logs")) for key in unexpected)

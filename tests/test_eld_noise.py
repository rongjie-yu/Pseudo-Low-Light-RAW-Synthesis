from pathlib import Path

import numpy as np
import pytest

from pllraw_synthesis.eld_noise import ELDNoiseModel, NoiseParams


def test_noise_params_are_deterministic_for_seed(tmp_path: Path):
    params_file = tmp_path / "CanonEOS5D4_params.npy"
    np.save(
        params_file,
        {
            "Kmin": np.float64(0.2),
            "Kmax": np.float64(2.0),
            "Profile-1": {
                "g_scale": {"slope": 0.5, "bias": 1.0, "sigma": 0.1},
            },
        },
    )
    model_a = ELDNoiseModel(params_file=params_file, seed=123)
    model_b = ELDNoiseModel(params_file=params_file, seed=123)

    assert model_a.sample_params(ratio=50.0) == model_b.sample_params(ratio=50.0)


def test_apply_noise_returns_same_shape_and_metadata(tmp_path: Path):
    params_file = tmp_path / "CanonEOS5D4_params.npy"
    np.save(
        params_file,
        {
            "Kmin": np.float64(1.0),
            "Kmax": np.float64(1.0),
            "Profile-1": {
                "g_scale": {"slope": 0.0, "bias": -30.0, "sigma": 0.0},
            },
        },
    )
    model = ELDNoiseModel(params_file=params_file, seed=0, saturation_level=100.0)
    clean = np.full((4, 4), 0.5, dtype=np.float32)

    noisy, params = model.apply(clean, ratio=25.0)

    assert noisy.shape == clean.shape
    assert noisy.dtype == np.float32
    assert np.isfinite(noisy).all()
    assert isinstance(params, NoiseParams)
    assert params.ratio == 25.0
    assert params.saturation_level == 100.0


def test_explicit_noise_params_skip_sampling(tmp_path: Path):
    params_file = tmp_path / "CanonEOS5D4_params.npy"
    np.save(
        params_file,
        {
            "Kmin": np.float64(1.0),
            "Kmax": np.float64(1.0),
            "Profile-1": {
                "g_scale": {"slope": 0.0, "bias": -30.0, "sigma": 0.0},
            },
        },
    )
    model = ELDNoiseModel(params_file=params_file, seed=0, saturation_level=100.0)
    clean = np.full((4, 4), 0.5, dtype=np.float32)
    explicit = NoiseParams(K=1.0, g_scale=0.0, ratio=10.0, saturation_level=100.0)

    _, used = model.apply(clean, ratio=10.0, params=explicit)

    assert used == explicit


def test_explicit_params_with_mismatched_ratio_raises(tmp_path: Path):
    params_file = tmp_path / "CanonEOS5D4_params.npy"
    np.save(
        params_file,
        {
            "Kmin": np.float64(1.0),
            "Kmax": np.float64(1.0),
            "Profile-1": {
                "g_scale": {"slope": 0.0, "bias": -30.0, "sigma": 0.0},
            },
        },
    )
    model = ELDNoiseModel(params_file=params_file, seed=0, saturation_level=100.0)
    clean = np.full((4, 4), 0.5, dtype=np.float32)
    explicit = NoiseParams(K=1.0, g_scale=0.0, ratio=10.0, saturation_level=100.0)

    with pytest.raises(ValueError, match="does not match ratio argument"):
        model.apply(clean, ratio=999.0, params=explicit)

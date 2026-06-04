from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np


DEFAULT_SATURATION_LEVEL = 16383.0 - 800.0


@dataclass(frozen=True)
class NoiseParams:
    K: float
    g_scale: float
    ratio: float
    saturation_level: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


class ELDNoiseModel:
    def __init__(
        self,
        params_file: str | Path,
        seed: Optional[int] = None,
        saturation_level: float = DEFAULT_SATURATION_LEVEL,
    ) -> None:
        self.params_file = Path(params_file)
        if not self.params_file.exists():
            raise FileNotFoundError(f"Camera params file not found: {self.params_file}")
        self.camera_params = np.load(self.params_file, allow_pickle=True).item()
        self.rng = np.random.default_rng(seed)
        self.saturation_level = float(saturation_level)

    def sample_params(self, ratio: float) -> NoiseParams:
        camera_params = self.camera_params
        profile = camera_params["Profile-1"]
        g_scale_params = profile["g_scale"]

        log_k = self.rng.uniform(np.log(float(camera_params["Kmin"])), np.log(float(camera_params["Kmax"])))
        log_g_scale = (
            self.rng.standard_normal() * float(g_scale_params["sigma"])
            + float(g_scale_params["slope"]) * log_k
            + float(g_scale_params["bias"])
        )
        return NoiseParams(
            K=float(np.exp(log_k)),
            g_scale=float(np.exp(log_g_scale)),
            ratio=float(ratio),
            saturation_level=self.saturation_level,
        )

    def apply(
        self,
        clean_raw: np.ndarray,
        ratio: float,
        params: Optional[NoiseParams] = None,
        *,
        amplify: bool = True,
    ) -> Tuple[np.ndarray, NoiseParams]:
        if ratio <= 0:
            raise ValueError(f"ratio must be positive, got {ratio}")
        if clean_raw.ndim != 2:
            raise ValueError(f"Expected 2D clean Bayer RAW, got shape {clean_raw.shape}")
        if params is not None and params.ratio != float(ratio):
            raise ValueError(
                f"Explicit params ratio ({params.ratio}) does not match ratio argument ({ratio}). "
                f"When providing explicit NoiseParams, ratio must match the apply() ratio argument."
            )

        used = params if params is not None else self.sample_params(ratio)
        y = clean_raw.astype(np.float32, copy=False)
        y = np.clip(y, 0.0, 1.0) * used.saturation_level
        y = y / used.ratio

        poisson_rate = np.maximum(y / max(used.K, 1e-10), 0.0)
        z = self.rng.poisson(poisson_rate).astype(np.float32) * used.K

        if used.g_scale > 0:
            z = z + self.rng.standard_normal(size=y.shape).astype(np.float32) * max(used.g_scale, 1e-10)

        if amplify:
            z = z * used.ratio
        z = z / used.saturation_level
        z = np.clip(z, 0.0, 1.0).astype(np.float32, copy=False)
        return z, used

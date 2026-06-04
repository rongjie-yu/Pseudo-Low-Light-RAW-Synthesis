from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import torch

from .invisp_model import InvISPNet


DEFAULT_CHECKPOINT = Path("assets/checkpoints/invisp_canon_eos_5d.pth")


class InvISPInverse:
    def __init__(
        self,
        checkpoint_path: str | Path = DEFAULT_CHECKPOINT,
        device: str = "auto",
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path)
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"InvISP checkpoint not found: {self.checkpoint_path}")
        self.device = self.resolve_device(device)
        self.model = InvISPNet(channel_in=3, channel_out=3, block_num=8).to(self.device)
        state_dict = torch.load(self.checkpoint_path, map_location=self.device)
        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
        if missing:
            warnings.warn(f"Missing keys in checkpoint '{self.checkpoint_path}': {missing}")
        non_actnorm_unexpected = [
            k for k in unexpected if not k.endswith((".actnorm.bias", ".actnorm.logs"))
        ]
        if non_actnorm_unexpected:
            warnings.warn(
                f"Unexpected keys in checkpoint '{self.checkpoint_path}': {non_actnorm_unexpected}"
            )
        self.model.eval()

    @staticmethod
    def resolve_device(device: str) -> torch.device:
        if device == "auto":
            return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        resolved = torch.device(device)
        if resolved.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        return resolved

    @staticmethod
    def prepare_rgb_tensor(rgb: np.ndarray, device: torch.device) -> torch.Tensor:
        if rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError(f"Expected HxWx3 RGB image, got shape {rgb.shape}")
        rgb = np.clip(rgb.astype(np.float32, copy=False), 0.0, 1.0)
        return torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).float().to(device)

    @staticmethod
    def postprocess_demosaiced_raw(raw_tensor: torch.Tensor) -> np.ndarray:
        raw = torch.clamp(raw_tensor.detach(), 0.0, 1.0)
        raw = raw.squeeze(0).permute(1, 2, 0).cpu().numpy()
        if not np.isfinite(raw).all():
            raise RuntimeError("InvISP inverse produced non-finite values in demosaiced RAW output")
        return raw.astype(np.float32, copy=False)

    @staticmethod
    def postprocess_rgb(rgb_tensor: torch.Tensor) -> np.ndarray:
        rgb = torch.clamp(rgb_tensor.detach(), 0.0, 1.0)
        rgb = rgb.squeeze(0).permute(1, 2, 0).cpu().numpy()
        if not np.isfinite(rgb).all():
            raise RuntimeError("InvISP forward produced non-finite values in RGB output")
        return rgb.astype(np.float32, copy=False)

    def rgb_to_demosaiced_raw(self, rgb: np.ndarray) -> np.ndarray:
        tensor = self.prepare_rgb_tensor(rgb, self.device)
        with torch.no_grad():
            raw = self.model(tensor, rev=True)
        return self.postprocess_demosaiced_raw(raw)

    def demosaiced_raw_to_rgb(self, demosaiced_raw: np.ndarray) -> np.ndarray:
        tensor = self.prepare_rgb_tensor(demosaiced_raw, self.device)
        with torch.no_grad():
            rgb = self.model(tensor, rev=False)
        return self.postprocess_rgb(rgb)

# Pseudo Low-Light RAW Synthesis Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a self-contained InvISP + ELD tool that converts RGB PNG/JPEG images to Canon-style pseudo clean Bayer RAW and synthesizes pseudo low-light RAW at arbitrary ratios.

**Architecture:** The implementation lives in a new project-owned package, `pllraw_synthesis/`, plus one CLI under `scripts/`. Reference code and assets are copied out of `third-party/` into project-owned paths; final runtime code must not import from `third-party/` or `reference/`.

**Tech Stack:** Python 3, NumPy, Pillow, PyTorch, pytest, copied InvISP model definitions, copied Canon InvISP checkpoint, copied ELD CanonEOS5D4 parameter file.

---

## File Structure

- Create `pllraw_synthesis/__init__.py`: package marker and public version.
- Create `pllraw_synthesis/cfa.py`: CFA validation and demosaiced RAW to Bayer extraction.
- Create `pllraw_synthesis/packing.py`: Bayer RAW pack/unpack helpers.
- Create `pllraw_synthesis/eld_noise.py`: ELD baseline Poisson + Gaussian noise synthesis.
- Create `pllraw_synthesis/io.py`: image discovery, image loading/cropping, output naming, `.npz` writing.
- Create `pllraw_synthesis/preview.py`: diagnostic preview rendering.
- Create `pllraw_synthesis/invisp_inverse.py`: InvISP checkpoint loading and RGB-to-demosaiced-RAW inference.
- Create `pllraw_synthesis/invisp_model/`: copied InvISP model package.
- Create `assets/checkpoints/invisp_canon_eos_5d.pth`: copied Canon checkpoint.
- Create `assets/camera_params/CanonEOS5D4_params.npy`: copied ELD camera params.
- Create `scripts/synthesize_pseudo_lowlight_raw.py`: CLI pipeline.
- Create `tests/test_cfa.py`, `tests/test_packing.py`, `tests/test_eld_noise.py`, `tests/test_io.py`, `tests/test_preview.py`, `tests/test_cli.py`.

Important repository condition: current git index already contains unrelated staged `third-party/*` and `.gitmodules`. Every commit in this plan must use explicit path-limited `git add` and `git commit -- <paths>` commands.

---

### Task 1: Package Skeleton, CFA Extraction, And Bayer Packing

**Files:**
- Create: `pllraw_synthesis/__init__.py`
- Create: `pllraw_synthesis/cfa.py`
- Create: `pllraw_synthesis/packing.py`
- Test: `tests/test_cfa.py`
- Test: `tests/test_packing.py`

- [ ] **Step 1: Write failing CFA tests**

Create `tests/test_cfa.py`:

```python
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
```

- [ ] **Step 2: Write failing packing tests**

Create `tests/test_packing.py`:

```python
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
```

- [ ] **Step 3: Run tests to verify they fail**

Run:

```bash
python -m pytest tests/test_cfa.py tests/test_packing.py -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'pllraw_synthesis'`.

- [ ] **Step 4: Implement package skeleton and CFA helpers**

Create `pllraw_synthesis/__init__.py`:

```python
"""Pseudo low-light RAW synthesis tools."""

__version__ = "0.1.0"
```

Create `pllraw_synthesis/cfa.py`:

```python
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np


SUPPORTED_CFA = {"RGGB"}


def crop_even(image: np.ndarray) -> Tuple[np.ndarray, Dict[str, int]]:
    if image.ndim < 2:
        raise ValueError(f"Expected image with at least 2 dimensions, got shape {image.shape}")

    height_before, width_before = image.shape[:2]
    height = height_before - (height_before % 2)
    width = width_before - (width_before % 2)
    cropped = image[:height, :width, ...]
    meta = {
        "height_before_crop": int(height_before),
        "width_before_crop": int(width_before),
        "height": int(height),
        "width": int(width),
    }
    return cropped, meta


def extract_bayer(demosaiced_raw: np.ndarray, cfa: str = "RGGB") -> np.ndarray:
    cfa = cfa.upper()
    if cfa not in SUPPORTED_CFA:
        raise ValueError(f"Unsupported CFA '{cfa}'. Supported patterns: {sorted(SUPPORTED_CFA)}")
    if demosaiced_raw.ndim != 3 or demosaiced_raw.shape[2] != 3:
        raise ValueError(f"Expected HxWx3 demosaiced RAW, got shape {demosaiced_raw.shape}")

    raw, _ = crop_even(demosaiced_raw)
    bayer = np.empty(raw.shape[:2], dtype=np.float32)
    bayer[0::2, 0::2] = raw[0::2, 0::2, 0]
    bayer[0::2, 1::2] = raw[0::2, 1::2, 1]
    bayer[1::2, 0::2] = raw[1::2, 0::2, 1]
    bayer[1::2, 1::2] = raw[1::2, 1::2, 2]
    return bayer
```

- [ ] **Step 5: Implement Bayer packing helpers**

Create `pllraw_synthesis/packing.py`:

```python
from __future__ import annotations

import numpy as np


def pack_bayer(bayer: np.ndarray) -> np.ndarray:
    if bayer.ndim != 2:
        raise ValueError(f"Expected 2D Bayer RAW, got shape {bayer.shape}")
    height, width = bayer.shape
    if height % 2 or width % 2:
        raise ValueError(f"Bayer RAW dimensions must be even, got {height}x{width}")

    return np.stack(
        (
            bayer[0:height:2, 0:width:2],
            bayer[0:height:2, 1:width:2],
            bayer[1:height:2, 1:width:2],
            bayer[1:height:2, 0:width:2],
        ),
        axis=0,
    ).astype(np.float32, copy=False)


def unpack_bayer(packed: np.ndarray) -> np.ndarray:
    if packed.ndim != 3 or packed.shape[0] != 4:
        raise ValueError(f"Expected packed RAW with shape 4xHxW, got {packed.shape}")

    _, half_h, half_w = packed.shape
    bayer = np.empty((half_h * 2, half_w * 2), dtype=np.float32)
    bayer[0::2, 0::2] = packed[0]
    bayer[0::2, 1::2] = packed[1]
    bayer[1::2, 1::2] = packed[2]
    bayer[1::2, 0::2] = packed[3]
    return bayer
```

- [ ] **Step 6: Run tests to verify they pass**

Run:

```bash
python -m pytest tests/test_cfa.py tests/test_packing.py -q
```

Expected: PASS, `5 passed`.

- [ ] **Step 7: Commit Task 1**

Run:

```bash
git add pllraw_synthesis/__init__.py pllraw_synthesis/cfa.py pllraw_synthesis/packing.py tests/test_cfa.py tests/test_packing.py
git commit -m "Add CFA and Bayer packing utilities" -- pllraw_synthesis/__init__.py pllraw_synthesis/cfa.py pllraw_synthesis/packing.py tests/test_cfa.py tests/test_packing.py
```

---

### Task 2: ELD Baseline Noise Model

**Files:**
- Create: `pllraw_synthesis/eld_noise.py`
- Test: `tests/test_eld_noise.py`

- [ ] **Step 1: Write failing ELD noise tests**

Create `tests/test_eld_noise.py`:

```python
from pathlib import Path

import numpy as np

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

    _, used = model.apply(clean, ratio=999.0, params=explicit)

    assert used == explicit
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
python -m pytest tests/test_eld_noise.py -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'pllraw_synthesis.eld_noise'`.

- [ ] **Step 3: Implement ELD noise model**

Create `pllraw_synthesis/eld_noise.py`:

```python
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
    ) -> Tuple[np.ndarray, NoiseParams]:
        if ratio <= 0:
            raise ValueError(f"ratio must be positive, got {ratio}")
        if clean_raw.ndim != 2:
            raise ValueError(f"Expected 2D clean Bayer RAW, got shape {clean_raw.shape}")

        used = params if params is not None else self.sample_params(ratio)
        y = clean_raw.astype(np.float32, copy=False)
        y = np.clip(y, 0.0, 1.0) * used.saturation_level
        y = y / used.ratio

        poisson_rate = np.maximum(y / max(used.K, 1e-10), 0.0)
        z = self.rng.poisson(poisson_rate).astype(np.float32) * used.K

        if used.g_scale > 0:
            z = z + self.rng.standard_normal(size=y.shape).astype(np.float32) * max(used.g_scale, 1e-10)

        z = z * used.ratio
        z = z / used.saturation_level
        z = np.clip(z, 0.0, 1.0).astype(np.float32, copy=False)
        return z, used
```

- [ ] **Step 4: Run ELD tests to verify they pass**

Run:

```bash
python -m pytest tests/test_eld_noise.py -q
```

Expected: PASS, `3 passed`.

- [ ] **Step 5: Commit Task 2**

Run:

```bash
git add pllraw_synthesis/eld_noise.py tests/test_eld_noise.py
git commit -m "Add ELD baseline noise model" -- pllraw_synthesis/eld_noise.py tests/test_eld_noise.py
```

---

### Task 3: IO, Metadata, And Preview Utilities

**Files:**
- Create: `pllraw_synthesis/io.py`
- Create: `pllraw_synthesis/preview.py`
- Test: `tests/test_io.py`
- Test: `tests/test_preview.py`

- [ ] **Step 1: Write failing IO tests**

Create `tests/test_io.py`:

```python
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
```

- [ ] **Step 2: Write failing preview tests**

Create `tests/test_preview.py`:

```python
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
```

- [ ] **Step 3: Run tests to verify they fail**

Run:

```bash
python -m pytest tests/test_io.py tests/test_preview.py -q
```

Expected: FAIL with missing `pllraw_synthesis.io` or `pllraw_synthesis.preview`.

- [ ] **Step 4: Implement IO helpers**

Create `pllraw_synthesis/io.py`:

```python
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
from PIL import Image

from .cfa import crop_even


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}


def discover_images(input_path: str | Path) -> List[Path]:
    path = Path(input_path)
    if path.is_file():
        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            raise ValueError(f"Unsupported image extension: {path}")
        return [path]
    if not path.exists():
        raise FileNotFoundError(f"Input path not found: {path}")
    if not path.is_dir():
        raise ValueError(f"Input path is neither file nor directory: {path}")

    return sorted(p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)


def load_rgb_image(path: str | Path) -> Tuple[np.ndarray, Dict[str, int]]:
    image_path = Path(path)
    with Image.open(image_path) as image:
        rgb = image.convert("RGB")
        array = np.asarray(rgb, dtype=np.float32) / 255.0
    return crop_even(array)


def _format_ratio(ratio: float) -> str:
    if float(ratio).is_integer():
        return str(int(ratio))
    return str(ratio).replace(".", "p")


def make_output_stem(source_path: str | Path, ratio: float) -> str:
    path = Path(source_path)
    digest = hashlib.sha1(str(path).encode("utf-8")).hexdigest()[:10]
    return f"{path.stem}_{digest}_ratio{_format_ratio(ratio)}"


def save_npz(
    output_path: str | Path,
    clean_raw: np.ndarray,
    low_light_raw: np.ndarray,
    metadata: Dict[str, Any],
) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        clean_raw=clean_raw.astype(np.float32, copy=False),
        low_light_raw=low_light_raw.astype(np.float32, copy=False),
        ratio=np.array(metadata["ratio"], dtype=np.float32),
        noise_params=np.array(metadata["noise_params"], dtype=object),
        source_path=np.array(str(metadata["source_path"]), dtype=object),
        cfa=np.array(str(metadata["cfa"]), dtype=object),
        height_before_crop=np.array(metadata.get("height_before_crop", -1), dtype=np.int32),
        width_before_crop=np.array(metadata.get("width_before_crop", -1), dtype=np.int32),
        height=np.array(metadata.get("height", -1), dtype=np.int32),
        width=np.array(metadata.get("width", -1), dtype=np.int32),
        saturation_level=np.array(metadata.get("saturation_level", -1.0), dtype=np.float32),
    )
```

- [ ] **Step 5: Implement preview helper**

Create `pllraw_synthesis/preview.py`:

```python
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def _to_uint8(raw: np.ndarray) -> np.ndarray:
    clipped = np.clip(raw, 0.0, 1.0)
    return (clipped * 255.0 + 0.5).astype(np.uint8)


def bayer_to_nearest_rgb(bayer: np.ndarray) -> np.ndarray:
    if bayer.ndim != 2:
        raise ValueError(f"Expected 2D Bayer RAW, got shape {bayer.shape}")
    height, width = bayer.shape
    if height % 2 or width % 2:
        raise ValueError(f"Bayer dimensions must be even, got {height}x{width}")

    rgb = np.zeros((height, width, 3), dtype=np.float32)
    r = bayer[0::2, 0::2]
    g_top = bayer[0::2, 1::2]
    g_bottom = bayer[1::2, 0::2]
    b = bayer[1::2, 1::2]

    rgb[0::2, 0::2, 0] = r
    rgb[0::2, 1::2, 0] = r
    rgb[1::2, 0::2, 0] = r
    rgb[1::2, 1::2, 0] = r
    rgb[0::2, 0::2, 1] = g_top
    rgb[0::2, 1::2, 1] = g_top
    rgb[1::2, 0::2, 1] = g_bottom
    rgb[1::2, 1::2, 1] = g_bottom
    rgb[0::2, 0::2, 2] = b
    rgb[0::2, 1::2, 2] = b
    rgb[1::2, 0::2, 2] = b
    rgb[1::2, 1::2, 2] = b
    return rgb


def save_preview(output_path: str | Path, clean_bayer: np.ndarray, low_light_bayer: np.ndarray) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    clean_rgb = bayer_to_nearest_rgb(clean_bayer)
    low_rgb = bayer_to_nearest_rgb(low_light_bayer)
    preview = np.concatenate((_to_uint8(clean_rgb), _to_uint8(low_rgb)), axis=1)
    Image.fromarray(preview, mode="RGB").save(output)
```

- [ ] **Step 6: Run IO and preview tests**

Run:

```bash
python -m pytest tests/test_io.py tests/test_preview.py -q
```

Expected: PASS, `5 passed`.

- [ ] **Step 7: Commit Task 3**

Run:

```bash
git add pllraw_synthesis/io.py pllraw_synthesis/preview.py tests/test_io.py tests/test_preview.py
git commit -m "Add synthesis IO and preview helpers" -- pllraw_synthesis/io.py pllraw_synthesis/preview.py tests/test_io.py tests/test_preview.py
```

---

### Task 4: Copy Project-Owned Assets And InvISP Model Code

**Files:**
- Create: `assets/checkpoints/invisp_canon_eos_5d.pth`
- Create: `assets/camera_params/CanonEOS5D4_params.npy`
- Create: `pllraw_synthesis/invisp_model/__init__.py`
- Create: `pllraw_synthesis/invisp_model/model.py`
- Create: `pllraw_synthesis/invisp_model/modules.py`
- Create: `pllraw_synthesis/invisp_model/utils.py`
- Test: `tests/test_assets.py`

- [ ] **Step 1: Write failing asset tests**

Create `tests/test_assets.py`:

```python
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

    assert isinstance(missing, list)
    assert unexpected == []
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
python -m pytest tests/test_assets.py -q
```

Expected: FAIL because copied assets and copied model package do not exist.

- [ ] **Step 3: Copy assets into project-owned paths**

Run:

```bash
mkdir -p assets/checkpoints assets/camera_params
cp third-party/Invertible-ISP/pretrained/canon.pth assets/checkpoints/invisp_canon_eos_5d.pth
cp third-party/ELD/camera_params/release/CanonEOS5D4_params.npy assets/camera_params/CanonEOS5D4_params.npy
```

- [ ] **Step 4: Copy InvISP model code into project package**

Run:

```bash
mkdir -p pllraw_synthesis/invisp_model
cp third-party/Invertible-ISP/model/model.py pllraw_synthesis/invisp_model/model.py
cp third-party/Invertible-ISP/model/modules.py pllraw_synthesis/invisp_model/modules.py
cp third-party/Invertible-ISP/model/utils.py pllraw_synthesis/invisp_model/utils.py
```

Create `pllraw_synthesis/invisp_model/__init__.py`:

```python
"""Copied InvISP model definitions used by pllraw_synthesis."""

from .model import InvISPNet

__all__ = ["InvISPNet"]
```

- [ ] **Step 5: Verify copied code has no runtime `third-party` imports**

Run:

```bash
rg -n "third-party|Invertible-ISP|DarkFeat|reference" pllraw_synthesis/invisp_model
```

Expected: no output.

- [ ] **Step 6: Run asset tests**

Run:

```bash
python -m pytest tests/test_assets.py -q
```

Expected: PASS, `2 passed`.

- [ ] **Step 7: Commit Task 4**

Run:

```bash
git add assets/checkpoints/invisp_canon_eos_5d.pth assets/camera_params/CanonEOS5D4_params.npy pllraw_synthesis/invisp_model tests/test_assets.py
git commit -m "Copy InvISP model and camera assets" -- assets/checkpoints/invisp_canon_eos_5d.pth assets/camera_params/CanonEOS5D4_params.npy pllraw_synthesis/invisp_model tests/test_assets.py
```

---

### Task 5: InvISP Inverse Inference Wrapper

**Files:**
- Create: `pllraw_synthesis/invisp_inverse.py`
- Test: `tests/test_invisp_inverse.py`

- [ ] **Step 1: Write failing InvISP wrapper tests**

Create `tests/test_invisp_inverse.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
python -m pytest tests/test_invisp_inverse.py -q
```

Expected: FAIL with missing `pllraw_synthesis.invisp_inverse`.

- [ ] **Step 3: Implement InvISP wrapper**

Create `pllraw_synthesis/invisp_inverse.py`:

```python
from __future__ import annotations

from pathlib import Path
from typing import Optional

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
        self.model.load_state_dict(state_dict, strict=False)
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
        return raw.astype(np.float32, copy=False)

    def rgb_to_demosaiced_raw(self, rgb: np.ndarray) -> np.ndarray:
        tensor = self.prepare_rgb_tensor(rgb, self.device)
        with torch.no_grad():
            raw = self.model(tensor, rev=True)
        return self.postprocess_demosaiced_raw(raw)
```

- [ ] **Step 4: Run wrapper tests**

Run:

```bash
python -m pytest tests/test_invisp_inverse.py -q
```

Expected: PASS, `3 passed`.

- [ ] **Step 5: Run a tiny CPU load smoke test**

Run:

```bash
python - <<'PY'
from pllraw_synthesis.invisp_inverse import InvISPInverse

model = InvISPInverse(device="cpu")
print(type(model).__name__, model.device)
PY
```

Expected: prints `InvISPInverse cpu`.

- [ ] **Step 6: Commit Task 5**

Run:

```bash
git add pllraw_synthesis/invisp_inverse.py tests/test_invisp_inverse.py
git commit -m "Add InvISP inverse inference wrapper" -- pllraw_synthesis/invisp_inverse.py tests/test_invisp_inverse.py
```

---

### Task 6: End-To-End Synthesis CLI

**Files:**
- Create: `scripts/synthesize_pseudo_lowlight_raw.py`
- Test: `tests/test_cli.py`

- [ ] **Step 1: Write failing CLI unit tests for argument parsing**

Create `tests/test_cli.py`:

```python
from scripts.synthesize_pseudo_lowlight_raw import parse_args


def test_parse_args_requires_input_output_and_ratios():
    args = parse_args([
        "--input",
        "/tmp/images",
        "--output",
        "/tmp/out",
        "--ratios",
        "50",
        "100",
        "--preview",
        "--limit",
        "3",
        "--seed",
        "7",
        "--device",
        "cpu",
    ])

    assert args.input == "/tmp/images"
    assert args.output == "/tmp/out"
    assert args.ratios == [50.0, 100.0]
    assert args.preview is True
    assert args.limit == 3
    assert args.seed == 7
    assert args.device == "cpu"
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
python -m pytest tests/test_cli.py -q
```

Expected: FAIL because `scripts/synthesize_pseudo_lowlight_raw.py` does not exist.

- [ ] **Step 3: Implement CLI pipeline**

Create `scripts/synthesize_pseudo_lowlight_raw.py`:

```python
#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from tqdm import tqdm

from pllraw_synthesis.cfa import extract_bayer
from pllraw_synthesis.eld_noise import ELDNoiseModel
from pllraw_synthesis.invisp_inverse import InvISPInverse
from pllraw_synthesis.io import discover_images, load_rgb_image, make_output_stem, save_npz
from pllraw_synthesis.packing import pack_bayer
from pllraw_synthesis.preview import save_preview


DEFAULT_PARAMS = Path("assets/camera_params/CanonEOS5D4_params.npy")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Synthesize pseudo low-light RAW from RGB PNG/JPEG images.")
    parser.add_argument("--input", required=True, help="Input image file or directory.")
    parser.add_argument("--output", required=True, help="Output directory.")
    parser.add_argument("--ratios", required=True, nargs="+", type=float, help="One or more low-light ratios.")
    parser.add_argument("--preview", action="store_true", help="Write diagnostic preview PNGs.")
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of input images to process.")
    parser.add_argument("--seed", type=int, default=None, help="Seed for deterministic ELD noise sampling.")
    parser.add_argument("--device", default="auto", help="Torch device: auto, cpu, cuda, or cuda:0.")
    parser.add_argument("--cfa", default="RGGB", help="CFA pattern. First version supports RGGB.")
    parser.add_argument("--checkpoint", default="assets/checkpoints/invisp_canon_eos_5d.pth", help="Copied InvISP checkpoint.")
    parser.add_argument("--camera-params", default=str(DEFAULT_PARAMS), help="Copied CanonEOS5D4 params .npy file.")
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> int:
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    preview_dir = output_dir / "preview"
    images = discover_images(args.input)
    if args.limit is not None:
        images = images[: args.limit]
    if not images:
        raise RuntimeError(f"No supported images found under {args.input}")

    invisp = InvISPInverse(checkpoint_path=args.checkpoint, device=args.device)
    noise_model = ELDNoiseModel(params_file=args.camera_params, seed=args.seed)

    for image_path in tqdm(images, desc="synthesizing"):
        rgb, crop_meta = load_rgb_image(image_path)
        demosaiced = invisp.rgb_to_demosaiced_raw(rgb)
        clean_bayer = extract_bayer(demosaiced, cfa=args.cfa)
        clean_packed = pack_bayer(clean_bayer)

        for ratio in args.ratios:
            low_bayer, noise_params = noise_model.apply(clean_bayer, ratio=ratio)
            low_packed = pack_bayer(low_bayer)
            stem = make_output_stem(image_path, ratio)
            metadata = {
                **crop_meta,
                "ratio": float(ratio),
                "source_path": str(image_path),
                "cfa": args.cfa.upper(),
                "noise_params": noise_params.to_dict(),
                "saturation_level": noise_params.saturation_level,
            }
            save_npz(output_dir / f"{stem}.npz", clean_raw=clean_packed, low_light_raw=low_packed, metadata=metadata)
            if args.preview:
                save_preview(preview_dir / f"{stem}.png", clean_bayer=clean_bayer, low_light_bayer=low_bayer)

    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run CLI unit test**

Run:

```bash
python -m pytest tests/test_cli.py -q
```

Expected: PASS, `1 passed`.

- [ ] **Step 5: Commit Task 6**

Run:

```bash
git add scripts/synthesize_pseudo_lowlight_raw.py tests/test_cli.py
git commit -m "Add pseudo low-light RAW synthesis CLI" -- scripts/synthesize_pseudo_lowlight_raw.py tests/test_cli.py
```

---

### Task 7: Full Test Suite And Point-Line Smoke Test

**Files:**
- Modify: files created in Tasks 1-6 when smoke-test failures identify a concrete defect
- Generate: `outputs/point_line_wireframe_smoke/` as untracked smoke-test artifacts

- [ ] **Step 1: Run all unit tests**

Run:

```bash
python -m pytest tests -q
```

Expected: all tests pass.

- [ ] **Step 2: Run Point-Line smoke test on CPU with one image**

Run:

```bash
python scripts/synthesize_pseudo_lowlight_raw.py \
  --input /home/rjyu/data/Point-Line/wireframe/images \
  --output outputs/point_line_wireframe_smoke \
  --ratios 50 100 \
  --preview \
  --limit 1 \
  --seed 0 \
  --device cpu
```

Expected: command exits 0 and writes two `.npz` files plus two preview PNGs.

- [ ] **Step 3: Inspect generated `.npz` files**

Run:

```bash
python - <<'PY'
from pathlib import Path
import numpy as np

files = sorted(Path("outputs/point_line_wireframe_smoke").glob("*.npz"))
print("npz_count", len(files))
for path in files:
    data = np.load(path, allow_pickle=True)
    print(path.name, data["clean_raw"].shape, data["low_light_raw"].shape, float(data["ratio"]))
    assert data["clean_raw"].shape[0] == 4
    assert data["low_light_raw"].shape == data["clean_raw"].shape
    assert np.isfinite(data["clean_raw"]).all()
    assert np.isfinite(data["low_light_raw"]).all()
assert len(files) == 2
PY
```

Expected: prints `npz_count 2`; assertions pass.

- [ ] **Step 4: Verify generated preview files exist**

Run:

```bash
find outputs/point_line_wireframe_smoke/preview -maxdepth 1 -type f -name '*.png' | sort | wc -l
```

Expected: `2`.

- [ ] **Step 5: Ensure generated outputs remain untracked**

Run:

```bash
git status --short outputs/point_line_wireframe_smoke
```

Expected: output begins with `?? outputs/point_line_wireframe_smoke/`. Do not add this directory.

- [ ] **Step 6: Commit any smoke-test fixes**

If Task 7 required code fixes, commit only the changed source/test files:

```bash
git add pllraw_synthesis scripts tests
git commit -m "Fix pseudo RAW smoke test issues" -- pllraw_synthesis scripts tests
```

If no code fixes were required, skip this commit.

---

### Task 8: Documentation And Final Verification

**Files:**
- Modify: `README.md`
- Modify: `.gitignore`

- [ ] **Step 1: Add generated-output ignore rules**

Append these lines to `.gitignore` if they are not already present:

```gitignore
outputs/
*.npz
```

- [ ] **Step 2: Document usage in README**

Add this section to `README.md`:

~~~markdown
## Pseudo Low-Light RAW Synthesis Tool

The repository provides a self-contained tool for converting RGB PNG/JPEG images into Canon-style pseudo clean RAW with InvISP and synthesizing pseudo low-light RAW with ELD CanonEOS5D4 noise parameters.

Required project-owned assets:

- `assets/checkpoints/invisp_canon_eos_5d.pth`
- `assets/camera_params/CanonEOS5D4_params.npy`

Smoke-test command:

```bash
python scripts/synthesize_pseudo_lowlight_raw.py \
  --input /home/rjyu/data/Point-Line/wireframe/images \
  --output outputs/point_line_wireframe_smoke \
  --ratios 50 100 \
  --preview \
  --limit 1 \
  --seed 0 \
  --device cpu
```

Each output `.npz` contains `clean_raw`, `low_light_raw`, `ratio`, `noise_params`, `source_path`, `cfa`, crop metadata, and `saturation_level`. RAW arrays are packed as `4 x H/2 x W/2` in RGGB order: R, G on the R row, B, G on the B row.
~~~

- [ ] **Step 3: Run final unit tests**

Run:

```bash
python -m pytest tests -q
```

Expected: all tests pass.

- [ ] **Step 4: Verify final code has no runtime third-party imports**

Run:

```bash
rg -n "third-party|reference|DarkFeat|Invertible-ISP" pllraw_synthesis scripts tests README.md
```

Expected: mentions are allowed in `README.md` only if they describe reference provenance; no `pllraw_synthesis/` or `scripts/` source file may import from or path into `third-party/` or `reference/`.

- [ ] **Step 5: Check git status before final handoff**

Run:

```bash
git status --short
```

Expected: source/docs changes from this task are tracked or committed; generated `outputs/` remains ignored or untracked and uncommitted; pre-existing unrelated staged `third-party/*` state is not altered by this plan.

- [ ] **Step 6: Commit documentation**

Run:

```bash
git add README.md .gitignore
git commit -m "Document pseudo low-light RAW synthesis tool" -- README.md .gitignore
```

---

## Self-Review

Spec coverage:

- Self-contained package and no runtime `third-party/` imports: covered by Tasks 4 and 8.
- Copied assets: covered by Task 4.
- InvISP demosaiced RAW to Bayer extraction before ELD: covered by Tasks 1, 5, and 6.
- Arbitrary ratios: covered by Task 6.
- `.npz` output and optional previews: covered by Tasks 3, 6, and 7.
- Point-Line smoke test: covered by Task 7.
- Tests for pure helpers and CLI: covered by Tasks 1-3 and 6.

Placeholder scan:

- This plan contains no placeholder markers or unspecified implementation steps.
- Generated outputs are explicitly not committed.
- Every commit command uses path-limited arguments to avoid unrelated staged `third-party/*` content.

Type consistency:

- `NoiseParams.to_dict()` is used by the CLI and defined in Task 2.
- `extract_bayer()`, `pack_bayer()`, `save_npz()`, and `save_preview()` signatures match their tests and CLI usage.
- Packed RAW layout is consistently `4 x H/2 x W/2`.

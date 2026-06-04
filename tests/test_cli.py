import subprocess
import sys

import pytest
import numpy as np

from scripts.preview_pseudo_lowlight_raw import DEFAULT_PREVIEW_RATIOS, parse_args as parse_preview_args, sort_npz_paths_by_ratio
from scripts.synthesize_pseudo_lowlight_raw import parse_args


def test_synthesize_parse_args_outputs_npz_only():
    args = parse_args(
        [
            "--input",
            "/tmp/images",
            "--output",
            "/tmp/out",
            "--ratios",
            "50",
            "100",
            "--limit",
            "3",
            "--seed",
            "7",
            "--device",
            "cpu",
        ]
    )

    assert args.input == "/tmp/images"
    assert args.output == "/tmp/out"
    assert args.ratios == [50.0, 100.0]
    assert not hasattr(args, "preview")
    assert args.limit == 3
    assert args.seed == 7
    assert args.device == "cpu"


def test_synthesize_parse_args_rejects_preview_flag():
    with pytest.raises(SystemExit):
        parse_args(["--input", "/tmp/images", "--output", "/tmp/out", "--ratios", "50", "--preview"])


def test_preview_parse_args_defaults_to_three_ratio_grid():
    args = parse_preview_args(["--input", "/tmp/image.png", "--output", "/tmp/preview.png", "--device", "cpu"])

    assert DEFAULT_PREVIEW_RATIOS == (100.0, 200.0, 300.0)
    assert args.ratios == [100.0, 200.0, 300.0]
    assert args.device == "cpu"


def test_sort_npz_paths_by_ratio_uses_numeric_metadata(tmp_path):
    ratio100 = tmp_path / "sample_ratio100.npz"
    ratio50 = tmp_path / "sample_ratio50.npz"
    np.savez(ratio100, ratio=np.array(100.0, dtype=np.float32))
    np.savez(ratio50, ratio=np.array(50.0, dtype=np.float32))

    sorted_paths = sort_npz_paths_by_ratio([ratio100, ratio50])

    assert [path.name for path in sorted_paths] == ["sample_ratio50.npz", "sample_ratio100.npz"]


def test_script_runs_from_repo_root_as_file():
    result = subprocess.run(
        [sys.executable, "scripts/synthesize_pseudo_lowlight_raw.py", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "Synthesize pseudo low-light RAW" in result.stdout


def test_preview_script_runs_from_repo_root_as_file():
    result = subprocess.run(
        [sys.executable, "scripts/preview_pseudo_lowlight_raw.py", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "Preview pseudo low-light RAW" in result.stdout

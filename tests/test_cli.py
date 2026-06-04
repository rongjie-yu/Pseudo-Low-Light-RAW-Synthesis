import subprocess
import sys

import pytest

from scripts.preview_pseudo_lowlight_raw import DEFAULT_PREVIEW_RATIOS, parse_args as parse_preview_args
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


def test_preview_parse_args_defaults_to_six_ratio_grid():
    args = parse_preview_args(["--input", "/tmp/image.png", "--output", "/tmp/preview.png", "--device", "cpu"])

    assert DEFAULT_PREVIEW_RATIOS == (50.0, 100.0, 150.0, 200.0, 250.0, 300.0)
    assert args.ratios == [50.0, 100.0, 150.0, 200.0, 250.0, 300.0]
    assert args.device == "cpu"


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

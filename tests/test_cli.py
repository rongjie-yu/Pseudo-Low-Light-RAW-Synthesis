import subprocess
import sys

import pytest

from scripts.preview_pseudo_lowlight_raw import DEFAULT_RATIO, parse_args as parse_preview_args
from scripts.synthesize_pseudo_lowlight_raw import parse_args


def test_synthesize_parse_args():
    args = parse_args(
        [
            "--input", "/tmp/images",
            "--output", "/tmp/out",
            "--ratio", "100",
            "--limit", "3",
            "--seed", "7",
            "--device", "cpu",
        ]
    )

    assert args.input == "/tmp/images"
    assert args.output == "/tmp/out"
    assert args.ratio == 100.0
    assert args.limit == 3
    assert args.seed == 7
    assert args.device == "cpu"


def test_synthesize_default_device_is_cuda():
    args = parse_args(["--input", "/tmp/i", "--output", "/tmp/o", "--ratio", "50"])

    assert args.device == "cuda"


def test_preview_parse_args_defaults():
    args = parse_preview_args(["--input", "/tmp/image.png", "--output", "/tmp/preview.png"])

    assert DEFAULT_RATIO == 100.0
    assert args.ratio == 100.0
    assert args.device == "cuda"


def test_preview_parse_args_custom_ratio():
    args = parse_preview_args(["--input", "/tmp/i.png", "--output", "/tmp/o.png", "--ratio", "250"])

    assert args.ratio == 250.0


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

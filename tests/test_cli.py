import subprocess
import sys

from scripts.synthesize_pseudo_lowlight_raw import parse_args


def test_parse_args_requires_input_output_and_ratios():
    args = parse_args(
        [
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
        ]
    )

    assert args.input == "/tmp/images"
    assert args.output == "/tmp/out"
    assert args.ratios == [50.0, 100.0]
    assert args.preview is True
    assert args.limit == 3
    assert args.seed == 7
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

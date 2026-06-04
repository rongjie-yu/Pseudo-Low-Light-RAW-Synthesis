# Repository Guidelines

## Project Structure & Module Organization

This repository combines research implementations to synthesize pseudo low-light RAW data from normal-light JPEG images using InvISP and ELD. The root `README.md` gives the project intent. Reference papers live in `reference/`.

Most executable code is under `third-party/`:

- `third-party/Invertible-ISP/`: JPEG/RGB-to-RAW reconstruction, training, testing, metrics, and camera dataset preprocessing.
- `third-party/ELD/`: physics-based low-light RAW noise synthesis, denoising models, dataset utilities, and scripts.
- `third-party/DarkFeat/`: low-light RAW feature extraction and matching utilities.

Keep new integration code at the repository root or in a clearly named top-level package. Avoid modifying vendored code unless required and documented.

## Build, Test, and Development Commands

There is no root build system. Run commands from the relevant component directory.

- `pip install -r third-party/DarkFeat/requirements.txt`: install DarkFeat dependencies.
- `cd third-party/Invertible-ISP && conda env create -f environment.yml`: create the InvISP reference environment.
- `cd third-party/Invertible-ISP && bash train.sh`: train InvISP using paths configured in `train.sh`.
- `cd third-party/Invertible-ISP && bash test.sh`: run InvISP testing/evaluation with configured checkpoints.
- `cd third-party/ELD && bash scripts/test_ELD.sh`: run ELD evaluation as configured by the upstream script.
- `cd third-party/ELD && bash scripts/train.sh`: launch ELD training or synthetic-data workflows.

Dataset and checkpoint paths are machine-specific; edit scripts before long jobs.

## Coding Style & Naming Conventions

The codebase is Python-heavy and follows upstream research-code conventions. Use 4-space indentation, `snake_case` for functions and variables, and `CamelCase` for classes. Prefer explicit path/config arguments over hard-coded local paths. Keep shell script configuration near the top.

## Testing Guidelines

There is no central test suite or coverage requirement. Use component scripts as smoke tests after changes. For Python modules, prefer small `test_*.py` files next to the affected component, matching names such as `test_rgb.py`, `test_raw.py`, and `test_ELD.py`. Record required datasets, checkpoints, camera names, and GPU/CUDA assumptions in PR notes.

## Commit & Pull Request Guidelines

Current history only contains `Initial commit`, so use concise imperative messages such as `Add RAW synthesis guide` or `Fix InvISP checkpoint path handling`. Pull requests should describe the pipeline stage affected, list exact commands run, note dataset/checkpoint requirements, and include sample outputs or metrics when behavior changes. Do not commit large datasets, checkpoints, or experiment outputs unless explicitly required.

## Security & Configuration Tips

Keep private dataset paths, credentials, and download tokens out of the repository. Store bulky local data outside the workspace or under ignored experiment directories, and document reproducible preprocessing steps instead.

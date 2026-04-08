# AGENTS.md

## Cursor Cloud specific instructions

### Overview

RAI is a Python monorepo for an Embodied AI agent framework integrating LLMs with ROS 2. It uses `uv` for dependency management and `colcon` for ROS 2 workspace builds.

### Environment prerequisites

-   **ROS 2 Jazzy** must be installed and sourced (`source /opt/ros/jazzy/setup.bash`) before any build or run step.
-   **`uv`** is the Python package manager (lockfile: `uv.lock`).
-   A C++ toolchain (`g++-14`, `libstdc++-14-dev`) is required for `colcon build` of `rai_interfaces`.

### Shell setup

Before running anything (tests, Python scripts, imports), always source the environment:

```bash
source /opt/ros/jazzy/setup.bash
export SHELL=/bin/bash
source ./setup_shell.sh
```

This activates the `.venv`, sources the colcon `install/` overlay, and sets `PYTHONPATH` correctly.

### Building

```bash
source /opt/ros/jazzy/setup.bash
colcon build --symlink-install
```

The `rai_interfaces` package (cloned via `vcs import src < ros_deps.repos`) must be present in `src/src/rai_interfaces` before building.

### Linting

Ruff is configured via `.pre-commit-config.yaml`. Run with:

```bash
pre-commit run ruff --all-files
pre-commit run ruff-format --all-files
```

Ruff uses `--fix` by default through pre-commit, so it auto-fixes import ordering issues.

### Testing

```bash
python -m pytest tests/ --timeout=60 -m "not billable and not ci_only and not manual" --ignore=src
```

The `pyproject.toml` already sets these default markers and `--ignore=src`.

**Sound device tests**: Tests under `tests/communication/sounds_device/` require virtual audio input/output devices. In headless environments without audio hardware, these tests will error during collection. Either skip them (`--ignore=tests/communication/sounds_device`) or create virtual PulseAudio/ALSA devices before running.

### Key gotchas

-   The `config.toml` in the repo root is the working config file; `rai-config-init` is only for pip-installed users, not developers using the repo.
-   The Streamlit configurator (`rai/frontend/configurator.py`) is an optional GUI; all configuration can be done by editing `config.toml` directly.
-   Many test modules import ROS 2 packages (`rclpy`, `geometry_msgs`, etc.) at module level, so ROS 2 must be sourced even to collect tests.

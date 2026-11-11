# Developer environment

The repository is a uv workspace of the RAI packages plus a colcon workspace for the ROS 2 parts (`rai_interfaces`, `rai_bringup`). The venv is created with `--system-site-packages` so it sees rclpy and the other ROS 2 Python packages of the system interpreter.

## Prerequisites

-   ROS 2 Lyrical or Jazzy, installed from apt. See [Install](install.md#requirements) for the supported platforms.
-   [uv](https://docs.astral.sh/uv/getting-started/installation/): `curl -LsSf https://astral.sh/uv/install.sh | sh`
-   colcon, vcstool, rosdep and a C++ toolchain for `rai_interfaces`: `sudo apt install python3-colcon-common-extensions python3-vcstool python3-rosdep g++`

## Set up

```bash
git clone https://github.com/RobotecAI/rai.git
cd rai
source /opt/ros/${ROS_DISTRO}/setup.bash
rosdep install --from-paths src --ignore-src -r -y
scripts/dev_env.sh --all-groups
```

`scripts/dev_env.sh` creates `.venv` with the system interpreter and `--system-site-packages`, runs `uv sync` with the arguments you pass, clones `rai_interfaces` from `ros_deps.repos` if it is not installed, and runs `colcon build --symlink-install`.

In every new shell:

```bash
source /opt/ros/${ROS_DISTRO}/setup.bash
source setup_shell.sh   # sources install/ and activates .venv
```

## Dependency groups

`uv sync` installs `rai_core[all]` and the dev tools. The other packages are groups:

| Group        | Installs                                                  |
| ------------ | --------------------------------------------------------- |
| `s2s`        | `rai_s2s[all]`: speech recognition and text to speech     |
| `semap`      | `rai_semap`                                               |
| `simbench`   | `rai_sim` and `rai_bench`                                 |
| `perception` | `rai_perception` (GroundingDINO, Grounded SAM 2, PyTorch) |
| `docs`       | mkdocs and its plugins                                    |

Pass them to the script (`scripts/dev_env.sh --group s2s`) or to `uv sync` later. `--all-groups` installs everything.

KokoroTTS is not in the `s2s` group because its dependencies are GPL-3.0 licensed. Install it explicitly with `uv pip install -e "src/rai_s2s[kokoro]"`; a later `uv sync` removes it again unless run with `--inexact`.

## Tests

```bash
uv sync --all-groups
pytest
```

The defaults in `pyproject.toml` skip tests marked `billable`, `ci_only`, `manual` and `benchmark`. Many test modules import ROS 2 packages at module level, so ROS 2 must be sourced even to collect tests. The tests in `tests/communication/sounds_device/` need audio devices; on a headless machine create virtual ones with `docker/virtual-audio.sh` or skip them with `--ignore=tests/communication/sounds_device`. Run concurrent test sessions on different `ROS_DOMAIN_ID`s.

`tests/docs/` extracts the Python snippets of these docs: it compiles every snippet, imports what it imports, and runs the snippets that need no ROS graph or API key.

## Lint and docs

```bash
sudo apt install shellcheck
pre-commit install
pre-commit run --all-files
```

To build the docs:

```bash
uv sync --group docs
mkdocs serve        # or: mkdocs build --strict
```

Markdown is formatted by prettier through pre-commit. Do not hard-wrap lines.

See [Contributing](../faq/contributing/CONTRIBUTING.md) for the review process and commit conventions.

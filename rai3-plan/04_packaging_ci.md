# RAI 3.0 plan: packaging, dev environment, CI, dead code, docs

Scope: packaging, dependencies, dev env, Docker, CI (tests, coverage, types, benchmarks), dead code, docs structure. Worktree: `/home/mmajek/.t3/worktrees/rai/t3-fd1a0102` on the uncommitted Lyrical spike. Date of measurements: 2026-11-11.

Legend: **[M]** measured here (command output in scratchpad `envtest/` and `bench/`), **[I]** inferred or recommended without a direct measurement.

Scratchpad evidence: `/tmp/claude-1000/-home-mmajek--t3-worktrees-rai-t3-fd1a0102/40ae5bcf-22ef-426d-9758-257b11dd26d6/scratchpad/envtest/` (pytest log, coverage json, ty/pyright/mypy outputs, outdated list) and `.../scratchpad/bench/` (3 benchmark runs).

---

## 0. Top findings (read this first)

1. **CI never fails on test failures.** [M] `build-and-test.yml` has `continue-on-error: true` at job level, so a red test job still yields a green run. Every gate proposed below is pointless until this is removed.
2. **Dropping rai_nomad silently breaks rai_core in Docker and CI.** [M] `nav2_msgs`, `tf_transformations` and `portaudio19-dev` are installed by `rosdep install --from-paths src` *only because* `src/rai_extensions/rai_nomad/package.xml` declares them. rai_core imports `nav2_msgs` and `tf_transformations` eagerly via `rai.tools.ros2.__init__`. The rosdep keys must move to a surviving `package.xml` in the same PR that deletes rai_nomad.
3. **`uv venv --system-site-packages` replaces all PYTHONPATH tricks and the spike's empy/osrf-pycommon deps.** [M] On Lyrical: 874 passed, 4 skipped, 11 xfailed in 90 s with only `source /opt/ros/lyrical/setup.bash && source install/setup.bash && source .venv/bin/activate`, with `empy` and `osrf-pycommon` uninstalled from the venv (they resolve from `/usr/lib/python3/dist-packages`). On Jazzy (rai-dev:jazzy container): 308 passed on the ROS-heavy subset with the same recipe.
4. **numpy<2 on Jazzy is caused only by cv_bridge.** [M] In the Jazzy container with numpy 2.5.4: ROS message (de)serialization including numpy fixed arrays (CameraInfo.k) and `sensor_msgs_py.point_cloud2` work; `cv_bridge.CvBridge().cv2_to_imgmsg` fails (`KeyError: 16`, its boost module is built against numpy 1.x) and system `cv2` fails to import. rai_core uses cv_bridge in only 3 files (11 call sites). Replacing them with a ~60-line numpy converter removes the numpy split and lets both distros use numpy>=2.
5. **rai_core installs ~770 MB of which ~550 MB is optional.** [M] Exclusive closure sizes from installed RECORDs (py3.14): streamlit 306 MB (18 pkgs), opencv-python 211 MB + opencv-python-headless 169 MB (both write the same `cv2/` directory), langchain-community 40 MB (unused in rai_core), langchain-google-genai 25 MB, langchain-aws 23 MB, langchain-openai 18 MB, langfuse 6 MB.
6. **Type checking**: [M] on rai_core with ROS paths resolved: pyright standard 128 errors (5.4 s), basic 74, strict 761; mypy default 191 errors (39.7 s cold, 1.0 s warm); ty 0.0.101 92 errors + 49 warnings (0.27 s). Recommendation: gate on **pyright standard, zero errors, rai_core only**, then strict per-subpackage. ruff is a linter, not a type checker.
7. **Coverage**: [M] rai_core line coverage on Lyrical = **67.6 %** (2738/4051). Without `rai/frontend` 75.9 %. Also deleting two unreferenced modules (plan_agent, ros_logs) gives 79.2 % (computed from the same data). 85 % needs a ratchet, not a day-one gate.
8. **Benchmark noise is metric-specific.** [M] 3 quick runs on Lyrical (20 to 26 s each): steady-state metrics have CV 0 to 10 %, first-call/cold/p99 metrics have CV 50 to 137 %. Thresholds must be per metric class, and A/B on the same runner.

---

## 1. Packaging audit

### 1.1 Layout and workspace

[M] The root `pyproject.toml` (`rai_framework`, `package = false`) is **not a uv workspace**: it has `[tool.uv.sources]` path entries but no `[tool.uv.workspace]`. Consequences:
- `uv export --package rai_core` fails: "The workspace does not have a member rai-core".
- `uv sync --package X`, `uv build --package X` and `{ workspace = true }` sources are unavailable.
- `src/rai_semap/pyproject.toml` duplicates `[tool.uv.sources]` with relative paths.

Packages (python, uv): rai_core 2.14.1, rai_whoami 0.2.0, rai_s2s 1.1.0, rai_semap 0.1.0, rai-sim 0.0.2, rai-bench 0.4.0, rai-perception 0.4.0 (hatchling, all others uv_build). Not in the workspace: rai_finetune (standalone), rai_nomad (ament_python, colcon).
Colcon packages: rai_bringup (launch files only, `exec_depend rai_interfaces`), rai_nomad, plus rai_interfaces from `ros_deps.repos` (`version: main`, unpinned).

### 1.2 Concrete packaging bugs

| # | Finding | Evidence |
|---|---|---|
| B1 | `continue-on-error: true` on the test job | [M] build-and-test.yml |
| B2 | rosdep keys for rai_core live in rai_nomad's package.xml (nav2_msgs, nav2_simple_commander, tf_transformations, portaudio19-dev) | [M] `grep depend src/rai_extensions/rai_nomad/package.xml`; rai_core has no package.xml |
| B3 | Spike bug: requirement strings `"empy>=4.2,<5; ..."`, `"osrf-pycommon; ..."` were pasted into `[tool.ruff.lint.isort] known-first-party` | [M] root pyproject.toml |
| B4 | `rai_bench`, `rai_sim`, `rai_perception` require `rai_core>=2.0.0a2,<3.0.0`: bumping rai_core to 3.0.0 makes the lock unresolvable | [M] pyproject files; [I] uv validates path-source versions against the requirement |
| B5 | `opencv-python` and `opencv-python-headless` are both direct deps of rai_core; both install into `cv2/`, last writer wins. rai-gsam2 (RobotecAI) also requires non-headless `opencv-python` | [M] both dist-infos in `.venv`; rai_gsam2 METADATA |
| B6 | `onnxruntime` (base) and `onnxruntime-gpu` (`all` extra) both install into `onnxruntime/` | [M] both dist-infos 1.31.1 present in `.venv` |
| B7 | rai_core imports other workspace packages: `rai_s2s` (frontend/configurator), `rai_perception` (`tools/ros2/manipulation/custom.py` re-exports `GetGrabbingPointTool`), `sounddevice`, `elevenlabs` | [M] grep + deptry DEP001 |
| B8 | Undeclared direct imports in rai_core: `pydantic`, `typing_extensions`, `langsmith` | [M] deptry DEP001 |
| B9 | Declared but unused in rai_core: `langchain-community`, `tqdm`, `lark`, `transforms3d`, `audioop-lts` (only needed transitively via pydub). `lark`/`transforms3d` are there for ROS (rosidl_parser, tf_transformations), not for rai | [M] deptry DEP002 + grep |
| B10 | `tomli` used although Python >=3.12 has `tomllib` | [M] 3 files import tomli |
| B11 | `smoke/import_test.py` imports every `src/*/rai*` package, including rai_nomad, which has no deps installed after the spike | [M] test source |
| B12 | Tests write `artifact_database.pkl` (pickle) into the CWD and it grows every run (1.7 MB to 3.4 MB after one run here). ToolRunner does the same in production (`store_artifacts` default `db_path="artifact_database.pkl"`), and loads it with `pickle.load` | [M] stat before/after; source |
| B13 | rai_finetune `requires-python = ">=3.10,<3.12"`, numpy<1.25, langfuse<3: not installable on either supported system Python (3.12/3.14) | [M] pyproject |
| B14 | root `dependencies` contain dev tools (pre-commit, pytest, pytest-cov, pytest-timeout, tabulate) | [M] |
| B15 | `requires-python` inconsistent: rai_bench `>=3.12` (no cap), others `>=3.12,<3.15` | [M] |
| B16 | No `py.typed` in rai_core, so downstream users get no types | [M] |
| B17 | Lock resolves for darwin, win32, emscripten and 3.13 (15 resolution forks; 615 macOS/Windows wheel lines) although ROS 2 targets are Linux + 3.12 (Jazzy) / 3.14 (Lyrical) | [M] uv.lock header |

### 1.3 Constraints blocking Lyrical / py3.14 and "latest everything"

| Constraint | Why it exists | Status / action |
|---|---|---|
| `numpy<2; python_version<'3.13'` | Jazzy cv_bridge (boost ext) and apt python3-opencv are built against numpy 1.x. [M] messages and point_cloud2 work on numpy 2; cv_bridge does not | Replace cv_bridge in rai_core, rai_perception (7 files), rai_semap (1 file) with a numpy converter, then `numpy>=2` on both distros. Venv opencv-python wheels shadow the apt cv2 [M] |
| cv_bridge | see above | Remove usage (11 call sites in rai_core) [M] |
| tf_transformations | Only `quaternion_from_euler(0,0,yaw)` and `euler_from_quaternion` yaw [M] | Replace with 10 lines of math; drops transforms3d dep too. ros-lyrical-tf-transformations is on apt [M], so it is not a blocker, just bloat |
| nav2_msgs / vision_msgs / geographic_msgs / backward_ros | Spike built them from source for lack of sudo | [M] all are on the Lyrical apt repo (`apt-cache policy ros-lyrical-nav2-msgs` -> 1.5.2). Not a blocker in Docker. Make nav2 tools lazily import nav2_msgs so `import rai.tools.ros2` does not require Nav2 [I] |
| rai_interfaces | Built from `ros_deps.repos` main branch | [M] `ros-jazzy-rai-interfaces 0.3.0` exists on the Jazzy buildfarm; no Lyrical package. Bloom-release to Lyrical, then drop `ros_deps.repos` and the colcon build of interfaces. Until then pin to tag `0.3.0` |
| empy, osrf-pycommon (spike) | Venv could not see system dist-packages | Remove; `--system-site-packages` covers them [M] |
| torch / torchvision | none on 3.14 | [M] torch 2.15.0 cp314 installs and tests pass. Lock has CUDA 13 deps; CI strips `nvidia-*`/`triton` and uses `--torch-backend cpu` |
| torchaudio (rai_s2s base dep) | not imported anywhere in rai_s2s [M] | Drop. torch itself is only used for Silero VAD via `torch.hub.load` (downloads and executes code from GitHub at runtime: a safety issue). Run Silero's ONNX model with onnxruntime instead [I], which removes torch from rai_s2s base |
| onnxruntime | none on 3.14 (1.31.1 installed) [M] | Fix B6: drop `onnxruntime-gpu` from `all`; keep `gpu` extra declared as a uv `conflicts` pair, or document a manual swap. kokoro-onnx depends on CPU onnxruntime unconditionally [M] |
| openwakeword | Pinned to a personal fork commit (`maciejmajek/openWakeWord@f2340d8`), 0.6.0, onnxruntime only | [M] builds on 3.14. Move the fork to the RobotecAI org or vendor the small inference code [I] |
| openai-whisper `>=20250625,<20250626` | Exact pin | Loosen to `>=20250625` [I] |
| kokoro-onnx `==0.3.3` | Exact pin; latest 0.6.1 [M] | Bump (API check needed) |
| elevenlabs `<2` | latest 2.77.0 [M] | Bump (v2 client API changed) [I] |
| langfuse `<5` | latest 5.2.0 [M] | Bump; it is behind an extra after the diet |
| opencv `<5` | latest 5.0.0.93 [M] | Bump to 5 |
| deprecated `<2` | latest 3.0.1 [M]; used in 2 files | Replace with `typing_extensions.deprecated` (PEP 702) and drop |
| huggingface-hub `<3` locked at 1.33 | latest 2.4.0 [M]; held by transformers | Follows transformers |
| pydub (rai_core) | Unmaintained since 2021; needs `audioop-lts` on 3.13+; emits 4 SyntaxWarnings on 3.14 [M] | Public API (`HRIMessage.audios: list[AudioSegment]`). Breaking change; coordinate with the core-API planner. Candidate: own small `Audio` dataclass over numpy + stdlib `wave` |
| visualnav_transformer (nomad) numpy<2, diffusion-policy override | nomad only | Delete with rai_nomad (user decision), including `override-dependencies` |
| rai_finetune pins | Standalone, unsloth | Keep (user decision) but outside the workspace and CI. Either move to `>=3.12` or document that it uses its own uv-managed Python [I] |

[M] `uv tree --outdated` (py3.12 resolution) lists 59 outdated packages; the ones held back by **our** caps are: deprecated, elevenlabs, kokoro-onnx, langfuse, opencv-python(-headless), numpy (Jazzy split), librosa 0.11 vs 1.0 (via kokoro 0.3.3). The rest are plain `uv lock --upgrade` material. Full list: `envtest/outdated312.txt`.

### 1.4 rai_core dependency diet (target)

[M] Current: 25 direct deps, 122 packages, ~770 MB installed (RECORD sum 935 MB minus the double-counted cv2).

Proposed `rai_core` base (all used by core code): `langchain-core`, `langchain`, `langgraph`, `langgraph-prebuilt`, `langsmith`, `pydantic`, `typing-extensions`, `numpy>=2`, `opencv-python-headless`, `pillow`, `requests`, `tomli-w`, `coloredlogs` (or drop), audio type (pydub until replaced).

Extras:
- `rai_core[openai]` langchain-openai; `[aws]` langchain-aws; `[google]` langchain-google-genai; `[ollama]` langchain-ollama; `[vendors]` = all four.
- `[langfuse]` langfuse.
- `[ui]` streamlit (`rai.frontend`, `rai.agents.integrations.streamlit`).
- `[all]`.

Code changes required for extras to work [M]: `rai/initialization/model_initialization.py` imports `ChatBedrock`, `ChatOllama`, `ChatOpenAI` at module top (lines 22, 26, 27); these must become lazy imports with a clear "install rai_core[aws]" error. Enforce layering with ruff `TID251`/`banned-module-level-imports` (ban `rai_s2s`, `rai_perception`, `streamlit`, vendor langchain packages at module level in rai_core).

Remove: `langchain-community`, `tqdm`, `lark`, `transforms3d`, `tomli`, `deprecated`, `opencv-python` (keep headless), `audioop-lts` (once pydub is gone). Remove ROS stand-ins from pip deps entirely (they come from apt via system site-packages).

Expected base size after diet: roughly 200 MB [I, from the exclusive-closure table].

Fix opencv duplication end to end: ask rai-gsam2 (RobotecAI-owned) to depend on `opencv-python-headless` [I].

---

## 2. Dev environment design

### 2.1 What was tested on Lyrical (this machine)

| Variant | Result |
|---|---|
| A. `uv venv --system-site-packages --python /usr/bin/python3.14` + `UV_PROJECT_ENVIRONMENT=... uv sync --frozen --all-groups`, then `uv pip uninstall empy osrf-pycommon`; clean env (`env -i`), `source /opt/ros/lyrical/setup.bash; source install/setup.bash; source venv/bin/activate`, **no** setup_shell.sh PYTHONPATH lines | [M] `import em` resolves to `/usr/lib/python3/dist-packages/em.py`; `rai` from editable `src/rai_core`; numpy 2.5.4 from the venv (cv_bridge built against apt numpy 2.3.5 works); full suite **874 passed, 4 skipped, 11 xfailed in 90 s** (PortAudio tests ignored, no libportaudio on host) |
| A on Jazzy (`rai-dev:jazzy`, worktree copied in) | [M] `em` from system, rai editable, **308 passed** on `tests/communication/ros2 tests/tools tests/messages tests/agents` in 73 s |
| B. Plain venv + `.pth` pointing at `/opt/ros/lyrical/lib/python3.14/site-packages`, ROS not sourced | [M] `import rclpy` fails: `librcl_action.so: cannot open shared object file`. ROS must be sourced anyway (LD_LIBRARY_PATH, AMENT_PREFIX_PATH), so a `.pth` adds nothing |
| C. Isolated venv (no system site), ROS sourced | [M] `rosidl_parser` -> `No module named 'lark'`; `rosidl_pycommon`, `ros2cli.cli`, `cv_bridge` -> `No module named 'em'`; `launch` -> `No module named 'typing_extensions'`. Requires mirroring every apt Python dep of ROS in pyproject (the spike's whack-a-mole) |

[M] No module name overlap between the venv and `/opt/ros/lyrical/lib/python3.14/site-packages`, so ROS's PYTHONPATH being ahead of the venv is harmless. Venv site-packages are ahead of system dist-packages, so pinned venv versions win.

[M] The existing `.venv` already contains editable `.pth` files for every rai package (`rai_core.pth` -> `src/rai_core/`), so `PYTHONPATH=src/rai_core:...` in setup_shell.sh is redundant, and prepending the venv site-packages is only a workaround for missing system packages.

### 2.2 Recommended recipe (both distros)

```bash
source /opt/ros/$ROS_DISTRO/setup.bash
uv venv --system-site-packages --python /usr/bin/python3   # once
uv sync                                                     # editable installs of all workspace members
colcon build --symlink-install                              # rai_bringup (+ rai_interfaces until released for Lyrical)
source install/setup.bash && source .venv/bin/activate
```

- [M] `uv sync` keeps an existing venv's `include-system-site-packages = true`.
- [I] uv has no project setting for system site-packages, so if uv decides to recreate the venv (interpreter mismatch) the flag is lost. Mitigate with `[tool.uv] python-preference = "only-system"` (prevents a managed CPython with a different build being picked) and a `scripts/dev_env.sh` (or `make venv`) that creates the venv correctly. A test in conftest can assert `sys.flags`/`site.ENABLE_USER_SITE`-independent check: `"/usr/lib/python3/dist-packages" in sys.path` and fail with a helpful message.
- setup_shell.sh shrinks to 3 lines (source ROS, source install, activate) or is deleted and documented. 14 references to update (docs, Dockerfile.manipulation-demo, CI, AGENTS.md) [M].
- Python must be the **system** interpreter (rclpy C extensions are built for it) [I, standard ROS constraint].

### 2.3 Workspace config changes (one PR)

```toml
[tool.uv.workspace]
members = ["src/rai_core", "src/rai_whoami", "src/rai_s2s", "src/rai_semap", "src/rai_sim", "src/rai_bench", "src/rai_extensions/rai_perception"]

[tool.uv]
python-preference = "only-system"
environments = ["sys_platform == 'linux'"]   # all arches: arm64 robots matter

[tool.uv.sources]
rai_core = { workspace = true }   # etc.

[dependency-groups]
dev = ["pytest", "pytest-timeout", "pytest-cov", "pre-commit", "pyright", "tabulate"]
```

[I] With `environments` limited to Linux the lock loses the darwin/win32/emscripten forks (B17). Keep `requires-python = ">=3.12,<3.15"`.

---

## 3. Docker images

### 3.1 Current state [M]

- `docker/Dockerfile`: `FROM osrf/ros:${ROS_DISTRO}-desktop-full` (4.59 GB base), `COPY --from=ghcr.io/astral-sh/uv:latest` (unpinned), installs `libnvidia-gl-470` and X11 libs, `COPY . /rai` **before** rosdep (any source change busts every later layer), rosdep from all `src` (including rai_nomad), `colcon build`, `uv sync --locked`.
- Local images: `rai-dev:jazzy` 6.07 GB (venv 868 MB), `rai-dev:jazzy-all` 13.4 GB, `rai-dev:humble` 5.34 GB.
- [M] `osrf/ros:lyrical-desktop-full`, `ros:lyrical-ros-base`, `ros:lyrical-ros-core` exist on Docker Hub. `ros:jazzy-ros-base-noble` is 879 MB locally.

### 3.2 Target

One `docker/Dockerfile`, `ARG ROS_DISTRO={jazzy,lyrical}`, multi-stage:

1. `base`: `FROM ros:${ROS_DISTRO}-ros-base`. Apt: git, libportaudio2/portaudio19-dev, jackd2 (dummy audio for s2s tests), ffmpeg (pydub), espeak-ng (kokoro) [I: check kokoro need]. `COPY` **only `package.xml` files** then `rosdep install`. uv from a pinned tag (`ghcr.io/astral-sh/uv:0.13.x`).
2. `deps`: `COPY pyproject.toml uv.lock src/*/pyproject.toml`, `uv venv --system-site-packages --python /usr/bin/python3 /opt/rai/venv`, `uv sync --frozen --no-install-workspace --all-groups` with `ARG TORCH_BACKEND=cpu` for the CI flavour (CI's current `uv export | grep -vE '^(nvidia-|triton==)'` + `--torch-backend cpu` hack moves here). Cache mount for uv.
3. `dev`: `COPY . /rai`, `colcon build`, `uv sync --frozen` (only installs the editable members).

[I] ros-base is enough for tests: they need rclpy, std/sensor/geometry/nav msgs, tf2_ros, launch and the `ros2` CLI (used by `rai.tools.ros2.cli` and `tests/rai_bringup`); no rviz/gazebo/demo nodes are referenced in tests [M grep]. Expected CI image ~3 to 4 GB with CPU torch vs 6 to 13 GB today.

Publish `ghcr.io/robotecai/rai-dev:{jazzy,lyrical}` (+ `-<sha>`) from a `docker-images.yml` workflow triggered by changes to `docker/**`, `uv.lock`, `**/package.xml`, `ros_deps.repos`, plus a weekly schedule (picks up ROS sync releases). PR CI consumes the image built from main; if a PR changes those paths, the workflow builds the image in the PR and the test job uses that tag [I].

---

## 4. CI design

### 4.1 Current state [M]

| Workflow | Wall time (latest successful run) | Notes |
|---|---|---|
| build-and-test | 8.4 to 10.1 min (19.6 min outlier) | Per job: container init 63 to 68 s (desktop-full pull), jackd apt 40 to 46 s, python deps 56 to 61 s, rosdep 15 to 42 s, colcon 19 to 24 s, tests 126 to 159 s, **"Post Install uv" cache save 182 to 208 s**. Jazzy: 919 passed. `continue-on-error: true` |
| mkdocs-build | **494 s** | `uv sync --frozen --all-groups` installs torch & co to build docs; mkdocstrings-python is static (griffe) [I] |
| pre-commit | 105 s | Runs in `osrf/ros:jazzy-desktop-full` and does `vcs import`, neither needed by the hooks [I] |
| uv-lock-check | 8 s | fine |

### 4.2 Target workflows

```
lint.yml         ubuntu-latest, no container: pre-commit (ruff, ruff-format, shellcheck, prettier), uv lock --check   ~1 min
docs.yml         uv sync --only-group docs; mkdocs build --strict                                                      ~1 min [I]
test.yml         matrix ros_distro: [jazzy, lyrical]; container ghcr.io/robotecai/rai-dev:${distro}
                   uv sync --frozen (no-op on unchanged lock), colcon build, pytest
                   lyrical leg: --cov=src/rai_core/rai --cov-fail-under=<floor>; junit + codecov upload             ~4-5 min each, parallel [I]
typecheck.yml    container rai-dev:lyrical; pyright -p pyrightconfig.json (rai_core)                                  ~1.5 min (pull dominated; pyright itself 5.4 s [M])
bench.yml        PR (paths: src/rai_core/rai/communication/**, bench/**) + nightly on main; container rai-dev:lyrical  ~5 min [I]
docker-images.yml   build + push images (paths filter + weekly)
release.yml      later
```

- Remove `continue-on-error`. Make `test (jazzy)`, `test (lyrical)`, `lint`, `typecheck`, `docs` required checks. `bench` required only after a burn-in period of ~2 weeks of non-blocking runs [I].
- setup-uv cache: save only on `push` to main (`save-cache: ${{ github.event_name == 'push' }}`) or rely on the image; this alone removes ~3 min per job [M timing of the save step].
- Audio tests: image ships jackd and PortAudio; start `jackd -d dummy` in a session fixture rather than a CI step [I].
- Test isolation: unique `ROS_DOMAIN_ID` per job is free (separate VMs). Test suite is 90 s on Lyrical locally [M]; no need for xdist yet.
- Fix B12 (pickle in CWD) via a conftest `tmp_path` override so CI workspaces stay clean.

Expected PR wall time: ~5 min (vs 8 to 10 min now; mkdocs from 8 min to ~1 min) [I].

### 4.3 Coverage gate

[M] Lyrical, rai_core line coverage 67.6 %. Per area: frontend 0 % (444 stmts), agents/langchain 54 %, communication/ros2 73 %, tools/ros2 84 %, initialization 79 %, messages 81 %, types/aggregators 100 %. Biggest uncovered files: configurator.py 410, plan_agent.py 75 (unreferenced), ros_logs.py 74 (unreferenced), nav2.py 71, state_based_agent.py 65, agents/langchain/agent.py 65.

Plan:
1. `[tool.coverage.run] source = ["src/rai_core/rai"]`, `omit = ["*/frontend/*"]` (until frontend moves behind `[ui]` and gets its own tests or leaves rai_core); `[tool.coverage.report] fail_under` in pyproject so local and CI agree.
2. Start the floor at the measured value after dead-code removal (~79 %, computed), raise by 2 points per PR that adds tests, reach 85 % before the 3.0 tag. Line coverage first; branch coverage later (would lower the number).
3. codecov.yml: `coverage.status.patch.target: 85%` so new code meets 85 % immediately; `project.threshold: 0.5%`.
4. Coverage measured on one leg (lyrical) to avoid double uploads.

### 4.4 Type-check gate (pyright vs mypy vs ty)

All three were run on `src/rai_core/rai` with the scratch venv (py3.14) and ROS + colcon install paths on the search path [M]:

| Tool | Version | Errors | Time | Notes |
|---|---|---|---|---|
| pyright basic | 1.1.414 | 74 | 5.3 s | |
| **pyright standard** | 1.1.414 | **128** | 5.4 s | top rules: ArgumentType 28, IncompatibleVariableOverride 26 (`args_schema` override pattern in tools), AttributeAccessIssue 24 (ROS msg attrs), IncompatibleMethodOverride 16, PossiblyUnbound 12, CallIssue 8, OptionalMemberAccess 7 (real bugs: `llm.bind_tools` on Optional, `goal_handle.cancel_goal` on None in `api/action.py`) |
| pyright strict | 1.1.414 | 761 | 5.3 s | dominated by UnknownMemberType 147, MissingTypeStubs 138 (ROS generated msgs and langchain), Unknown* 209 |
| mypy (default + ignore-missing-imports) | 2.4.1 | 191 | 39.7 s cold / 1.0 s warm | arg-type 44, attr-defined 41, assignment 32, union-attr 21 |
| ty | 0.0.101 | 92 errors + 49 warnings | 0.27 s | 39 `unused-type-ignore-comment`, 21 incompatible-mutable-override; pre-1.0, diagnostics change between releases |

By area (pyright standard): communication 40, agents 38, tools 33, messages 7, frontend 6, initialization 4.
rclpy ships `py.typed` on Lyrical [M]; ROS generated message modules have no stubs. rai_core has 50 `# type: ignore` comments [M].

Recommendation:
- Gate on **pyright `standard`**, rai_core only, **zero errors**. Most accurate on pydantic/langchain generics among the three, fast, and pyright is what VS Code users already see (Pylance). Add `py.typed`.
- Get to zero in 3 to 4 PRs split by subpackage (tools, communication, agents, rest). Fix the `args_schema` override once in a base tool class. Then enable `strict = ["src/rai_core/rai/communication"]` (new connector code) and grow it per subpackage; never global strict (ROS msgs have no stubs).
- Do not gate on ty in 3.0 (0.0.x); optionally run it non-blocking to track. mypy: no advantage here, slowest cold, most noise.
- Other workspace packages: run pyright `basic` non-blocking, report-only.
- Add ruff rules that help typing hygiene: `UP` (py312 syntax), `B`, `TID` (layering), `PGH003` (blanket type: ignore), `target-version = "py312"`.

### 4.5 Benchmark job with regression thresholds

Base: `tests/communication/ros2/bench/ros2_connector_bench.py` from branch `feat/ros2-connector-hardening` (commits 17237a4, dfcbe3b; 615 lines, `--quick`, `--out`, `--compare`). Services/actions/streams run in subprocesses, so discovery is real.

[M] 3 quick runs on Lyrical against the current spike code (not the hardening branch); wall 19.6 / 25.8 / 25.4 s:

| Metric | Run 1 / 2 / 3 | CV | Class |
|---|---|---|---|
| publish.string.per_s | 85 357 / 85 860 / 87 583 | 1 % | stable |
| publish.image_vga.p50_ms | 0.087 / 0.076 / 0.089 | 7 % | stable |
| publish.hri_vga.p50_ms | 339 / 337 / 344 | 1 % | stable (slow; hardening commit 7e47ada targets it) |
| dispatch_hri_30hz.latency.p50_ms | 0.83 / 0.84 / 0.80 | 2 % | stable |
| dispatch_hri_30hz.dropped | 0 / 0 / 0 | 0 | exact |
| dispatch_100hz.delivered (of 100) | 19 / 13 / 44 | 53 % | noisy, and a real bug on this code |
| dispatch_1khz.latency.p99_ms | 228 / 66 / 26 | 82 % | noisy |
| receive_message.cold.p99_ms | 517 / 3002 / 2702 | 53 % | noisy |
| service_call.reused.p50_ms | 0.36 / 0.35 / 0.43 | 9 % | stable |
| service_call.reused.p99_ms | 0.55 / 0.63 / 0.60 | 6 % | stable |
| service_call.first_call.p50_ms | 259 / 261 / 1510 | 87 % | noisy |
| service_call.fresh.p50_ms | 0.84 / 0.80 / 81.9 | 137 % | noisy |
| service_call.concurrent_8x2.p50_ms | 3.6 / 4.0 / 5.3 | 18 % | medium |
| start_action.result.p50_ms | 17.98 / 17.99 / 18.02 | 0 % | stable |
| leak.after_service_calls.clients (200 fresh calls) | 200 / 200 / 200 | 0 | exact (shows a client leak on this code) |
| leak.after_actions.action_entries | 40 / 40 / 40 | 0 | exact |
| leak.*.rss_kb | 304 668 -> 345 896 | 0 % | stable |
| shutdown.duration_ms | 2.41 / 2.53 / 2.23 | 5 % | stable |

Design:
- **A/B on one runner**: check out base and head into two dirs in the same job, run `bench --quick` interleaved base, head, base, head, base, head (3 repeats each, ~2.5 min total at 25 s/run [M]); take per-metric medians. This cancels runner-to-runner variance that would make absolute thresholds flaky on GitHub-hosted runners.
- `tests/communication/ros2/bench/thresholds.toml` with three metric kinds:
  - `exact`: entity counts (clients, waitables, action_entries, threads after N ops), `errors`, `inversions`, `dispatch_hri_30hz.dropped`. Fail if head > base (or > 0 for errors).
  - `ratio`: stable metrics (CV <= 10 %). Latency fails at head/base > 1.25, warns at > 1.10; throughput fails at head/base < 0.80. `concurrent_8x2` gets 1.5.
  - `report`: cold, first-call, fresh, p99 of high-rate dispatch, `elapsed_s`. Printed in the job summary, never fail.
  - Plus generous absolute ceilings (about 5x the values above, e.g. `service_call.reused.p50_ms <= 2`, `publish.string.per_s >= 15000`) to catch slow drift that A/B cannot see.
- `bench_gate.py` (new, ~100 lines) reads both JSON sets and thresholds, writes a Markdown table to `$GITHUB_STEP_SUMMARY`, exits non-zero on failures. The existing `--compare` stays for local use.
- Nightly on main: full (non-quick) run, upload JSON as an artifact; optionally feed `benchmark-action/github-action-benchmark` for a trend page [I].
- Set the dispatch/leak thresholds after the hardening branch lands; on current code those metrics show bugs (drops, 200 leaked clients), so the gate would only freeze bad behaviour.

---

## 5. Dead code and things to delete in 3.0

| Item | Evidence | Action |
|---|---|---|
| `src/rai_extensions/rai_nomad` (+ `docs/extensions/nomad.md`, nomad group, `override-dependencies` diffusion-policy, mkdocs nav entry, README/install.md mentions) | user decision; smoke test imports it [M] | delete; move rosdep keys (B2) in the same PR |
| `rai/agents/langchain/core/plan_agent.py` (275 lines) | [M] not imported anywhere in src/tests/examples/docs; 0 % coverage | delete (confirm with core planner) |
| `rai/communication/ros2/ros_logs.py` (170 lines) | [M] unreferenced; 0 % coverage | delete |
| `rai/frontend/configurator.py` (1064 lines) + `rai/frontend/streamlit.py` + `agents/integrations/streamlit.py` | [M] configurator only launched via `streamlit run` (install.md); streamlit app used by `examples/debugging_assistant.py`, `examples/rosbot-xl-demo.py`; excluded from codecov and from ruff | not dead: move behind `rai_core[ui]` extra (or a separate `rai_ui` package later); remove its rai_s2s/sounddevice/elevenlabs imports or make them guarded |
| Unused private methods in `communication/ros2/api/topic.py`: `_verify_receive_args`, `_is_topic_available`, `_get_message_class`, `_verify_publisher_exists`; `publisher_exists` used once | [M] vulture 60 % + grep (1 occurrence = definition) | delete |
| `NodeDiscovery` / `add_setter` in `communication/ros2/context.py` | [M] vulture; NodeDiscovery 2 occurrences (def + export) | check exports, likely delete |
| `rai/types/rai_interfaces.py`: `ManipulatorMoveToResponse`, `RAIGroundedSamResponse`, `RAIGroundingDinoResponse` | [M] vulture, 1 occurrence each | delete if not public API |
| `tools/ros2/manipulation/custom.py:35` re-export of `rai_perception.tools.GetGrabbingPointTool` | [M] vulture 90 % unused import; layering violation | delete |
| `messages/multimodal.py:141` unreachable code after return | [M] vulture 100 % | delete |
| `tests/messages/` | [M] not dead: tests `rai.messages` (multimodal, utils, artifacts). `test_transport.py` is a QoS smoke test of ROS2Connector receive | keep the dir; move `test_transport.py` to `tests/communication/ros2/test_qos_transport.py`; drop its `ROS_DISTRO` match (both distros support all presets) |
| `setup_shell.sh` PYTHONPATH lines | [M] redundant (editable .pth exist) | replace (section 2) |
| Humble leftovers in docs | [M] 10 files mention humble/jammy/22.04/3.10 | docs rewrite |
| `scripts/download_demo.sh`, demos | user: leave demos alone | keep |
| `.github/workflows/pre-commit.yml` ROS container + `vcs import` | [M] | remove |
| `ros_deps.repos` | [M] rai_interfaces on Jazzy apt | delete once Lyrical release exists |
| pydub `AudioSegment` in public API | [M] | candidate, coordinate with core planner |
| `artifact_database.pkl` pickle store in CWD | [M] B12 | replace (safety): coordinate with core planner |

[M] Command: `uvx vulture src/rai_core/rai src/rai_bench src/rai_sim src/rai_s2s src/rai_semap src/rai_whoami src/rai_extensions/rai_perception src/rai_bringup examples tests --min-confidence 60`, then each candidate grepped across src/tests/examples/docs. Pydantic validators, LangChain callbacks and `model_post_init` hits are framework hooks (false positives). Module-level check: only `rai.frontend.configurator`, `rai.communication.ros2.ros_logs`, `rai.agents.langchain.core.plan_agent` are never imported.

---

## 6. Docs rewrite outline (structure only, mkdocs-material)

```
Home
  What is RAI · What's new in 3.0 · Supported platforms (Lyrical primary / Jazzy, Python 3.14 / 3.12, x86_64 / arm64)
Getting started
  Install with Docker (official images)
  Install natively (apt ROS + uv venv --system-site-packages + uv sync + colcon)
  Configure models (config.toml, vendor extras: openai/aws/google/ollama)
  Your first agent (10-minute quickstart)
Concepts
  Architecture (connectors, agents, tools, aggregators, runners)
  Messages (HRI, multimodal)
  Configuration · Observability (langfuse, langsmith)
Guides
  ROS 2 connector (topics, services, actions, QoS, threading)
  Writing tools · Building agents
  Voice (rai_s2s) · Perception (rai_perception) · Semantic map (rai_semap) · Whoami (rai_whoami)
  Benchmarking (rai_bench) · Simulation (rai_sim) · Fine-tuning (rai_finetune, experimental)
Reference
  API (mkdocstrings per package) · CLI entry points · config.toml schema · Extras matrix
Demos (existing pages, unchanged)
Development
  Dev environment · Testing · CI and quality gates (coverage, pyright, benchmarks) · Release process · Contributing
Troubleshooting / FAQ
  numpy/cv_bridge, PortAudio, DDS discovery, venv not seeing ROS packages
(Migration 2.x -> 3.0: placeholder, later)
```

Drop: `extensions/nomad.md`, the commented `ROS_2/*` pages, all Humble content. Docs CI builds with `--only-group docs`.

---

## 7. Ordered PR sequence (packaging / CI area)

| # | PR | Depends on | Size |
|---|---|---|---|
| 0 | **Land the Lyrical spike cleanly**: py 3.12 to 3.14, Humble removal, TopicEndpointInfo import, fix B3 (ruff known-first-party), keep nomad group deleted | none | S |
| 1 | **Remove `continue-on-error`** from build-and-test (Jazzy only until images exist) | 0 | XS |
| 2 | **Drop rai_nomad + move rosdep keys** (nav2_msgs, tf_transformations, portaudio19-dev, cv_bridge, tf2_ros, vision_msgs...) into `rai_bringup/package.xml` (or a no-code `rai_ros_deps` metapackage); fix smoke test | 0 | S |
| 3 | **uv workspace + dev env**: `[tool.uv.workspace]`, `{workspace = true}`, `environments = linux`, `python-preference = only-system`, dev group, remove empy/osrf-pycommon/lark from deps, `--system-site-packages` recipe, shrink `setup_shell.sh`, conftest sanity check, fix B12 test pollution | 0 | M |
| 4 | **rai_core dependency diet + extras** (vendors, langfuse, ui), lazy imports in model_initialization, remove rai_core -> rai_s2s/rai_perception imports, ruff TID layering ban, headless opencv only, declare pydantic/typing_extensions/langsmith, tomllib, drop deprecated/tqdm/langchain-community | 3 | M |
| 5 | **Drop cv_bridge + tf_transformations** (numpy image converter in rai_core, used by perception and semap; yaw helpers); lazy nav2_msgs import; unify `numpy>=2` | 4 | M |
| 6 | **Bump everything**: `uv lock --upgrade`; raise caps (opencv 5, langfuse 5, elevenlabs 2, kokoro-onnx 0.6, librosa 1, openai-whisper unpin, hf-hub 2); fix B4 (`rai_core<4`), B6 (onnxruntime), drop torchaudio; pin uv in Docker; pre-commit autoupdate (prettier mirror v2.7.1 is archived). Split the API-breaking bumps (elevenlabs, langfuse, kokoro) into their own PRs if they need code changes | 4, 5 | M (x3) |
| 7 | **Docker images**: multi-stage on ros-base, jazzy + lyrical, CPU-torch CI flavour, ghcr publish workflow; pin `ros_deps.repos` to `0.3.0` and start the bloom release of rai_interfaces for Lyrical (external) | 2, 3 | M |
| 8 | **CI on images**: test matrix jazzy+lyrical, lint without ROS container, docs with `--only-group docs`, setup-uv cache saved only on main, jackd/PortAudio from image | 7 | M |
| 9 | **Dead code removal** (section 5) | 0 (coordinate with core planner) | S |
| 10 | **Coverage gate**: `[tool.coverage]`, floor at measured value (~79 %), codecov patch 85 %, ratchet plan | 8, 9 | S |
| 11 | **Type-check gate**: pyrightconfig, `py.typed`, CI job; fix 128 errors in 3 to 4 PRs (tools, communication, agents, rest); then strict for `rai.communication` | 8 (job), 4 (frontend out) | M (x4) |
| 12 | **Benchmark CI**: bring bench script from `feat/ros2-connector-hardening`, `--repeat`, `bench_gate.py`, `thresholds.toml`, A/B workflow + nightly; non-blocking for ~2 weeks, then required | 8, hardening branch merged | M |
| 13 | **Docs 3.0 rewrite** (structure in section 6) | 3, 7 | L |
| 14 | **Release 3.0.0**: versions of all packages, `rai_core<4` everywhere, changelog, one tag | all | S |

Critical path: 0 -> 2/3 -> 7 -> 8 -> (10, 11, 12). PRs 4 to 6 run in parallel with 7.

---

## 8. Open questions for the user

1. rai_interfaces Lyrical release via bloom: who owns it? Until then `ros_deps.repos` stays (pinned).
2. Is replacing `pydub.AudioSegment` in the HRI message API acceptable for 3.0 (API break)?
3. Configurator: keep inside rai_core behind `[ui]`, or split into its own package later?
4. arm64 (Jetson-class robots): should CI build arm64 images too? (GitHub has arm64 runners; doubles image build time.)
5. Coverage target: is 85 % at the 3.0 tag a hard requirement, or "floor + patch 85 %" enough?

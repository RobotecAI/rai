# PyPI Release Workflow

RAI packages are published to PyPI via GitHub Actions using Trusted Publishers (OIDC authentication). Each package uses independent versioning.

## Release Flow

1. Run `list-packages.yaml` workflow to identify packages needing release
2. Run `pkg-publish.yaml` workflow (manual dispatch)
3. Select package(s), target (`test-pypi` or `pypi`), and wheel building option
4. Workflow builds source distributions and publishes
5. For PyPI target, manual approval required via GitHub Environment

## Release Strategy

Each package maintains independent semantic versioning. Use version ranges in dependencies: `rai_core = ">=2.0.0,<3.0.0"`.

## Checking Package Versions

Run `list-packages.yaml` workflow (manual or scheduled weekly) to view package versions across local repo, PyPI, and Test PyPI in workflow summary.

## Publishing Packages

Run `pkg-publish.yaml` workflow with these inputs:

-   **package**: Single (`rai_core`) or multiple comma-separated (`rai_core,rai-perception`)
-   **publish_target**: `test-pypi` or `pypi`

Workflow validates packages, builds distributions, publishes to target, and verifies installation.

**Notes:**

-   Publishing fails if version exists on target. Bump version before republishing.
-   PyPI publishes require manual approval via GitHub Environment.

## Building Wheels

Wheels are pre-built binary distributions that install faster than source distributions. The workflow automatically detects package type and uses the appropriate build method. Currently, only Linux wheels are released.

### Pure Python Packages

Pure Python packages contain only Python code with no compiled C/C++/Rust extensions. Most RAI packages are pure Python (e.g., `rai_core`, `rai-perception`, `rai_s2s`).

**Build process:**

-   Workflow auto-detects pure Python packages by checking for C extension files (`.c`, `.cpp`, `.pyx`) or `setup.py` with `ext_modules`
-   Uses `python -m build --wheel` to create universal wheels tagged `py3-none-any.whl`
-   Universal wheels work on all platforms (Linux, macOS, Windows) and are accepted by PyPI

**Explicit marking:** Add to `pyproject.toml`:

```toml
[tool.rai]
has_c_extensions = false
```

### Packages with C Extensions

Packages with C extensions contain compiled code.

**Build process:**
This is enabled for future-proofing; currently, RAI doesn't have any packages with C extensions.

-   Workflow detects C extensions automatically or via `[tool.rai] has_c_extensions = true` marker
-   Uses `cibuildwheel` to build platform-specific wheels with proper `manylinux*` tags
-   Builds wheels for Python 3.10 and 3.12 on Linux (x86_64 and ARM64)
-   Skips 32-bit, Windows, macOS, and musllinux (Alpine) builds
-   Includes Rust compiler support for dependencies like `tiktoken`

**Platform support:**

-   Linux: `manylinux_2_28` (x86_64 and aarch64)
-   Python versions: 3.10 and 3.12
-   Excluded: 32-bit, Windows, macOS, musllinux (Alpine)

**Explicit marking:** Add to `pyproject.toml`:

```toml
[tool.rai]
has_c_extensions = true
```

## For Maintainers

### Workflow Components

The publishing workflow consists of several Python scripts in `scripts/`:

-   `discover_packages.py`: Scans `src/` for packages and extracts metadata from `pyproject.toml`
-   `validate_packages.py`: Validates package names, checks PyPI versions, and supports variant matching (`-` and `_`)
-   `pypi_query.py`: Queries PyPI and Test PyPI for package versions (commands: `check`, `list`)

The workflow YAML (`.github/workflows/pkg-publish.yaml`) orchestrates discovery, validation, building, and publishing.

### Running Tests

Test the publishing scripts locally:

```bash
pytest tests/pkg_publish/ -v
```

Key test files:

-   `test_discover_packages.py`: Tests package discovery and metadata extraction
-   `test_validate_packages.py`: Tests validation, variant matching, and PyPI version checks
-   `test_pypi_query.py`: Tests PyPI version checking and listing functionality

### GitHub Environment Setup

Create two environments in repository Settings > Environments:

-   **test-pypi**: No protection rules required
-   **pypi**: Configure required reviewers and restrict to main branch

When publishing to PyPI, workflow pauses for approval. Configured reviewers must approve in Actions tab > workflow run > "Review deployments".

### Trusted Publishers Setup

Configure Trusted Publishers on [Test PyPI](https://test.pypi.org/manage/account/publishing/) and [PyPI](https://pypi.org/manage/account/publishing/):

**Settings:**

-   Publisher: GitHub
-   Owner: `RobotecAI` (your username/org)
-   Repository: `rai`
-   Workflow: `pkg-publish.yaml`
-   Environment: `pypi` (for PyPI) or `test-pypi` (for Test PyPI)

Add publisher before running the publish workflow. If the trusted publisher is configured per project/package, for any new package, manually push the first time, then update the authentication on that project. See [PyPI Trusted Publishers docs](https://docs.pypi.org/trusted-publishers/) for details.

### Troubleshooting

**Package not found errors:**

-   Verify package name matches exactly what's in `pyproject.toml` (the workflow supports both `-` and `_` variants)
-   Check that `pyproject.toml` exists in `src/<package_dir>/`
-   Ensure `name` and `version` fields are present in `[tool.poetry]` section

**Artifact download failures:**

-   Check that `build_wheels` and `build_sdist` jobs completed successfully
-   Verify artifact names match the pattern `wheels-*` or `sdist-*`

**Version conflicts:**

-   The workflow checks PyPI versions before publishing
-   Bump the version in `pyproject.toml` if the version already exists
-   For Test PyPI, warnings are shown but don't block publishing

**Wheel building issues:**

-   Pure Python packages: Check for accidental C extension markers or files
-   C extension packages: Verify `[tool.rai] has_c_extensions = true` is set if auto-detection fails
-   Review build logs for compilation errors or missing dependencies

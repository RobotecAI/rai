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
-   **build_wheels**: Enable for C extensions (default: false, work in progress)

Workflow validates packages, builds distributions, publishes to target, and verifies installation.

**Notes:**

-   Publishing fails if version exists on target. Bump version before republishing.
-   PyPI publishes require manual approval via GitHub Environment.
-   Wheel building is experimental and only needed for packages with C extensions.

## GitHub Environment Setup

Create two environments in repository Settings > Environments:

-   **test-pypi**: No protection rules required
-   **pypi**: Configure required reviewers and restrict to main branch

When publishing to PyPI, workflow pauses for approval. Configured reviewers must approve in Actions tab > workflow run > "Review deployments".

## Trusted Publishers Setup

Configure Trusted Publishers on [Test PyPI](https://test.pypi.org/manage/account/publishing/) and [PyPI](https://pypi.org/manage/account/publishing/):

**Settings:**

-   Publisher: GitHub
-   Owner: `RobotecAI` (your username/org)
-   Repository: `rai`
-   Workflow: `pkg-publish.yaml`
-   Environment: `pypi` (for PyPI) or `test-pypi` (for Test PyPI)

Add publisher before first publish. See [PyPI Trusted Publishers docs](https://docs.pypi.org/trusted-publishers/) for details.

# PyPI Release Workflow

This document explains the build and publish process for RAI packages to PyPI. Publishing is handled via GitHub Actions workflows using PyPI Trusted Publishers with OpenID Connect (OIDC) for secure authentication.

Typical release flow:

```
PR changes in src/*
  |
  +--> bump package version(s) in src/*/pyproject.toml
  |
  +--> CI build + tests
         |
         +--> Actions: pkg-publish.yaml
                |
                +--> publish to Test PyPI
                |
                +--> install/import smoke test
                |
                +--> manual approval (GitHub Environment: pypi)
                |
                +--> publish to PyPI
                |
                +--> install/import smoke test
```

## Release Strategy

RAI uses independent versioning for each package in the multi-package repo. Each module (rai-core, rai-perception, rai-bench, etc.) maintains its own version number based on semantic versioning, allowing mature packages like rai-core to be at 2.x while newer modules start at 0.x. During development, tests run against the current repo state using local path dependencies. For published packages, compatibility testing focuses on dependency boundary versions rather than exhaustive version combinations.

Applications that depend on RAI packages should use version ranges (not exact pins) to stay compatible across patch/minor updates. For example, depend on a supported major series: `rai_core = ">=2.0.0.a2,<3.0.0"`.

## Building Distributions

The `build-distro.yml` workflow builds wheels and source distributions for all packages without publishing them. Use this to create distributions for local testing or manual distribution.

Trigger: PR with label `build-distro`, manual dispatch, or git tags matching `[0-9]*`
Output: All distributions are collected into a single artifact that can be downloaded from the workflow run.

## Publishing

The `pkg-publish.yaml` workflow publishes in two stages: Test PyPI first, then production PyPI after approval.

Trigger: PR with label `publish-testpypi` or manual dispatch

Steps:

1. Run tests: `pytest -m "not billable"`
2. Build wheels and source distributions for all packages (including `rai-tiny`)
3. Publish `rai-tiny` to Test PyPI first (validates Trusted Publisher authentication)
4. Publish other packages to Test PyPI and verify installation
5. Manual approval (GitHub Environment: `pypi`)
6. Publish `rai-tiny` to PyPI first (validates Trusted Publisher authentication)
7. Publish other packages to PyPI and verify installation

Dry-run (no publishing):

1. Go to the repo Actions tab
2. Select the workflow "Publish packages (Test PyPI to PyPI)"
3. Click "Run workflow"
4. Set input `dry_run` to `true`
5. Click "Run workflow"

In dry-run mode, the workflow skips publishing to Test PyPI and PyPI. It builds all distributions (including `rai-tiny`), installs from locally built wheels, and runs import verification tests. Use `dry_run=true` to validate workflow changes without consuming Test PyPI/PyPI versions.

Notes:

-   `rai-tiny` is a minimal test package used to validate the Trusted Publisher authentication path before publishing real packages. It is published to both Test PyPI and production PyPI as a canary to catch authentication issues early.
-   Production publish is gated by a GitHub Environment named `pypi`. Configure this environment with required reviewers for manual approval.
-   Publishing fails if the same package version already exists on Test PyPI or PyPI. Bump the version in the relevant `pyproject.toml` files before publishing again.

## GitHub Environment Setup

The `pypi` environment must be configured to require manual approval before publishing to production PyPI.

To set up the environment:

1. Go to repository Settings > Environments
2. Create or edit the `pypi` environment
3. Configure protection rules:
    - Required reviewers: Add trusted team members who must approve deployments
    - Wait timer: Optional delay to allow cancellation of accidental deployments
    - Deployment branches: Restrict to `main` branch only
4. Save the configuration

### Approving a Production Release

When the workflow reaches production PyPI publishing, it pauses for approval:

1. Go to the repository Actions tab
2. Find the workflow run with "Waiting" status
3. Click the workflow run and locate the "Review deployments" section
4. Review deployment details (packages, versions, test results)
5. Click "Review deployments" and select "Approve and deploy" or "Reject"

Only configured reviewers can approve deployments. All approvals are logged for audit purposes.

## Trusted Publishers Setup

This project uses PyPI Trusted Publishers with OpenID Connect (OIDC) for authentication. This eliminates the need for manually managed API tokens and provides better security with short-lived credentials.

### Setting Up Trusted Publishers

Configure a Trusted Publisher at the organization level (recommended for multiple projects) or per-project:

1. Go to your organization's page on [pypi.org](https://pypi.org) (e.g., `https://pypi.org/organizations/<org-name>/`) or your project's settings page on [test.pypi.org](https://test.pypi.org) or [pypi.org](https://pypi.org)
2. Navigate to "Publishing" > "Add a new trusted publisher"
3. Configure:
    - Publisher: GitHub
    - Repository: `RobotecAI/rai`
    - Workflow filename: `pkg-publish.yaml`
    - Environment name: leave empty for Test PyPI, `pypi` for production PyPI

For new packages, create the project first, then add the Trusted Publisher. Organization-level publishers apply to all projects in the organization.

For detailed instructions, see PyPI's documentation on [Trusted Publishers](https://docs.pypi.org/trusted-publishers/).

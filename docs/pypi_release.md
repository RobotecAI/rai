# PyPI Release Workflow

This document explains the build and publish process for RAI packages to PyPI. Publishing is handled via GitHub Actions workflows to keep API tokens and credentials secure using GitHub Secrets.

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
2. Build wheels and source distributions for all packages
3. Publish to Test PyPI and verify installation
4. Manual approval (GitHub Environment: `pypi`)
5. Publish to PyPI and verify installation

Dry-run (no publishing):

1. Go to the repo Actions tab
2. Select the workflow "Publish to Test PyPI" (from `pkg-publish.yaml`)
3. Click "Run workflow"
4. Set input `dry_run` to `true`
5. Click the final "Run workflow" button

In dry-run mode, the workflow skips `twine upload` to Test PyPI and PyPI. It installs from the locally built wheels and runs the import smoke test. Use `dry_run=true` when changing `pkg-publish.yaml` (or related scripts) to validate the workflow logic without consuming Test PyPI/PyPI versions.

Notes:

-   Production publish is gated by a GitHub Environment named `pypi`. Configure this environment with required reviewers for manual approval.
-   Publishing fails if the same package version already exists on Test PyPI or PyPI. Bump the version in the relevant `pyproject.toml` files before publishing again.

## Required GitHub Secrets

-   TEST_PYPI_API_TOKEN: Test PyPI API token
-   PYPI_API_TOKEN: Production PyPI API token

### Getting API Tokens

Test PyPI requires a separate account from production PyPI:

1. Create an account at [test.pypi.org](https://test.pypi.org/account/register/)
2. Go to Account settings > API tokens
3. Create a new API token with "Upload packages" scope
4. Copy the token (it's only shown once)

Production PyPI:

1. Go to [pypi.org](https://pypi.org) and log in
2. Go to Account settings > API tokens
3. Create a new API token with "Upload packages" scope
4. Copy the token (it's only shown once)

### Setting Up Secrets in GitHub

To add these tokens as secrets, go to your repository Settings > Secrets and variables > Actions, then add a new repository secret. See GitHub's documentation on [encrypted secrets](https://docs.github.com/actions/security-guides/encrypted-secrets) for detailed instructions.

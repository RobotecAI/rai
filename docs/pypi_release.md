# PyPI Release Workflow

This document explains the build and publish process for RAI packages to PyPI. Publishing is handled via GitHub Actions workflows to keep API tokens and credentials secure using GitHub Secrets.

## Release Strategy

RAI uses independent versioning for each package in the monorepo. Each module (rai-core, rai-perception, rai-bench, etc.) maintains its own version number based on semantic versioning, allowing mature packages like rai-core to be at 2.x while newer modules start at 0.x. Dependencies use version ranges (e.g., `rai_core = ">=2.0.0.a2,<3.0.0"`) to maintain compatibility. During development, tests run against the current monorepo state using local path dependencies. For published packages, compatibility testing focuses on dependency boundary versions rather than exhaustive version combinations.

## Building Distributions

The `build-distro.yml` workflow builds wheels and source distributions for all packages without publishing them. Use this to create distributions for local testing or manual distribution.

Trigger: PR with label `build-distro`, manual dispatch, or git tags matching `[0-9]*`
Output: All distributions are collected into a single artifact that can be downloaded from the workflow run.

## Publishing to Test PyPI

The `publish-testpypi.yml` workflow builds, tests, and publishes packages to test.pypi.org for validation before production release.

Trigger: PR with label `publish-testpypi` or manual dispatch

Steps:

1. Run tests: `pytest -m "not billable"`
2. Build wheels and source distributions for all packages
3. Publish to test.pypi.org
4. Verify installation from Test PyPI

## Publishing to Production PyPI

To be implemented.

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

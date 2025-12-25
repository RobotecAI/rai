# Copyright (C) 2025 Julia Jia
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""List package versions and compare with PyPI/Test PyPI.

Outputs package version status in various formats for GitHub Actions workflows.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys

try:
    from packaging import version as pkg_version
except ImportError:
    pkg_version = None

# Status recommendation constants
STATUS_AWAITING_RELEASE = "📦 Awaiting release"
STATUS_UP_TO_DATE = "✅ Up to date"
STATUS_LOCAL_OLDER = "Local < PyPI"


def _get_pypi_versions(package_name: str, test_pypi: bool = False) -> list[str]:
    """Get versions from PyPI or Test PyPI.

    Args:
        package_name: Name of the package
        test_pypi: If True, query Test PyPI instead of PyPI

    Returns:
        List of versions sorted newest first, empty list on error
    """
    try:
        cmd = [sys.executable, "scripts/pypi_query.py"]
        if test_pypi:
            cmd.append("--test-pypi")
        cmd.extend(["list", package_name, "--json"])
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        if result.returncode == 0 and result.stdout.strip():
            return json.loads(result.stdout)
    except (json.JSONDecodeError, subprocess.TimeoutExpired):
        pass
    return []


def _get_recommendation(
    local_version: str,
    pypi_latest: str,
    testpypi_latest: str,
    pkg_name: str | None = None,
) -> str:
    """Generate status recommendation based on version comparison.

    Args:
        local_version: Local package version
        pypi_latest: Latest version on PyPI (or "-" if none)
        testpypi_latest: Latest version on Test PyPI (or "-" if none)
        pkg_name: Optional package name for error messages

    Returns:
        Status recommendation string
    """
    if pypi_latest and pypi_latest != "-":
        if pkg_version is None:
            return "-"
        try:
            if pkg_version.parse(local_version) > pkg_version.parse(pypi_latest):
                return STATUS_AWAITING_RELEASE
            elif local_version == pypi_latest:
                return STATUS_UP_TO_DATE
            else:
                return STATUS_LOCAL_OLDER
        except (pkg_version.InvalidVersion, ValueError) as e:
            if pkg_name:
                print(
                    f"Warning: Version comparison failed for {pkg_name}: {e}",
                    file=sys.stderr,
                )
            return "-"
    else:
        return STATUS_AWAITING_RELEASE


def _get_latest_version(versions: list[str], default: str | None = "-") -> str | None:
    """Get latest version from list or return default.

    Args:
        versions: List of versions (sorted newest first)
        default: Default value if list is empty

    Returns:
        Latest version string or default
    """
    return versions[0] if versions else default


def _get_console_status(
    local_version: str, pypi_versions: list[str], testpypi_versions: list[str]
) -> str:
    """Get status string for console output.

    Args:
        local_version: Local package version
        pypi_versions: List of versions on PyPI
        testpypi_versions: List of versions on Test PyPI

    Returns:
        Status string like "[PUBLISHED on PyPI]" or "[NOT PUBLISHED]"
    """
    if local_version in pypi_versions:
        return " [PUBLISHED on PyPI]"
    if local_version in testpypi_versions:
        return " [PUBLISHED on Test PyPI]"
    return " [NOT PUBLISHED]"


def list_package_versions(
    packages_json_path: pathlib.Path,
) -> list[dict[str, str | list[str]]]:
    """Collect version information for all packages.

    Args:
        packages_json_path: Path to packages.json file

    Returns:
        List of dicts with package info: name, local_version, pypi_versions, testpypi_versions
    """
    with open(packages_json_path) as f:
        local_packages = json.load(f)

    results = []
    for pkg_name in sorted(local_packages.keys()):
        pkg_info = local_packages[pkg_name]
        local_version = pkg_info["version"]

        pypi_versions = _get_pypi_versions(pkg_name, test_pypi=False)
        testpypi_versions = _get_pypi_versions(pkg_name, test_pypi=True)

        results.append(
            {
                "name": pkg_name,
                "local_version": local_version,
                "pypi_versions": pypi_versions,
                "testpypi_versions": testpypi_versions,
            }
        )

    return results


def output_markdown(results: list[dict[str, str | list[str]]]) -> None:
    """Output results in markdown table format."""
    print("## Package Version Status")
    print("")
    print("| Package | Local Version | PyPI Latest | Test PyPI Latest | Status |")
    print("|---------|---------------|-------------|------------------|--------|")

    for result in results:
        pkg_name = result["name"]
        local_version = result["local_version"]
        pypi_latest = _get_latest_version(result["pypi_versions"])
        testpypi_latest = _get_latest_version(result["testpypi_versions"])

        local_display = f"**{local_version}**"
        recommendation = _get_recommendation(
            local_version, pypi_latest, testpypi_latest, pkg_name
        )

        print(
            f"| {pkg_name} | {local_display} | {pypi_latest} | {testpypi_latest} | {recommendation} |"
        )

    print("")
    print("Legend:")
    print("- **bold** = current local version")


def output_console(results: list[dict[str, str | list[str]]]) -> None:
    """Output results in console format."""
    print("\nPackage Version Status:\n")

    for result in results:
        pkg_name = result["name"]
        local_version = result["local_version"]
        pypi_versions = result["pypi_versions"]
        testpypi_versions = result["testpypi_versions"]

        pypi_latest = _get_latest_version(pypi_versions)
        testpypi_latest = _get_latest_version(testpypi_versions)
        status = _get_console_status(local_version, pypi_versions, testpypi_versions)

        print(f"{pkg_name}:")
        print(f"  Local:     {local_version}{status}")
        print(f"  PyPI:      {pypi_latest}")
        print(f"  Test PyPI: {testpypi_latest}")
        print()


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description="List package versions and compare with PyPI/Test PyPI"
    )
    parser.add_argument(
        "--output-format",
        choices=["markdown", "console"],
        default="console",
        help="Output format",
    )
    parser.add_argument(
        "--packages-json",
        type=pathlib.Path,
        default=pathlib.Path("packages.json"),
        help="Path to packages.json file",
    )
    args = parser.parse_args(argv)

    if not args.packages_json.exists():
        print(f"Error: {args.packages_json} not found", file=sys.stderr)
        return 1

    results = list_package_versions(args.packages_json)

    output_formatters = {
        "markdown": output_markdown,
        "console": output_console,
    }
    output_formatters[args.output_format](results)

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

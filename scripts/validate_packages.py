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
"""Validate packages for publishing and check PyPI versions."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


def _find_package_variant(
    pkg_name: str, all_packages: dict[str, dict[str, str]]
) -> tuple[str | None, bool]:
    """Find package by trying both - and _ variants.

    Args:
        pkg_name: Package name to look up
        all_packages: Dictionary of all discovered packages

    Returns:
        Tuple of (actual_package_name, was_variant_used)
        Returns (None, False) if package not found
    """
    # Try exact match first (preferred)
    if pkg_name in all_packages:
        return pkg_name, False

    # Try swapping - and _ (only if name contains one of them)
    if "-" in pkg_name:
        variant = pkg_name.replace("-", "_")
        if variant in all_packages:
            return variant, True
    elif "_" in pkg_name:
        variant = pkg_name.replace("_", "-")
        if variant in all_packages:
            return variant, True

    return None, False


def validate_packages(
    packages_json_path: str | Path,
    package_input: str,
    publish_target: str,
    github_output: str | None = None,
) -> list[dict[str, str]]:
    """Validate packages and check PyPI versions.

    Args:
        packages_json_path: Path to JSON file with discovered packages
        package_input: Comma-separated list of package names
        publish_target: Either 'pypi' or 'test-pypi'
        github_output: Path to GitHub Actions output file (optional)

    Returns:
        List of validated package dictionaries with 'name', 'path', and 'version'

    Raises:
        SystemExit: If validation fails
    """
    # Load discovered packages
    with open(packages_json_path) as f:
        all_packages = json.load(f)

    # Parse input packages (comma-separated)
    package_names = [p.strip() for p in package_input.split(",") if p.strip()]

    validated = []
    errors = []

    for pkg_name in package_names:
        actual_name, was_variant = _find_package_variant(pkg_name, all_packages)
        if actual_name is None:
            errors.append(f"Package '{pkg_name}' not found in repository")
            continue

        pkg_info = all_packages[actual_name]
        # Use actual package name from repository (not the input variant)
        validated.append(
            {
                "name": actual_name,
                "path": pkg_info["path"],
                "version": pkg_info["version"],
            }
        )
        if was_variant:
            print(
                f"Found package: {actual_name} v{pkg_info['version']} at {pkg_info['path']} "
                f"(matched '{pkg_name}')"
            )
        else:
            print(
                f"Found package: {actual_name} v{pkg_info['version']} at {pkg_info['path']}"
            )

    if errors:
        print("\nErrors:", file=sys.stderr)
        for err in errors:
            print(f"  - {err}", file=sys.stderr)
        print("\nAvailable packages:", file=sys.stderr)
        for name in sorted(all_packages.keys()):
            print(f"  - {name}", file=sys.stderr)
        sys.exit(1)

    # Check PyPI versions
    for pkg in validated:
        pkg_name = pkg["name"]
        pkg_version = pkg["version"]

        if publish_target == "pypi":
            result = subprocess.run(
                ["python", "scripts/pypi_query.py", "check", pkg_name, pkg_version],
                capture_output=True,
                text=True,
            )
            if result.returncode == 1:
                print(
                    f"::error::Version {pkg_version} of {pkg_name} already exists on PyPI. "
                    "Bump version in pyproject.toml before publishing.",
                    file=sys.stderr,
                )
                errors.append(
                    f"Version {pkg_version} of {pkg_name} already exists on PyPI"
                )
        else:
            result = subprocess.run(
                [
                    "python",
                    "scripts/pypi_query.py",
                    "--test-pypi",
                    "check",
                    pkg_name,
                    pkg_version,
                ],
                capture_output=True,
                text=True,
            )
            if result.returncode == 1:
                print(
                    f"::warning::Version {pkg_version} of {pkg_name} already exists on Test PyPI"
                )

    if errors:
        sys.exit(1)

    # Output validated packages as JSON
    packages_json = json.dumps(validated)
    if github_output:
        with open(github_output, "a") as f:
            print(f"packages_json={packages_json}", file=f)
    print(f"Validated {len(validated)} package(s): {packages_json}")

    return validated


def extract_package_names(packages_json: str) -> str:
    """Extract package names from packages JSON string.

    Args:
        packages_json: JSON string with package information

    Returns:
        Space-separated package names
    """
    pkgs = json.loads(packages_json)
    return " ".join([p["name"] for p in pkgs])


def check_c_extensions(package_json: str, github_output: str | None = None) -> bool:
    """Check if package has C extensions.

    Args:
        package_json: JSON string with package information
        github_output: Path to GitHub Actions output file (optional)

    Returns:
        True if package has C extensions, False otherwise
    """
    pkg = json.loads(package_json)
    has_c_ext = pkg.get("has_c_extensions", False)

    if github_output:
        with open(github_output, "a") as f:
            f.write(f"has_c_extensions={'true' if has_c_ext else 'false'}\n")

    print(f"Package {pkg.get('name', 'unknown')} has_c_extensions={has_c_ext}")

    return has_c_ext


def main() -> None:
    """Main entry point for CLI usage."""
    if len(sys.argv) < 2:
        print(
            "Usage:\n"
            "  validate_packages.py validate <packages.json> <package_input> <publish_target> [github_output]\n"
            "  validate_packages.py extract-names <packages_json>\n"
            "  validate_packages.py check-c-ext [package_json]",
            file=sys.stderr,
        )
        sys.exit(1)

    command = sys.argv[1]

    if command == "extract-names":
        if len(sys.argv) < 3:
            print(
                "Usage: validate_packages.py extract-names <packages_json>",
                file=sys.stderr,
            )
            sys.exit(1)
        packages_json = sys.argv[2]
        print(extract_package_names(packages_json))
    elif command == "check-c-ext":
        package_json = os.environ.get(
            "PACKAGE_JSON", sys.argv[2] if len(sys.argv) > 2 else "{}"
        )
        github_output = os.environ.get("GITHUB_OUTPUT")
        check_c_extensions(package_json, github_output)
    elif command == "validate":
        if len(sys.argv) < 5:
            print(
                "Usage: validate_packages.py validate <packages.json> <package_input> <publish_target> [github_output]",
                file=sys.stderr,
            )
            sys.exit(1)
        packages_json_path = sys.argv[2]
        package_input = sys.argv[3]
        publish_target = sys.argv[4]
        github_output = sys.argv[5] if len(sys.argv) > 5 else None
        validate_packages(
            packages_json_path, package_input, publish_target, github_output
        )
    else:
        # Backward compatibility: if no command, assume validate mode
        if len(sys.argv) < 4:
            print(
                "Usage: validate_packages.py <packages.json> <package_input> <publish_target> [github_output]",
                file=sys.stderr,
            )
            sys.exit(1)
        packages_json_path = sys.argv[1]
        package_input = sys.argv[2]
        publish_target = sys.argv[3]
        github_output = sys.argv[4] if len(sys.argv) > 4 else None
        validate_packages(
            packages_json_path, package_input, publish_target, github_output
        )


if __name__ == "__main__":
    main()

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
"""Query PyPI and Test PyPI for package versions."""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from urllib.error import HTTPError, URLError


def _fetch_pypi_data(package_name: str, test_pypi: bool = False) -> dict | None:
    """Fetch package data from PyPI or Test PyPI.

    Args:
        package_name: Name of the package
        test_pypi: If True, query Test PyPI instead of PyPI

    Returns:
        Package data dictionary or None if not found/error
    """
    base_url = "https://test.pypi.org" if test_pypi else "https://pypi.org"
    url = f"{base_url}/pypi/{package_name}/json"

    try:
        with urllib.request.urlopen(url, timeout=10) as response:
            return json.loads(response.read())
    except HTTPError as e:
        if e.code == 404:
            # Package doesn't exist yet
            return None
        return None
    except (URLError, TimeoutError, json.JSONDecodeError):
        return None


def check_version(package_name: str, version: str, test_pypi: bool = False) -> bool:
    """Check if version exists on PyPI or Test PyPI.

    Args:
        package_name: Name of the package
        version: Version to check
        test_pypi: If True, check Test PyPI instead of PyPI

    Returns:
        True if version exists, False otherwise
    """
    data = _fetch_pypi_data(package_name, test_pypi)
    if data is None:
        return False
    releases = data.get("releases", {})
    return version in releases


def get_pypi_versions(package_name: str, test_pypi: bool = False) -> list[str]:
    """Get all versions of a package from PyPI or Test PyPI.

    Args:
        package_name: Name of the package
        test_pypi: If True, query Test PyPI instead of PyPI

    Returns:
        List of versions sorted newest first
    """
    data = _fetch_pypi_data(package_name, test_pypi)
    if data is None:
        return []
    releases = data.get("releases", {})
    return sorted(releases.keys(), reverse=True)


def main(argv: list[str]) -> int:
    """Main entry point for CLI usage."""
    parser = argparse.ArgumentParser(
        description="Query PyPI and Test PyPI for package versions"
    )
    parser.add_argument(
        "--test-pypi", action="store_true", help="Query Test PyPI instead of PyPI"
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Check command
    check_parser = subparsers.add_parser("check", help="Check if version exists")
    check_parser.add_argument("package", help="Package name")
    check_parser.add_argument("version", help="Version to check")

    # List command
    list_parser = subparsers.add_parser("list", help="List all versions")
    list_parser.add_argument("package", help="Package name")
    list_parser.add_argument("--json", action="store_true", help="Output as JSON")

    args = parser.parse_args(argv)

    if not args.command:
        parser.print_help()
        return 1

    if args.command == "check":
        exists = check_version(args.package, args.version, test_pypi=args.test_pypi)
        if exists:
            print(
                f"Version {args.version} of {args.package} exists on {'Test PyPI' if args.test_pypi else 'PyPI'}"
            )
            return 1
        else:
            print(
                f"Version {args.version} of {args.package} does not exist on {'Test PyPI' if args.test_pypi else 'PyPI'}"
            )
            return 0

    elif args.command == "list":
        versions = get_pypi_versions(args.package, test_pypi=args.test_pypi)
        if args.json:
            print(json.dumps(versions))
        else:
            if versions:
                print(
                    f"Versions on {'Test PyPI' if args.test_pypi else 'PyPI'}: {', '.join(versions)}"
                )
            else:
                print(
                    f"No versions found on {'Test PyPI' if args.test_pypi else 'PyPI'}"
                )
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

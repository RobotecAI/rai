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
"""Check if a package version exists on PyPI or Test PyPI."""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from urllib.error import HTTPError, URLError


def check_version(package_name: str, version: str, test_pypi: bool = False) -> bool:
    """Check if version exists on PyPI or Test PyPI."""
    base_url = "https://test.pypi.org" if test_pypi else "https://pypi.org"
    url = f"{base_url}/pypi/{package_name}/json"

    try:
        with urllib.request.urlopen(url, timeout=10) as response:
            data = json.loads(response.read())
            releases = data.get("releases", {})
            return version in releases
    except HTTPError as e:
        if e.code == 404:
            # Package doesn't exist yet
            return False
        # Other HTTP errors - assume version doesn't exist
        return False
    except (URLError, TimeoutError, json.JSONDecodeError):
        # Network issues or invalid response - assume version doesn't exist
        return False


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description="Check if package version exists on PyPI"
    )
    parser.add_argument("package", help="Package name")
    parser.add_argument("version", help="Version to check")
    parser.add_argument(
        "--test-pypi", action="store_true", help="Check Test PyPI instead of PyPI"
    )
    args = parser.parse_args(argv)

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


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

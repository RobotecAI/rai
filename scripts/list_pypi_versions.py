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
"""List versions of a package on PyPI or Test PyPI."""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from urllib.error import HTTPError, URLError


def get_pypi_versions(package_name: str, test_pypi: bool = False) -> list[str]:
    """Get all versions of a package from PyPI or Test PyPI."""
    base_url = "https://test.pypi.org" if test_pypi else "https://pypi.org"
    url = f"{base_url}/pypi/{package_name}/json"

    try:
        with urllib.request.urlopen(url, timeout=10) as response:
            data = json.loads(response.read())
            releases = data.get("releases", {})
            # Return sorted versions (newest first)
            versions = sorted(releases.keys(), reverse=True)
            return versions
    except HTTPError as e:
        if e.code == 404:
            # Package doesn't exist yet
            return []
        return []
    except (URLError, TimeoutError, json.JSONDecodeError):
        return []


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="List versions of a package on PyPI")
    parser.add_argument("package", help="Package name")
    parser.add_argument(
        "--test-pypi", action="store_true", help="Check Test PyPI instead of PyPI"
    )
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args(argv)

    versions = get_pypi_versions(args.package, test_pypi=args.test_pypi)

    if args.json:
        print(json.dumps(versions))
    else:
        if versions:
            print(
                f"Versions on {'Test PyPI' if args.test_pypi else 'PyPI'}: {', '.join(versions)}"
            )
        else:
            print(f"No versions found on {'Test PyPI' if args.test_pypi else 'PyPI'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

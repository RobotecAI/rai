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
"""Discover packages and map package names to paths.

Outputs package information for GitHub Actions workflow inputs.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys

VERSION_RE = re.compile(r'^\s*version\s*=\s*"([^"]+)"\s*$', re.MULTILINE)
POETRY_NAME_RE = re.compile(r'^\s*name\s*=\s*"([^"]+)"\s*$', re.MULTILINE)
HAS_C_EXTENSIONS_RE = re.compile(
    r"^\s*has_c_extensions\s*=\s*(true|false)\s*$", re.MULTILINE
)


def _extract_poetry_name(pyproject_text: str) -> str | None:
    m = POETRY_NAME_RE.search(pyproject_text)
    return None if m is None else m.group(1)


def _extract_version(pyproject_text: str) -> str | None:
    m = VERSION_RE.search(pyproject_text)
    return None if m is None else m.group(1)


def _has_c_extensions_marker(pyproject_text: str) -> bool | None:
    """Check for explicit marker in [tool.rai] section."""
    # Look for [tool.rai] section
    tool_rai_match = re.search(r"\[tool\.rai\].*?(?=\[|$)", pyproject_text, re.DOTALL)
    if tool_rai_match:
        m = HAS_C_EXTENSIONS_RE.search(tool_rai_match.group(0))
        if m:
            return m.group(1).lower() == "true"
    return None


def _detect_c_extensions(package_dir: pathlib.Path) -> bool:
    """Auto-detect if package has C extensions."""
    # Check for setup.py with ext_modules
    setup_py = package_dir / "setup.py"
    if setup_py.exists():
        try:
            content = setup_py.read_text(encoding="utf-8")
            if "ext_modules" in content or "Extension(" in content:
                return True
        except Exception:
            pass

    # Check for C/C++/Cython source files in package directory
    for ext in (".c", ".cpp", ".cxx", ".cc", ".pyx"):
        if any(package_dir.rglob(f"*{ext}")):
            return True

    return False


def discover_packages(repo_root: pathlib.Path) -> dict[str, dict[str, str]]:
    """Discover all packages and return dict mapping name -> {path, version}."""
    src_root = repo_root / "src"
    if not src_root.exists():
        return {}

    packages: dict[str, dict[str, str]] = {}
    for pyproject in sorted(src_root.rglob("pyproject.toml")):
        rel_dir = pyproject.parent.relative_to(repo_root).as_posix()
        try:
            text = pyproject.read_text(encoding="utf-8")
        except Exception:
            continue

        name = _extract_poetry_name(text)
        version = _extract_version(text)
        if name is None or version is None:
            continue

        # Check for C extensions: explicit marker first, then auto-detect
        has_c_ext = _has_c_extensions_marker(text)
        if has_c_ext is None:
            has_c_ext = _detect_c_extensions(pyproject.parent)

        packages[name] = {
            "path": rel_dir,
            "version": version,
            "has_c_extensions": has_c_ext,
        }

    return packages


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Discover packages in monorepo")
    parser.add_argument(
        "--output-format",
        choices=["json", "list"],
        default="json",
        help="Output format",
    )
    args = parser.parse_args(argv)

    repo_root = pathlib.Path(__file__).resolve().parents[1]
    packages = discover_packages(repo_root)

    if args.output_format == "json":
        print(json.dumps(packages, indent=2))
    else:
        for name in sorted(packages.keys()):
            print(name)

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

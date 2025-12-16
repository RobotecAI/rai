# Copyright (C) 2025 Robotec.AI
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
"""CI helper: require version bumps for changed packages in this monorepo.

Policy:
- Detect which package directories changed between BASE..HEAD.
- For each changed package, require `tool.poetry.version` in that package's
  `pyproject.toml` to change between BASE and HEAD.

Optional local helper:
- If --bump is provided, bump the version in the working tree for each changed
  package that did not already bump its version.
"""

from __future__ import annotations

import argparse
import logging
import pathlib
import re
import subprocess
import sys
from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class Package:
    name: str
    path: str  # repo-relative path

    @property
    def pyproject_path(self) -> str:
        return f"{self.path}/pyproject.toml"


VERSION_RE = re.compile(r'^\s*version\s*=\s*"([^"]+)"\s*$', re.MULTILINE)
SEMVER_CORE_RE = re.compile(
    r"^(?P<major>\d+)\.(?P<minor>\d+)\.(?P<patch>\d+)(?:[.\-+].*)?$"
)
POETRY_NAME_RE = re.compile(r'^\s*name\s*=\s*"([^"]+)"\s*$', re.MULTILINE)

LOG = logging.getLogger(__name__)


def _run(cmd: list[str]) -> str:
    return subprocess.check_output(cmd, text=True).strip()


def _git_show(sha: str, path: str) -> str | None:
    try:
        return subprocess.check_output(["git", "show", f"{sha}:{path}"], text=True)
    except subprocess.CalledProcessError:
        return None


def _extract_version(pyproject_text: str) -> str | None:
    m = VERSION_RE.search(pyproject_text)
    return None if m is None else m.group(1)


def _extract_poetry_name(pyproject_text: str) -> str | None:
    m = POETRY_NAME_RE.search(pyproject_text)
    return None if m is None else m.group(1)


def _discover_packages(repo_root: pathlib.Path) -> tuple[Package, ...]:
    """
    Discover packages by scanning for src/**/pyproject.toml with [tool.poetry].
    This avoids hardcoding a list, so adding a new package only requires adding
    its pyproject.toml under src/.
    """
    src_root = repo_root / "src"
    if not src_root.exists():
        return tuple()

    packages: list[Package] = []
    for pyproject in sorted(src_root.rglob("pyproject.toml")):
        rel_dir = pyproject.parent.relative_to(repo_root).as_posix()
        try:
            text = pyproject.read_text(encoding="utf-8")
        except Exception:
            LOG.debug("skip unreadable pyproject: %s", rel_dir)
            continue

        # Only treat it as a Poetry package if it has a name and version.
        name = _extract_poetry_name(text)
        version = _extract_version(text)
        if name is None or version is None:
            LOG.debug("skip non-poetry pyproject (missing name/version): %s", rel_dir)
            continue

        packages.append(Package(name=name, path=rel_dir))

    LOG.debug("discovered %d packages under src/", len(packages))
    return tuple(packages)


def _bump_semver(version: str, part: str) -> str | None:
    """
    Bump semver-like versions (MAJOR.MINOR.PATCH), ignoring any suffix.
    Returns None if the version is not semver-like.
    """
    m = SEMVER_CORE_RE.match(version)
    if m is None:
        return None
    major = int(m.group("major"))
    minor = int(m.group("minor"))
    patch = int(m.group("patch"))
    if part == "major":
        major += 1
        minor = 0
        patch = 0
    elif part == "minor":
        minor += 1
        patch = 0
    elif part == "patch":
        patch += 1
    else:
        raise ValueError(f"unknown bump part: {part}")
    return f"{major}.{minor}.{patch}"


def _replace_version_line(pyproject_text: str, new_version: str) -> str:
    # Replace the first matching version line.
    def _repl(match: re.Match[str]) -> str:
        prefix = match.group(0).split("=", 1)[0]
        return f'{prefix}= "{new_version}"'

    updated, n = VERSION_RE.subn(_repl, pyproject_text, count=1)
    if n != 1:
        raise ValueError("expected exactly one version line to replace")
    return updated


def _changed_files(base_sha: str, head_sha: str) -> set[str]:
    out = _run(["git", "diff", "--name-only", f"{base_sha}..{head_sha}"])
    if not out:
        return set()
    return {line.strip() for line in out.splitlines() if line.strip()}


def _is_file_under_dir(file_path: str, dir_path: str) -> bool:
    # Normalize to forward slashes for git paths.
    file_path = file_path.replace("\\", "/")
    dir_path = dir_path.rstrip("/").replace("\\", "/")
    return file_path == dir_path or file_path.startswith(dir_path + "/")


def _changed_packages(changed_files: Iterable[str]) -> list[Package]:
    raise RuntimeError(
        "_changed_packages requires packages list; use _changed_packages_for_repo"
    )


def _changed_packages_for_repo(
    repo_root: pathlib.Path, changed_files: Iterable[str]
) -> list[Package]:
    pkgs = _discover_packages(repo_root)
    changed = []
    for pkg in pkgs:
        if any(_is_file_under_dir(f, pkg.path) for f in changed_files):
            changed.append(pkg)
            LOG.debug("package changed: %s (%s)", pkg.name, pkg.path)
    return changed


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-sha", required=True)
    parser.add_argument("--head-sha", required=True)
    parser.add_argument("--bump", choices=["patch", "minor", "major"])
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging (useful when troubleshooting CI behavior).",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)s: %(message)s",
    )

    # Ensure we run from repo root (common in CI, but also allow local use).
    repo_root = pathlib.Path(__file__).resolve().parents[1]
    try:
        subprocess.check_call(
            ["git", "rev-parse", "--show-toplevel"], stdout=subprocess.DEVNULL
        )
    except Exception:
        print("error: must be run inside a git repository", file=sys.stderr)
        return 2

    changed_files = _changed_files(args.base_sha, args.head_sha)
    LOG.debug("changed files (%d): %s", len(changed_files), sorted(changed_files))
    pkgs = _changed_packages_for_repo(repo_root, changed_files)

    if not pkgs:
        print("No package changes detected; skipping version bump check.")
        return 0

    failures: list[str] = []
    print(f"Changed packages ({len(pkgs)}): " + ", ".join(p.name for p in pkgs))

    for pkg in pkgs:
        base_text = _git_show(args.base_sha, pkg.pyproject_path)
        head_text = _git_show(args.head_sha, pkg.pyproject_path)

        base_ver = None if base_text is None else _extract_version(base_text)
        head_ver = None if head_text is None else _extract_version(head_text)

        print(f"- {pkg.name}: base={base_ver!r} head={head_ver!r}")

        if base_ver is None or head_ver is None:
            failures.append(
                f"{pkg.name}: could not read version from {pkg.pyproject_path} in "
                f"{'base' if base_ver is None else 'head'}"
            )
            continue

        if base_ver == head_ver:
            if args.bump is None:
                failures.append(
                    f"{pkg.name}: version not bumped (still {head_ver}); please bump {pkg.pyproject_path}"
                )
            else:
                bumped = _bump_semver(head_ver, args.bump)
                if bumped is None:
                    failures.append(
                        f"{pkg.name}: cannot auto-bump non-semver version {head_ver!r}; please bump {pkg.pyproject_path}"
                    )
                    continue
                pyproject_file = repo_root / pkg.pyproject_path
                try:
                    text = pyproject_file.read_text(encoding="utf-8")
                    pyproject_file.write_text(
                        _replace_version_line(text, bumped), encoding="utf-8"
                    )
                    print(f"  -> bumped {pkg.name} to {bumped} (edited working tree)")
                except Exception as e:
                    failures.append(
                        f"{pkg.name}: failed to bump version in working tree: {e}"
                    )

    if failures:
        print("\nVersion bump check failed:", file=sys.stderr)
        for msg in failures:
            print(f"- {msg}", file=sys.stderr)
        return 1

    if args.bump is None:
        print("All changed packages have version bumps.")
    else:
        print("Done. Review changes and commit the bumped pyproject.toml files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

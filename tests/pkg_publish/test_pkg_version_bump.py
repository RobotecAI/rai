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


import importlib


def _load_module():
    # Import via file path relative to repo layout.
    # scripts/ is not a package, so we import by module name after adjusting sys.path.
    import sys
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root))
    try:
        return importlib.import_module("scripts.pkg_version_bump")
    finally:
        sys.path.pop(0)


def test_extract_version():
    m = _load_module()
    text = """
[tool.poetry]
name = "x"
version = "1.2.3"
"""
    assert m._extract_version(text) == "1.2.3"


def test_bump_semver_patch_minor_major():
    m = _load_module()
    assert m._bump_semver("1.2.3", "patch") == "1.2.4"
    assert m._bump_semver("1.2.3", "minor") == "1.3.0"
    assert m._bump_semver("1.2.3", "major") == "2.0.0"


def test_bump_semver_with_suffix_ignored():
    m = _load_module()
    assert m._bump_semver("2.0.0.a2", "patch") == "2.0.1"
    assert m._bump_semver("2.0.0-alpha.1", "minor") == "2.1.0"


def test_bump_semver_non_semver_returns_none():
    m = _load_module()
    assert m._bump_semver("dev", "patch") is None
    assert m._bump_semver("1.2", "patch") is None


def test_replace_version_line():
    m = _load_module()
    text = """
[tool.poetry]
name = "x"
version = "1.2.3"
description = "d"
"""
    out = m._replace_version_line(text, "1.2.4")
    assert 'version = "1.2.4"' in out
    assert 'version = "1.2.3"' not in out


def test_discover_packages_finds_expected(monkeypatch):
    m = _load_module()
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[2]
    pkgs = m._discover_packages(repo_root)
    assert pkgs, "expected at least one package to be discovered under src/"

    by_name = {p.name: p.path for p in pkgs}
    # Spot-check a few known packages.
    assert by_name["rai_core"] == "src/rai_core"
    assert by_name["rai-perception"] == "src/rai_extensions/rai_perception"
    assert by_name["rai-bench"] == "src/rai_bench"


def test_changed_packages_detection():
    m = _load_module()
    changed = {
        "src/rai_core/rai/something.py",
        "src/rai_extensions/rai_perception/pyproject.toml",
        "README.md",
    }
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[2]
    pkgs = m._changed_packages_for_repo(repo_root, changed)
    names = {p.name for p in pkgs}
    assert "rai_core" in names
    assert "rai-perception" in names
    assert "rai-bench" not in names


def test_main_no_package_changes(monkeypatch, capsys):
    m = _load_module()
    monkeypatch.setattr(m.subprocess, "check_call", lambda *a, **k: 0)
    monkeypatch.setattr(m, "_changed_files", lambda base, head: set())
    rc = m.main(["--base-sha", "a", "--head-sha", "b"])
    assert rc == 0
    assert "No package changes detected" in capsys.readouterr().out


def test_main_fails_when_changed_package_not_bumped(monkeypatch, capsys):
    m = _load_module()
    monkeypatch.setattr(m.subprocess, "check_call", lambda *a, **k: 0)
    monkeypatch.setattr(m, "_changed_files", lambda base, head: {"src/rai_core/x.py"})

    def fake_show(sha, path):
        assert path == "src/rai_core/pyproject.toml"
        return '[tool.poetry]\nversion = "1.2.3"\n'

    monkeypatch.setattr(m, "_git_show", fake_show)
    rc = m.main(["--base-sha", "a", "--head-sha", "b"])
    assert rc == 1
    err = capsys.readouterr().err
    assert "version not bumped" in err


def test_main_passes_when_changed_package_bumped(monkeypatch, capsys):
    m = _load_module()
    monkeypatch.setattr(m.subprocess, "check_call", lambda *a, **k: 0)
    monkeypatch.setattr(m, "_changed_files", lambda base, head: {"src/rai_core/x.py"})

    def fake_show(sha, path):
        assert path == "src/rai_core/pyproject.toml"
        if sha == "base":
            return '[tool.poetry]\nversion = "1.2.3"\n'
        return '[tool.poetry]\nversion = "1.2.4"\n'

    monkeypatch.setattr(m, "_git_show", fake_show)
    rc = m.main(["--base-sha", "base", "--head-sha", "head"])
    assert rc == 0
    assert "All changed packages have version bumps" in capsys.readouterr().out


def test_main_fails_when_version_missing(monkeypatch, capsys):
    m = _load_module()
    monkeypatch.setattr(m.subprocess, "check_call", lambda *a, **k: 0)
    monkeypatch.setattr(m, "_changed_files", lambda base, head: {"src/rai_core/x.py"})
    monkeypatch.setattr(m, "_git_show", lambda sha, path: "[tool.poetry]\nname='x'\n")
    rc = m.main(["--base-sha", "a", "--head-sha", "b"])
    assert rc == 1
    assert "could not read version" in capsys.readouterr().err

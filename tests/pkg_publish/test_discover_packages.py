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

"""Unit tests for discover_packages.py script."""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from scripts.discover_packages import (
    _detect_c_extensions,
    _extract_poetry_name,
    _extract_version,
    _has_c_extensions_marker,
    discover_packages,
    main,
)


class TestExtractFunctions:
    """Tests for extraction functions."""

    @pytest.mark.parametrize(
        "text,expected",
        [
            ('name = "test-package"', "test-package"),
            ('  name = "test-package"  ', "test-package"),
            ("no name here", None),
            ('[tool.poetry]\nname = "test-package"\nversion = "1.0.0"', "test-package"),
        ],
    )
    def test_extract_poetry_name(self, text, expected):
        assert _extract_poetry_name(text) == expected

    @pytest.mark.parametrize(
        "text,expected",
        [
            ('version = "1.0.0"', "1.0.0"),
            ('  version = "2.3.4"  ', "2.3.4"),
            ("no version here", None),
            ('[tool.poetry]\nname = "test"\nversion = "1.2.3"', "1.2.3"),
        ],
    )
    def test_extract_version(self, text, expected):
        assert _extract_version(text) == expected


class TestHasCExtensionsMarker:
    """Tests for _has_c_extensions_marker function."""

    @pytest.mark.parametrize(
        "text,expected",
        [
            (
                '[tool.poetry]\nname = "test"\n\n[tool.rai]\nhas_c_extensions = true',
                True,
            ),
            (
                '[tool.poetry]\nname = "test"\n\n[tool.rai]\nhas_c_extensions = false',
                False,
            ),
            ('[tool.poetry]\nname = "test"', None),
            ('[tool.poetry]\nname = "test"\nhas_c_extensions = true', None),
        ],
    )
    def test_marker_detection(self, text, expected):
        assert _has_c_extensions_marker(text) == expected


class TestDetectCExtensions:
    """Tests for _detect_c_extensions function."""

    @pytest.mark.parametrize(
        "setup_content,has_ext",
        [
            (
                "from setuptools import Extension, setup\nsetup(ext_modules=[Extension('test', ['test.c'])])",
                True,
            ),
            ("from setuptools import setup\nsetup(name='test')", False),
        ],
    )
    def test_detect_setup_py(self, tmp_path, setup_content, has_ext):
        package_dir = tmp_path / "test_pkg"
        package_dir.mkdir()
        (package_dir / "setup.py").write_text(setup_content)
        assert _detect_c_extensions(package_dir) is has_ext

    @pytest.mark.parametrize("ext", [".c", ".cpp", ".pyx"])
    def test_detect_source_files(self, tmp_path, ext):
        package_dir = tmp_path / "test_pkg"
        package_dir.mkdir()
        (package_dir / f"test{ext}").write_text(
            "int main() { return 0; }" if ext != ".pyx" else "def test(): pass"
        )
        assert _detect_c_extensions(package_dir) is True

    def test_no_c_extensions(self, tmp_path):
        package_dir = tmp_path / "test_pkg"
        package_dir.mkdir()
        (package_dir / "test.py").write_text("def test(): pass")
        assert _detect_c_extensions(package_dir) is False


class TestDiscoverPackages:
    """Tests for discover_packages function."""

    def test_discover_packages_basic(self, tmp_path):
        src_root = tmp_path / "src"
        src_root.mkdir()
        pkg_dir = src_root / "test_pkg"
        pkg_dir.mkdir()
        (pkg_dir / "pyproject.toml").write_text('name = "test-pkg"\nversion = "1.0.0"')

        packages = discover_packages(tmp_path)

        assert "test-pkg" in packages
        assert packages["test-pkg"]["version"] == "1.0.0"
        assert packages["test-pkg"]["path"] == "src/test_pkg"
        assert packages["test-pkg"]["has_c_extensions"] is False

    def test_discover_multiple_packages(self, tmp_path):
        src_root = tmp_path / "src"
        src_root.mkdir()
        for name, version in [("pkg1", "1.0.0"), ("pkg2", "2.0.0")]:
            pkg_dir = src_root / name
            pkg_dir.mkdir()
            (pkg_dir / "pyproject.toml").write_text(
                f'name = "{name}"\nversion = "{version}"'
            )

        packages = discover_packages(tmp_path)

        assert len(packages) == 2
        assert all(name in packages for name in ["pkg1", "pkg2"])

    @pytest.mark.parametrize(
        "detection_method,content",
        [
            ("marker", "[tool.rai]\nhas_c_extensions = true"),
            ("c_file", None),
            ("setup_py", None),
        ],
    )
    def test_discover_package_with_c_extensions(
        self, tmp_path, detection_method, content
    ):
        src_root = tmp_path / "src"
        src_root.mkdir()
        pkg_dir = src_root / "test_pkg"
        pkg_dir.mkdir()
        pyproject_content = 'name = "test-pkg"\nversion = "1.0.0"'
        if detection_method == "marker" and content:
            pyproject_content += f"\n\n{content}"
        (pkg_dir / "pyproject.toml").write_text(pyproject_content)
        if detection_method == "c_file":
            (pkg_dir / "test.c").write_text("int main() { return 0; }")
        elif detection_method == "setup_py":
            (pkg_dir / "setup.py").write_text(
                "from setuptools import Extension, setup\nsetup(ext_modules=[Extension('test', ['test.c'])])"
            )

        packages = discover_packages(tmp_path)
        assert packages["test-pkg"]["has_c_extensions"] is True

    @pytest.mark.parametrize("missing_field", ["name", "version"])
    def test_discover_package_missing_fields(self, tmp_path, missing_field):
        src_root = tmp_path / "src"
        src_root.mkdir()
        pkg_dir = src_root / "test_pkg"
        pkg_dir.mkdir()
        content = 'name = "test-pkg"\nversion = "1.0.0"'.replace(
            f"{missing_field} = ", f"# {missing_field} = "
        )
        (pkg_dir / "pyproject.toml").write_text(content)

        packages = discover_packages(tmp_path)
        assert len(packages) == 0

    def test_discover_edge_cases(self, tmp_path):
        # No src directory
        assert discover_packages(tmp_path) == {}

        # Nested package
        src_root = tmp_path / "src"
        src_root.mkdir()
        nested_dir = src_root / "extensions" / "nested_pkg"
        nested_dir.mkdir(parents=True)
        (nested_dir / "pyproject.toml").write_text(
            'name = "nested-pkg"\nversion = "1.0.0"'
        )

        packages = discover_packages(tmp_path)
        assert "nested-pkg" in packages
        assert "extensions/nested_pkg" in packages["nested-pkg"]["path"]


class TestMain:
    """Tests for main function."""

    @pytest.mark.parametrize("output_format", ["json", "list"])
    def test_main_output_formats(self, tmp_path, capsys, output_format):
        with patch("scripts.discover_packages.discover_packages") as mock_discover:
            mock_discover.return_value = {
                "test-pkg": {
                    "path": "src/test_pkg",
                    "version": "1.0.0",
                    "has_c_extensions": False,
                }
            }

            mock_file = tmp_path / "scripts" / "discover_packages.py"
            mock_file.parent.mkdir(parents=True)
            mock_file.touch()

            with patch("scripts.discover_packages.__file__", str(mock_file)):
                result = main(["--output-format", output_format])

                assert result == 0
                output = capsys.readouterr().out
                if output_format == "json":
                    data = json.loads(output.strip())
                    assert "test-pkg" in data
                else:
                    assert "test-pkg" in output

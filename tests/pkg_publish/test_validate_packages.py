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
"""Unit tests for validate_packages.py script."""

from __future__ import annotations

import json
from unittest.mock import Mock, patch

import pytest

from scripts.validate_packages import (
    check_c_extensions,
    extract_package_names,
    main,
    validate_packages,
)


class TestExtractPackageNames:
    """Tests for extract_package_names function."""

    def test_extract_single_package(self):
        packages_json = json.dumps([{"name": "test-package", "version": "1.0.0"}])
        result = extract_package_names(packages_json)
        assert result == "test-package"

    def test_extract_multiple_packages(self):
        packages_json = json.dumps(
            [
                {"name": "package1", "version": "1.0.0"},
                {"name": "package2", "version": "2.0.0"},
                {"name": "package3", "version": "3.0.0"},
            ]
        )
        result = extract_package_names(packages_json)
        assert result == "package1 package2 package3"

    def test_extract_empty_list(self):
        packages_json = json.dumps([])
        result = extract_package_names(packages_json)
        assert result == ""


class TestCheckCExtensions:
    """Tests for check_c_extensions function."""

    def test_package_with_c_extensions(self, capsys):
        package_json = json.dumps(
            {"name": "test-package", "version": "1.0.0", "has_c_extensions": True}
        )
        result = check_c_extensions(package_json)
        assert result is True
        captured = capsys.readouterr()
        assert "has_c_extensions=True" in captured.out

    def test_package_without_c_extensions(self, capsys):
        package_json = json.dumps(
            {"name": "test-package", "version": "1.0.0", "has_c_extensions": False}
        )
        result = check_c_extensions(package_json)
        assert result is False
        captured = capsys.readouterr()
        assert "has_c_extensions=False" in captured.out

    def test_package_missing_c_extensions_key(self, capsys):
        package_json = json.dumps({"name": "test-package", "version": "1.0.0"})
        result = check_c_extensions(package_json)
        assert result is False

    def test_write_to_github_output(self, tmp_path):
        github_output = tmp_path / "output.txt"
        package_json = json.dumps(
            {"name": "test-package", "version": "1.0.0", "has_c_extensions": True}
        )
        check_c_extensions(package_json, str(github_output))
        content = github_output.read_text()
        assert "has_c_extensions=true" in content


class TestValidatePackages:
    """Tests for validate_packages function."""

    def test_validate_single_package_success(self, tmp_path, capsys):
        packages_json = {
            "test-package": {
                "name": "test-package",
                "path": "src/test_package",
                "version": "1.0.0",
            }
        }
        packages_file = tmp_path / "packages.json"
        packages_file.write_text(json.dumps(packages_json))

        with patch("scripts.validate_packages.subprocess.run") as mock_run:
            mock_run.return_value = Mock(returncode=0)
            result = validate_packages(str(packages_file), "test-package", "test-pypi")

        assert len(result) == 1
        assert result[0]["name"] == "test-package"
        assert result[0]["version"] == "1.0.0"
        captured = capsys.readouterr()
        assert "Found package: test-package" in captured.out

    def test_validate_multiple_packages_success(self, tmp_path):
        packages_json = {
            "package1": {
                "name": "package1",
                "path": "src/package1",
                "version": "1.0.0",
            },
            "package2": {
                "name": "package2",
                "path": "src/package2",
                "version": "2.0.0",
            },
        }
        packages_file = tmp_path / "packages.json"
        packages_file.write_text(json.dumps(packages_json))

        with patch("scripts.validate_packages.subprocess.run") as mock_run:
            mock_run.return_value = Mock(returncode=0)
            result = validate_packages(
                str(packages_file), "package1,package2", "test-pypi"
            )

        assert len(result) == 2
        assert result[0]["name"] == "package1"
        assert result[1]["name"] == "package2"

    def test_validate_package_not_found(self, tmp_path, capsys):
        packages_json = {
            "package1": {
                "name": "package1",
                "path": "src/package1",
                "version": "1.0.0",
            }
        }
        packages_file = tmp_path / "packages.json"
        packages_file.write_text(json.dumps(packages_json))

        with pytest.raises(SystemExit) as exc_info:
            validate_packages(str(packages_file), "nonexistent", "test-pypi")

        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "Package 'nonexistent' not found" in captured.err

    def test_validate_package_with_spaces(self, tmp_path):
        """Test that spaces around package names are handled correctly."""
        packages_json = {
            "package1": {
                "name": "package1",
                "path": "src/package1",
                "version": "1.0.0",
            }
        }
        packages_file = tmp_path / "packages.json"
        packages_file.write_text(json.dumps(packages_json))

        with patch("scripts.validate_packages.subprocess.run") as mock_run:
            mock_run.return_value = Mock(returncode=0)
            result = validate_packages(str(packages_file), " package1 ", "test-pypi")

        assert len(result) == 1
        assert result[0]["name"] == "package1"

    def test_validate_package_hyphen_to_underscore_variant(self, tmp_path, capsys):
        """Test that package with hyphen can be found using underscore."""
        packages_json = {
            "rai-perception": {
                "name": "rai-perception",
                "path": "src/rai_perception",
                "version": "1.0.0",
            }
        }
        packages_file = tmp_path / "packages.json"
        packages_file.write_text(json.dumps(packages_json))

        with patch("scripts.validate_packages.subprocess.run") as mock_run:
            mock_run.return_value = Mock(returncode=0)
            result = validate_packages(
                str(packages_file), "rai_perception", "test-pypi"
            )

        assert len(result) == 1
        assert result[0]["name"] == "rai-perception"  # Uses actual name from repo
        captured = capsys.readouterr()
        assert "rai-perception" in captured.out
        assert "matched 'rai_perception'" in captured.out

    def test_validate_package_underscore_to_hyphen_variant(self, tmp_path, capsys):
        """Test that package with underscore can be found using hyphen."""
        packages_json = {
            "rai_core": {
                "name": "rai_core",
                "path": "src/rai_core",
                "version": "1.0.0",
            }
        }
        packages_file = tmp_path / "packages.json"
        packages_file.write_text(json.dumps(packages_json))

        with patch("scripts.validate_packages.subprocess.run") as mock_run:
            mock_run.return_value = Mock(returncode=0)
            result = validate_packages(str(packages_file), "rai-core", "test-pypi")

        assert len(result) == 1
        assert result[0]["name"] == "rai_core"  # Uses actual name from repo
        captured = capsys.readouterr()
        assert "rai_core" in captured.out
        assert "matched 'rai-core'" in captured.out

    def test_validate_package_exact_match_preferred(self, tmp_path, capsys):
        """Test that exact match is preferred when both variants exist."""
        packages_json = {
            "rai-perception": {
                "name": "rai-perception",
                "path": "src/rai_perception",
                "version": "1.0.0",
            },
            "rai_perception": {
                "name": "rai_perception",
                "path": "src/rai_perception_alt",
                "version": "2.0.0",
            },
        }
        packages_file = tmp_path / "packages.json"
        packages_file.write_text(json.dumps(packages_json))

        with patch("scripts.validate_packages.subprocess.run") as mock_run:
            mock_run.return_value = Mock(returncode=0)
            # Input with hyphen should match hyphen variant
            result_hyphen = validate_packages(
                str(packages_file), "rai-perception", "test-pypi"
            )
            # Input with underscore should match underscore variant
            result_underscore = validate_packages(
                str(packages_file), "rai_perception", "test-pypi"
            )

        assert len(result_hyphen) == 1
        assert result_hyphen[0]["name"] == "rai-perception"
        assert result_hyphen[0]["version"] == "1.0.0"

        assert len(result_underscore) == 1
        assert result_underscore[0]["name"] == "rai_perception"
        assert result_underscore[0]["version"] == "2.0.0"

    def test_validate_package_no_variant_matching_when_no_separator(self, tmp_path):
        """Test that variant matching only works when name contains - or _."""
        packages_json = {
            "package": {
                "name": "package",
                "path": "src/package",
                "version": "1.0.0",
            }
        }
        packages_file = tmp_path / "packages.json"
        packages_file.write_text(json.dumps(packages_json))

        with patch("scripts.validate_packages.subprocess.run") as mock_run:
            mock_run.return_value = Mock(returncode=0)
            # Should work with exact match
            result = validate_packages(str(packages_file), "package", "test-pypi")

        assert len(result) == 1
        assert result[0]["name"] == "package"

    def test_validate_pypi_version_exists(self, tmp_path, capsys):
        packages_json = {
            "test-package": {
                "name": "test-package",
                "path": "src/test_package",
                "version": "1.0.0",
            }
        }
        packages_file = tmp_path / "packages.json"
        packages_file.write_text(json.dumps(packages_json))

        with patch("scripts.validate_packages.subprocess.run") as mock_run:
            mock_run.return_value = Mock(returncode=1)
            with pytest.raises(SystemExit) as exc_info:
                validate_packages(str(packages_file), "test-package", "pypi")

        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "already exists on PyPI" in captured.err

    def test_validate_test_pypi_version_exists_warning(self, tmp_path, capsys):
        packages_json = {
            "test-package": {
                "name": "test-package",
                "path": "src/test_package",
                "version": "1.0.0",
            }
        }
        packages_file = tmp_path / "packages.json"
        packages_file.write_text(json.dumps(packages_json))

        with patch("scripts.validate_packages.subprocess.run") as mock_run:
            mock_run.return_value = Mock(returncode=1)
            result = validate_packages(str(packages_file), "test-package", "test-pypi")

        # Test PyPI warnings don't fail validation
        assert len(result) == 1
        captured = capsys.readouterr()
        assert "already exists on Test PyPI" in captured.out

    def test_write_to_github_output(self, tmp_path):
        packages_json = {
            "test-package": {
                "name": "test-package",
                "path": "src/test_package",
                "version": "1.0.0",
            }
        }
        packages_file = tmp_path / "packages.json"
        packages_file.write_text(json.dumps(packages_json))
        github_output = tmp_path / "output.txt"

        with patch("scripts.validate_packages.subprocess.run") as mock_run:
            mock_run.return_value = Mock(returncode=0)
            validate_packages(
                str(packages_file),
                "test-package",
                "test-pypi",
                str(github_output),
            )

        content = github_output.read_text()
        assert "packages_json=" in content
        assert "test-package" in content


class TestMain:
    """Tests for main CLI function."""

    def test_main_validate_command(self, tmp_path, capsys):
        packages_json = {
            "test-package": {
                "name": "test-package",
                "path": "src/test_package",
                "version": "1.0.0",
            }
        }
        packages_file = tmp_path / "packages.json"
        packages_file.write_text(json.dumps(packages_json))

        with (
            patch("scripts.validate_packages.subprocess.run") as mock_run,
            patch(
                "sys.argv",
                [
                    "validate_packages.py",
                    "validate",
                    str(packages_file),
                    "test-package",
                    "test-pypi",
                ],
            ),
        ):
            mock_run.return_value = Mock(returncode=0)
            main()

        captured = capsys.readouterr()
        assert "Validated" in captured.out

    def test_main_extract_names_command(self, capsys):
        packages_json = json.dumps([{"name": "package1"}, {"name": "package2"}])

        with patch(
            "sys.argv", ["validate_packages.py", "extract-names", packages_json]
        ):
            main()

        captured = capsys.readouterr()
        assert captured.out.strip() == "package1 package2"

    def test_main_check_c_ext_command(self, capsys, monkeypatch):
        package_json = json.dumps({"name": "test-package", "has_c_extensions": True})
        monkeypatch.setenv("PACKAGE_JSON", package_json)

        with patch("sys.argv", ["validate_packages.py", "check-c-ext"]):
            main()

        captured = capsys.readouterr()
        assert "has_c_extensions=True" in captured.out

    def test_main_backward_compatibility(self, tmp_path, capsys):
        packages_json = {
            "test-package": {
                "name": "test-package",
                "path": "src/test_package",
                "version": "1.0.0",
            }
        }
        packages_file = tmp_path / "packages.json"
        packages_file.write_text(json.dumps(packages_json))

        with (
            patch("scripts.validate_packages.subprocess.run") as mock_run,
            patch(
                "sys.argv",
                [
                    "validate_packages.py",
                    str(packages_file),
                    "test-package",
                    "test-pypi",
                ],
            ),
        ):
            mock_run.return_value = Mock(returncode=0)
            main()

        captured = capsys.readouterr()
        assert "Validated" in captured.out

    def test_main_insufficient_args(self, capsys):
        with (
            patch("sys.argv", ["validate_packages.py"]),
            pytest.raises(SystemExit) as exc_info,
        ):
            main()

        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "Usage:" in captured.err

    def test_main_extract_names_insufficient_args(self, capsys):
        with (
            patch("sys.argv", ["validate_packages.py", "extract-names"]),
            pytest.raises(SystemExit) as exc_info,
        ):
            main()

        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "Usage:" in captured.err

    def test_main_validate_insufficient_args(self, capsys):
        with (
            patch("sys.argv", ["validate_packages.py", "validate", "file.json"]),
            pytest.raises(SystemExit) as exc_info,
        ):
            main()

        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "Usage:" in captured.err

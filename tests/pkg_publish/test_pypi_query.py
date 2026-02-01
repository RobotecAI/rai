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
"""Unit tests for pypi_query.py script."""

from __future__ import annotations

import json
from unittest.mock import patch

from scripts.pypi_query import check_version, get_pypi_versions, main


class TestCheckVersion:
    """Tests for check_version function."""

    @patch("scripts.pypi_query._fetch_pypi_data")
    def test_version_exists(self, mock_fetch):
        mock_fetch.return_value = {
            "releases": {
                "1.0.0": [],
                "2.0.0": [],
                "3.0.0": [],
            }
        }
        assert check_version("test-package", "2.0.0") is True

    @patch("scripts.pypi_query._fetch_pypi_data")
    def test_version_not_exists(self, mock_fetch):
        mock_fetch.return_value = {
            "releases": {
                "1.0.0": [],
                "2.0.0": [],
            }
        }
        assert check_version("test-package", "3.0.0") is False

    @patch("scripts.pypi_query._fetch_pypi_data")
    def test_package_not_found(self, mock_fetch):
        mock_fetch.return_value = None
        assert check_version("nonexistent", "1.0.0") is False

    @patch("scripts.pypi_query._fetch_pypi_data")
    def test_test_pypi_flag(self, mock_fetch):
        mock_fetch.return_value = {
            "releases": {
                "1.0.0": [],
            }
        }
        check_version("test-package", "1.0.0", test_pypi=True)
        mock_fetch.assert_called_once_with("test-package", True)


class TestGetPypiVersions:
    """Tests for get_pypi_versions function."""

    @patch("scripts.pypi_query._fetch_pypi_data")
    def test_get_versions_sorted(self, mock_fetch):
        mock_fetch.return_value = {
            "releases": {
                "1.0.0": [],
                "3.0.0": [],
                "2.0.0": [],
            }
        }
        versions = get_pypi_versions("test-package")
        assert versions == ["3.0.0", "2.0.0", "1.0.0"]

    @patch("scripts.pypi_query._fetch_pypi_data")
    def test_no_versions(self, mock_fetch):
        mock_fetch.return_value = {"releases": {}}
        versions = get_pypi_versions("test-package")
        assert versions == []

    @patch("scripts.pypi_query._fetch_pypi_data")
    def test_package_not_found(self, mock_fetch):
        mock_fetch.return_value = None
        versions = get_pypi_versions("nonexistent")
        assert versions == []

    @patch("scripts.pypi_query._fetch_pypi_data")
    def test_test_pypi_flag(self, mock_fetch):
        mock_fetch.return_value = {
            "releases": {
                "1.0.0": [],
            }
        }
        get_pypi_versions("test-package", test_pypi=True)
        mock_fetch.assert_called_once_with("test-package", True)


class TestMain:
    """Tests for main CLI function."""

    @patch("scripts.pypi_query.check_version")
    def test_check_command_version_exists(self, mock_check, capsys):
        mock_check.return_value = True
        with patch("sys.argv", ["pypi_query.py", "check", "test-package", "1.0.0"]):
            result = main(["check", "test-package", "1.0.0"])
        assert result == 1
        captured = capsys.readouterr()
        assert "exists on PyPI" in captured.out

    @patch("scripts.pypi_query.check_version")
    def test_check_command_version_not_exists(self, mock_check, capsys):
        mock_check.return_value = False
        with patch("sys.argv", ["pypi_query.py", "check", "test-package", "1.0.0"]):
            result = main(["check", "test-package", "1.0.0"])
        assert result == 0
        captured = capsys.readouterr()
        assert "does not exist on PyPI" in captured.out

    @patch("scripts.pypi_query.check_version")
    def test_check_command_test_pypi(self, mock_check, capsys):
        mock_check.return_value = True
        with patch(
            "sys.argv",
            ["pypi_query.py", "--test-pypi", "check", "test-package", "1.0.0"],
        ):
            result = main(["--test-pypi", "check", "test-package", "1.0.0"])
        assert result == 1
        captured = capsys.readouterr()
        assert "exists on Test PyPI" in captured.out
        mock_check.assert_called_once_with("test-package", "1.0.0", test_pypi=True)

    @patch("scripts.pypi_query.get_pypi_versions")
    def test_list_command(self, mock_list, capsys):
        mock_list.return_value = ["3.0.0", "2.0.0", "1.0.0"]
        with patch("sys.argv", ["pypi_query.py", "list", "test-package"]):
            result = main(["list", "test-package"])
        assert result == 0
        captured = capsys.readouterr()
        assert "Versions on PyPI: 3.0.0, 2.0.0, 1.0.0" in captured.out

    @patch("scripts.pypi_query.get_pypi_versions")
    def test_list_command_json(self, mock_list, capsys):
        mock_list.return_value = ["3.0.0", "2.0.0", "1.0.0"]
        with patch("sys.argv", ["pypi_query.py", "list", "test-package", "--json"]):
            result = main(["list", "test-package", "--json"])
        assert result == 0
        captured = capsys.readouterr()
        assert json.loads(captured.out) == ["3.0.0", "2.0.0", "1.0.0"]

    @patch("scripts.pypi_query.get_pypi_versions")
    def test_list_command_no_versions(self, mock_list, capsys):
        mock_list.return_value = []
        with patch("sys.argv", ["pypi_query.py", "list", "test-package"]):
            result = main(["list", "test-package"])
        assert result == 0
        captured = capsys.readouterr()
        assert "No versions found on PyPI" in captured.out

    @patch("scripts.pypi_query.get_pypi_versions")
    def test_list_command_test_pypi(self, mock_list, capsys):
        mock_list.return_value = ["1.0.0"]
        with patch(
            "sys.argv", ["pypi_query.py", "--test-pypi", "list", "test-package"]
        ):
            result = main(["--test-pypi", "list", "test-package"])
        assert result == 0
        captured = capsys.readouterr()
        assert "Versions on Test PyPI: 1.0.0" in captured.out
        mock_list.assert_called_once_with("test-package", test_pypi=True)

    def test_no_command(self, capsys):
        with patch("sys.argv", ["pypi_query.py"]):
            result = main([])
        assert result == 1
        captured = capsys.readouterr()
        assert "Command to run" in captured.out or "usage" in captured.out.lower()

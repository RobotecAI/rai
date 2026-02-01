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
"""Unit tests for list_package_versions.py script."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from scripts.list_package_versions import (
    STATUS_AWAITING_RELEASE,
    STATUS_LOCAL_OLDER,
    STATUS_UP_TO_DATE,
    _get_console_status,
    _get_pypi_versions,
    _get_recommendation,
    list_package_versions,
    main,
    output_console,
    output_markdown,
)


class TestGetPypiVersions:
    """Tests for _get_pypi_versions function."""

    @patch("scripts.list_package_versions.subprocess.run")
    def test_get_versions_success(self, mock_run):
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = json.dumps(["3.0.0", "2.0.0", "1.0.0"])
        mock_run.return_value = mock_result

        versions = _get_pypi_versions("test-package", test_pypi=False)

        assert versions == ["3.0.0", "2.0.0", "1.0.0"]
        cmd = mock_run.call_args[0][0]
        assert "--test-pypi" not in cmd

    @patch("scripts.list_package_versions.subprocess.run")
    def test_get_versions_error_handling(self, mock_run):
        import subprocess

        mock_run.side_effect = subprocess.TimeoutExpired("cmd", 10)

        versions = _get_pypi_versions("test-package")

        assert versions == []


class TestGetRecommendation:
    """Tests for _get_recommendation function."""

    def test_recommendation_version_comparison(self):
        assert (
            _get_recommendation("2.0.0", "1.0.0", "-", "test-pkg")
            == STATUS_AWAITING_RELEASE
        )
        assert (
            _get_recommendation("1.0.0", "1.0.0", "-", "test-pkg") == STATUS_UP_TO_DATE
        )
        assert (
            _get_recommendation("1.0.0", "2.0.0", "-", "test-pkg") == STATUS_LOCAL_OLDER
        )

    def test_recommendation_no_versions(self):
        assert (
            _get_recommendation("1.0.0", "-", "-", "test-pkg")
            == STATUS_AWAITING_RELEASE
        )

    def test_recommendation_invalid_version(self, capsys):
        from packaging.version import InvalidVersion

        with patch("scripts.list_package_versions.pkg_version") as mock_pkg_version:
            mock_pkg_version.parse.side_effect = InvalidVersion("Invalid version")
            mock_pkg_version.InvalidVersion = InvalidVersion

            result = _get_recommendation("invalid", "1.0.0", "-", "test-pkg")

            assert result == "-"
            captured = capsys.readouterr()
            assert "Warning: Version comparison failed for test-pkg" in captured.err


class TestGetConsoleStatus:
    """Tests for _get_console_status function."""

    def test_status_published(self):
        assert (
            _get_console_status("1.0.0", ["1.0.0", "0.9.0"], [])
            == " [PUBLISHED on PyPI]"
        )
        assert (
            _get_console_status("1.0.0", [], ["1.0.0", "0.9.0"])
            == " [PUBLISHED on Test PyPI]"
        )

    def test_status_not_published(self):
        assert _get_console_status("1.0.0", ["0.9.0"], ["0.8.0"]) == " [NOT PUBLISHED]"


class TestListPackageVersions:
    """Tests for list_package_versions function."""

    @patch("scripts.list_package_versions._get_pypi_versions")
    def test_list_package_versions_basic(self, mock_get_versions, tmp_path):
        packages_json = tmp_path / "packages.json"
        packages_json.write_text(
            json.dumps(
                {
                    "pkg1": {"version": "1.0.0", "path": "src/pkg1"},
                    "pkg2": {"version": "2.0.0", "path": "src/pkg2"},
                }
            )
        )

        mock_get_versions.side_effect = [
            ["1.0.0", "0.9.0"],  # pypi for pkg1
            ["0.8.0"],  # testpypi for pkg1
            ["2.0.0"],  # pypi for pkg2
            [],  # testpypi for pkg2
        ]

        results = list_package_versions(packages_json)

        assert len(results) == 2
        assert results[0]["name"] == "pkg1"
        assert results[0]["local_version"] == "1.0.0"
        assert results[0]["pypi_versions"] == ["1.0.0", "0.9.0"]
        assert results[1]["name"] == "pkg2"
        assert results[1]["pypi_versions"] == ["2.0.0"]


class TestOutputFormats:
    """Tests for output functions."""

    @patch("scripts.list_package_versions._get_recommendation")
    def test_output_markdown(self, mock_recommendation, capsys):
        mock_recommendation.return_value = STATUS_UP_TO_DATE
        results = [
            {
                "name": "test-pkg",
                "local_version": "1.0.0",
                "pypi_versions": ["1.0.0"],
                "testpypi_versions": [],
            }
        ]

        output_markdown(results)

        captured = capsys.readouterr()
        assert "## Package Version Status" in captured.out
        assert "test-pkg" in captured.out
        assert "**1.0.0**" in captured.out

    @patch("scripts.list_package_versions._get_console_status")
    def test_output_console(self, mock_status, capsys):
        mock_status.return_value = " [PUBLISHED on PyPI]"
        results = [
            {
                "name": "test-pkg",
                "local_version": "1.0.0",
                "pypi_versions": ["1.0.0"],
                "testpypi_versions": [],
            }
        ]

        output_console(results)

        captured = capsys.readouterr()
        assert "Package Version Status" in captured.out
        assert "test-pkg:" in captured.out


class TestMain:
    """Tests for main function."""

    @patch("scripts.list_package_versions.list_package_versions")
    @patch("scripts.list_package_versions.output_markdown")
    def test_main_markdown_format(self, mock_output, mock_list, tmp_path):
        packages_json = tmp_path / "packages.json"
        packages_json.write_text(json.dumps({"test-pkg": {"version": "1.0.0"}}))
        mock_list.return_value = []

        result = main(
            [
                "--output-format",
                "markdown",
                "--packages-json",
                str(packages_json),
            ]
        )

        assert result == 0
        mock_output.assert_called_once()

    @patch("scripts.list_package_versions.list_package_versions")
    @patch("scripts.list_package_versions.output_console")
    def test_main_console_format(self, mock_output, mock_list, tmp_path):
        packages_json = tmp_path / "packages.json"
        packages_json.write_text(json.dumps({"test-pkg": {"version": "1.0.0"}}))
        mock_list.return_value = []

        result = main(
            [
                "--output-format",
                "console",
                "--packages-json",
                str(packages_json),
            ]
        )

        assert result == 0
        mock_output.assert_called_once()

    def test_main_file_not_found(self, tmp_path, capsys):
        packages_json = tmp_path / "nonexistent.json"

        result = main(
            [
                "--output-format",
                "console",
                "--packages-json",
                str(packages_json),
            ]
        )

        assert result == 1
        captured = capsys.readouterr()
        assert "Error:" in captured.err
        assert "not found" in captured.err

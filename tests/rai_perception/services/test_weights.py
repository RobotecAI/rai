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

"""Tests for weight management helper functions."""

from io import BytesIO
from unittest.mock import MagicMock, patch

import pytest
from rai_perception.services.weights import (
    download_weights,
    load_model_with_error_handling,
    remove_weights,
)

from tests.rai_perception.conftest import create_valid_weights_file


class MockUrlopenResponse(BytesIO):
    """Minimal urlopen-like response for download tests."""

    def __init__(self, data: bytes, content_length: int | None = None):
        super().__init__(data)
        self.headers = {}
        if content_length is not None:
            self.headers["Content-Length"] = str(content_length)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False


class TestDownloadWeights:
    """Test cases for download_weights function."""

    def test_download_weights_success(self, tmp_path):
        """Test successful weight download."""
        weights_path = tmp_path / "weights.pth"
        logger = MagicMock()
        data = b"weights-data"

        with patch(
            "rai_perception.services.weights.urllib.request.urlopen",
            return_value=MockUrlopenResponse(data, content_length=len(data)),
        ):
            download_weights(weights_path, logger, "https://example.com/weights.pth")

            assert weights_path.exists()
            assert weights_path.read_bytes() == data
            logger.info.assert_called()

    def test_download_weights_failure(self, tmp_path):
        """Test weight download failure raises exception."""
        weights_path = tmp_path / "weights.pth"
        logger = MagicMock()

        with patch(
            "rai_perception.services.weights.urllib.request.urlopen",
            side_effect=RuntimeError("Download failed"),
        ):
            with pytest.raises(Exception, match="Could not download weights"):
                download_weights(
                    weights_path, logger, "https://example.com/weights.pth"
                )

    def test_download_weights_empty_download(self, tmp_path):
        """Test download success when server returns an empty file."""
        weights_path = tmp_path / "weights.pth"
        logger = MagicMock()

        with patch(
            "rai_perception.services.weights.urllib.request.urlopen",
            return_value=MockUrlopenResponse(b"", content_length=0),
        ):
            download_weights(weights_path, logger, "https://example.com/weights.pth")
            assert weights_path.exists()
            assert weights_path.read_bytes() == b""


class TestLoadModelWithErrorHandling:
    """Test cases for load_model_with_error_handling function."""

    def test_load_model_success(self, tmp_path):
        """Test successful model loading."""
        weights_path = tmp_path / "weights.pth"
        create_valid_weights_file(weights_path)
        logger = MagicMock()

        class MockModel:
            def __init__(self, weights_path):
                self.weights_path = weights_path

        model = load_model_with_error_handling(
            MockModel, weights_path, logger, "https://example.com/weights.pth"
        )
        assert model.weights_path == weights_path

    def test_load_model_corrupted_weights(self, tmp_path):
        """Test model loading with corrupted weights triggers redownload."""
        weights_path = tmp_path / "weights.pth"
        weights_path.write_bytes(b"corrupted")
        logger = MagicMock()

        call_count = 0

        class MockModel:
            def __init__(self, weights_path):
                nonlocal call_count
                call_count += 1
                self.weights_path = weights_path
                if call_count == 1:
                    raise RuntimeError("PytorchStreamReader failed")

        with patch(
            "rai_perception.services.weights.download_weights",
            side_effect=lambda path, *_args, **_kwargs: create_valid_weights_file(path),
        ):
            model = load_model_with_error_handling(
                MockModel, weights_path, logger, "https://example.com/weights.pth"
            )
            assert model.weights_path == weights_path
            assert call_count == 2

    def test_load_model_other_runtime_error(self, tmp_path):
        """Test that non-corruption RuntimeErrors are re-raised."""
        weights_path = tmp_path / "weights.pth"
        create_valid_weights_file(weights_path)
        logger = MagicMock()

        class MockModel:
            def __init__(self, weights_path):
                raise RuntimeError("Some other error")

        with pytest.raises(RuntimeError, match="Some other error"):
            load_model_with_error_handling(
                MockModel, weights_path, logger, "https://example.com/weights.pth"
            )


class TestRemoveWeights:
    """Test cases for remove_weights function."""

    def test_remove_weights(self, tmp_path):
        """Test remove_weights removes file."""
        weights_path = tmp_path / "weights.pth"
        weights_path.write_bytes(b"test")

        assert weights_path.exists()
        remove_weights(weights_path)
        assert not weights_path.exists()

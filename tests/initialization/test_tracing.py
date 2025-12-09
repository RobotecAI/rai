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

import os
from unittest.mock import patch

from rai.initialization import get_tracing_callbacks


class TestInitializationTracing:
    """Test the initialization module's tracing functionality."""

    def test_tracing_with_missing_config_file(self):
        """Test that tracing gracefully handles missing config.toml file."""
        # Mock load_config to simulate missing config file scenario.
        # Without mocking, get_tracing_callbacks() would load the workspace's config.toml,
        # If tracing is enabled, it'll prevent us from testing how it handles missing config files.
        with patch("rai.initialization.model_initialization.load_config") as mock_load:
            mock_load.side_effect = FileNotFoundError("Config file not found")
            callbacks = get_tracing_callbacks()
            assert len(callbacks) == 0

    def test_tracing_with_config_file_present_tracing_disabled(self, test_config_toml):
        """Test that tracing works when config.toml is present but tracing is disabled."""
        config_path, cleanup = test_config_toml(
            langfuse_enabled=False, langsmith_enabled=False
        )

        try:
            callbacks = get_tracing_callbacks(config_path=config_path)
            # Should return 0 callbacks since both langfuse and langsmith are disabled
            assert len(callbacks) == 0
        finally:
            cleanup()

    def test_tracing_with_config_file_present_tracing_enabled(self, test_config_toml):
        """Test that tracing works when config.toml is present and tracing is enabled."""
        config_path, cleanup = test_config_toml(
            langfuse_enabled=True, langsmith_enabled=False
        )

        try:
            # Mock environment variables to avoid actual API calls
            with patch.dict(
                os.environ,
                {
                    "LANGFUSE_PUBLIC_KEY": "test_key",
                    "LANGFUSE_SECRET_KEY": "test_secret",
                },
            ):
                callbacks = get_tracing_callbacks(config_path=config_path)
                # Should return 1 callback for langfuse
                assert len(callbacks) == 1
        finally:
            cleanup()

    def test_tracing_with_valid_config_file_no_tracing(self, test_config_no_tracing):
        """Test that tracing works when config.toml is valid but has no tracing sections."""
        config_path, cleanup = test_config_no_tracing()

        try:
            # This should not crash, should return empty callbacks
            callbacks = get_tracing_callbacks(config_path=config_path)
            assert len(callbacks) == 0
        finally:
            cleanup()

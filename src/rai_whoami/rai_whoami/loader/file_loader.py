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

import json
import os
from datetime import datetime
from pathlib import Path

from .base import ConfigData, ConfigLoader


class ConfigLoadError(Exception):
    """Raised when there's an error loading the configuration."""

    pass


class ConfigValidationError(Exception):
    """Raised when there's an error validating the configuration."""

    pass


class FileConfigLoader(ConfigLoader):
    """Configuration loader that reads from the filesystem.

    This loader supports JSON configuration files and includes basic validation
    of the configuration structure.
    """

    def __init__(self, config_path: str):
        """Initialize the file configuration loader.

        Args:
            config_path: Path to the configuration file.
        """
        self.config_path = Path(config_path)
        if not self.config_path.exists():
            raise ConfigLoadError(f"Configuration file not found: {config_path}")

    def load(self) -> ConfigData:
        """Load configuration from the filesystem.

        Returns:
            ConfigData: The loaded configuration data.

        Raises:
            ConfigLoadError: If there's an error reading or parsing the file.
        """
        try:
            with open(self.config_path, "r") as f:
                config_data = json.load(f)

            # Get file metadata
            stat = os.stat(self.config_path)
            last_modified = datetime.fromtimestamp(stat.st_mtime).isoformat()

            return ConfigData(
                data=config_data,
                source=str(self.config_path),
                last_modified=last_modified,
            )
        except json.JSONDecodeError as e:
            raise ConfigLoadError(f"Invalid JSON in configuration file: {e}")
        except Exception as e:
            raise ConfigLoadError(f"Error loading configuration: {e}")

    def validate(self, config: ConfigData) -> bool:
        """Validate the configuration data.

        Args:
            config: The configuration data to validate.

        Returns:
            bool: True if the configuration is valid.

        Raises:
            ConfigValidationError: If the configuration is invalid.
        """
        if not isinstance(config.data, dict):
            raise ConfigValidationError("Configuration must be a dictionary")

        # Add specific validation rules here
        required_fields = ["version", "environment"]
        for field in required_fields:
            if field not in config.data:
                raise ConfigValidationError(f"Missing required field: {field}")

        return True

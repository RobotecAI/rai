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

from enum import Enum

from .base import ConfigData, ConfigLoader
from .file_loader import FileConfigLoader
from .mongo_loader import MongoConfigLoader
from .schema import ConfigValidationError, RobotConfig


class ConfigSource(Enum):
    """Enumeration of supported configuration sources.

    Attributes
    ----------
    FILE : str
        Configuration source is a file.
    MONGODB : str
        Configuration source is MongoDB.
    """

    FILE = "file"
    MONGODB = "mongodb"


class ConfigManager:
    """Manages configuration loading from different sources.

    This class uses the Strategy pattern to handle different configuration sources.
    It provides a simple interface for loading and validating configurations.

    Parameters
    ----------
    source : ConfigSource
        The type of configuration source to use.
    **kwargs : dict
        Additional arguments required by the specific loader.

    Raises
    ------
    ValueError
        If the source type is not supported or required arguments are missing.

    Methods
    -------
    load() -> ConfigData
        Load configuration data from the source.
    validate(config: ConfigData) -> bool
        Validate the configuration data.
    get_config() -> RobotConfig
        Load and validate configuration, returning a RobotConfig object.
    """

    def __init__(self, source: ConfigSource, **kwargs):
        """Initialize the configuration manager.

        Parameters
        ----------
        source : ConfigSource
            The type of configuration source to use.
        **kwargs : dict
            Additional arguments required by the specific loader.

        Raises
        ------
        ValueError
            If the source type is not supported or required arguments are missing.
        """
        self._loader = self._create_loader(source, **kwargs)

    def _create_loader(self, source: ConfigSource, **kwargs) -> ConfigLoader:
        """Create the appropriate configuration loader.

        Parameters
        ----------
        source : ConfigSource
            The type of configuration source.
        **kwargs : dict
            Additional arguments required by the specific loader.

        Returns
        -------
        ConfigLoader
            An instance of the appropriate configuration loader.

        Raises
        ------
        ValueError
            If the source type is not supported or required arguments are missing.
        """
        if source == ConfigSource.FILE:
            if "config_path" not in kwargs:
                raise ValueError("config_path is required for file source")
            return FileConfigLoader(kwargs["config_path"])

        elif source == ConfigSource.MONGODB:
            required_args = ["connection_string", "database", "collection", "config_id"]
            missing_args = [arg for arg in required_args if arg not in kwargs]
            if missing_args:
                raise ValueError(
                    f"Missing required arguments for MongoDB source: {missing_args}"
                )

            return MongoConfigLoader(
                connection_string=kwargs["connection_string"],
                database=kwargs["database"],
                collection=kwargs["collection"],
                config_id=kwargs["config_id"],
                username=kwargs.get("username"),
                password=kwargs.get("password"),
            )

        else:
            raise ValueError(f"Unsupported configuration source: {source}")

    def load(self) -> ConfigData:
        """Load configuration data from the source.

        Returns
        -------
        ConfigData
            The loaded configuration data.

        Raises
        ------
        ConfigLoadError
            If there's an error loading the configuration.
        """
        return self._loader.load()

    def validate(self, config: ConfigData) -> bool:
        """Validate the configuration data.

        Parameters
        ----------
        config : ConfigData
            The configuration data to validate.

        Returns
        -------
        bool
            True if the configuration is valid.

        Raises
        ------
        ConfigValidationError
            If the configuration is invalid.
        """
        return self._loader.validate(config)

    def get_config(self) -> RobotConfig:
        """Load and validate configuration, returning a RobotConfig object.

        Returns
        -------
        RobotConfig
            The validated configuration as a RobotConfig object.

        Raises
        ------
        ConfigLoadError
            If there's an error loading the configuration.
        ConfigValidationError
            If the configuration is invalid.
        """
        config_data = self.load()
        if self.validate(config_data):
            return RobotConfig(**config_data.data)
        raise ConfigValidationError("Configuration validation failed")

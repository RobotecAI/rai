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

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class ConfigData:
    """Data class to hold configuration data with metadata.

    Parameters
    ----------
    data : Dict[str, Any]
        The actual configuration data as a dictionary.
    source : str
        The source of the configuration (e.g., file path, MongoDB URI).
    version : Optional[str], default=None
        Version of the configuration.
    last_modified : Optional[str], default=None
        Timestamp of when the configuration was last modified.
    """

    data: Dict[str, Any]
    source: str
    version: Optional[str] = None
    last_modified: Optional[str] = None


class ConfigLoader(ABC):
    """Abstract base class for configuration loaders.

    This class defines the interface that all configuration loaders must implement.
    Each loader is responsible for loading configuration data from a specific source
    (e.g., filesystem, MongoDB, etc.).

    Methods
    -------
    load() -> ConfigData
        Load configuration data from the source.
    validate(config: ConfigData) -> bool
        Validate the loaded configuration data.
    """

    @abstractmethod
    def load(self) -> ConfigData:
        """Load configuration data from the source.

        Returns
        -------
        ConfigData
            A dataclass containing the loaded configuration data and metadata.

        Raises
        ------
        ConfigLoadError
            If there's an error loading the configuration.
        """
        pass

    @abstractmethod
    def validate(self, config: ConfigData) -> bool:
        """Validate the loaded configuration data.

        Parameters
        ----------
        config : ConfigData
            The configuration data to validate.

        Returns
        -------
        bool
            True if the configuration is valid, False otherwise.

        Raises
        ------
        ConfigValidationError
            If there's an error validating the configuration.
        """
        pass

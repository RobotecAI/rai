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

from pathlib import Path
from typing import Tuple

from .base import ConfigData, ConfigLoader
from .file_loader import FileConfigLoader
from .mongo_loader import MongoConfigLoader
from .schema import RobotConfig, RobotConstitution, RobotIdentity, VectorDBConfig

__all__ = [
    "ConfigData",
    "ConfigLoader",
    "FileConfigLoader",
    "MongoConfigLoader",
    "RobotConfig",
    "RobotConstitution",
    "RobotIdentity",
    "VectorDBConfig",
    "load_configs_from_dir",
]


def load_configs_from_dir(
    config_dir: str | Path,
) -> Tuple[RobotIdentity, RobotConstitution]:
    """Load both identity and constitution configurations from a directory.

    Parameters
    ----------
    config_dir : str | Path
        Path to the directory containing configuration files.

    Returns
    -------
    Tuple[RobotIdentity, RobotConstitution]
        A tuple containing the loaded identity and constitution configurations.

    Raises
    ------
    FileNotFoundError
        If the configuration files are not found.
    """
    config_dir = Path(config_dir)

    # Load identity configuration
    identity_loader = FileConfigLoader(config_dir / "identity.json")
    identity_data = identity_loader.load()
    identity = RobotIdentity(**identity_data.data)

    # Load constitution configuration
    constitution_loader = FileConfigLoader(config_dir / "constitution.json")
    constitution_data = constitution_loader.load()
    constitution = RobotConstitution(**constitution_data.data)

    return identity, constitution

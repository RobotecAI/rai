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

"""RAI Whoami package."""

from .docs_processors.config_generator import ConfigGenerator
from .docs_processors.generate_configs import generate_configs_from_docs
from .loader.file_loader import FileConfigLoader
from .loader.mongo_loader import MongoConfigLoader
from .loader.schema import RobotConfig, RobotConstitution, RobotIdentity

__all__ = [
    "ConfigGenerator",
    "FileConfigLoader",
    "MongoConfigLoader",
    "RobotConfig",
    "RobotConstitution",
    "RobotIdentity",
    "generate_configs_from_docs",
]

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
from typing import Any, Set, Type

import yaml

from rai_sim.engine_connector import SimulationConfig, SimulationConfigT
from rai_sim.o3de.o3de_connector import O3DExROS2SimulationConfig

CONFIG_REGISTRY: Set[Type[SimulationConfig]] = {O3DExROS2SimulationConfig}


def load_base_config(path: Path) -> SimulationConfig:
    """Load the base configuration file"""
    with open(path) as f:
        content = yaml.safe_load(f)
    return SimulationConfig(**content)


def load_simulation_config(
    base_config_path: Path,
    connector_config_path: Path,
    config_type: Type[SimulationConfigT],
) -> SimulationConfigT:
    """Combine base config with connector-specific config"""

    if config_type not in CONFIG_REGISTRY:
        raise ValueError(
            f"Invalid config type: {config_type}. Must be one of {CONFIG_REGISTRY}"
        )

    base_config = load_base_config(base_config_path)

    with open(connector_config_path) as f:
        connector_content: dict[str, Any] = yaml.safe_load(f)

    return config_type(**base_config.model_dump(), **connector_content)

from pathlib import Path
from typing import Any, Set, Type

import yaml

from rai_sim.engine_connector import SimulationConfig, SimulationConfigT
from rai_sim.o3de.o3de_connector import O3DESimulationConfig

CONFIG_REGISTRY: Set[Type[SimulationConfig]] = {O3DESimulationConfig}


def load_base_config(path: Path) -> SimulationConfig:
    """Load the base configuration file"""
    with open(path) as f:
        content = yaml.safe_load(f)
    print(content)
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

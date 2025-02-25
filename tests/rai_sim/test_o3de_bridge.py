from pathlib import Path

from rai_sim.o3de.o3de_bridge import O3DExROS2SimulationConfig
from rai_sim.simulation_bridge import (
    Entity,
)


def test_load_config(sample_base_yaml_config: Path, sample_o3dexros2_config: Path):
    config = O3DExROS2SimulationConfig.load_config(
        sample_base_yaml_config, sample_o3dexros2_config
    )
    assert isinstance(config, O3DExROS2SimulationConfig)
    assert config.binary_path == Path("/path/to/binary")
    assert config.robotic_stack_command == "ros2 launch robotic_stack.launch.py"
    assert isinstance(config.entities, list)
    assert all(isinstance(e, Entity) for e in config.entities)

    assert len(config.entities) == 2

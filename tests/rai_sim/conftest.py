from pathlib import Path

import pytest


@pytest.fixture
def sample_base_yaml_config(tmp_path: Path) -> Path:
    yaml_content = """
    entities:
      - name: entity1
        prefab_name: cube
        pose:
          translation:
            x: 1.0
            y: 2.0
            z: 3.0

      - name: entity2
        prefab_name: carrot
        pose:
          translation:
            x: 1.0
            y: 2.0
            z: 3.0
          rotation:
            x: 0.1
            y: 0.2
            z: 0.3
            w: 0.4
    """
    file_path = tmp_path / "test_config.yaml"
    file_path.write_text(yaml_content)
    return file_path


@pytest.fixture
def sample_o3dexros2_config(tmp_path: Path) -> Path:
    yaml_content = """
    binary_path: /path/to/binary
    robotic_stack_command: "ros2 launch robotic_stack.launch.py"
    """
    file_path = tmp_path / "test_o3dexros2_config.yaml"
    file_path.write_text(yaml_content)
    return file_path

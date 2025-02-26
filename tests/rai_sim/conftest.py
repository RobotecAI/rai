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

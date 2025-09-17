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
import pytest
from rai.types import Header, Point, Pose, PoseStamped, Quaternion

from rai_bench.tool_calling_agent.interfaces import TaskArgs
from rai_sim.simulation_bridge import Entity


def create_entity(
    name: str,
    prefab: str,
    x: float,
    y: float,
    z: float,
    orientation: Quaternion | None = None,
) -> Entity:
    if orientation is None:
        orientation = Quaternion()
    return Entity(
        name=name,
        prefab_name=prefab,
        pose=PoseStamped(
            pose=Pose(position=Point(x=x, y=y, z=z), orientation=orientation),
            header=Header(frame_id="/test_frame"),
        ),
    )


@pytest.fixture
def task_args() -> TaskArgs:
    """Create basic task arguments for testing."""
    return TaskArgs(
        extra_tool_calls=0,
        prompt_detail="brief",
        examples_in_system_prompt=0,
    )

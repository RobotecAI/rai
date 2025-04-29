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

from typing import Type

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field
from rai.types import Pose

from rai_sim.simulation_bridge import SimulationBridge


class GetObjectPositionsGroundTruthToolInput(BaseModel):
    object_name: str = Field(
        ..., description="The name of the object to get the positions of"
    )


class GetObjectPositionsGroundTruthTool(BaseTool):
    name: str = "get_object_positions"
    description: str = (
        "Retrieve the positions of all objects of a specified type in the target frame. "
        "This tool provides accurate positional data but does not distinguish between different colors of the same object type. "
        "While position detection is reliable, please note that object classification may occasionally be inaccurate."
    )

    simulation: SimulationBridge

    args_schema: Type[GetObjectPositionsGroundTruthToolInput] = (
        GetObjectPositionsGroundTruthToolInput
    )

    @staticmethod
    def format_pose(pose: Pose) -> str:
        return f"Centroid(x={pose.position.x:.2f}, y={pose.position.y:.2f}, z={pose.position.z:.2f})"

    @staticmethod
    def match_name(object_name: str, prefab_name: str) -> bool:
        return all([word in prefab_name for word in object_name.split()])

    def _run(self, object_name: str):
        poses = []
        for entity in self.simulation.get_scene_state().entities:
            if self.match_name(object_name, entity.prefab_name):
                poses.append(entity.pose)

        if len(poses) == 0:
            return f"No {object_name}s detected."
        else:
            return f"Centroids of detected {object_name}s in manipulator frame: [{', '.join(map(self.format_pose, poses))}]. Sizes of the detected objects are unknown."

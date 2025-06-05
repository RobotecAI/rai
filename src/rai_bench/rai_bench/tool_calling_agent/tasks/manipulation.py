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
from typing import Any, Dict, List

import inflect
from langchain_core.tools import BaseTool
from rai.tools.ros2 import MoveToPointToolInput
from rai.types import Point

from rai_bench.tool_calling_agent.interfaces import Task, TaskArgs, Validator
from rai_bench.tool_calling_agent.mocked_ros2_interfaces import (
    COMMON_INTERFACES,
    COMMON_SERVICES_AND_TYPES,
    COMMON_TOPICS_AND_TYPES,
    MANIPULATION_ACTIONS_AND_TYPES,
    MANIPULATION_INTERFACES,
    MANIPULATION_SERVICES_AND_TYPES,
    MANIPULATION_TOPICS_AND_TYPES,
)
from rai_bench.tool_calling_agent.mocked_tools import (
    MockGetObjectPositionsTool,
    MockGetROS2MessageInterfaceTool,
    MockGetROS2ServicesNamesAndTypesTool,
    MockGetROS2TopicsNamesAndTypesTool,
    MockMoveToPointTool,
)

INTERFACES = COMMON_INTERFACES | MANIPULATION_INTERFACES
TOPCIS_AND_TYPES = COMMON_TOPICS_AND_TYPES | MANIPULATION_TOPICS_AND_TYPES
SERVICES_AND_TYPES = COMMON_SERVICES_AND_TYPES | MANIPULATION_SERVICES_AND_TYPES

TOPIC_STRINGS = [
    f"topic: {topic}\ntype: {topic_type}\n"
    for topic, topic_type in COMMON_TOPICS_AND_TYPES.items()
]

ACTION_STRINGS = [
    f"action: {action}\ntype: {act_type}\n"
    for action, act_type in MANIPULATION_ACTIONS_AND_TYPES.items()
]

SERVICE_STRINGS = [
    f"service: {service}\ntype: {srv_type}\n"
    for service, srv_type in SERVICES_AND_TYPES.items()
]

PROACTIVE_ROS2_EXPERT_SYSTEM_PROMPT_0_SHOT = """
        You are a robotic arm with interfaces to detect and manipulate objects.
        Here are the coordinates information:
        x - front to back (positive is forward)
        y - left to right (positive is right)
        z - up to down (positive is up).
        """

PROACTIVE_ROS2_EXPERT_SYSTEM_PROMPT_2_SHOT = (
    PROACTIVE_ROS2_EXPERT_SYSTEM_PROMPT_0_SHOT
    + """
Example of tool calls:
- get_object_positions, args: {}
- move_to_point, args: {'x': 0.5, 'y': 0.2, 'z': 0.3, 'task': 'grab'}"""
)

PROACTIVE_ROS2_EXPERT_SYSTEM_PROMPT_5_SHOT = (
    PROACTIVE_ROS2_EXPERT_SYSTEM_PROMPT_2_SHOT
    + """
- move_to_point, args: {'x': 1.7, 'y': 1.8, 'z': 1.9, 'task': 'drop'}
- move_to_point, args: {'x': 0.1, 'y': -0.2, 'z': 0.1, 'task': 'grab'}
- move_to_point, args: {'x': 0.7, 'y': 0.8, 'z': 0.9, 'task': 'drop'}
"""
)


class TaskParametrizationError(Exception):
    """Exception raised when the task parameters are not valid."""

    pass


class ManipulationTask(Task, ABC):
    type = "manipulation"

    def __init__(
        self,
        objects: Dict[str, List[Point]],
        validators: List[Validator],
        task_args: TaskArgs,
        **kwargs: Any,
    ) -> None:
        super().__init__(validators=validators, task_args=task_args, **kwargs)
        self.objects = objects
        self._verify_args()

    @property
    def optional_tool_calls_number(self) -> int:
        return 0

    @property
    def available_tools(self) -> List[BaseTool]:
        return [
            MockGetROS2TopicsNamesAndTypesTool(
                mock_topics_names_and_types=TOPIC_STRINGS
            ),
            MockGetObjectPositionsTool(
                target_frame="panda_link0",
                source_frame="RGBDCamera5",
                camera_topic="/color_image5",
                depth_topic="/depth_image5",
                camera_info_topic="/color_camera_info5",
                mock_objects=self.objects,
            ),
            MockMoveToPointTool(manipulator_frame="panda_link0"),
            MockGetROS2ServicesNamesAndTypesTool(
                mock_service_names_and_types=SERVICE_STRINGS
            ),
            MockGetROS2MessageInterfaceTool(mock_interfaces=INTERFACES),
        ]

    def _verify_args(self) -> None:
        pass

    def get_system_prompt(self) -> str:
        if self.n_shots == 0:
            return PROACTIVE_ROS2_EXPERT_SYSTEM_PROMPT_0_SHOT
        elif self.n_shots == 2:
            return PROACTIVE_ROS2_EXPERT_SYSTEM_PROMPT_2_SHOT
        else:
            return PROACTIVE_ROS2_EXPERT_SYSTEM_PROMPT_5_SHOT


class GrabTask(ManipulationTask, ABC):
    def __init__(
        self,
        objects: Dict[str, List[Point]],
        object_to_grab: str,
        validators: List[Validator],
        task_args: TaskArgs,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            validators=validators, objects=objects, task_args=task_args, **kwargs
        )
        self.object_to_grab = object_to_grab
        self._verify_args()

    @abstractmethod
    def _verify_args(self) -> None:
        pass


class MoveToPointTask(ManipulationTask):
    complexity = "easy"

    def __init__(
        self,
        objects: Dict[str, List[Point]],
        move_to_tool_input: MoveToPointToolInput,
        validators: List[Validator],
        task_args: TaskArgs,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            validators=validators, objects=objects, task_args=task_args, **kwargs
        )
        self.move_to_tool_input = move_to_tool_input

    def get_prompt(self) -> str:
        base_prompt = (
            f"Move the arm to point x={self.move_to_tool_input.x}, "
            f"y={self.move_to_tool_input.y}, z={self.move_to_tool_input.z} "
            f"to {self.move_to_tool_input.task} an object"
        )

        if self.prompt_detail == "brief":
            return base_prompt
        elif self.prompt_detail == "moderate":
            return f"{base_prompt} using robotic arm control"
        else:
            return (
                f"{base_prompt} using the robotic manipulation system. "
                "You can control the arm movement to the specified coordinates "
                f"and perform the {self.move_to_tool_input.task} action at that location."
            )


class GetObjectPositionsTask(ManipulationTask):
    complexity = "easy"

    def __init__(
        self,
        objects: Dict[str, List[Point]],
        validators: List[Validator],
        task_args: TaskArgs,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            validators=validators, objects=objects, task_args=task_args, **kwargs
        )
        self.objects = objects

    def get_prompt(self) -> str:
        inflector = inflect.engine()
        object_counts = {obj: len(positions) for obj, positions in self.objects.items()}
        formatted_objects = [
            inflector.plural(obj) if count > 1 else obj
            for obj, count in object_counts.items()
        ]
        if len(formatted_objects) > 1:
            objects_list = (
                ", ".join(formatted_objects[:-1]) + f", and {formatted_objects[-1]}"
            )
        else:
            objects_list = formatted_objects[0]

        base_prompt = f"Get the {objects_list} positions"

        if self.prompt_detail == "brief":
            return base_prompt
        elif self.prompt_detail == "moderate":
            return f"{base_prompt} in the workspace"
        else:
            return (
                f"{base_prompt} in the robotic workspace environment. "
                "You can detect all objects and retrieve their 3D coordinates "
                "for manipulation planning."
            )


class GrabExistingObjectTask(GrabTask):
    complexity = "medium"

    def get_prompt(self) -> str:
        base_prompt = f"Grab {self.object_to_grab}"

        if self.prompt_detail == "brief":
            return base_prompt
        elif self.prompt_detail == "moderate":
            return f"{base_prompt} from the workspace"
        else:
            return (
                f"{base_prompt} using robotic manipulation. "
                "You can locate the object in the workspace and move the arm "
                "to grab it at the correct coordinates."
            )

    def _verify_args(self):
        if self.object_to_grab not in self.objects:
            error_message = f"Requested object to grab {self.object_to_grab} is not present in defined objects: {self.objects}."
            self.logger.error(msg=error_message)
            raise TaskParametrizationError(error_message)

        if len(self.objects[self.object_to_grab]) > 1:
            error_message = f"Requested object to grab {self.object_to_grab} has more than one position in defined objects: {self.objects[self.object_to_grab]}."
            self.logger.error(msg=error_message)
            raise TaskParametrizationError(error_message)


class GrabNotExistingObjectTask(GrabTask):
    complexity = "medium"

    def get_prompt(self) -> str:
        base_prompt = f"Grab {self.object_to_grab}"

        if self.prompt_detail == "brief":
            return base_prompt
        elif self.prompt_detail == "moderate":
            return f"{base_prompt} if it exists in the workspace"
        else:
            return (
                f"{base_prompt} if available in the robotic workspace. "
                "You can check if the object exists in the environment and "
                "attempt to grab it if found."
            )

    def _verify_args(self):
        if self.object_to_grab in self.objects:
            error_message = f"Requested object to grab {self.object_to_grab} is present in defined objects: {self.objects} but should not be."
            self.logger.error(msg=error_message)
            raise TaskParametrizationError(error_message)


class MoveExistingObjectLeftTask(GrabTask):
    complexity = "hard"

    def get_prompt(self) -> str:
        base_prompt = f"Move {self.object_to_grab} 20 cm to the left"

        if self.prompt_detail == "brief":
            return base_prompt
        elif self.prompt_detail == "moderate":
            return f"{base_prompt} using robotic manipulation"
        else:
            return (
                f"{base_prompt} using the robotic arm system. "
                "You can locate the object, grab it with the manipulator, "
                "and move it to a position 20 cm to the left of its current location."
            )

    def _verify_args(self):
        if self.object_to_grab not in self.objects:
            error_message = f"Requested object to grab {self.object_to_grab} is not present in defined objects: {self.objects}."
            self.logger.error(msg=error_message)
            raise TaskParametrizationError(error_message)

        if len(self.objects[self.object_to_grab]) > 1:
            error_message = f"Requested object to grab {self.object_to_grab} has more than one position in defined objects: {self.objects[self.object_to_grab]}."
            self.logger.error(msg=error_message)
            raise TaskParametrizationError(error_message)


class MoveExistingObjectFrontTask(GrabTask):
    complexity = "hard"

    def get_prompt(self) -> str:
        base_prompt = f"Move {self.object_to_grab} 60 cm to the front"

        if self.prompt_detail == "brief":
            return base_prompt
        elif self.prompt_detail == "moderate":
            return f"{base_prompt} using robotic manipulation"
        else:
            return (
                f"{base_prompt} using the robotic arm system. "
                "You can locate the object, grab it with the manipulator, "
                "and move it to a position 60 cm forward from its current location."
            )

    def _verify_args(self):
        if self.object_to_grab not in self.objects:
            error_message = f"Requested object to grab {self.object_to_grab} is not present in defined objects: {self.objects}."
            self.logger.error(msg=error_message)
            raise TaskParametrizationError(error_message)

        if len(self.objects[self.object_to_grab]) > 1:
            error_message = f"Requested object to grab {self.object_to_grab} has more than one position in defined objects: {self.objects[self.object_to_grab]}."
            self.logger.error(msg=error_message)
            raise TaskParametrizationError(error_message)


class SwapObjectsTask(ManipulationTask):
    complexity = "hard"

    def __init__(
        self,
        objects: Dict[str, List[Point]],
        objects_to_swap: List[str],
        validators: List[Validator],
        task_args: TaskArgs,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            validators=validators, objects=objects, task_args=task_args, **kwargs
        )
        self.objects = objects
        self.objects_to_swap = objects_to_swap
        self._verify_args()

    def _verify_args(self):
        for obj in self.objects_to_swap:
            if obj not in self.objects:
                error_message = f"Requested object to swap {obj} is not present in defined objects: {self.objects}."
                self.logger.error(msg=error_message)
                raise TaskParametrizationError(error_message)
            if len(self.objects[obj]) != 1:
                error_message = f"Number of positions for object to swap ({obj}) should be equal to 1."
                self.logger.error(msg=error_message)
                raise TaskParametrizationError(error_message)
        if len(self.objects_to_swap) != 2:
            error_message = f"Number of requested objects to swap {len(self.objects_to_swap)} should be equal to 2."
            self.logger.error(msg=error_message)
            raise TaskParametrizationError(error_message)

    def get_prompt(self) -> str:
        base_prompt = f"Swap {self.objects_to_swap[0]} and {self.objects_to_swap[1]}"

        if self.prompt_detail == "brief":
            return base_prompt
        elif self.prompt_detail == "moderate":
            return f"{base_prompt} positions using robotic manipulation"
        else:
            return (
                f"{base_prompt} positions using the robotic manipulation system. "
                "You can locate both objects in the workspace, then perform a sequence "
                f"of grab and move operations to swap the positions of {self.objects_to_swap[0]} "
                f"and {self.objects_to_swap[1]}."
            )

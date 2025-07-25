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
import logging
from abc import ABC
from typing import Any, Dict, List, Optional

import inflect
from langchain_core.tools import BaseTool
from rai.tools.ros2 import MoveToPointToolInput
from rai.types import Point

from rai_bench.tool_calling_agent.interfaces import SubTask, Task, TaskArgs, Validator
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
from rai_bench.tool_calling_agent.subtasks import (
    CheckArgsToolCallSubTask,
)
from rai_bench.tool_calling_agent.validators import (
    NotOrderedCallsValidator,
    OneFromManyValidator,
    OrderedCallsValidator,
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

        1 unit in system is equal to 1 meter in real environment.
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
- get_ros2_topics_names_and_types, args: {}
- get_ros2_message_interface, args: {'msg_type': 'moveit_msgs/srv/ExecuteKnownTrajectory'}
- move_to_point, args: {'x': 0.7, 'y': 0.8, 'z': 0.9, 'task': 'drop'}
"""
)

LEFT_DISTANCE = 0.2  # 20cm
FRONT_DISTANCE = 0.6  # 60cm


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
        logger: Optional[logging.Logger] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            validators=validators, task_args=task_args, logger=logger, **kwargs
        )
        self.objects = objects

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
        logger: Optional[logging.Logger] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            validators=validators,
            objects=objects,
            task_args=task_args,
            logger=logger,
            **kwargs,
        )
        self.object_to_grab = object_to_grab
        self._verify_args()

    def _verify_args(self):
        if self.object_to_grab not in self.objects:
            error_message = f"Requested object to grab {self.object_to_grab} is not present in defined objects: {self.objects}."
            self.logger.error(msg=error_message)
            raise TaskParametrizationError(error_message)

        if len(self.objects[self.object_to_grab]) > 1:
            error_message = f"Requested object to grab {self.object_to_grab} has more than one position in defined objects: {self.objects[self.object_to_grab]}."
            self.logger.error(msg=error_message)
            raise TaskParametrizationError(error_message)


class MoveToPointTask(ManipulationTask):
    complexity = "easy"

    def __init__(
        self,
        objects: Dict[str, List[Point]],
        move_to_tool_input: MoveToPointToolInput,
        task_args: TaskArgs,
        validators: Optional[List[Validator]] = None,
        logger: Optional[logging.Logger] = None,
        **kwargs: Any,
    ) -> None:
        self.move_to_tool_input = move_to_tool_input

        if validators is None:
            move_to_point_subtask = CheckArgsToolCallSubTask(
                expected_tool_name="move_to_point",
                expected_args={
                    "x": move_to_tool_input.x,
                    "y": move_to_tool_input.y,
                    "z": move_to_tool_input.z,
                    "task": move_to_tool_input.task,
                },
            )
            validators = [OrderedCallsValidator(subtasks=[move_to_point_subtask])]

        super().__init__(
            validators=validators,
            objects=objects,
            task_args=task_args,
            logger=logger,
            **kwargs,
        )

    def get_base_prompt(self) -> str:
        return (
            f"Move the arm to point x={self.move_to_tool_input.x}, "
            f"y={self.move_to_tool_input.y}, z={self.move_to_tool_input.z} "
            f"to {self.move_to_tool_input.task} an object."
        )

    def get_prompt(self) -> str:
        if self.prompt_detail == "brief":
            return self.get_base_prompt()
        else:
            return (
                f"{self.get_base_prompt()} "
                "You can control the arm movement to the specified coordinates "
                f"and perform the {self.move_to_tool_input.task} action at that location."
            )


class GetObjectPositionsTask(ManipulationTask):
    complexity = "easy"

    def __init__(
        self,
        objects: Dict[str, List[Point]],
        task_args: TaskArgs,
        validators: Optional[List[Validator]] = None,
        logger: Optional[logging.Logger] = None,
        **kwargs: Any,
    ) -> None:
        if validators is None:
            subtasks: List[SubTask] = []
            for obj_name in objects.keys():
                subtask = CheckArgsToolCallSubTask(
                    expected_tool_name="get_object_positions",
                    expected_args={"object_name": obj_name},
                )
                subtasks.append(subtask)

            validators = [NotOrderedCallsValidator(subtasks=subtasks)]

        super().__init__(
            validators=validators,
            objects=objects,
            task_args=task_args,
            logger=logger,
            **kwargs,
        )

    def get_base_prompt(self) -> str:
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

        return (
            f"Get the {objects_list} positions. Object name should be in singular form."
        )

    def get_prompt(self) -> str:
        if self.prompt_detail == "brief":
            return self.get_base_prompt()
        else:
            return (
                f"{self.get_base_prompt()} in the robotic workspace environment. "
                "You can detect all objects and retrieve their 3D coordinates."
            )


class GrabExistingObjectTask(GrabTask):
    complexity = "medium"

    def __init__(
        self,
        objects: Dict[str, List[Point]],
        object_to_grab: str,
        task_args: TaskArgs,
        validators: Optional[List[Validator]] = None,
        logger: Optional[logging.Logger] = None,
        **kwargs: Any,
    ) -> None:
        if validators is None:
            get_object_positions_subtask = CheckArgsToolCallSubTask(
                expected_tool_name="get_object_positions",
                expected_args={"object_name": object_to_grab},
            )
            grab_move_subtask = CheckArgsToolCallSubTask(
                expected_tool_name="move_to_point",
                expected_args={
                    "x": objects[object_to_grab][0].x,
                    "y": objects[object_to_grab][0].y,
                    "z": objects[object_to_grab][0].z,
                    "task": "grab",
                },
            )
            validators = [
                OrderedCallsValidator(
                    subtasks=[get_object_positions_subtask, grab_move_subtask]
                )
            ]

        super().__init__(
            objects=objects,
            object_to_grab=object_to_grab,
            validators=validators,
            task_args=task_args,
            logger=logger,
            **kwargs,
        )

    def get_base_prompt(self) -> str:
        return f"Grab {self.object_to_grab}."

    def get_prompt(self) -> str:
        if self.prompt_detail == "brief":
            return self.get_base_prompt()
        else:
            return (
                f"{self.get_base_prompt()} "
                "You can locate the object in the workspace and move the arm "
                "to grab it at the correct coordinates."
            )


class MoveExistingObjectLeftTask(GrabTask):
    complexity = "medium"

    def __init__(
        self,
        objects: Dict[str, List[Point]],
        object_to_grab: str,
        task_args: TaskArgs,
        validators: Optional[List[Validator]] = None,
        logger: Optional[logging.Logger] = None,
        **kwargs: Any,
    ) -> None:
        if validators is None:
            get_object_positions_subtask = CheckArgsToolCallSubTask(
                expected_tool_name="get_object_positions",
                expected_args={"object_name": object_to_grab},
            )
            grab_subtask = CheckArgsToolCallSubTask(
                expected_tool_name="move_to_point",
                expected_args={
                    "x": objects[object_to_grab][0].x,
                    "y": objects[object_to_grab][0].y,
                    "z": objects[object_to_grab][0].z,
                    "task": "grab",
                },
            )
            drop_subtask = CheckArgsToolCallSubTask(
                expected_tool_name="move_to_point",
                expected_args={
                    "x": objects[object_to_grab][0].x,
                    "y": round(objects[object_to_grab][0].y - LEFT_DISTANCE, 2),
                    "z": objects[object_to_grab][0].z,
                    "task": "drop",
                },
            )
            validators = [
                OrderedCallsValidator(
                    subtasks=[get_object_positions_subtask, grab_subtask, drop_subtask]
                )
            ]

        super().__init__(
            objects=objects,
            object_to_grab=object_to_grab,
            validators=validators,
            task_args=task_args,
            logger=logger,
            **kwargs,
        )

    def get_base_prompt(self) -> str:
        return f"Move {self.object_to_grab} 20 cm to the left."

    def get_prompt(self) -> str:
        if self.prompt_detail == "brief":
            return self.get_base_prompt()
        else:
            return (
                f"{self.get_base_prompt()} "
                "You can locate the object, grab it with the manipulator, "
                "and move it to a position 20 cm to the left of its current location."
            )


class MoveExistingObjectFrontTask(GrabTask):
    complexity = "medium"

    def __init__(
        self,
        objects: Dict[str, List[Point]],
        object_to_grab: str,
        task_args: TaskArgs,
        validators: Optional[List[Validator]] = None,
        logger: Optional[logging.Logger] = None,
        **kwargs: Any,
    ) -> None:
        if validators is None:
            get_object_positions_subtask = CheckArgsToolCallSubTask(
                expected_tool_name="get_object_positions",
                expected_args={"object_name": object_to_grab},
            )
            grab_subtask = CheckArgsToolCallSubTask(
                expected_tool_name="move_to_point",
                expected_args={
                    "x": objects[object_to_grab][0].x,
                    "y": objects[object_to_grab][0].y,
                    "z": objects[object_to_grab][0].z,
                    "task": "grab",
                },
            )
            drop_subtask = CheckArgsToolCallSubTask(
                expected_tool_name="move_to_point",
                expected_args={
                    "x": round(objects[object_to_grab][0].x + FRONT_DISTANCE, 2),
                    "y": objects[object_to_grab][0].y,
                    "z": objects[object_to_grab][0].z,
                    "task": "drop",
                },
            )
            validators = [
                OrderedCallsValidator(
                    subtasks=[get_object_positions_subtask, grab_subtask, drop_subtask]
                )
            ]

        super().__init__(
            objects=objects,
            object_to_grab=object_to_grab,
            validators=validators,
            task_args=task_args,
            logger=logger,
            **kwargs,
        )

    def get_base_prompt(self) -> str:
        return f"Move {self.object_to_grab} 60 cm to the front."

    def get_prompt(self) -> str:
        if self.prompt_detail == "brief":
            return self.get_base_prompt()
        else:
            return (
                f"{self.get_base_prompt()} "
                "You can locate the object, grab it with the manipulator, "
                "and move it to a position 60 cm forward from its current location."
            )


class AlignTwoObjectsTask(ManipulationTask):
    complexity = "hard"

    def __init__(
        self,
        objects: Dict[str, List[Point]],
        task_args: TaskArgs,
        validators: Optional[List[Validator]] = None,
        logger: Optional[logging.Logger] = None,
        **kwargs: Any,
    ) -> None:
        if validators is None:
            # Get the two objects from the objects dict (first and second object)
            object_names = list(objects.keys())
            obj1_name, obj2_name = object_names[0], object_names[1]
            obj1_pos, obj2_pos = objects[obj1_name][0], objects[obj2_name][0]

            get_object_positions_subtask = CheckArgsToolCallSubTask(
                expected_tool_name="get_object_positions",
                expected_args={},
            )

            # Two possible positions: 0.5 units to the right or left of obj2
            target_x_pos1 = round(obj2_pos.x + 0.5, 2)
            target_x_pos2 = round(obj2_pos.x - 0.5, 2)

            grab_subtask = CheckArgsToolCallSubTask(
                expected_tool_name="move_to_point",
                expected_args={
                    "x": obj1_pos.x,
                    "y": obj1_pos.y,
                    "z": obj1_pos.z,
                    "task": "grab",
                },
            )

            # Create subtasks for dropping the first object at either valid position
            drop_pos1 = CheckArgsToolCallSubTask(
                expected_tool_name="move_to_point",
                expected_args={
                    "x": target_x_pos1,
                    "y": obj1_pos.y,
                    "z": obj1_pos.z,
                    "task": "drop",
                },
            )
            drop_pos2 = CheckArgsToolCallSubTask(
                expected_tool_name="move_to_point",
                expected_args={
                    "x": target_x_pos2,
                    "y": obj1_pos.y,
                    "z": obj1_pos.z,
                    "task": "drop",
                },
            )

            val1 = OrderedCallsValidator(
                subtasks=[
                    get_object_positions_subtask,
                    grab_subtask,
                ]
            )
            val2 = OneFromManyValidator(subtasks=[drop_pos1, drop_pos2])
            validators = [val1, val2]
        super().__init__(
            validators=validators,
            objects=objects,
            task_args=task_args,
            logger=logger,
            **kwargs,
        )

    def get_base_prompt(self) -> str:
        object_names = list(self.objects.keys())
        return f"Move the first object ({object_names[0]}) so it is 50 cm apart from the second object ({object_names[1]}) along the x-axis."

    def get_prompt(self) -> str:
        if self.prompt_detail == "brief":
            return self.get_base_prompt()
        else:
            object_names = list(self.objects.keys())
            return (
                f"{self.get_base_prompt()} "
                f"You can locate both objects, grab the first object ({object_names[0]}) with the manipulator, "
                f"and position it so that the distance between {object_names[0]} and {object_names[1]} along the x-axis is exactly 50 cm (0.5 units). "
                f"You can move {object_names[0]} to either side of {object_names[1]} to achieve this distance."
            )

    def _verify_args(self) -> None:
        if len(self.objects) < 2:
            error_message = f"AlignTwoObjectsTask requires at least 2 objects, but got {len(self.objects)}: {list(self.objects.keys())}"
            if self.logger:
                self.logger.error(msg=error_message)
            raise TaskParametrizationError(error_message)

        # Verify that objects are different so they can be distinguished
        for obj_name, positions in self.objects.items():
            if len(positions) != 1:
                error_message = f"Object {obj_name} must have exactly 1 position, but got {len(positions)}: {positions}"
                if self.logger:
                    self.logger.error(msg=error_message)
                raise TaskParametrizationError(error_message)

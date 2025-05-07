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
from abc import ABC, abstractmethod
from typing import Dict, List

import inflect
from langchain_core.tools import BaseTool
from rai.tools.ros2 import MoveToPointToolInput
from rai.types import Point

from rai_bench.tool_calling_agent.interfaces import Task, Validator
from rai_bench.tool_calling_agent.mocked_tools import (
    MockGetObjectPositionsTool,
    MockGetROS2TopicsNamesAndTypesTool,
    MockMoveToPointTool,
)

loggers_type = logging.Logger

PROACTIVE_ROS2_EXPERT_SYSTEM_PROMPT = """
        You are a robotic arm with interfaces to detect and manipulate objects.
        Here are the coordinates information:
        x - front to back (positive is forward)
        y - left to right (positive is right)
        z - up to down (positive is up).
        """


class TaskParametrizationError(Exception):
    """Exception raised when the task parameters are not valid."""

    pass


class ManipulationTask(Task, ABC):
    @property
    def type(self) -> str:
        return "manipulation"

    def get_system_prompt(self) -> str:
        return PROACTIVE_ROS2_EXPERT_SYSTEM_PROMPT


class GrabTask(ManipulationTask, ABC):
    def __init__(
        self,
        objects: Dict[str, List[Point]],
        object_to_grab: str,
        validators: List[Validator],
        extra_tool_calls: int = 0,
        logger: loggers_type | None = None,
    ) -> None:
        super().__init__(
            validators=validators,
            extra_tool_calls=extra_tool_calls,
            logger=logger,
        )
        self.objects = objects
        self.object_to_grab = object_to_grab
        self._verify_args()

    @abstractmethod
    def _verify_args(self) -> None:
        pass

    @property
    def available_tools(self) -> List[BaseTool]:
        return [
            MockGetROS2TopicsNamesAndTypesTool(
                mock_topics_names_and_types=[
                    "topic: /attached_collision_object\ntype: moveit_msgs/msg/AttachedCollisionObject\n",
                    "topic: /camera_image_color\ntype: sensor_msgs/msg/Image\n",
                    "topic: /camera_image_depth\ntype: sensor_msgs/msg/Image\n",
                    "topic: /clock\ntype: rosgraph_msgs/msg/Clock\n",
                    "topic: /collision_object\ntype: moveit_msgs/msg/CollisionObject\n",
                    "topic: /color_camera_info\ntype: sensor_msgs/msg/CameraInfo\n",
                ]
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
        ]


class MoveToPointTask(ManipulationTask):
    complexity = "easy"

    def __init__(
        self,
        move_to_tool_input: MoveToPointToolInput,
        validators: List[Validator],
        extra_tool_calls: int = 0,
        logger: loggers_type | None = None,
    ) -> None:
        super().__init__(
            validators=validators, extra_tool_calls=extra_tool_calls, logger=logger
        )

        self.move_to_tool_input = move_to_tool_input

    @property
    def available_tools(self) -> List[BaseTool]:
        return [
            MockGetROS2TopicsNamesAndTypesTool(
                mock_topics_names_and_types=[
                    "topic: /pointcloud\ntype: sensor_msgs/msg/PointCloud2\n",
                    "topic: /robot_description\ntype: std_msgs/msg/String\n",
                    "topic: /rosout\ntype: rcl_interfaces/msg/Log\n",
                    "topic: /tf\ntype: tf2_msgs/msg/TFMessage\n",
                ]
            ),
            MockMoveToPointTool(manipulator_frame="base_link"),
        ]

    def get_prompt(self) -> str:
        return f"Move the arm to a point x={self.move_to_tool_input.x}, y={self.move_to_tool_input.y}, z={self.move_to_tool_input.z} to {self.move_to_tool_input.task} an object."


class GetObjectPositionsTask(ManipulationTask):
    complexity = "easy"

    def __init__(
        self,
        objects: Dict[str, List[Point]],
        validators: List[Validator],
        extra_tool_calls: int = 0,
        logger: loggers_type | None = None,
    ) -> None:
        super().__init__(
            validators=validators, extra_tool_calls=extra_tool_calls, logger=logger
        )
        """Task to get the positions of the objects

        Examples
        --------
        objects = {
            "banana": [(0.1, 0.2, 0.3), (0.4, 0.5, 0.6)],
            "cube": [(0.7, 0.8, 0.9)],
        }
        """
        self.objects = objects

    @property
    def available_tools(self) -> List[BaseTool]:
        return [
            MockGetROS2TopicsNamesAndTypesTool(
                mock_topics_names_and_types=[
                    "topic: /pointcloud\ntype: sensor_msgs/msg/PointCloud2\n",
                    "topic: /robot_description\ntype: std_msgs/msg/String\n",
                    "topic: /rosout\ntype: rcl_interfaces/msg/Log\n",
                    "topic: /tf\ntype: tf2_msgs/msg/TFMessage\n",
                ]
            ),
            MockGetObjectPositionsTool(mock_objects=self.objects),
        ]

    def get_prompt(self) -> str:
        """Generates a prompt based on the objects provided in the task. If there is more than one object, the object in the prompt will be pluralized.
        Returns:
            str: Formatted prompt for the task
        """
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
        return f"Get the {objects_list} positions."


class GrabExistingObjectTask(GrabTask):
    complexity = "medium"
    """
    Task to grab an object.

    Parameters
    ----------
    objects : Dict[str, List[dict[str, float]]]
        Dictionary of object types and their positions.
    object_to_grab : str
        The object to be grabbed (must have a single position).
    """

    def get_prompt(self) -> str:
        return f"Grab {self.object_to_grab}."

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
    """
    Task to attempt grabbing an object that does not exist.

    Parameters
    ----------
    objects : Dict[str, List[dict[str, float]]]
        Available objects and their positions.
    object_to_grab : str
        Object that should not be present in the list.
    """

    def get_prompt(self) -> str:
        return f"Grab {self.object_to_grab}."

    def _verify_args(self):
        if self.object_to_grab in self.objects:
            error_message = f"Requested object to grab {self.object_to_grab} is present in defined objects: {self.objects} but should not be."
            self.logger.error(msg=error_message)
            raise TaskParametrizationError(error_message)


class MoveExistingObjectLeftTask(GrabTask):
    """Task to move an existing object to the left.

    Parameters
    ----------
    objects : Dict[str, List[dict[str, float]]]
        Dictionary containing the object types and their positions. Object type should be passed as singular.
    object_to_grab : str
        Object type should be passed as singular. Object to be grabbed should be defined in the objects argument with only one instance (one position).
    logger : loggers_type | None, optional
        Logger, by default None

    Examples
    --------
    objects = {
        "banana": [(0.1, 0.2, 0.3), (0.4, 0.5, 0.6)],
        "cube": [(0.7, 0.8, 0.9)],
    }
    object_to_grab = "cube"
    """

    complexity = "medium"

    def get_prompt(self) -> str:
        return f"Move {self.object_to_grab} 20 cm to the left."

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
    """Task to move an existing object to the front

    Parameters
    ----------
    objects : Dict[str, List[dict[str, float]]]
        Dictionary containing the object types and their positions. Object type should be passed as singular.
    object_to_grab : str
        Object to grab. Object type should be passed as singular. Object to be grabbed should be defined in the objects argument with only one instance (one position).
    logger : loggers_type | None, optional
        Logger, by default None
    """

    complexity = "medium"

    def get_prompt(self) -> str:
        return f"Move {self.object_to_grab} 60 cm to the front."

    def _verify_args(self):
        if self.object_to_grab not in self.objects:
            error_message = f"Requested object to grab {self.object_to_grab} is not present in defined objects: {self.objects}."
            self.logger.error(msg=error_message)
            raise TaskParametrizationError(error_message)

        if len(self.objects[self.object_to_grab]) > 1:
            error_message = f"Requested object to grab {self.object_to_grab} has more than one position in defined objects: {self.objects[self.object_to_grab]}."
            self.logger.error(msg=error_message)
            raise TaskParametrizationError(error_message)


class SwapObjectsTask(Task):
    """Task to swap objects

    Parameters
    ----------
    objects : Dict[str, List[Dict[str, float]]]
        Dictionary containing the object types and their positions. Object type should be passed as singular.
    objects_to_swap : List[str]
        Objects to be swapped. Object type should be passed as singular. Objects to be swapped should be defined in the objects argument with only one instance (one position).
    logger : loggers_type | None, optional
        Logger, by default None

    Examples
    --------
    objects = {
        "banana": [(0.1, 0.2, 0.1)],
        "cube": [(0.7, 0.8, 0.1)],
        "apple": [(0.3, 0.4, 0.1), (0.5, 0.6, 0.1)],

    }
    objects_to_swap = ["cube", "banana"]
    """

    complexity = "hard"

    def __init__(
        self,
        objects: Dict[str, List[Point]],
        objects_to_swap: str,
        validators: List[Validator],
        extra_tool_calls: int = 0,
        logger: loggers_type | None = None,
    ) -> None:
        super().__init__(
            validators=validators,
            extra_tool_calls=extra_tool_calls,
            logger=logger,
        )
        self.objects = objects
        self.objects_to_swap = objects_to_swap
        self._verify_args()

    @property
    def available_tools(self) -> List[BaseTool]:
        return [
            MockGetROS2TopicsNamesAndTypesTool(
                mock_topics_names_and_types=[
                    "topic: /attached_collision_object\ntype: moveit_msgs/msg/AttachedCollisionObject\n",
                    "topic: /camera_image_color\ntype: sensor_msgs/msg/Image\n",
                    "topic: /camera_image_depth\ntype: sensor_msgs/msg/Image\n",
                    "topic: /clock\ntype: rosgraph_msgs/msg/Clock\n",
                    "topic: /collision_object\ntype: moveit_msgs/msg/CollisionObject\n",
                    "topic: /color_camera_info\ntype: sensor_msgs/msg/CameraInfo\n",
                ]
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
        ]

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
        return f"Move {self.objects_to_swap[0]} to the initial position of {self.objects_to_swap[1]}, and move {self.objects_to_swap[1]} to the initial position of {self.objects_to_swap[0]}."

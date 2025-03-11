import logging
from abc import ABC, abstractmethod
from typing import Any, List

from langchain_core.messages import AIMessage, ToolCall
from langchain_core.tools import BaseTool

from rai_bench.agent_bench.mocked_tools import (
    MockGetROS2ImageTool,
    MockGetROS2TopicsNamesAndTypesTool,
)

loggers_type = logging.Logger


class AgentTask(ABC):
    def __init__(
        self,
        logger: loggers_type | None = None,
    ) -> None:
        if logger:
            self.logger = logger
        else:
            self.logger = logging.getLogger(__name__)
        self.expected_tools: List[BaseTool] = []

    @abstractmethod
    def get_prompt(self) -> str:
        """Returns the task instruction - the prompt that will be passed to agent"""
        pass

    @abstractmethod
    def verify_tool_calls(self, response: dict[str, Any]) -> bool:
        pass


class ROS2AgentTask(AgentTask):
    def __init__(self, logger: loggers_type | None = None) -> None:
        super().__init__(logger)

    def _is_ai_message_requesting_get_ros2_topics_and_types(
        self, ai_message: AIMessage
    ) -> bool:
        """Helper method to check if the given AIMessage is calling the only tool
        to get ROS2 topics names and types correctly.
        """
        if len(ai_message.tool_calls) != 1:
            self.logger.info(
                f"Number of tool calls in AIMessage should be 1, but got {len(ai_message.tool_calls)}."
            )
            return False

        expected_tool_call: ToolCall = ai_message.tool_calls[0]

        if expected_tool_call["name"] != "get_ros2_topics_names_and_types":
            self.logger.info(
                f"Expected tool call name should be 'get_ros2_topics_names_and_types', but got {expected_tool_call['name']}."
            )
            return False

        if expected_tool_call["args"] != {}:
            self.logger.info(
                f"Expected args for tool call should be empty, but got {expected_tool_call['args']}."
            )
            return False

        return True

    def _is_ai_message_requesting_get_ros2_camera(
        self, ai_message: AIMessage, camera_topic: str
    ) -> bool:
        if len(ai_message.tool_calls) != 1:
            self.logger.info(
                f"Number of tool calls in AIMessage should be 1, but got {len(ai_message.tool_calls)}."
            )
            return False

        expected_tool_call: ToolCall = ai_message.tool_calls[0]

        if expected_tool_call["name"] != "get_ros2_image":
            self.logger.info(
                f"Expected tool call name should be 'get_ros2_camera_image', but got {expected_tool_call['name']}."
            )
            return False

        if expected_tool_call["args"] != {"topic": camera_topic}:
            self.logger.info(
                f"Expected args for tool call should be {{'topic': '{camera_topic}'}}, but got {expected_tool_call['args']}."
            )
            return False

        return True


class GetROS2TopicsTask(ROS2AgentTask):
    def __init__(self, logger: loggers_type | None = None) -> None:
        super().__init__(logger=logger)
        self.expected_tools = [
            MockGetROS2TopicsNamesAndTypesTool(
                mock_topics_names_and_types=[
                    "topic: /attached_collision_object\ntype: moveit_msgs/msg/AttachedCollisionObject\n",
                    "topic: /camera_image_color\ntype: sensor_msgs/msg/Image\n",
                    "topic: /camera_image_depth\ntype: sensor_msgs/msg/Image\n",
                    "topic: /clock\ntype: rosgraph_msgs/msg/Clock\n",
                    "topic: /collision_object\ntype: moveit_msgs/msg/CollisionObject\n",
                    "topic: /color_camera_info\ntype: sensor_msgs/msg/CameraInfo\n",
                    "topic: /color_camera_info5\ntype: sensor_msgs/msg/CameraInfo\n",
                    "topic: /color_image5\ntype: sensor_msgs/msg/Image\n",
                    "topic: /depth_camera_info5\ntype: sensor_msgs/msg/CameraInfo\n",
                    "topic: /depth_image5\ntype: sensor_msgs/msg/Image\n",
                    "topic: /display_contacts\ntype: visualization_msgs/msg/MarkerArray\n",
                    "topic: /display_planned_path\ntype: moveit_msgs/msg/DisplayTrajectory\n",
                    "topic: /execute_trajectory/_action/feedback\ntype: moveit_msgs/action/ExecuteTrajectory_FeedbackMessage\n",
                    "topic: /execute_trajectory/_action/status\ntype: action_msgs/msg/GoalStatusArray\n",
                    "topic: /joint_states\ntype: sensor_msgs/msg/JointState\n",
                    "topic: /monitored_planning_scene\ntype: moveit_msgs/msg/PlanningScene\n",
                    "topic: /motion_plan_request\ntype: moveit_msgs/msg/MotionPlanRequest\n",
                    "topic: /move_action/_action/feedback\ntype: moveit_msgs/action/MoveGroup_FeedbackMessage\n",
                    "topic: /move_action/_action/status\ntype: action_msgs/msg/GoalStatusArray\n",
                    "topic: /panda_arm_controller/follow_joint_trajectory/_action/feedback\ntype: control_msgs/action/FollowJointTrajectory_FeedbackMessage\n",
                    "topic: /panda_arm_controller/follow_joint_trajectory/_action/status\ntype: action_msgs/msg/GoalStatusArray\n",
                    "topic: /panda_hand_controller/gripper_cmd/_action/feedback\ntype: control_msgs/action/GripperCommand_FeedbackMessage\n",
                    "topic: /panda_hand_controller/gripper_cmd/_action/status\ntype: action_msgs/msg/GoalStatusArray\n",
                    "topic: /parameter_events\ntype: rcl_interfaces/msg/ParameterEvent\n",
                    "topic: /planning_scene\ntype: moveit_msgs/msg/PlanningScene\n",
                    "topic: /planning_scene_world\ntype: moveit_msgs/msg/PlanningSceneWorld\n",
                    "topic: /pointcloud\ntype: sensor_msgs/msg/PointCloud2\n",
                    "topic: /robot_description\ntype: std_msgs/msg/String\n",
                    "topic: /robot_description_semantic\ntype: std_msgs/msg/String\n",
                    "topic: /rosout\ntype: rcl_interfaces/msg/Log\n",
                    "topic: /tf\ntype: tf2_msgs/msg/TFMessage\n",
                    "topic: /tf_static\ntype: tf2_msgs/msg/TFMessage\n",
                    "topic: /trajectory_execution_event\ntype: std_msgs/msg/String\n",
                ]
            )
        ]

    def get_prompt(self) -> str:
        return "Get the names and types of all ROS2 topics"

    def verify_tool_calls(self, response: dict[str, Any]) -> bool:
        messages = response["messages"]
        ai_messages: List[AIMessage] = [
            message for message in messages if isinstance(message, AIMessage)
        ]

        if (
            not ai_messages
            or not self._is_ai_message_requesting_get_ros2_topics_and_types(
                ai_messages[0]
            )
        ):
            return False

        total_tool_calls = sum(len(message.tool_calls) for message in ai_messages)
        if total_tool_calls != 1:
            self.logger.info(
                f"Total number of tool calls across all AI messages should be 1, but got {total_tool_calls}."
            )
            return False

        return True


class GetROS2CameraTask(ROS2AgentTask):
    def __init__(self, logger: loggers_type | None = None) -> None:
        super().__init__(logger=logger)
        self.expected_tools: List[BaseTool] = [
            MockGetROS2TopicsNamesAndTypesTool(
                mock_topics_names_and_types=[
                    "topic: /attached_collision_object\ntype: moveit_msgs/msg/AttachedCollisionObject\n",
                    "topic: /camera_image_color\ntype: sensor_msgs/msg/Image\n",
                    "topic: /camera_image_depth\ntype: sensor_msgs/msg/Image\n",
                    "topic: /clock\ntype: rosgraph_msgs/msg/Clock\n",
                    "topic: /collision_object\ntype: moveit_msgs/msg/CollisionObject\n",
                    "topic: /color_camera_info\ntype: sensor_msgs/msg/CameraInfo\n",
                    "topic: /color_camera_info5\ntype: sensor_msgs/msg/CameraInfo\n",
                    "topic: /color_image5\ntype: sensor_msgs/msg/Image\n",
                    "topic: /depth_camera_info5\ntype: sensor_msgs/msg/CameraInfo\n",
                    "topic: /depth_image5\ntype: sensor_msgs/msg/Image\n",
                    "topic: /display_contacts\ntype: visualization_msgs/msg/MarkerArray\n",
                    "topic: /display_planned_path\ntype: moveit_msgs/msg/DisplayTrajectory\n",
                    "topic: /execute_trajectory/_action/feedback\ntype: moveit_msgs/action/ExecuteTrajectory_FeedbackMessage\n",
                    "topic: /execute_trajectory/_action/status\ntype: action_msgs/msg/GoalStatusArray\n",
                    "topic: /joint_states\ntype: sensor_msgs/msg/JointState\n",
                    "topic: /monitored_planning_scene\ntype: moveit_msgs/msg/PlanningScene\n",
                    "topic: /motion_plan_request\ntype: moveit_msgs/msg/MotionPlanRequest\n",
                    "topic: /move_action/_action/feedback\ntype: moveit_msgs/action/MoveGroup_FeedbackMessage\n",
                    "topic: /move_action/_action/status\ntype: action_msgs/msg/GoalStatusArray\n",
                    "topic: /panda_arm_controller/follow_joint_trajectory/_action/feedback\ntype: control_msgs/action/FollowJointTrajectory_FeedbackMessage\n",
                    "topic: /panda_arm_controller/follow_joint_trajectory/_action/status\ntype: action_msgs/msg/GoalStatusArray\n",
                    "topic: /panda_hand_controller/gripper_cmd/_action/feedback\ntype: control_msgs/action/GripperCommand_FeedbackMessage\n",
                    "topic: /panda_hand_controller/gripper_cmd/_action/status\ntype: action_msgs/msg/GoalStatusArray\n",
                    "topic: /parameter_events\ntype: rcl_interfaces/msg/ParameterEvent\n",
                    "topic: /planning_scene\ntype: moveit_msgs/msg/PlanningScene\n",
                    "topic: /planning_scene_world\ntype: moveit_msgs/msg/PlanningSceneWorld\n",
                    "topic: /pointcloud\ntype: sensor_msgs/msg/PointCloud2\n",
                    "topic: /robot_description\ntype: std_msgs/msg/String\n",
                    "topic: /robot_description_semantic\ntype: std_msgs/msg/String\n",
                    "topic: /rosout\ntype: rcl_interfaces/msg/Log\n",
                    "topic: /tf\ntype: tf2_msgs/msg/TFMessage\n",
                    "topic: /tf_static\ntype: tf2_msgs/msg/TFMessage\n",
                    "topic: /trajectory_execution_event\ntype: std_msgs/msg/String\n",
                ]
            ),
            MockGetROS2ImageTool(),
        ]

    def get_prompt(self) -> str:
        return "Get the image from the camera."

    def verify_tool_calls(self, response: dict[str, Any]) -> bool:
        messages = response["messages"]
        ai_messages: List[AIMessage] = [
            message for message in messages if isinstance(message, AIMessage)
        ]

        if len(ai_messages) < 3:
            self.logger.info("Expected at least 3 AI messages, but got fewer.")
            return False

        if not self._is_ai_message_requesting_get_ros2_topics_and_types(ai_messages[0]):
            return False

        if not self._is_ai_message_requesting_get_ros2_camera(
            ai_messages[1], camera_topic="/camera_image_color"
        ):
            return False
        return True

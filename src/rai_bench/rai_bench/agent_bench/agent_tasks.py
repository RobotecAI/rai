import logging
from abc import ABC, abstractmethod
from typing import Any, List

from langchain_core.messages import AIMessage, ToolCall
from langchain_core.tools import BaseTool
from pydantic import BaseModel

from rai_bench.agent_bench.mocked_tools import (
    MockGetROS2ImageTool,
    MockGetROS2TopicsNamesAndTypesTool,
)

loggers_type = logging.Logger


class Result(BaseModel):
    success: bool = False
    errors: list[str] = []


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
    def verify_tool_calls(self, response: dict[str, Any]) -> Result:
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
                f"Expected tool call name should be 'get_ros2_image', but got {expected_tool_call['name']}."
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
                    "topic: /camera_image_color\ntype: sensor_msgs/msg/Image\n",
                    "topic: /camera_image_depth\ntype: sensor_msgs/msg/Image\n",
                    "topic: /tf\ntype: tf2_msgs/msg/TFMessage\n",
                ]
            )
        ]

    def get_prompt(self) -> str:
        return "Get the names and types of all ROS2 topics"

    def verify_tool_calls(self, response: dict[str, Any]) -> Result:
        result = Result()
        messages = response["messages"]
        ai_messages: List[AIMessage] = [
            message for message in messages if isinstance(message, AIMessage)
        ]

        if not ai_messages:
            error_msg = "No AI messages found in the response."
            self.logger.error(error_msg)
            result.errors.append(error_msg)

        if not self._is_ai_message_requesting_get_ros2_topics_and_types(ai_messages[0]):
            error_msg = (
                "First AI message did not request ROS2 topics and types correctly."
            )
            self.logger.error(error_msg)
            result.errors.append(error_msg)

        total_tool_calls = sum(len(message.tool_calls) for message in ai_messages)
        if total_tool_calls != 1:
            error_msg = f"Total number of tool calls across all AI messages should be 1, but got {total_tool_calls}."
            self.logger.error(error_msg)
            result.errors.append(error_msg)

        if not result.errors:
            result.success = True

        return result


class GetROS2CameraTask(ROS2AgentTask):
    def __init__(self, logger: loggers_type | None = None) -> None:
        super().__init__(logger=logger)
        self.expected_tools: List[BaseTool] = [
            MockGetROS2TopicsNamesAndTypesTool(
                mock_topics_names_and_types=[
                    "topic: /camera_image_color\ntype: sensor_msgs/msg/Image\n",
                    "topic: /camera_image_depth\ntype: sensor_msgs/msg/Image\n",
                    "topic: /tf\ntype: tf2_msgs/msg/TFMessage\n",
                ]
            ),
            MockGetROS2ImageTool(),
        ]

    def get_prompt(self) -> str:
        return "Get the image from the camera."

    def verify_tool_calls(self, response: dict[str, Any]) -> Result:
        result = Result()
        messages = response["messages"]
        ai_messages: List[AIMessage] = [
            message for message in messages if isinstance(message, AIMessage)
        ]

        if len(ai_messages) < 3:
            error_msg = f"Expected at least 3 AI messages, but got {len(ai_messages)}."
            self.logger.error(error_msg)
            result.errors.append(error_msg)

        if ai_messages and not self._is_ai_message_requesting_get_ros2_topics_and_types(
            ai_messages[0]
        ):
            error_msg = (
                "First AI message did not request ROS2 topics and types correctly."
            )
            self.logger.error(error_msg)
            result.errors.append(error_msg)

        if len(ai_messages) > 1 and not self._is_ai_message_requesting_get_ros2_camera(
            ai_messages[1], camera_topic="/camera_image_color"
        ):
            error_msg = (
                "Second AI message did not request the ROS2 camera image correctly."
            )
            self.logger.error(error_msg)
            result.errors.append(error_msg)

        if not result.errors:
            result.success = True

        return result

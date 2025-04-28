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

from typing import Any, Dict

import pytest
from langchain_core.messages import ToolCall

from rai_bench.tool_calling_agent.interfaces import SubTask, SubTaskValidationError
from rai_bench.tool_calling_agent.tasks.subtasks import (
    CheckActionFieldsToolCallSubTask,
    CheckArgsToolCallSubTask,
    CheckServiceFieldsToolCallSubTask,
    CheckTopicFieldsToolCallSubTask,
)


class TestSubTaskHelpers:
    """Test the helper methods in the abstract SubTask class."""

    @pytest.fixture
    def mock_subtask(self):
        """Create a concrete implementation of the abstract SubTask for testing"""

        class ConcreteSubTask(SubTask):
            def validate(self, tool_call: ToolCall) -> bool:
                return True

            def dump(self) -> Dict[str, Any]:
                return {}

            @property
            def info(self) -> Dict[str, Any]:
                return {"name": "blybly"}

        return ConcreteSubTask()

    def test_check_tool_call_valid(self, mock_subtask):
        """Test _check_tool_call with valid inputs."""
        tool_call = {"name": "test_tool", "args": {"arg1": "value1", "arg2": 42}}

        expected_args = {"arg1": "value1", "arg2": 42}

        assert mock_subtask._check_tool_call(
            tool_call=tool_call, expected_name="test_tool", expected_args=expected_args
        )

    def test_check_tool_call_wrong_name(self, mock_subtask):
        """Test _check_tool_call fails with wrong tool name."""
        tool_call = {"name": "wrong_tool", "args": {"arg1": "value1", "arg2": 42}}

        expected_args = {"arg1": "value1", "arg2": 42}

        with pytest.raises(
            SubTaskValidationError,
            match="Expected tool call name should be 'test_tool'",
        ):
            mock_subtask._check_tool_call(
                tool_call=tool_call,
                expected_name="test_tool",
                expected_args=expected_args,
            )

    def test_check_tool_call_missing_arg(self, mock_subtask):
        """Test _check_tool_call fails with missing argument."""
        tool_call = {"name": "test_tool", "args": {"arg1": "value1"}}

        expected_args = {"arg1": "value1", "arg2": 42}

        with pytest.raises(
            SubTaskValidationError, match="Required argument 'arg2' missing"
        ):
            mock_subtask._check_tool_call(
                tool_call=tool_call,
                expected_name="test_tool",
                expected_args=expected_args,
            )

    def test_check_tool_call_wrong_arg_value(self, mock_subtask):
        """Test _check_tool_call fails with wrong argument value."""
        tool_call = {"name": "test_tool", "args": {"arg1": "wrong_value", "arg2": 42}}

        expected_args = {"arg1": "value1", "arg2": 42}

        with pytest.raises(
            SubTaskValidationError,
            match="Expected argument 'arg1' should have value 'value1'",
        ):
            mock_subtask._check_tool_call(
                tool_call=tool_call,
                expected_name="test_tool",
                expected_args=expected_args,
            )

    def test_check_tool_call_unexpected_arg(self, mock_subtask):
        """Test _check_tool_call fails with unexpected argument."""
        tool_call = {
            "name": "test_tool",
            "args": {"arg1": "value1", "arg2": 42, "unexpected": "surprise"},
        }

        expected_args = {"arg1": "value1", "arg2": 42}

        with pytest.raises(
            SubTaskValidationError, match="Unexpected argument 'unexpected' found"
        ):
            mock_subtask._check_tool_call(
                tool_call=tool_call,
                expected_name="test_tool",
                expected_args=expected_args,
            )

    def test_check_topic_tool_call_field_valid(self, mock_subtask):
        """Test _check_topic_tool_call_field with valid inputs."""
        tool_call = {
            "name": "publish_ros2_message",
            "args": {
                "topic": "/test_topic",
                "message_type": "std_msgs/msg/String",
                "message": {
                    "header": {
                        "frame_id": "base_link",
                        "stamp": {"sec": 10, "nanosec": 500},
                    },
                    "data": "test message",
                },
            },
        }

        assert mock_subtask._check_topic_tool_call_field(
            tool_call=tool_call,
            expected_name="publish_ros2_message",
            expected_topic="/test_topic",
            expected_message_type="std_msgs/msg/String",
            field_path="header.frame_id",
            expected_value="base_link",
        )

    def test_check_topic_tool_call_field_wrong_name(self, mock_subtask):
        """Test _check_topic_tool_call_field fails with wrong tool name."""
        tool_call = {
            "name": "wrong_name",
            "args": {
                "topic": "/test_topic",
                "message_type": "std_msgs/msg/String",
                "message": {"data": "test"},
            },
        }

        with pytest.raises(SubTaskValidationError, match="Expected tool call name"):
            mock_subtask._check_topic_tool_call_field(
                tool_call=tool_call,
                expected_name="publish_ros2_message",
                expected_topic="/test_topic",
                expected_message_type="std_msgs/msg/String",
                field_path="data",
                expected_value="test",
            )

    def test_check_topic_tool_call_field_wrong_topic(self, mock_subtask):
        """Test _check_topic_tool_call_field fails with wrong topic."""
        tool_call = {
            "name": "publish_ros2_message",
            "args": {
                "topic": "/wrong_topic",
                "message_type": "std_msgs/msg/String",
                "message": {"data": "test"},
            },
        }

        with pytest.raises(SubTaskValidationError, match="Expected topic"):
            mock_subtask._check_topic_tool_call_field(
                tool_call=tool_call,
                expected_name="publish_ros2_message",
                expected_topic="/test_topic",
                expected_message_type="std_msgs/msg/String",
                field_path="data",
                expected_value="test",
            )

    def test_check_topic_tool_call_field_wrong_message_type(self, mock_subtask):
        """Test _check_topic_tool_call_field fails with wrong message type."""
        tool_call = {
            "name": "publish_ros2_message",
            "args": {
                "topic": "/test_topic",
                "message_type": "wrong_type",
                "message": {"data": "test"},
            },
        }

        with pytest.raises(SubTaskValidationError, match="Expected message type"):
            mock_subtask._check_topic_tool_call_field(
                tool_call=tool_call,
                expected_name="publish_ros2_message",
                expected_topic="/test_topic",
                expected_message_type="std_msgs/msg/String",
                field_path="data",
                expected_value="test",
            )

    def test_check_topic_tool_call_field_missing_message(self, mock_subtask):
        """Test _check_topic_tool_call_field fails with missing message."""
        tool_call = {
            "name": "publish_ros2_message",
            "args": {
                "topic": "/test_topic",
                "message_type": "std_msgs/msg/String",
                # missing message
            },
        }

        with pytest.raises(
            SubTaskValidationError, match="does not contain a 'message' argument"
        ):
            mock_subtask._check_topic_tool_call_field(
                tool_call=tool_call,
                expected_name="publish_ros2_message",
                expected_topic="/test_topic",
                expected_message_type="std_msgs/msg/String",
                field_path="data",
                expected_value="test",
            )

    def test_check_topic_tool_call_field_invalid_path(self, mock_subtask):
        """Test _check_topic_tool_call_field fails with invalid field path."""
        tool_call = {
            "name": "publish_ros2_message",
            "args": {
                "topic": "/test_topic",
                "message_type": "std_msgs/msg/String",
                "message": {"data": "test"},
            },
        }

        with pytest.raises(
            SubTaskValidationError, match="Field path 'non_existent.field' not found"
        ):
            mock_subtask._check_topic_tool_call_field(
                tool_call=tool_call,
                expected_name="publish_ros2_message",
                expected_topic="/test_topic",
                expected_message_type="std_msgs/msg/String",
                field_path="non_existent.field",
                expected_value="test",
            )

    def test_check_topic_tool_call_field_wrong_value(self, mock_subtask):
        """Test _check_topic_tool_call_field fails with wrong field value."""
        tool_call = {
            "name": "publish_ros2_message",
            "args": {
                "topic": "/test_topic",
                "message_type": "std_msgs/msg/String",
                "message": {"data": "wrong value"},
            },
        }

        with pytest.raises(
            SubTaskValidationError, match="Expected value for field 'data'"
        ):
            mock_subtask._check_topic_tool_call_field(
                tool_call=tool_call,
                expected_name="publish_ros2_message",
                expected_topic="/test_topic",
                expected_message_type="std_msgs/msg/String",
                field_path="data",
                expected_value="test",
            )

    def test_check_service_tool_call_field_valid(self, mock_subtask):
        """Test _check_service_tool_call_field with valid inputs."""
        tool_call = {
            "name": "call_ros2_service",
            "args": {
                "service_name": "/test_service",
                "service_type": "std_srvs/srv/SetBool",
                "service_args": {"data": True},
            },
        }

        assert mock_subtask._check_service_tool_call_field(
            tool_call=tool_call,
            expected_name="call_ros2_service",
            expected_service="/test_service",
            expected_service_type="std_srvs/srv/SetBool",
            field_path="data",
            expected_value=True,
        )

    def test_check_service_tool_call_field_empty_args(self, mock_subtask):
        """Test _check_service_tool_call_field with empty service_args."""
        tool_call = {
            "name": "call_ros2_service",
            "args": {
                "service_name": "/test_service",
                "service_type": "std_srvs/srv/Empty",
                "service_args": {},
            },
        }

        assert mock_subtask._check_service_tool_call_field(
            tool_call=tool_call,
            expected_name="call_ros2_service",
            expected_service="/test_service",
            expected_service_type="std_srvs/srv/Empty",
            field_path="",
            expected_value={},
        )

    def test_check_service_tool_call_field_wrong_service_name(self, mock_subtask):
        """Test _check_service_tool_call_field fails with wrong service name."""
        tool_call = {
            "name": "call_ros2_service",
            "args": {
                "service_name": "/wrong_service",
                "service_type": "std_srvs/srv/SetBool",
                "service_args": {"data": True},
            },
        }

        with pytest.raises(SubTaskValidationError, match="Expected service"):
            mock_subtask._check_service_tool_call_field(
                tool_call=tool_call,
                expected_name="call_ros2_service",
                expected_service="/test_service",
                expected_service_type="std_srvs/srv/SetBool",
                field_path="data",
                expected_value=True,
            )

    def test_check_service_tool_call_field_missing_service_args(self, mock_subtask):
        """Test _check_service_tool_call_field fails with missing service_args."""
        tool_call = {
            "name": "call_ros2_service",
            "args": {
                "service_name": "/test_service",
                "service_type": "std_srvs/srv/SetBool",
                # missing service_args
            },
        }

        with pytest.raises(
            SubTaskValidationError, match="does not contain a 'service_args' argument"
        ):
            mock_subtask._check_service_tool_call_field(
                tool_call=tool_call,
                expected_name="call_ros2_service",
                expected_service="/test_service",
                expected_service_type="std_srvs/srv/SetBool",
                field_path="data",
                expected_value=True,
            )

    def test_check_action_tool_call_field_valid(self, mock_subtask):
        """Test _check_action_tool_call_field with valid inputs."""
        tool_call = {
            "name": "call_ros2_action",
            "args": {
                "action_name": "/test_action",
                "action_type": "control_msgs/action/FollowJointTrajectory",
                "action_args": {
                    "trajectory": {
                        "joint_names": ["joint1", "joint2"],
                        "points": [{"positions": [0.1, 0.2]}],
                    }
                },
            },
        }

        assert mock_subtask._check_action_tool_call_field(
            tool_call=tool_call,
            expected_name="call_ros2_action",
            expected_action="/test_action",
            expected_action_type="control_msgs/action/FollowJointTrajectory",
            field_path="trajectory.joint_names",
            expected_value=["joint1", "joint2"],
        )

    def test_check_action_tool_call_field_empty_args(self, mock_subtask):
        """Test _check_action_tool_call_field with empty action_args."""
        tool_call = {
            "name": "call_ros2_action",
            "args": {
                "action_name": "/test_action",
                "action_type": "test_msgs/action/Empty",
                "action_args": {},
            },
        }

        assert mock_subtask._check_action_tool_call_field(
            tool_call=tool_call,
            expected_name="call_ros2_action",
            expected_action="/test_action",
            expected_action_type="test_msgs/action/Empty",
            field_path="",
            expected_value={},
        )

    def test_check_action_tool_call_field_wrong_action_name(self, mock_subtask):
        """Test _check_action_tool_call_field fails with wrong action name."""
        tool_call = {
            "name": "call_ros2_action",
            "args": {
                "action_name": "/wrong_action",
                "action_type": "test_msgs/action/Test",
                "action_args": {"data": True},
            },
        }

        with pytest.raises(SubTaskValidationError, match="Expected action name"):
            mock_subtask._check_action_tool_call_field(
                tool_call=tool_call,
                expected_name="call_ros2_action",
                expected_action="/test_action",
                expected_action_type="test_msgs/action/Test",
                field_path="data",
                expected_value=True,
            )

    def test_check_tool_call_with_type_check_optional_args(self, mock_subtask):
        """Test _check_tool_call with type checking for optional arguments."""
        tool_call = {
            "name": "test_tool",
            "args": {
                "arg1": "value1",
                "arg2": 42,
                "optional1": "any string value",  # string type
                "optional2": 123,  # int type
                "optional3": [1, 2, 3],  # list type
                "optional4": {"key": "value"},  # dict type
            },
        }

        expected_args = {"arg1": "value1", "arg2": 42}

        expected_optional_args = {
            "optional1": str,  # expect string type
            "optional2": int,  # expect int type
            "optional3": list,  # expect list type
            "optional4": dict,  # expect dict type
            "optional5": None,  # not provided but would accept any type
        }

        assert mock_subtask._check_tool_call(
            tool_call=tool_call,
            expected_name="test_tool",
            expected_args=expected_args,
            expected_optional_args=expected_optional_args,
        )

    def test_check_tool_call_wrong_optional_arg_type(self, mock_subtask):
        """Test _check_tool_call fails with wrong optional argument type."""
        tool_call = {
            "name": "test_tool",
            "args": {
                "arg1": "value1",
                "arg2": 42,
                "optional1": 123,  # int type when str expected
            },
        }

        expected_args = {"arg1": "value1", "arg2": 42}

        expected_optional_args = {"optional1": str}  # expect string type

        with pytest.raises(SubTaskValidationError, match="has incorrect type"):
            mock_subtask._check_tool_call(
                tool_call=tool_call,
                expected_name="test_tool",
                expected_args=expected_args,
                expected_optional_args=expected_optional_args,
            )

    def test_check_tool_call_multiple_types(self, mock_subtask):
        """Test _check_tool_call with optional arguments accepting multiple types."""
        tool_call = {
            "name": "test_tool",
            "args": {"arg1": "value1", "arg2": 42, "optional1": 123},  # int type
        }

        expected_args = {"arg1": "value1", "arg2": 42}

        expected_optional_args = {
            "optional1": (str, int)  # accept either string or int
        }

        assert mock_subtask._check_tool_call(
            tool_call=tool_call,
            expected_name="test_tool",
            expected_args=expected_args,
            expected_optional_args=expected_optional_args,
        )


class TestCheckArgsToolCallSubTask:
    """Test the CheckArgsToolCallSubTask implementation."""

    def test_validate_valid_args(self):
        """Test validate with valid arguments."""
        subtask = CheckArgsToolCallSubTask(
            expected_tool_name="test_tool", expected_args={"arg1": "value1", "arg2": 42}
        )

        tool_call = {"name": "test_tool", "args": {"arg1": "value1", "arg2": 42}}

        assert subtask.validate(tool_call)

    def test_validate_invalid_args(self):
        """Test validate with invalid arguments."""
        subtask = CheckArgsToolCallSubTask(
            expected_tool_name="test_tool", expected_args={"arg1": "value1", "arg2": 42}
        )

        tool_call = {"name": "test_tool", "args": {"arg1": "wrong_value", "arg2": 42}}

        with pytest.raises(SubTaskValidationError):
            subtask.validate(tool_call)


class TestCheckTopicFieldsToolCallSubTask:
    """Test the CheckTopicFieldsToolCallSubTask implementation."""

    def test_validate_valid_fields(self):
        """Test validate with valid fields."""
        subtask = CheckTopicFieldsToolCallSubTask(
            expected_tool_name="publish_ros2_message",
            expected_topic="/test_topic",
            expected_message_type="std_msgs/msg/String",
            expected_fields={"data": "test message"},
        )

        tool_call = {
            "name": "publish_ros2_message",
            "args": {
                "topic": "/test_topic",
                "message_type": "std_msgs/msg/String",
                "message": {"data": "test message"},
            },
        }

        assert subtask.validate(tool_call)

    def test_validate_multiple_fields(self):
        """Test validate with multiple fields."""
        subtask = CheckTopicFieldsToolCallSubTask(
            expected_tool_name="publish_ros2_message",
            expected_topic="/test_topic",
            expected_message_type="geometry_msgs/msg/Pose",
            expected_fields={
                "position.x": 1.0,
                "position.y": 2.0,
                "orientation.w": 1.0,
            },
        )

        tool_call = {
            "name": "publish_ros2_message",
            "args": {
                "topic": "/test_topic",
                "message_type": "geometry_msgs/msg/Pose",
                "message": {
                    "position": {"x": 1.0, "y": 2.0, "z": 0.0},
                    "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
                },
            },
        }

        assert subtask.validate(tool_call)

    def test_validate_invalid_field(self):
        """Test validate with invalid field value."""
        subtask = CheckTopicFieldsToolCallSubTask(
            expected_tool_name="publish_ros2_message",
            expected_topic="/test_topic",
            expected_message_type="std_msgs/msg/String",
            expected_fields={"data": "expected message"},
        )

        tool_call = {
            "name": "publish_ros2_message",
            "args": {
                "topic": "/test_topic",
                "message_type": "std_msgs/msg/String",
                "message": {"data": "wrong message"},
            },
        }

        with pytest.raises(SubTaskValidationError):
            subtask.validate(tool_call)


class TestCheckServiceFieldsToolCallSubTask:
    """Test the CheckServiceFieldsToolCallSubTask implementation."""

    def test_validate_valid_fields(self):
        """Test validate with valid fields."""
        subtask = CheckServiceFieldsToolCallSubTask(
            expected_tool_name="call_ros2_service",
            expected_service="/test_service",
            expected_service_type="std_srvs/srv/SetBool",
            expected_fields={"data": True},
        )

        tool_call = {
            "name": "call_ros2_service",
            "args": {
                "service_name": "/test_service",
                "service_type": "std_srvs/srv/SetBool",
                "service_args": {"data": True},
            },
        }

        assert subtask.validate(tool_call)

    def test_validate_multiple_fields(self):
        """Test validate with multiple fields."""
        subtask = CheckServiceFieldsToolCallSubTask(
            expected_tool_name="call_ros2_service",
            expected_service="/test_service",
            expected_service_type="test_msgs/srv/Complex",
            expected_fields={"request_field.subfield": "value", "flag": True},
        )

        tool_call = {
            "name": "call_ros2_service",
            "args": {
                "service_name": "/test_service",
                "service_type": "test_msgs/srv/Complex",
                "service_args": {"request_field": {"subfield": "value"}, "flag": True},
            },
        }

        assert subtask.validate(tool_call)

    def test_validate_empty_args(self):
        """Test validate with empty service args."""
        subtask = CheckServiceFieldsToolCallSubTask(
            expected_tool_name="call_ros2_service",
            expected_service="/test_service",
            expected_service_type="std_srvs/srv/Empty",
            expected_fields={"": {}},
        )

        tool_call = {
            "name": "call_ros2_service",
            "args": {
                "service_name": "/test_service",
                "service_type": "std_srvs/srv/Empty",
                "service_args": {},
            },
        }

        assert subtask.validate(tool_call)


class TestCheckActionFieldsToolCallSubTask:
    """Test the CheckActionFieldsToolCallSubTask implementation."""

    def test_validate_valid_fields(self):
        """Test validate with valid fields."""
        subtask = CheckActionFieldsToolCallSubTask(
            expected_tool_name="call_ros2_action",
            expected_action="/test_action",
            expected_action_type="control_msgs/action/GripperCommand",
            expected_fields={"command.position": 0.5},
        )

        tool_call = {
            "name": "call_ros2_action",
            "args": {
                "action_name": "/test_action",
                "action_type": "control_msgs/action/GripperCommand",
                "action_args": {"command": {"position": 0.5, "max_effort": 10.0}},
            },
        }

        assert subtask.validate(tool_call)

    def test_validate_multiple_fields(self):
        """Test validate with multiple fields."""
        subtask = CheckActionFieldsToolCallSubTask(
            expected_tool_name="call_ros2_action",
            expected_action="/test_action",
            expected_action_type="test_msgs/action/Navigate",
            expected_fields={"goal.x": 1.0, "goal.y": 2.0, "speed": 0.5},
        )

        tool_call = {
            "name": "call_ros2_action",
            "args": {
                "action_name": "/test_action",
                "action_type": "test_msgs/action/Navigate",
                "action_args": {"goal": {"x": 1.0, "y": 2.0, "z": 0.0}, "speed": 0.5},
            },
        }

        assert subtask.validate(tool_call)

    def test_validate_empty_args(self):
        """Test validate with empty action args."""
        subtask = CheckActionFieldsToolCallSubTask(
            expected_tool_name="call_ros2_action",
            expected_action="/test_action",
            expected_action_type="test_msgs/action/Empty",
            expected_fields={"": {}},
        )

        tool_call = {
            "name": "call_ros2_action",
            "args": {
                "action_name": "/test_action",
                "action_type": "test_msgs/action/Empty",
                "action_args": {},
            },
        }

        assert subtask.validate(tool_call)

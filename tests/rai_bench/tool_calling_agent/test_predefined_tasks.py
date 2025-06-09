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

from typing import Any, Dict, List

import pytest

from rai_bench.tool_calling_agent.interfaces import (
    TaskArgs,
)
from rai_bench.tool_calling_agent.subtasks import (
    CheckArgsToolCallSubTask,
    CheckServiceFieldsToolCallSubTask,
)
from rai_bench.tool_calling_agent.tasks.basic import (
    GetAllROS2CamerasTask,
    GetPointcloudTask,
    GetRobotDescriptionTask,
    GetROS2DepthCameraTask,
    GetROS2RGBCameraTask,
    GetROS2TopicsTask,
    SetRobotParameterTask,
)
from rai_bench.tool_calling_agent.validators import (
    NotOrderedCallsValidator,
    OrderedCallsValidator,
)


@pytest.fixture
def task_args() -> TaskArgs:
    """Create basic task arguments for testing."""
    return TaskArgs(
        extra_tool_calls=0,
        prompt_detail="brief",
        examples_in_system_prompt=0,
    )


class TestSetParameterTask:
    """Test SetRobotParameterTask validation."""

    def test_set_parameter_task_valid(self, task_args: TaskArgs) -> None:
        set_robot_state_params_subtask = CheckServiceFieldsToolCallSubTask(
            expected_tool_name="call_ros2_service",
            expected_service="/robot_state_publisher/set_parameters",
            expected_service_type="rcl_interfaces/srv/SetParameters",
            expected_fields={
                "parameters.0.name": "publish_frequency",
                "parameters.0.value.type": 3,
                "parameters.0.value.double_value": 30.0,
            },
        )
        set_param_val = OrderedCallsValidator(subtasks=[set_robot_state_params_subtask])

        tool_calls: List[Dict[str, Any]] = [
            {"name": "get_ros2_services_names_and_types", "args": {}},
            {
                "name": "get_ros2_service_interface",
                "args": {"service_type": "rcl_interfaces/srv/SetParameters"},
            },
            {
                "name": "call_ros2_service",
                "args": {
                    "service_name": "/robot_state_publisher/get_parameters",
                    "service_type": "rcl_interfaces/srv/GetParameters",
                    "service_args": {"names": ["publish_frequency"]},
                },
            },
            {
                "name": "call_ros2_service",
                "args": {
                    "service_name": "/robot_state_publisher/set_parameters",
                    "service_type": "rcl_interfaces/srv/SetParameters",
                    "service_args": {
                        "parameters": [
                            {
                                "name": "publish_frequency",
                                "value": {
                                    "type": 3,
                                    "double_value": 30.0,
                                },
                            }
                        ]
                    },
                },
            },
        ]

        task = SetRobotParameterTask(validators=[set_param_val], task_args=task_args)
        score = task.validate(tool_calls)
        assert score == 1.0  # All validators should pass

    def test_set_parameter_task_wrong_parameter_type(self, task_args: TaskArgs) -> None:
        set_robot_state_params_subtask = CheckServiceFieldsToolCallSubTask(
            expected_tool_name="call_ros2_service",
            expected_service="/robot_state_publisher/set_parameters",
            expected_service_type="rcl_interfaces/srv/SetParameters",
            expected_fields={
                "parameters.0.name": "publish_frequency",
                "parameters.0.value.type": 3,  # Expecting double
                "parameters.0.value.double_value": 30.0,
            },
        )
        set_param_val = OrderedCallsValidator(subtasks=[set_robot_state_params_subtask])

        tool_calls: List[Dict[str, Any]] = [
            {
                "name": "call_ros2_service",
                "args": {
                    "service_name": "/robot_state_publisher/set_parameters",
                    "service_type": "rcl_interfaces/srv/SetParameters",
                    "service_args": {
                        "parameters": [
                            {
                                "name": "publish_frequency",
                                "value": {
                                    "type": 2,  # Wrong type (integer instead of double)
                                    "integer_value": 30,
                                },
                            }
                        ]
                    },
                },
            },
        ]

        task = SetRobotParameterTask(validators=[set_param_val], task_args=task_args)
        score = task.validate(tool_calls)
        assert score == 0.0  # Validator should fail

    def test_set_parameter_task_wrong_parameter_name(self, task_args: TaskArgs) -> None:
        set_robot_state_params_subtask = CheckServiceFieldsToolCallSubTask(
            expected_tool_name="call_ros2_service",
            expected_service="/robot_state_publisher/set_parameters",
            expected_service_type="rcl_interfaces/srv/SetParameters",
            expected_fields={
                "parameters.0.name": "publish_frequency",
                "parameters.0.value.type": 3,
                "parameters.0.value.double_value": 30.0,
            },
        )
        set_param_val = OrderedCallsValidator(subtasks=[set_robot_state_params_subtask])

        tool_calls: List[Dict[str, Any]] = [
            {
                "name": "call_ros2_service",
                "args": {
                    "service_name": "/robot_state_publisher/set_parameters",
                    "service_type": "rcl_interfaces/srv/SetParameters",
                    "service_args": {
                        "parameters": [
                            {
                                "name": "wrong_parameter_name",  # Wrong parameter name
                                "value": {
                                    "type": 3,
                                    "double_value": 30.0,
                                },
                            }
                        ]
                    },
                },
            },
        ]

        task = SetRobotParameterTask(validators=[set_param_val], task_args=task_args)
        score = task.validate(tool_calls)
        assert score == 0.0  # Validator should fail

    def test_set_parameter_task_wrong_tools(self, task_args: TaskArgs) -> None:
        set_robot_state_params_subtask = CheckServiceFieldsToolCallSubTask(
            expected_tool_name="call_ros2_service",
            expected_service="/robot_state_publisher/set_parameters",
            expected_service_type="rcl_interfaces/srv/SetParameters",
            expected_fields={
                "parameters.0.name": "publish_frequency",
                "parameters.0.value.type": 3,
                "parameters.0.value.double_value": 30.0,
            },
        )
        set_param_val = OrderedCallsValidator(subtasks=[set_robot_state_params_subtask])

        tool_calls: List[Dict[str, Any]] = [
            {"name": "get_ros2_services_names_and_types", "args": {}},
            {
                "name": "get_ros2_service_interface",
                "args": {"service_type": "rcl_interfaces/srv/SetParameters"},
            },
            {"name": "get_ros2_services_names_and_types", "args": {}},
            {"name": "get_ros2_services_names_and_types", "args": {}},
        ]

        task = SetRobotParameterTask(validators=[set_param_val], task_args=task_args)
        score = task.validate(tool_calls)
        assert score == 0.0  # Validator should fail


class TestGetTopicsTask:
    """Test GetROS2TopicsTask validation."""

    def test_get_topics_task_valid(self, task_args: TaskArgs) -> None:
        get_topics_subtask = CheckArgsToolCallSubTask(
            expected_tool_name="get_ros2_topics_names_and_types", expected_args={}
        )
        topics_val = OrderedCallsValidator(subtasks=[get_topics_subtask])

        tool_calls: List[Dict[str, Any]] = [
            {"name": "get_ros2_topics_names_and_types", "args": {}}
        ]

        task = GetROS2TopicsTask(validators=[topics_val], task_args=task_args)
        score = task.validate(tool_calls)
        assert score == 1.0

    def test_get_topics_task_wrong_tool(self, task_args: TaskArgs) -> None:
        """Test get ROS2 topics task with wrong tool name."""
        get_topics_subtask = CheckArgsToolCallSubTask(
            expected_tool_name="get_ros2_topics_names_and_types", expected_args={}
        )
        topics_val = OrderedCallsValidator(subtasks=[get_topics_subtask])

        tool_calls: List[Dict[str, Any]] = [
            {"name": "wrong_tool_name", "args": {}}  # Wrong tool name
        ]

        task = GetROS2TopicsTask(validators=[topics_val], task_args=task_args)
        score = task.validate(tool_calls)
        assert score == 0.0

    def test_get_topics_task_unexpected_args(self, task_args: TaskArgs) -> None:
        get_topics_subtask = CheckArgsToolCallSubTask(
            expected_tool_name="get_ros2_topics_names_and_types", expected_args={}
        )
        topics_val = OrderedCallsValidator(subtasks=[get_topics_subtask])

        tool_calls: List[Dict[str, Any]] = [
            {
                "name": "get_ros2_topics_names_and_types",
                "args": {"unexpected": "arg"},
            }  # Unexpected args
        ]

        task = GetROS2TopicsTask(validators=[topics_val], task_args=task_args)
        score = task.validate(tool_calls)
        assert score == 0.0


class TestGetRGBCameraTask:
    """Test GetROS2RGBCameraTask validation."""

    def test_get_rgb_camera_task_valid(self, task_args: TaskArgs) -> None:
        color_image_subtask = CheckArgsToolCallSubTask(
            expected_tool_name="get_ros2_image",
            expected_args={"topic": "/color_image5"},
            expected_optional_args={"timeout_sec": int},
        )
        color_image_val = OrderedCallsValidator(subtasks=[color_image_subtask])

        tool_calls: List[Dict[str, Any]] = [
            {"name": "get_ros2_topics_names_and_types", "args": {}},
            {
                "name": "get_ros2_image",
                "args": {"topic": "/color_image5", "timeout_sec": 5},
            },
        ]

        task = GetROS2RGBCameraTask(validators=[color_image_val], task_args=task_args)
        score = task.validate(tool_calls)
        assert score == 1.0

    def test_get_rgb_camera_task_wrong_topic(self, task_args: TaskArgs) -> None:
        color_image_subtask = CheckArgsToolCallSubTask(
            expected_tool_name="get_ros2_image",
            expected_args={"topic": "/color_image5"},
            expected_optional_args={"timeout_sec": int},
        )
        color_image_val = OrderedCallsValidator(subtasks=[color_image_subtask])

        tool_calls: List[Dict[str, Any]] = [
            {
                "name": "get_ros2_image",
                "args": {"topic": "/wrong_topic", "timeout_sec": 5},  # Wrong topic
            }
        ]

        task = GetROS2RGBCameraTask(validators=[color_image_val], task_args=task_args)
        score = task.validate(tool_calls)
        assert score == 0.0

    def test_get_rgb_camera_task_missing_required_arg(
        self, task_args: TaskArgs
    ) -> None:
        color_image_subtask = CheckArgsToolCallSubTask(
            expected_tool_name="get_ros2_image",
            expected_args={"topic": "/color_image5"},
            expected_optional_args={"timeout_sec": int},
        )
        color_image_val = OrderedCallsValidator(subtasks=[color_image_subtask])

        tool_calls: List[Dict[str, Any]] = [
            {
                "name": "get_ros2_image",
                "args": {"timeout_sec": 5},  # Missing required topic arg
            }
        ]

        task = GetROS2RGBCameraTask(validators=[color_image_val], task_args=task_args)
        score = task.validate(tool_calls)
        assert score == 0.0

    def test_get_rgb_camera_task_wrong_timeout_type(self, task_args: TaskArgs) -> None:
        color_image_subtask = CheckArgsToolCallSubTask(
            expected_tool_name="get_ros2_image",
            expected_args={"topic": "/color_image5"},
            expected_optional_args={"timeout_sec": int},
        )
        color_image_val = OrderedCallsValidator(subtasks=[color_image_subtask])

        tool_calls: List[Dict[str, Any]] = [
            {
                "name": "get_ros2_image",
                "args": {
                    "topic": "/color_image5",
                    "timeout_sec": "not_an_int",
                },  # Wrong type
            }
        ]

        task = GetROS2RGBCameraTask(validators=[color_image_val], task_args=task_args)
        score = task.validate(tool_calls)
        assert score == 0.0


class TestGetDepthCameraTask:
    """Test GetROS2DepthCameraTask validation."""

    def test_get_depth_camera_task_valid(self, task_args: TaskArgs) -> None:
        depth_image_subtask = CheckArgsToolCallSubTask(
            expected_tool_name="get_ros2_image",
            expected_args={"topic": "/depth_image5"},
            expected_optional_args={"timeout_sec": int},
        )
        depth_image_val = OrderedCallsValidator(subtasks=[depth_image_subtask])

        tool_calls: List[Dict[str, Any]] = [
            {"name": "get_ros2_topics_names_and_types", "args": {}},
            {
                "name": "get_ros2_image",
                "args": {"topic": "/depth_image5", "timeout_sec": 5},
            },
        ]

        task = GetROS2DepthCameraTask(validators=[depth_image_val], task_args=task_args)
        score = task.validate(tool_calls)
        assert score == 1.0

    def test_get_depth_camera_task_wrong_topic(self, task_args: TaskArgs) -> None:
        depth_image_subtask = CheckArgsToolCallSubTask(
            expected_tool_name="get_ros2_image",
            expected_args={"topic": "/depth_image5"},
            expected_optional_args={"timeout_sec": int},
        )
        depth_image_val = OrderedCallsValidator(subtasks=[depth_image_subtask])

        tool_calls: List[Dict[str, Any]] = [
            {
                "name": "get_ros2_image",
                "args": {
                    "topic": "/color_image5",
                    "timeout_sec": 5,
                },  # Wrong topic (color instead of depth)
            }
        ]

        task = GetROS2DepthCameraTask(validators=[depth_image_val], task_args=task_args)
        score = task.validate(tool_calls)
        assert score == 0.0


class TestGetAllCamerasTask:
    """Test GetAllROS2CamerasTask validation."""

    def test_get_all_cameras_task_valid(self, task_args: TaskArgs) -> None:
        color_image_subtask = CheckArgsToolCallSubTask(
            expected_tool_name="get_ros2_image",
            expected_args={"topic": "/color_image5"},
            expected_optional_args={"timeout_sec": int},
        )
        depth_image_subtask = CheckArgsToolCallSubTask(
            expected_tool_name="get_ros2_image",
            expected_args={"topic": "/depth_image5"},
            expected_optional_args={"timeout_sec": int},
        )
        all_cameras_val = NotOrderedCallsValidator(
            subtasks=[color_image_subtask, depth_image_subtask]
        )

        tool_calls: List[Dict[str, Any]] = [
            {"name": "get_ros2_topics_names_and_types", "args": {}},
            {
                "name": "get_ros2_image",
                "args": {"topic": "/color_image5", "timeout_sec": 5},
            },
            {
                "name": "get_ros2_image",
                "args": {"topic": "/depth_image5", "timeout_sec": 5},
            },
        ]

        task = GetAllROS2CamerasTask(validators=[all_cameras_val], task_args=task_args)
        score = task.validate(tool_calls)
        assert score == 1.0

    def test_get_all_cameras_task_missing_depth(self, task_args: TaskArgs) -> None:
        color_image_subtask = CheckArgsToolCallSubTask(
            expected_tool_name="get_ros2_image",
            expected_args={"topic": "/color_image5"},
            expected_optional_args={"timeout_sec": int},
        )
        depth_image_subtask = CheckArgsToolCallSubTask(
            expected_tool_name="get_ros2_image",
            expected_args={"topic": "/depth_image5"},
            expected_optional_args={"timeout_sec": int},
        )
        all_cameras_val = NotOrderedCallsValidator(
            subtasks=[color_image_subtask, depth_image_subtask]
        )

        tool_calls: List[Dict[str, Any]] = [
            {
                "name": "get_ros2_image",
                "args": {"topic": "/color_image5", "timeout_sec": 5},
            }
            # Missing depth camera call
        ]

        task = GetAllROS2CamerasTask(validators=[all_cameras_val], task_args=task_args)
        score = task.validate(tool_calls)
        assert score == 0.0

    def test_get_all_cameras_task_wrong_order_should_pass(
        self, task_args: TaskArgs
    ) -> None:
        color_image_subtask = CheckArgsToolCallSubTask(
            expected_tool_name="get_ros2_image",
            expected_args={"topic": "/color_image5"},
            expected_optional_args={"timeout_sec": int},
        )
        depth_image_subtask = CheckArgsToolCallSubTask(
            expected_tool_name="get_ros2_image",
            expected_args={"topic": "/depth_image5"},
            expected_optional_args={"timeout_sec": int},
        )
        all_cameras_val = NotOrderedCallsValidator(
            subtasks=[color_image_subtask, depth_image_subtask]
        )

        tool_calls: List[Dict[str, Any]] = [
            # Reversed order - depth first, then color
            {
                "name": "get_ros2_image",
                "args": {"topic": "/depth_image5", "timeout_sec": 5},
            },
            {
                "name": "get_ros2_image",
                "args": {"topic": "/color_image5", "timeout_sec": 5},
            },
        ]

        task = GetAllROS2CamerasTask(validators=[all_cameras_val], task_args=task_args)
        score = task.validate(tool_calls)
        assert score == 1.0  # Should pass with NotOrderedCallsValidator


class TestGetPointcloudTask:
    """Test GetPointcloudTask validation."""

    def test_get_pointcloud_task_valid(self, task_args: TaskArgs) -> None:
        pointcloud_subtask = CheckArgsToolCallSubTask(
            expected_tool_name="receive_ros2_message",
            expected_args={"topic": "/pointcloud"},
            expected_optional_args={"timeout_sec": int},
        )
        pointcloud_val = OrderedCallsValidator(subtasks=[pointcloud_subtask])

        tool_calls: List[Dict[str, Any]] = [
            {"name": "get_ros2_topics_names_and_types", "args": {}},
            {
                "name": "receive_ros2_message",
                "args": {"topic": "/pointcloud", "timeout_sec": 10},
            },
        ]

        task = GetPointcloudTask(validators=[pointcloud_val], task_args=task_args)
        score = task.validate(tool_calls)
        assert score == 1.0

    def test_get_pointcloud_task_wrong_topic(self, task_args: TaskArgs) -> None:
        pointcloud_subtask = CheckArgsToolCallSubTask(
            expected_tool_name="receive_ros2_message",
            expected_args={"topic": "/pointcloud"},
            expected_optional_args={"timeout_sec": int},
        )
        pointcloud_val = OrderedCallsValidator(subtasks=[pointcloud_subtask])

        tool_calls: List[Dict[str, Any]] = [
            {
                "name": "receive_ros2_message",
                "args": {
                    "topic": "/wrong_pointcloud_topic",
                    "timeout_sec": 10,
                },  # Wrong topic
            }
        ]

        task = GetPointcloudTask(validators=[pointcloud_val], task_args=task_args)
        score = task.validate(tool_calls)
        assert score == 0.0


class TestGetRobotDescriptionTask:
    """Test GetRobotDescriptionTask validation."""

    def test_get_robot_description_task_valid(self, task_args: TaskArgs) -> None:
        robot_desc_subtask = CheckArgsToolCallSubTask(
            expected_tool_name="receive_ros2_message",
            expected_args={"topic": "/robot_description"},
            expected_optional_args={"timeout_sec": int},
        )
        robot_desc_val = OrderedCallsValidator(subtasks=[robot_desc_subtask])

        tool_calls: List[Dict[str, Any]] = [
            {"name": "get_ros2_topics_names_and_types", "args": {}},
            {
                "name": "receive_ros2_message",
                "args": {"topic": "/robot_description", "timeout_sec": 10},
            },
        ]

        task = GetRobotDescriptionTask(validators=[robot_desc_val], task_args=task_args)
        score = task.validate(tool_calls)
        assert score == 1.0

    def test_get_robot_description_task_wrong_tool_name(
        self, task_args: TaskArgs
    ) -> None:
        robot_desc_subtask = CheckArgsToolCallSubTask(
            expected_tool_name="receive_ros2_message",
            expected_args={"topic": "/robot_description"},
            expected_optional_args={"timeout_sec": int},
        )
        robot_desc_val = OrderedCallsValidator(subtasks=[robot_desc_subtask])

        tool_calls: List[Dict[str, Any]] = [
            {
                "name": "get_ros2_message",  # Wrong tool name (missing "receive_")
                "args": {"topic": "/robot_description", "timeout_sec": 10},
            }
        ]

        task = GetRobotDescriptionTask(validators=[robot_desc_val], task_args=task_args)
        score = task.validate(tool_calls)
        assert score == 0.0


class TestMultiValidatorScoring:
    """Test scoring with multiple validators to ensure proper fraction calculation."""

    def test_three_validators_all_pass(self, task_args: TaskArgs) -> None:
        # Create 3 simple validators that should all pass
        topics_subtask = CheckArgsToolCallSubTask(
            expected_tool_name="get_ros2_topics_names_and_types", expected_args={}
        )
        color_subtask = CheckArgsToolCallSubTask(
            expected_tool_name="get_ros2_image",
            expected_args={"topic": "/color_image5"},
            expected_optional_args={"timeout_sec": int},
        )
        depth_subtask = CheckArgsToolCallSubTask(
            expected_tool_name="get_ros2_image",
            expected_args={"topic": "/depth_image5"},
            expected_optional_args={"timeout_sec": int},
        )

        val1 = OrderedCallsValidator(subtasks=[topics_subtask])
        val2 = OrderedCallsValidator(subtasks=[color_subtask])
        val3 = OrderedCallsValidator(subtasks=[depth_subtask])

        tool_calls: List[Dict[str, Any]] = [
            {"name": "get_ros2_topics_names_and_types", "args": {}},
            {
                "name": "get_ros2_image",
                "args": {"topic": "/color_image5", "timeout_sec": 5},
            },
            {
                "name": "get_ros2_image",
                "args": {"topic": "/depth_image5", "timeout_sec": 5},
            },
        ]

        task = GetROS2RGBCameraTask(validators=[val1, val2, val3], task_args=task_args)
        score = task.validate(tool_calls)
        assert score == 1.0  # 3/3 validators pass

    def test_three_validators_two_pass(self, task_args: TaskArgs) -> None:
        topics_subtask = CheckArgsToolCallSubTask(
            expected_tool_name="get_ros2_topics_names_and_types", expected_args={}
        )
        color_subtask = CheckArgsToolCallSubTask(
            expected_tool_name="get_ros2_image",
            expected_args={"topic": "/color_image5"},
            expected_optional_args={"timeout_sec": int},
        )
        wrong_subtask = CheckArgsToolCallSubTask(
            expected_tool_name="get_ros2_image",
            expected_args={"topic": "/wrong_topic"},  # This will fail
            expected_optional_args={"timeout_sec": int},
        )

        val1 = OrderedCallsValidator(subtasks=[topics_subtask])
        val2 = OrderedCallsValidator(subtasks=[color_subtask])
        val3 = OrderedCallsValidator(subtasks=[wrong_subtask])

        tool_calls: List[Dict[str, Any]] = [
            {"name": "get_ros2_topics_names_and_types", "args": {}},
            {
                "name": "get_ros2_image",
                "args": {"topic": "/color_image5", "timeout_sec": 5},
            },
        ]

        task = GetROS2RGBCameraTask(validators=[val1, val2, val3], task_args=task_args)
        score = task.validate(tool_calls)
        # Should be 2/3 = 0.6666...
        assert abs(score - 0.6666666666666666) < 0.01

    def test_three_validators_one_pass(self, task_args: TaskArgs) -> None:
        topics_subtask = CheckArgsToolCallSubTask(
            expected_tool_name="get_ros2_topics_names_and_types", expected_args={}
        )
        wrong_subtask1 = CheckArgsToolCallSubTask(
            expected_tool_name="get_ros2_image",
            expected_args={"topic": "/wrong_topic1"},  # This will fail
            expected_optional_args={"timeout_sec": int},
        )
        wrong_subtask2 = CheckArgsToolCallSubTask(
            expected_tool_name="get_ros2_image",
            expected_args={"topic": "/wrong_topic2"},  # This will fail
            expected_optional_args={"timeout_sec": int},
        )

        val1 = OrderedCallsValidator(subtasks=[topics_subtask])
        val2 = OrderedCallsValidator(subtasks=[wrong_subtask1])
        val3 = OrderedCallsValidator(subtasks=[wrong_subtask2])

        tool_calls: List[Dict[str, Any]] = [
            {"name": "get_ros2_topics_names_and_types", "args": {}},
        ]

        task = GetROS2RGBCameraTask(validators=[val1, val2, val3], task_args=task_args)
        score = task.validate(tool_calls)
        # Should be 1/3 = 0.3333...
        assert abs(score - 0.3333333333333333) < 0.01

    def test_three_validators_none_pass(self, task_args: TaskArgs) -> None:
        wrong_subtask1 = CheckArgsToolCallSubTask(
            expected_tool_name="wrong_tool1", expected_args={}
        )
        wrong_subtask2 = CheckArgsToolCallSubTask(
            expected_tool_name="wrong_tool2", expected_args={}
        )
        wrong_subtask3 = CheckArgsToolCallSubTask(
            expected_tool_name="wrong_tool3", expected_args={}
        )

        val1 = OrderedCallsValidator(subtasks=[wrong_subtask1])
        val2 = OrderedCallsValidator(subtasks=[wrong_subtask2])
        val3 = OrderedCallsValidator(subtasks=[wrong_subtask3])

        tool_calls: List[Dict[str, Any]] = [
            {"name": "get_ros2_topics_names_and_types", "args": {}},
        ]

        task = GetROS2RGBCameraTask(validators=[val1, val2, val3], task_args=task_args)
        score = task.validate(tool_calls)
        assert score == 0.0  # 0/3 validators pass

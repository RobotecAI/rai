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
from rai_bench.tool_calling_agent.predefined.basic_tasks import (
    BOX1_ENTITY,
    BOX1_POSITION,
    BOX2_ENTITY,
    BOX2_POSITION,
    COLOR_IMAGE_TOPIC,
    DEFAULT_DINO_CONFIDENCE,
    DEFAULT_FPS,
    DEFAULT_PUBLISH_FREQUENCY,
    DEFAULT_SAM_CONFIDENCE,
    DELETE_ENTITY_SERVICE,
    DELETE_ENTITY_TYPE,
    DEPTH_IMAGE_TOPIC,
    DINO_CONFIDENCE_2,
    FPS_2,
    GET_PARAMETERS_TYPE,
    GET_SPAWNABLE_NAMES_SERVICE,
    GET_WORLD_PROPERTIES_TYPE,
    GROUNDED_SAM_SET_PARAMS,
    GROUNDED_SAM_SET_PARAMS_ATOMICALLY,
    GROUNDING_DINO_SET_PARAMS,
    GROUNDING_DINO_SET_PARAMS_ATOMICALLY,
    LIST_PARAMETERS_TYPE,
    O3DE_SET_PARAMS,
    POINTCLOUD_TOPIC,
    ROBOT_DESCRIPTION_TOPIC,
    ROBOT_STATE_PUBLISHER_GET_PARAMS,
    ROBOT_STATE_PUBLISHER_LIST_PARAMS,
    ROBOT_STATE_PUBLISHER_SET_PARAMS,
    SAM_CONFIDENCE_2,
    SET_PARAMETERS_ATOMICALLY_TYPE,
    SET_PARAMETERS_TYPE,
    SPAWN_ENTITY_SERVICE,
    SPAWN_ENTITY_TYPE,
    TOMATO_ENTITY,
    all_camera_images_notord_val,
    check_spawnable_entities_val,
    color_image_ord_val,
    delete_both_val,
    depth_image_ord_val,
    get_parameters_val,
    get_pointcloud_ord_val,
    get_robot_desc_ord_val,
    list_parameters_val,
    services_ord_val,
    set_grounded_dino_opt_val_1,
    set_grounded_dino_opt_val_2,
    set_grounded_sam_opt_val_1,
    set_grounded_sam_opt_val_2,
    set_o3de_fps_opt_val_1,
    set_o3de_fps_opt_val_2,
    set_param_val,
    spawn_both_val,
    spawn_entity_val,
    topics_ord_val,
)
from rai_bench.tool_calling_agent.tasks.basic import (
    CheckSpawnableEntitiesTask,
    ConfigureVisionPipelineTask,
    GetAllROS2CamerasTask,
    GetPointcloudTask,
    GetRobotDescriptionTask,
    GetROS2DepthCameraTask,
    GetROS2RGBCameraTask,
    GetROS2ServicesTask,
    GetROS2TopicsTask,
    GetSpecificParameterTask,
    ListRobotParametersTask,
    RespawnEntitiesTask,
    SetRobotParameterTask,
    SpawnEntityTask,
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
        tool_calls: List[Dict[str, Any]] = [
            {"name": "get_ros2_services_names_and_types", "args": {}},
            {
                "name": "get_ros2_message_interface",
                "args": {"msg_type": SET_PARAMETERS_TYPE},
            },
            {
                "name": "call_ros2_service",
                "args": {
                    "service_name": ROBOT_STATE_PUBLISHER_SET_PARAMS,
                    "service_type": SET_PARAMETERS_TYPE,
                    "service_args": {
                        "parameters": [
                            {
                                "name": "publish_frequency",
                                "value": {
                                    "type": "3",
                                    "double_value": DEFAULT_PUBLISH_FREQUENCY,
                                },
                            }
                        ]
                    },
                },
            },
        ]

        task = SetRobotParameterTask(
            value=DEFAULT_PUBLISH_FREQUENCY,
            validators=[set_param_val],
            task_args=task_args,
        )
        score = task.validate(tool_calls)
        assert score == 1.0  # All validators should pass

    def test_set_parameter_task_wrong_parameter_type(self, task_args: TaskArgs) -> None:
        tool_calls: List[Dict[str, Any]] = [
            {
                "name": "call_ros2_service",
                "args": {
                    "service_name": ROBOT_STATE_PUBLISHER_SET_PARAMS,
                    "service_type": SET_PARAMETERS_TYPE,
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

        task = SetRobotParameterTask(
            value=DEFAULT_PUBLISH_FREQUENCY,
            validators=[set_param_val],
            task_args=task_args,
        )
        score = task.validate(tool_calls)
        assert score == 0.0

    def test_set_parameter_task_wrong_parameter_missing_type(
        self, task_args: TaskArgs
    ) -> None:
        tool_calls: List[Dict[str, Any]] = [
            {
                "name": "call_ros2_service",
                "args": {
                    "service_name": ROBOT_STATE_PUBLISHER_SET_PARAMS,
                    "service_type": SET_PARAMETERS_TYPE,
                    "service_args": {
                        "parameters": [
                            {
                                "name": "publish_frequency",
                                "value": {
                                    # missing type field
                                    "integer_value": 30,
                                },
                            }
                        ]
                    },
                },
            },
        ]

        task = SetRobotParameterTask(
            value=DEFAULT_PUBLISH_FREQUENCY,
            validators=[set_param_val],
            task_args=task_args,
        )
        score = task.validate(tool_calls)
        assert score == 0.0

    def test_set_parameter_task_wrong_parameter_name(self, task_args: TaskArgs) -> None:
        tool_calls: List[Dict[str, Any]] = [
            {
                "name": "call_ros2_service",
                "args": {
                    "service_name": ROBOT_STATE_PUBLISHER_SET_PARAMS,
                    "service_type": SET_PARAMETERS_TYPE,
                    "service_args": {
                        "parameters": [
                            {
                                "name": "wrong_parameter_name",  # Wrong parameter name
                                "value": {
                                    "type": "3",
                                    "double_value": DEFAULT_PUBLISH_FREQUENCY,
                                },
                            }
                        ]
                    },
                },
            },
        ]

        task = SetRobotParameterTask(
            value=DEFAULT_PUBLISH_FREQUENCY,
            validators=[set_param_val],
            task_args=task_args,
        )
        score = task.validate(tool_calls)
        assert score == 0.0

    def test_set_parameter_task_wrong_tools(self, task_args: TaskArgs) -> None:
        tool_calls: List[Dict[str, Any]] = [
            {"name": "get_ros2_services_names_and_types", "args": {}},
            {
                "name": "get_ros2_message_interface",
                "args": {"msg_type": SET_PARAMETERS_TYPE},
            },
            {"name": "get_ros2_services_names_and_types", "args": {}},
            {"name": "get_ros2_services_names_and_types", "args": {}},
        ]

        task = SetRobotParameterTask(
            value=DEFAULT_PUBLISH_FREQUENCY,
            validators=[set_param_val],
            task_args=task_args,
        )
        score = task.validate(tool_calls)
        assert score == 0.0


class TestGetTopicsTask:
    """Test GetROS2TopicsTask validation."""

    def test_get_topics_task_valid(self, task_args: TaskArgs) -> None:
        tool_calls: List[Dict[str, Any]] = [
            {"name": "get_ros2_topics_names_and_types", "args": {}}
        ]

        task = GetROS2TopicsTask(validators=[topics_ord_val], task_args=task_args)
        score = task.validate(tool_calls)
        assert score == 1.0

    def test_get_topics_task_wrong_tool(self, task_args: TaskArgs) -> None:
        """Test get ROS2 topics task with wrong tool name."""
        tool_calls: List[Dict[str, Any]] = [
            {"name": "wrong_tool_name", "args": {}}  # Wrong tool name
        ]

        task = GetROS2TopicsTask(validators=[topics_ord_val], task_args=task_args)
        score = task.validate(tool_calls)
        assert score == 0.0

    def test_get_topics_task_unexpected_args(self, task_args: TaskArgs) -> None:
        tool_calls: List[Dict[str, Any]] = [
            {
                "name": "get_ros2_topics_names_and_types",
                "args": {"unexpected": "arg"},
            }  # Unexpected args
        ]

        task = GetROS2TopicsTask(validators=[topics_ord_val], task_args=task_args)
        score = task.validate(tool_calls)
        assert score == 0.0


class TestGetRGBCameraTask:
    """Test GetROS2RGBCameraTask validation."""

    def test_get_rgb_camera_task_valid(self, task_args: TaskArgs) -> None:
        tool_calls: List[Dict[str, Any]] = [
            {"name": "get_ros2_topics_names_and_types", "args": {}},
            {
                "name": "get_ros2_image",
                "args": {"topic": COLOR_IMAGE_TOPIC, "timeout_sec": 5},
            },
        ]

        task = GetROS2RGBCameraTask(
            validators=[color_image_ord_val], task_args=task_args
        )
        score = task.validate(tool_calls)
        assert score == 1.0

    def test_get_rgb_camera_task_wrong_topic(self, task_args: TaskArgs) -> None:
        tool_calls: List[Dict[str, Any]] = [
            {
                "name": "get_ros2_image",
                "args": {"topic": "/wrong_topic", "timeout_sec": 5},  # Wrong topic
            }
        ]

        task = GetROS2RGBCameraTask(
            validators=[color_image_ord_val], task_args=task_args
        )
        score = task.validate(tool_calls)
        assert score == 0.0

    def test_get_rgb_camera_task_missing_required_arg(
        self, task_args: TaskArgs
    ) -> None:
        tool_calls: List[Dict[str, Any]] = [
            {
                "name": "get_ros2_image",
                "args": {"timeout_sec": 5},  # Missing required topic arg
            }
        ]

        task = GetROS2RGBCameraTask(
            validators=[color_image_ord_val], task_args=task_args
        )
        score = task.validate(tool_calls)
        assert score == 0.0

    def test_get_rgb_camera_task_wrong_timeout_type(self, task_args: TaskArgs) -> None:
        tool_calls: List[Dict[str, Any]] = [
            {
                "name": "get_ros2_image",
                "args": {
                    "topic": COLOR_IMAGE_TOPIC,
                    "timeout_sec": "not_an_int",
                },  # Wrong type
            }
        ]

        task = GetROS2RGBCameraTask(
            validators=[color_image_ord_val], task_args=task_args
        )
        score = task.validate(tool_calls)
        assert score == 0.0


class TestGetDepthCameraTask:
    """Test GetROS2DepthCameraTask validation."""

    def test_get_depth_camera_task_valid(self, task_args: TaskArgs) -> None:
        tool_calls: List[Dict[str, Any]] = [
            {"name": "get_ros2_topics_names_and_types", "args": {}},
            {
                "name": "get_ros2_image",
                "args": {"topic": DEPTH_IMAGE_TOPIC, "timeout_sec": 5},
            },
        ]

        task = GetROS2DepthCameraTask(
            validators=[depth_image_ord_val], task_args=task_args
        )
        score = task.validate(tool_calls)
        assert score == 1.0

    def test_get_depth_camera_task_wrong_topic(self, task_args: TaskArgs) -> None:
        tool_calls: List[Dict[str, Any]] = [
            {
                "name": "get_ros2_image",
                "args": {
                    "topic": COLOR_IMAGE_TOPIC,  # Wrong topic (color instead of depth)
                    "timeout_sec": 5,
                },
            }
        ]

        task = GetROS2DepthCameraTask(
            validators=[depth_image_ord_val], task_args=task_args
        )
        score = task.validate(tool_calls)
        assert score == 0.0


class TestGetAllCamerasTask:
    """Test GetAllROS2CamerasTask validation."""

    def test_get_all_cameras_task_valid(self, task_args: TaskArgs) -> None:
        tool_calls: List[Dict[str, Any]] = [
            {"name": "get_ros2_topics_names_and_types", "args": {}},
            {
                "name": "get_ros2_image",
                "args": {"topic": COLOR_IMAGE_TOPIC, "timeout_sec": 5},
            },
            {
                "name": "get_ros2_image",
                "args": {"topic": DEPTH_IMAGE_TOPIC, "timeout_sec": 5},
            },
        ]

        task = GetAllROS2CamerasTask(
            validators=[all_camera_images_notord_val], task_args=task_args
        )
        score = task.validate(tool_calls)
        assert score == 1.0

    def test_get_all_cameras_task_missing_depth(self, task_args: TaskArgs) -> None:
        tool_calls: List[Dict[str, Any]] = [
            {
                "name": "get_ros2_image",
                "args": {"topic": COLOR_IMAGE_TOPIC, "timeout_sec": 5},
            }
            # Missing depth camera call
        ]

        task = GetAllROS2CamerasTask(
            validators=[all_camera_images_notord_val], task_args=task_args
        )
        score = task.validate(tool_calls)
        assert score == 0.0

    def test_get_all_cameras_task_wrong_order_should_pass(
        self, task_args: TaskArgs
    ) -> None:
        tool_calls: List[Dict[str, Any]] = [
            # Reversed order - depth first, then color
            {
                "name": "get_ros2_image",
                "args": {"topic": DEPTH_IMAGE_TOPIC, "timeout_sec": 5},
            },
            {
                "name": "get_ros2_image",
                "args": {"topic": COLOR_IMAGE_TOPIC, "timeout_sec": 5},
            },
        ]

        task = GetAllROS2CamerasTask(
            validators=[all_camera_images_notord_val], task_args=task_args
        )
        score = task.validate(tool_calls)
        assert score == 1.0  # Should pass with NotOrderedCallsValidator


class TestGetPointcloudTask:
    """Test GetPointcloudTask validation."""

    def test_get_pointcloud_task_valid(self, task_args: TaskArgs) -> None:
        tool_calls: List[Dict[str, Any]] = [
            {"name": "get_ros2_topics_names_and_types", "args": {}},
            {
                "name": "receive_ros2_message",
                "args": {"topic": POINTCLOUD_TOPIC, "timeout_sec": 10},
            },
        ]

        task = GetPointcloudTask(
            validators=[get_pointcloud_ord_val], task_args=task_args
        )
        score = task.validate(tool_calls)
        assert score == 1.0

    def test_get_pointcloud_task_wrong_topic(self, task_args: TaskArgs) -> None:
        tool_calls: List[Dict[str, Any]] = [
            {
                "name": "receive_ros2_message",
                "args": {
                    "topic": "/wrong_pointcloud_topic",
                    "timeout_sec": 10,
                },  # Wrong topic
            }
        ]

        task = GetPointcloudTask(
            validators=[get_pointcloud_ord_val], task_args=task_args
        )
        score = task.validate(tool_calls)
        assert score == 0.0


class TestGetRobotDescriptionTask:
    """Test GetRobotDescriptionTask validation."""

    def test_get_robot_description_task_valid(self, task_args: TaskArgs) -> None:
        tool_calls: List[Dict[str, Any]] = [
            {"name": "get_ros2_topics_names_and_types", "args": {}},
            {
                "name": "receive_ros2_message",
                "args": {"topic": ROBOT_DESCRIPTION_TOPIC, "timeout_sec": 10},
            },
        ]

        task = GetRobotDescriptionTask(
            validators=[get_robot_desc_ord_val], task_args=task_args
        )
        score = task.validate(tool_calls)
        assert score == 1.0

    def test_get_robot_description_task_wrong_tool_name(
        self, task_args: TaskArgs
    ) -> None:
        tool_calls: List[Dict[str, Any]] = [
            {
                "name": "get_ros2_message",  # Wrong tool name (missing "receive_")
                "args": {"topic": ROBOT_DESCRIPTION_TOPIC, "timeout_sec": 10},
            }
        ]

        task = GetRobotDescriptionTask(
            validators=[get_robot_desc_ord_val], task_args=task_args
        )
        score = task.validate(tool_calls)
        assert score == 0.0


class TestGetROS2ServicesTask:
    """Test GetROS2ServicesTask validation."""

    def test_get_services_task_valid(self, task_args: TaskArgs) -> None:
        """Test get ROS2 services task with valid call."""
        tool_calls: List[Dict[str, Any]] = [
            {"name": "get_ros2_services_names_and_types", "args": {}}
        ]

        task = GetROS2ServicesTask(validators=[services_ord_val], task_args=task_args)
        score = task.validate(tool_calls)
        assert score == 1.0

    def test_get_services_task_wrong_tool_name(self, task_args: TaskArgs) -> None:
        """Test get ROS2 services task with wrong tool name."""
        tool_calls: List[Dict[str, Any]] = [
            {"name": "wrong_tool_name", "args": {}}  # Wrong tool name
        ]

        task = GetROS2ServicesTask(validators=[services_ord_val], task_args=task_args)
        score = task.validate(tool_calls)
        assert score == 0.0

    def test_get_services_task_unexpected_args(self, task_args: TaskArgs) -> None:
        """Test get ROS2 services task with unexpected arguments."""
        tool_calls: List[Dict[str, Any]] = [
            {
                "name": "get_ros2_services_names_and_types",
                "args": {"unexpected": "arg"},
            }  # Unexpected args
        ]

        task = GetROS2ServicesTask(validators=[services_ord_val], task_args=task_args)
        score = task.validate(tool_calls)
        assert score == 0.0


class TestListRobotParametersTask:
    """Test ListRobotParametersTask validation."""

    def test_list_parameters_task_valid(self, task_args: TaskArgs) -> None:
        """Test list parameters task with valid service call."""
        tool_calls: List[Dict[str, Any]] = [
            {"name": "get_ros2_services_names_and_types", "args": {}},
            {
                "name": "call_ros2_service",
                "args": {
                    "service_name": ROBOT_STATE_PUBLISHER_LIST_PARAMS,
                    "service_type": LIST_PARAMETERS_TYPE,
                    "service_args": {},
                },
            },
        ]

        task = ListRobotParametersTask(
            validators=[list_parameters_val], task_args=task_args
        )
        score = task.validate(tool_calls)
        assert score == 1.0

    def test_list_parameters_task_wrong_service_name(self, task_args: TaskArgs) -> None:
        """Test list parameters task with wrong service name."""
        tool_calls: List[Dict[str, Any]] = [
            {
                "name": "call_ros2_service",
                "args": {
                    "service_name": "/wrong_node/list_parameters",  # Wrong service name
                    "service_type": LIST_PARAMETERS_TYPE,
                    "service_args": {},
                },
            }
        ]

        task = ListRobotParametersTask(
            validators=[list_parameters_val], task_args=task_args
        )
        score = task.validate(tool_calls)
        assert score == 0.0

    def test_list_parameters_task_wrong_tool_name(self, task_args: TaskArgs) -> None:
        """Test list parameters task with wrong tool name."""
        tool_calls: List[Dict[str, Any]] = [
            {
                "name": "wrong_tool_name",  # Wrong tool name
                "args": {
                    "service_name": ROBOT_STATE_PUBLISHER_LIST_PARAMS,
                    "service_type": LIST_PARAMETERS_TYPE,
                    "service_args": {},
                },
            }
        ]

        task = ListRobotParametersTask(
            validators=[list_parameters_val], task_args=task_args
        )
        score = task.validate(tool_calls)
        assert score == 0.0


class TestGetSpecificParameterTask:
    """Test GetSpecificParameterTask validation."""

    def test_get_parameter_task_valid(self, task_args: TaskArgs) -> None:
        """Test get specific parameter task with valid service call."""
        tool_calls: List[Dict[str, Any]] = [
            {"name": "get_ros2_services_names_and_types", "args": {}},
            {
                "name": "get_ros2_message_interface",
                "args": {"msg_type": GET_PARAMETERS_TYPE},
            },
            {
                "name": "call_ros2_service",
                "args": {
                    "service_name": ROBOT_STATE_PUBLISHER_GET_PARAMS,
                    "service_type": GET_PARAMETERS_TYPE,
                    "service_args": {"names": ["publish_frequency"]},
                },
            },
        ]

        task = GetSpecificParameterTask(
            parameter="publish_frequency",
            validators=[get_parameters_val],
            task_args=task_args,
        )
        score = task.validate(tool_calls)
        assert score == 1.0

    def test_get_parameter_task_wrong_parameter_name(self, task_args: TaskArgs) -> None:
        """Test get specific parameter task with wrong parameter name."""
        tool_calls: List[Dict[str, Any]] = [
            {
                "name": "call_ros2_service",
                "args": {
                    "service_name": ROBOT_STATE_PUBLISHER_GET_PARAMS,
                    "service_type": GET_PARAMETERS_TYPE,
                    "service_args": {
                        "names": ["wrong_parameter_name"]  # Wrong parameter name
                    },
                },
            }
        ]

        task = GetSpecificParameterTask(
            parameter="publish_frequency",
            validators=[get_parameters_val],
            task_args=task_args,
        )
        score = task.validate(tool_calls)
        assert score == 0.0

    def test_get_parameter_task_missing_names_field(self, task_args: TaskArgs) -> None:
        """Test get specific parameter task with missing names field."""
        tool_calls: List[Dict[str, Any]] = [
            {
                "name": "call_ros2_service",
                "args": {
                    "service_name": ROBOT_STATE_PUBLISHER_GET_PARAMS,
                    "service_type": GET_PARAMETERS_TYPE,
                    "service_args": {},  # Missing names field
                },
            }
        ]

        task = GetSpecificParameterTask(
            parameter="publish_frequency",
            validators=[get_parameters_val],
            task_args=task_args,
        )
        score = task.validate(tool_calls)
        assert score == 0.0


class TestCheckSpawnableEntitiesTask:
    """Test CheckSpawnableEntitiesTask validation."""

    def test_check_spawnable_entities_task_valid(self, task_args: TaskArgs) -> None:
        """Test check spawnable entities task with valid service call."""
        tool_calls: List[Dict[str, Any]] = [
            {"name": "get_ros2_services_names_and_types", "args": {}},
            {
                "name": "call_ros2_service",
                "args": {
                    "service_name": GET_SPAWNABLE_NAMES_SERVICE,
                    "service_type": GET_WORLD_PROPERTIES_TYPE,
                    "service_args": {},
                },
            },
        ]

        task = CheckSpawnableEntitiesTask(
            validators=[check_spawnable_entities_val], task_args=task_args
        )
        score = task.validate(tool_calls)
        assert score == 1.0

    def test_check_spawnable_entities_task_wrong_service_name(
        self, task_args: TaskArgs
    ) -> None:
        """Test check spawnable entities task with wrong service name."""
        tool_calls: List[Dict[str, Any]] = [
            {
                "name": "call_ros2_service",
                "args": {
                    "service_name": "/wrong_service_name",  # Wrong service name
                    "service_type": GET_WORLD_PROPERTIES_TYPE,
                    "service_args": {},
                },
            }
        ]

        task = CheckSpawnableEntitiesTask(
            validators=[check_spawnable_entities_val], task_args=task_args
        )
        score = task.validate(tool_calls)
        assert score == 0.0

    def test_check_spawnable_entities_task_wrong_tool_name(
        self, task_args: TaskArgs
    ) -> None:
        """Test check spawnable entities task with wrong tool name."""
        tool_calls: List[Dict[str, Any]] = [
            {
                "name": "wrong_tool_name",  # Wrong tool name
                "args": {
                    "service_name": GET_SPAWNABLE_NAMES_SERVICE,
                    "service_type": GET_WORLD_PROPERTIES_TYPE,
                    "service_args": {},
                },
            }
        ]

        task = CheckSpawnableEntitiesTask(
            validators=[check_spawnable_entities_val], task_args=task_args
        )
        score = task.validate(tool_calls)
        assert score == 0.0


class TestSpawnEntityTask:
    """Test SpawnEntityTask validation."""

    def test_spawn_entity_task_valid_tomato(self, task_args: TaskArgs) -> None:
        """Test spawn entity task with tomato entity."""
        tool_calls: List[Dict[str, Any]] = [
            {"name": "get_ros2_services_names_and_types", "args": {}},
            {
                "name": "get_ros2_message_interface",
                "args": {"msg_type": SPAWN_ENTITY_TYPE},
            },
            {
                "name": "call_ros2_service",
                "args": {
                    "service_name": SPAWN_ENTITY_SERVICE,
                    "service_type": SPAWN_ENTITY_TYPE,
                    "service_args": {
                        "name": TOMATO_ENTITY,
                        "xml": "<sdf>tomato model</sdf>",
                    },
                },
            },
        ]

        task = SpawnEntityTask(
            entity=TOMATO_ENTITY, validators=[spawn_entity_val], task_args=task_args
        )
        score = task.validate(tool_calls)
        assert score == 1.0

    def test_spawn_entity_task_wrong_service_name(self, task_args: TaskArgs) -> None:
        """Test spawn entity task with wrong service name."""
        tool_calls: List[Dict[str, Any]] = [
            {
                "name": "call_ros2_service",
                "args": {
                    "service_name": "/wrong_spawn_service",  # Wrong service name
                    "service_type": SPAWN_ENTITY_TYPE,
                    "service_args": {
                        "name": TOMATO_ENTITY,
                        "xml": "<sdf>test</sdf>",
                    },
                },
            }
        ]

        task = SpawnEntityTask(
            entity=TOMATO_ENTITY, validators=[spawn_entity_val], task_args=task_args
        )
        score = task.validate(tool_calls)
        assert score == 0.0

    def test_spawn_entity_task_wrong_entity_name(self, task_args: TaskArgs) -> None:
        """Test spawn entity task with wrong entity name."""
        tool_calls: List[Dict[str, Any]] = [
            {
                "name": "call_ros2_service",
                "args": {
                    "service_name": SPAWN_ENTITY_SERVICE,
                    "service_type": SPAWN_ENTITY_TYPE,
                    "service_args": {
                        "name": "wrong_entity_name",  # Wrong entity name
                        "xml": "<sdf>test</sdf>",
                    },
                },
            }
        ]

        task = SpawnEntityTask(
            entity=TOMATO_ENTITY, validators=[spawn_entity_val], task_args=task_args
        )
        score = task.validate(tool_calls)
        assert score == 0.0

    def test_spawn_entity_task_missing_service_args(self, task_args: TaskArgs) -> None:
        """Test spawn entity task with missing service args."""
        tool_calls: List[Dict[str, Any]] = [
            {
                "name": "call_ros2_service",
                "args": {
                    "service_name": SPAWN_ENTITY_SERVICE,
                    "service_type": SPAWN_ENTITY_TYPE,
                    # Missing service_args
                },
            }
        ]

        task = SpawnEntityTask(
            entity=TOMATO_ENTITY, validators=[spawn_entity_val], task_args=task_args
        )
        score = task.validate(tool_calls)
        assert score == 0.0

    def test_spawn_entity_task_wrong_tool_name(self, task_args: TaskArgs) -> None:
        """Test spawn entity task with wrong tool name."""
        tool_calls: List[Dict[str, Any]] = [
            {
                "name": "wrong_tool_name",  # Wrong tool name
                "args": {
                    "service_name": SPAWN_ENTITY_SERVICE,
                    "service_type": SPAWN_ENTITY_TYPE,
                    "service_args": {
                        "name": TOMATO_ENTITY,
                        "xml": "<sdf>test</sdf>",
                    },
                },
            }
        ]

        task = SpawnEntityTask(
            entity=TOMATO_ENTITY, validators=[spawn_entity_val], task_args=task_args
        )
        score = task.validate(tool_calls)
        assert score == 0.0


class TestConfigureVisionPipelineTask:
    """Test ConfigureVisionPipelineTask validation."""

    def test_configure_vision_pipeline_task_valid_config1(
        self, task_args: TaskArgs
    ) -> None:
        """Test configure vision pipeline task with first configuration."""
        tool_calls: List[Dict[str, Any]] = [
            {"name": "get_ros2_services_names_and_types", "args": {}},
            {
                "name": "get_ros2_message_interface",
                "args": {"msg_type": SET_PARAMETERS_TYPE},
            },
            {
                "name": "call_ros2_service",
                "args": {
                    "service_name": GROUNDED_SAM_SET_PARAMS_ATOMICALLY,
                    "service_type": SET_PARAMETERS_ATOMICALLY_TYPE,
                    "service_args": {
                        "parameters": [
                            {
                                "name": "confidence_threshold",
                                "value": {
                                    "type": 3,
                                    "double_value": DEFAULT_SAM_CONFIDENCE,
                                },
                            }
                        ]
                    },
                },
            },
            {
                "name": "call_ros2_service",
                "args": {
                    "service_name": GROUNDING_DINO_SET_PARAMS_ATOMICALLY,
                    "service_type": SET_PARAMETERS_ATOMICALLY_TYPE,
                    "service_args": {
                        "parameters": [
                            {
                                "name": "confidence_threshold",
                                "value": {
                                    "type": 3,
                                    "double_value": DEFAULT_DINO_CONFIDENCE,
                                },
                            }
                        ]
                    },
                },
            },
            {
                "name": "call_ros2_service",
                "args": {
                    "service_name": O3DE_SET_PARAMS,
                    "service_type": SET_PARAMETERS_TYPE,
                    "service_args": {
                        "parameters": [
                            {
                                "name": "fps",
                                "value": {"type": 2, "integer_value": DEFAULT_FPS},
                            }
                        ]
                    },
                },
            },
        ]

        task = ConfigureVisionPipelineTask(
            sam_confidence_threshold=DEFAULT_SAM_CONFIDENCE,
            dino_confidence_threshold=DEFAULT_DINO_CONFIDENCE,
            fps=DEFAULT_FPS,
            validators=[
                set_grounded_sam_opt_val_1,
                set_grounded_dino_opt_val_1,
                set_o3de_fps_opt_val_1,
            ],
            task_args=task_args,
        )
        score = task.validate(tool_calls)
        assert score == 1.0

    def test_configure_vision_pipeline_task_valid_config2(
        self, task_args: TaskArgs
    ) -> None:
        """Test configure vision pipeline task with second configuration."""
        tool_calls: List[Dict[str, Any]] = [
            {"name": "get_ros2_services_names_and_types", "args": {}},
            {
                "name": "get_ros2_message_interface",
                "args": {"msg_type": SET_PARAMETERS_TYPE},
            },
            {
                "name": "call_ros2_service",
                "args": {
                    "service_name": GROUNDED_SAM_SET_PARAMS,
                    "service_type": SET_PARAMETERS_TYPE,
                    "service_args": {
                        "parameters": [
                            {
                                "name": "confidence_threshold",
                                "value": {
                                    "type": 3,
                                    "double_value": SAM_CONFIDENCE_2,
                                },
                            }
                        ]
                    },
                },
            },
            {
                "name": "call_ros2_service",
                "args": {
                    "service_name": GROUNDING_DINO_SET_PARAMS,
                    "service_type": SET_PARAMETERS_TYPE,
                    "service_args": {
                        "parameters": [
                            {
                                "name": "confidence_threshold",
                                "value": {
                                    "type": 3,
                                    "double_value": DINO_CONFIDENCE_2,
                                },
                            }
                        ]
                    },
                },
            },
            {
                "name": "call_ros2_service",
                "args": {
                    "service_name": O3DE_SET_PARAMS,
                    "service_type": SET_PARAMETERS_TYPE,
                    "service_args": {
                        "parameters": [
                            {
                                "name": "fps",
                                "value": {"type": 2, "integer_value": FPS_2},
                            }
                        ]
                    },
                },
            },
        ]

        task = ConfigureVisionPipelineTask(
            sam_confidence_threshold=SAM_CONFIDENCE_2,
            dino_confidence_threshold=DINO_CONFIDENCE_2,
            fps=FPS_2,
            validators=[
                set_grounded_sam_opt_val_2,
                set_grounded_dino_opt_val_2,
                set_o3de_fps_opt_val_2,
            ],
            task_args=task_args,
        )
        score = task.validate(tool_calls)
        assert score == 1.0

    def test_configure_vision_pipeline_task_missing_calls(
        self, task_args: TaskArgs
    ) -> None:
        """Test configure vision pipeline task with missing service calls."""

        tool_calls: List[Dict[str, Any]] = [
            {
                "name": "call_ros2_service",
                "args": {
                    "service_name": GROUNDED_SAM_SET_PARAMS,
                    "service_type": SET_PARAMETERS_TYPE,
                    "service_args": {
                        "parameters": [
                            {
                                "name": "confidence_threshold",
                                "value": {
                                    "type": 3,
                                    "double_value": DEFAULT_SAM_CONFIDENCE,
                                },
                            }
                        ]
                    },
                },
            }
        ]

        task = ConfigureVisionPipelineTask(
            sam_confidence_threshold=DEFAULT_SAM_CONFIDENCE,
            dino_confidence_threshold=DEFAULT_DINO_CONFIDENCE,
            fps=DEFAULT_FPS,
            validators=[
                set_grounded_sam_opt_val_1,
                set_grounded_dino_opt_val_1,
                set_o3de_fps_opt_val_1,
            ],
            task_args=task_args,
        )
        score = task.validate(tool_calls)
        assert abs(score - 0.3333333333333333) < 0.01

    def test_configure_vision_pipeline_task_setting_in_one_call(
        self, task_args: TaskArgs
    ) -> None:
        tool_calls: List[Dict[str, Any]] = [
            {
                "name": "call_ros2_service",
                "args": {
                    "service_name": GROUNDED_SAM_SET_PARAMS,
                    "service_type": SET_PARAMETERS_TYPE,
                    "service_args": {
                        "parameters": [
                            {
                                "name": "confidence_threshold",
                                "value": {
                                    "type": 3,
                                    "double_value": DEFAULT_SAM_CONFIDENCE,
                                },
                            },
                            {
                                "name": "fps",
                                "value": {"type": 2, "integer_value": FPS_2},
                            },
                            {
                                "name": "confidence_threshold",
                                "value": {
                                    "type": 3,
                                    "double_value": DEFAULT_DINO_CONFIDENCE,
                                },
                            },
                        ]
                    },
                },
            }
        ]

        task = ConfigureVisionPipelineTask(
            sam_confidence_threshold=DEFAULT_SAM_CONFIDENCE,
            dino_confidence_threshold=DEFAULT_DINO_CONFIDENCE,
            fps=DEFAULT_FPS,
            validators=[
                set_grounded_sam_opt_val_1,
                set_grounded_dino_opt_val_1,
                set_o3de_fps_opt_val_1,
            ],
            task_args=task_args,
        )
        score = task.validate(tool_calls)
        assert abs(score - 0.3333333333333333) < 0.01

    def test_configure_vision_pipeline_task_empty_call(
        self, task_args: TaskArgs
    ) -> None:
        tool_calls: List[Dict[str, Any]] = [
            {
                "name": "call_ros2_service",
                "args": {
                    "service_name": GROUNDED_SAM_SET_PARAMS,
                    "service_type": SET_PARAMETERS_TYPE,
                    "service_args": {"parameters": []},
                },
            }
        ]

        task = ConfigureVisionPipelineTask(
            sam_confidence_threshold=DEFAULT_SAM_CONFIDENCE,
            dino_confidence_threshold=DEFAULT_DINO_CONFIDENCE,
            fps=DEFAULT_FPS,
            validators=[
                set_grounded_sam_opt_val_1,
                set_grounded_dino_opt_val_1,
                set_o3de_fps_opt_val_1,
            ],
            task_args=task_args,
        )
        score = task.validate(tool_calls)
        assert score == 0


class TestRespawnEntitiesTask:
    """Test RespawnEntitiesTask validation."""

    def test_respawn_entities_task_valid(self, task_args: TaskArgs) -> None:
        """Test respawn entities task with valid calls."""

        tool_calls: List[Dict[str, Any]] = [
            {"name": "get_ros2_services_names_and_types", "args": {}},
            {
                "name": "get_ros2_message_interface",
                "args": {"msg_type": DELETE_ENTITY_TYPE},
            },
            {
                "name": "get_ros2_message_interface",
                "args": {"msg_type": SPAWN_ENTITY_TYPE},
            },
            {
                "name": "call_ros2_service",
                "args": {
                    "service_name": DELETE_ENTITY_SERVICE,
                    "service_type": DELETE_ENTITY_TYPE,
                    "service_args": {"name": BOX1_ENTITY},
                },
            },
            {
                "name": "call_ros2_service",
                "args": {
                    "service_name": DELETE_ENTITY_SERVICE,
                    "service_type": DELETE_ENTITY_TYPE,
                    "service_args": {"name": BOX2_ENTITY},
                },
            },
            {
                "name": "call_ros2_service",
                "args": {
                    "service_name": SPAWN_ENTITY_SERVICE,
                    "service_type": SPAWN_ENTITY_TYPE,
                    "service_args": {
                        "name": BOX1_ENTITY,
                        "xml": "<sdf>box1 model</sdf>",
                        "initial_pose": {
                            "position": {
                                "x": BOX1_POSITION[0],
                                "y": BOX1_POSITION[1],
                                "z": BOX1_POSITION[2],
                            }
                        },
                    },
                },
            },
            {
                "name": "call_ros2_service",
                "args": {
                    "service_name": SPAWN_ENTITY_SERVICE,
                    "service_type": SPAWN_ENTITY_TYPE,
                    "service_args": {
                        "name": BOX2_ENTITY,
                        "xml": "<sdf>box2 model</sdf>",
                        "initial_pose": {
                            "position": {
                                "x": BOX2_POSITION[0],
                                "y": BOX2_POSITION[1],
                                "z": BOX2_POSITION[2],
                            }
                        },
                    },
                },
            },
        ]

        task = RespawnEntitiesTask(
            names=[BOX1_ENTITY, BOX2_ENTITY],
            coords=[BOX1_POSITION, BOX2_POSITION],
            validators=[delete_both_val, spawn_both_val],
            task_args=task_args,
        )
        score = task.validate(tool_calls)
        assert score == 1.0

    def test_respawn_entities_task_missing_delete(self, task_args: TaskArgs) -> None:
        """Test respawn entities task with missing delete calls."""
        tool_calls: List[Dict[str, Any]] = [
            {
                "name": "call_ros2_service",
                "args": {
                    "service_name": DELETE_ENTITY_SERVICE,
                    "service_type": DELETE_ENTITY_TYPE,
                    "service_args": {"name": BOX1_ENTITY},
                },
            },
            {
                "name": "call_ros2_service",
                "args": {
                    "service_name": SPAWN_ENTITY_SERVICE,
                    "service_type": SPAWN_ENTITY_TYPE,
                    "service_args": {
                        "name": BOX1_ENTITY,
                        "xml": "<sdf>box1 model</sdf>",
                        "initial_pose": {
                            "position": {
                                "x": BOX1_POSITION[0],
                                "y": BOX1_POSITION[1],
                                "z": BOX1_POSITION[2],
                            }
                        },
                    },
                },
            },
            {
                "name": "call_ros2_service",
                "args": {
                    "service_name": SPAWN_ENTITY_SERVICE,
                    "service_type": SPAWN_ENTITY_TYPE,
                    "service_args": {
                        "name": BOX2_ENTITY,
                        "xml": "<sdf>box2 model</sdf>",
                        "initial_pose": {
                            "position": {
                                "x": BOX2_POSITION[0],
                                "y": BOX2_POSITION[1],
                                "z": BOX2_POSITION[2],
                            }
                        },
                    },
                },
            },
        ]

        task = RespawnEntitiesTask(
            names=[BOX1_ENTITY, BOX2_ENTITY],
            coords=[BOX1_POSITION, BOX2_POSITION],
            validators=[delete_both_val, spawn_both_val],
            task_args=task_args,
        )
        score = task.validate(tool_calls)
        assert score == 0.0

    def test_respawn_entities_task_missing_spawn(self, task_args: TaskArgs) -> None:
        """Test respawn entities task with missing spawn calls."""

        tool_calls: List[Dict[str, Any]] = [
            {
                "name": "call_ros2_service",
                "args": {
                    "service_name": DELETE_ENTITY_SERVICE,
                    "service_type": DELETE_ENTITY_TYPE,
                    "service_args": {"name": BOX1_ENTITY},
                },
            },
            {
                "name": "call_ros2_service",
                "args": {
                    "service_name": DELETE_ENTITY_SERVICE,
                    "service_type": DELETE_ENTITY_TYPE,
                    "service_args": {"name": BOX2_ENTITY},
                },
            },
            {
                "name": "call_ros2_service",
                "args": {
                    "service_name": SPAWN_ENTITY_SERVICE,
                    "service_type": SPAWN_ENTITY_TYPE,
                    "service_args": {
                        "name": BOX1_ENTITY,
                        "xml": "<sdf>box1 model</sdf>",
                        "initial_pose": {
                            "position": {
                                "x": BOX1_POSITION[0],
                                "y": BOX1_POSITION[1],
                                "z": BOX1_POSITION[2],
                            }
                        },
                    },
                },
            },
        ]

        task = RespawnEntitiesTask(
            names=[BOX1_ENTITY, BOX2_ENTITY],
            coords=[BOX1_POSITION, BOX2_POSITION],
            validators=[delete_both_val, spawn_both_val],
            task_args=task_args,
        )
        score = task.validate(tool_calls)
        assert score == 0.5


class TestMultiValidatorScoring:
    """Test scoring with multiple validators to ensure proper fraction calculation."""

    def test_three_validators_all_pass(self, task_args: TaskArgs) -> None:
        tool_calls: List[Dict[str, Any]] = [
            {"name": "get_ros2_topics_names_and_types", "args": {}},
            {
                "name": "get_ros2_image",
                "args": {"topic": COLOR_IMAGE_TOPIC, "timeout_sec": 5},
            },
            {
                "name": "get_ros2_image",
                "args": {"topic": DEPTH_IMAGE_TOPIC, "timeout_sec": 5},
            },
        ]

        # Create multiple validators for testing

        val1 = topics_ord_val
        val2 = color_image_ord_val
        val3 = depth_image_ord_val

        task = GetROS2RGBCameraTask(validators=[val1, val2, val3], task_args=task_args)
        score = task.validate(tool_calls)
        assert score == 1.0  # 3/3 validators pass

    def test_three_validators_two_pass(self, task_args: TaskArgs) -> None:
        tool_calls: List[Dict[str, Any]] = [
            {"name": "get_ros2_topics_names_and_types", "args": {}},
            {
                "name": "get_ros2_image",
                "args": {"topic": COLOR_IMAGE_TOPIC, "timeout_sec": 5},
            },
        ]

        # Create multiple validators for testing
        val1 = topics_ord_val
        val2 = color_image_ord_val
        val3 = depth_image_ord_val

        task = GetROS2RGBCameraTask(validators=[val1, val2, val3], task_args=task_args)
        score = task.validate(tool_calls)
        # Should be 2/3 = 0.6666...
        assert abs(score - 0.6666666666666666) < 0.01

    def test_three_validators_one_pass(self, task_args: TaskArgs) -> None:
        tool_calls: List[Dict[str, Any]] = [
            {"name": "get_ros2_topics_names_and_types", "args": {}},
        ]

        # Create multiple validators for testing
        val1 = topics_ord_val
        val2 = color_image_ord_val
        val3 = depth_image_ord_val

        task = GetROS2RGBCameraTask(validators=[val1, val2, val3], task_args=task_args)
        score = task.validate(tool_calls)
        # Should be 1/3 = 0.3333...
        assert abs(score - 0.3333333333333333) < 0.01

    def test_three_validators_none_pass(self, task_args: TaskArgs) -> None:
        tool_calls: List[Dict[str, Any]] = [
            {
                "name": "get_ros2_image",
                "args": {"topic": COLOR_IMAGE_TOPIC, "timeout_sec": 5},
            },
        ]

        # Create multiple validators for testing
        val1 = topics_ord_val
        val2 = color_image_ord_val
        val3 = depth_image_ord_val

        task = GetROS2RGBCameraTask(validators=[val1, val2, val3], task_args=task_args)
        score = task.validate(tool_calls)
        assert score == 0.0

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

from typing import List, Literal

from rai_bench.tool_calling_agent.interfaces import (
    Task,
    TaskArgs,
)
from rai_bench.tool_calling_agent.subtasks import (
    CheckArgsToolCallSubTask,
    CheckServiceFieldsToolCallSubTask,
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
from rai_bench.tool_calling_agent.validators import (
    NotOrderedCallsValidator,
    OptionalValidator,
    OrderedCallsValidator,
)

COLOR_IMAGE_TOPIC = "/color_image5"
DEPTH_IMAGE_TOPIC = "/depth_image5"
COLOR_CAMERA_INFO_TOPIC = "/color_camera_info5"
DEPTH_CAMERA_INFO_TOPIC = "/depth_camera_info5"
ROBOT_DESCRIPTION_TOPIC = "/robot_description"
POINTCLOUD_TOPIC = "/pointcloud"
SCAN_TOPIC = "/scan"

ROBOT_STATE_PUBLISHER_LIST_PARAMS = "/robot_state_publisher/list_parameters"
ROBOT_STATE_PUBLISHER_GET_PARAMS = "/robot_state_publisher/get_parameters"
ROBOT_STATE_PUBLISHER_SET_PARAMS = "/robot_state_publisher/set_parameters"
ROBOT_STATE_PUBLISHER_SET_PARAMS_ATOMICALLY = (
    "/robot_state_publisher/set_parameters_atomically"
)

SPAWN_ENTITY_SERVICE = "/spawn_entity"
DELETE_ENTITY_SERVICE = "/delete_entity"
GET_SPAWNABLE_NAMES_SERVICE = "/get_available_spawnable_names"
GROUNDED_SAM_SET_PARAMS = "/grounded_sam/set_parameters"
GROUNDED_SAM_SET_PARAMS_ATOMICALLY = "/grounded_sam/set_parameters_atomically"
GROUNDING_DINO_SET_PARAMS = "/grounding_dino/set_parameters"
GROUNDING_DINO_SET_PARAMS_ATOMICALLY = "/grounding_dino/set_parameters_atomically"
O3DE_SET_PARAMS = "/o3de_ros2_node/set_parameters"
O3DE_SET_PARAMS_ATOMICALLY = "/o3de_ros2_node/set_parameters_atomically"

LIST_PARAMETERS_TYPE = "rcl_interfaces/srv/ListParameters"
SET_PARAMETERS_TYPE = "rcl_interfaces/srv/SetParameters"
SET_PARAMETERS_ATOMICALLY_TYPE = "rcl_interfaces/srv/SetParametersAtomically"
GET_PARAMETERS_TYPE = "rcl_interfaces/srv/GetParameters"
SPAWN_ENTITY_TYPE = "gazebo_msgs/srv/SpawnEntity"
DELETE_ENTITY_TYPE = "gazebo_msgs/srv/DeleteEntity"
GET_WORLD_PROPERTIES_TYPE = "gazebo_msgs/srv/GetWorldProperties"

DEFAULT_PUBLISH_FREQUENCY = 30.0
DEFAULT_FPS = 30
DEFAULT_SAM_CONFIDENCE = 0.8
DEFAULT_DINO_CONFIDENCE = 0.7
SAM_CONFIDENCE_2 = 0.6
DINO_CONFIDENCE_2 = 0.6
FPS_2 = 10

TOMATO_ENTITY = "tomato"
BOX1_ENTITY = "box1"
BOX2_ENTITY = "box2"
BOX1_POSITION = (0.2, 0.2, 0.2)
BOX2_POSITION = (0.4, 0.4, 0.2)

CAMERA_TOPICS_AND_TYPES = [
    f"topic: {COLOR_CAMERA_INFO_TOPIC}\ntype: sensor_msgs/msg/CameraInfo\n",
    f"topic: {COLOR_IMAGE_TOPIC}\ntype: sensor_msgs/msg/Image\n",
    f"topic: {DEPTH_CAMERA_INFO_TOPIC}\ntype: sensor_msgs/msg/CameraInfo\n",
    f"topic: {DEPTH_IMAGE_TOPIC}\ntype: sensor_msgs/msg/Image\n",
]

CAMERA_TOPICS = [
    COLOR_CAMERA_INFO_TOPIC,
    COLOR_IMAGE_TOPIC,
    DEPTH_CAMERA_INFO_TOPIC,
    DEPTH_IMAGE_TOPIC,
]
########## SUBTASKS #################################################################
get_topics_subtask = CheckArgsToolCallSubTask(
    expected_tool_name="get_ros2_topics_names_and_types", expected_args={}
)

color_image5_subtask = CheckArgsToolCallSubTask(
    expected_tool_name="get_ros2_image",
    expected_args={"topic": COLOR_IMAGE_TOPIC},
    expected_optional_args={"timeout_sec": int},
)
depth_image5_subtask = CheckArgsToolCallSubTask(
    expected_tool_name="get_ros2_image",
    expected_args={"topic": DEPTH_IMAGE_TOPIC},
    expected_optional_args={"timeout_sec": int},
)

color_camera_info5_subtask = CheckArgsToolCallSubTask(
    expected_tool_name="receive_ros2_message",
    expected_args={"topic": COLOR_CAMERA_INFO_TOPIC},
    expected_optional_args={"timeout_sec": int},
)
depth_camera_info5_subtask = CheckArgsToolCallSubTask(
    expected_tool_name="receive_ros2_message",
    expected_args={"topic": DEPTH_CAMERA_INFO_TOPIC},
    expected_optional_args={"timeout_sec": int},
)

receive_robot_desc_subtask = CheckArgsToolCallSubTask(
    expected_tool_name="receive_ros2_message",
    expected_args={"topic": ROBOT_DESCRIPTION_TOPIC},
    expected_optional_args={"timeout_sec": int},
)

receive_pointcloud_subtask = CheckArgsToolCallSubTask(
    expected_tool_name="receive_ros2_message",
    expected_args={"topic": POINTCLOUD_TOPIC},
    expected_optional_args={"timeout_sec": int},
)

robot_description_subtask = CheckArgsToolCallSubTask(
    expected_tool_name="receive_ros2_message",
    expected_args={"topic": ROBOT_DESCRIPTION_TOPIC},
    expected_optional_args={"timeout_sec": int},
)

scan_subtask = CheckArgsToolCallSubTask(
    expected_tool_name="receive_ros2_message",
    expected_args={"topic": SCAN_TOPIC},
    expected_optional_args={"timeout_sec": int},
)
pointcloud_subtask = CheckArgsToolCallSubTask(
    expected_tool_name="receive_ros2_message",
    expected_args={"topic": POINTCLOUD_TOPIC},
    expected_optional_args={"timeout_sec": int},
)


get_services_subtask = CheckArgsToolCallSubTask(
    expected_tool_name="get_ros2_services_names_and_types", expected_args={}
)

list_parameters_subtask = CheckServiceFieldsToolCallSubTask(
    expected_tool_name="call_ros2_service",
    expected_service=ROBOT_STATE_PUBLISHER_LIST_PARAMS,
    expected_service_type=LIST_PARAMETERS_TYPE,
    expected_fields={"": {}},
)

get_parameters_subtask = CheckServiceFieldsToolCallSubTask(
    expected_tool_name="call_ros2_service",
    expected_service=ROBOT_STATE_PUBLISHER_GET_PARAMS,
    expected_service_type=GET_PARAMETERS_TYPE,
    expected_fields={"names.0": "publish_frequency"},
)

check_spawnable_entities_subtask = CheckServiceFieldsToolCallSubTask(
    expected_tool_name="call_ros2_service",
    expected_service=GET_SPAWNABLE_NAMES_SERVICE,
    expected_service_type=GET_WORLD_PROPERTIES_TYPE,
    expected_fields={"": {}},
)

spawn_entity_subtask = CheckServiceFieldsToolCallSubTask(
    expected_tool_name="call_ros2_service",
    expected_service=SPAWN_ENTITY_SERVICE,
    expected_service_type=SPAWN_ENTITY_TYPE,
    expected_fields={
        "name": TOMATO_ENTITY,
    },
)

set_robot_state_params_subtask = CheckServiceFieldsToolCallSubTask(
    expected_tool_name="call_ros2_service",
    expected_service=ROBOT_STATE_PUBLISHER_SET_PARAMS,
    expected_service_type=SET_PARAMETERS_TYPE,
    expected_fields={
        "parameters.0.name": "publish_frequency",
        "parameters.0.value.type": "3",
        "parameters.0.value.double_value": DEFAULT_PUBLISH_FREQUENCY,
    },
)
set_robot_state_params_atomically_subtask = CheckServiceFieldsToolCallSubTask(
    expected_tool_name="call_ros2_service",
    expected_service=ROBOT_STATE_PUBLISHER_SET_PARAMS_ATOMICALLY,
    expected_service_type=SET_PARAMETERS_ATOMICALLY_TYPE,
    expected_fields={
        "parameters.0.name": "publish_frequency",
        "parameters.0.value.type": "3",
        "parameters.0.value.double_value": DEFAULT_PUBLISH_FREQUENCY,
    },
)


set_grounded_sam_subtask_1 = CheckServiceFieldsToolCallSubTask(
    expected_tool_name="call_ros2_service",
    expected_service=GROUNDED_SAM_SET_PARAMS,
    expected_service_type=SET_PARAMETERS_TYPE,
    expected_fields={
        "parameters.0.name": "confidence_threshold",
        "parameters.0.value.type": 3,
        "parameters.0.value.double_value": DEFAULT_SAM_CONFIDENCE,
    },
)
set_grounded_sam_atomically_subtask_1 = CheckServiceFieldsToolCallSubTask(
    expected_tool_name="call_ros2_service",
    expected_service=GROUNDED_SAM_SET_PARAMS_ATOMICALLY,
    expected_service_type=SET_PARAMETERS_ATOMICALLY_TYPE,
    expected_fields={
        "parameters.0.name": "confidence_threshold",
        "parameters.0.value.type": 3,
        "parameters.0.value.double_value": DEFAULT_SAM_CONFIDENCE,
    },
)

set_grounded_dino_subtask_1 = CheckServiceFieldsToolCallSubTask(
    expected_tool_name="call_ros2_service",
    expected_service=GROUNDING_DINO_SET_PARAMS,
    expected_service_type=SET_PARAMETERS_TYPE,
    expected_fields={
        "parameters.0.name": "confidence_threshold",
        "parameters.0.value.type": 3,
        "parameters.0.value.double_value": DEFAULT_DINO_CONFIDENCE,
    },
)
set_grounding_dino_atomically_subtask_1 = CheckServiceFieldsToolCallSubTask(
    expected_tool_name="call_ros2_service",
    expected_service=GROUNDING_DINO_SET_PARAMS_ATOMICALLY,
    expected_service_type=SET_PARAMETERS_ATOMICALLY_TYPE,
    expected_fields={
        "parameters.0.name": "confidence_threshold",
        "parameters.0.value.type": 3,
        "parameters.0.value.double_value": DEFAULT_DINO_CONFIDENCE,
    },
)

set_o3de_fps_subtask_1 = CheckServiceFieldsToolCallSubTask(
    expected_tool_name="call_ros2_service",
    expected_service=O3DE_SET_PARAMS,
    expected_service_type=SET_PARAMETERS_TYPE,
    expected_fields={
        "parameters.0.name": "fps",
        "parameters.0.value.type": 2,
        "parameters.0.value.integer_value": DEFAULT_FPS,
    },
)
set_o3de_fps_atomically_subtask_1 = CheckServiceFieldsToolCallSubTask(
    expected_tool_name="call_ros2_service",
    expected_service=O3DE_SET_PARAMS_ATOMICALLY,
    expected_service_type=SET_PARAMETERS_ATOMICALLY_TYPE,
    expected_fields={
        "parameters.0.name": "fps",
        "parameters.0.value.type": 2,
        "parameters.0.value.integer_value": DEFAULT_FPS,
    },
)

set_grounded_sam_subtask_2 = CheckServiceFieldsToolCallSubTask(
    expected_tool_name="call_ros2_service",
    expected_service=GROUNDED_SAM_SET_PARAMS,
    expected_service_type=SET_PARAMETERS_TYPE,
    expected_fields={
        "parameters.0.name": "confidence_threshold",
        "parameters.0.value.type": 3,
        "parameters.0.value.double_value": SAM_CONFIDENCE_2,
    },
)
set_grounded_sam_atomically_subtask_2 = CheckServiceFieldsToolCallSubTask(
    expected_tool_name="call_ros2_service",
    expected_service=GROUNDED_SAM_SET_PARAMS_ATOMICALLY,
    expected_service_type=SET_PARAMETERS_ATOMICALLY_TYPE,
    expected_fields={
        "parameters.0.name": "confidence_threshold",
        "parameters.0.value.type": 3,
        "parameters.0.value.double_value": SAM_CONFIDENCE_2,
    },
)

set_grounding_dino_subtask_2 = CheckServiceFieldsToolCallSubTask(
    expected_tool_name="call_ros2_service",
    expected_service=GROUNDING_DINO_SET_PARAMS,
    expected_service_type=SET_PARAMETERS_TYPE,
    expected_fields={
        "parameters.0.name": "confidence_threshold",
        "parameters.0.value.type": 3,
        "parameters.0.value.double_value": DINO_CONFIDENCE_2,
    },
)
set_grounding_dino_atomically_subtask_2 = CheckServiceFieldsToolCallSubTask(
    expected_tool_name="call_ros2_service",
    expected_service=GROUNDING_DINO_SET_PARAMS_ATOMICALLY,
    expected_service_type=SET_PARAMETERS_ATOMICALLY_TYPE,
    expected_fields={
        "parameters.0.name": "confidence_threshold",
        "parameters.0.value.type": 3,
        "parameters.0.value.double_value": DINO_CONFIDENCE_2,
    },
)

set_o3de_fps_subtask_2 = CheckServiceFieldsToolCallSubTask(
    expected_tool_name="call_ros2_service",
    expected_service=O3DE_SET_PARAMS,
    expected_service_type=SET_PARAMETERS_TYPE,
    expected_fields={
        "parameters.0.name": "fps",
        "parameters.0.value.type": 2,
        "parameters.0.value.integer_value": FPS_2,
    },
)
set_o3de_fps_atomically_subtask_2 = CheckServiceFieldsToolCallSubTask(
    expected_tool_name="call_ros2_service",
    expected_service=O3DE_SET_PARAMS_ATOMICALLY,
    expected_service_type=SET_PARAMETERS_ATOMICALLY_TYPE,
    expected_fields={
        "parameters.0.name": "fps",
        "parameters.0.value.type": 2,
        "parameters.0.value.integer_value": FPS_2,
    },
)


delete_entity_subtask1 = CheckServiceFieldsToolCallSubTask(
    expected_tool_name="call_ros2_service",
    expected_service=DELETE_ENTITY_SERVICE,
    expected_service_type=DELETE_ENTITY_TYPE,
    expected_fields={
        "name": BOX1_ENTITY,
    },
)
delete_entity_subtask2 = CheckServiceFieldsToolCallSubTask(
    expected_tool_name="call_ros2_service",
    expected_service=DELETE_ENTITY_SERVICE,
    expected_service_type=DELETE_ENTITY_TYPE,
    expected_fields={
        "name": BOX2_ENTITY,
    },
)
spawn_entity_subtask1 = CheckServiceFieldsToolCallSubTask(
    expected_tool_name="call_ros2_service",
    expected_service=SPAWN_ENTITY_SERVICE,
    expected_service_type=SPAWN_ENTITY_TYPE,
    expected_fields={
        "name": BOX1_ENTITY,
        "initial_pose.position.x": BOX1_POSITION[0],
        "initial_pose.position.y": BOX1_POSITION[1],
        "initial_pose.position.z": BOX1_POSITION[2],
    },
)
spawn_entity_subtask2 = CheckServiceFieldsToolCallSubTask(
    expected_tool_name="call_ros2_service",
    expected_service=SPAWN_ENTITY_SERVICE,
    expected_service_type=SPAWN_ENTITY_TYPE,
    expected_fields={
        "name": BOX2_ENTITY,
        "initial_pose.position.x": BOX2_POSITION[0],
        "initial_pose.position.y": BOX2_POSITION[1],
        "initial_pose.position.z": BOX2_POSITION[2],
    },
)

delete_both_val = NotOrderedCallsValidator(
    subtasks=[delete_entity_subtask1, delete_entity_subtask2]
)
spawn_both_val = NotOrderedCallsValidator(
    subtasks=[spawn_entity_subtask1, spawn_entity_subtask2]
)
######### VALIDATORS #########################################################################################
topics_ord_val = OrderedCallsValidator(subtasks=[get_topics_subtask])

color_image_ord_val = OrderedCallsValidator(subtasks=[color_image5_subtask])
depth_image_ord_val = OrderedCallsValidator(subtasks=[depth_image5_subtask])

color_camera_info_ord_val = OrderedCallsValidator(subtasks=[color_camera_info5_subtask])
depth_camera_info_ord_val = OrderedCallsValidator(subtasks=[depth_camera_info5_subtask])

all_camera_images_notord_val = NotOrderedCallsValidator(
    subtasks=[
        color_image5_subtask,
        depth_image5_subtask,
    ]
)


get_pointcloud_ord_val = OrderedCallsValidator(subtasks=[receive_pointcloud_subtask])
get_robot_desc_ord_val = OrderedCallsValidator(subtasks=[receive_robot_desc_subtask])

set_param_val = OptionalValidator(
    subtasks=[set_robot_state_params_subtask, set_robot_state_params_atomically_subtask]
)
services_ord_val = OrderedCallsValidator(subtasks=[get_services_subtask])
list_parameters_val = OrderedCallsValidator(subtasks=[list_parameters_subtask])
get_parameters_val = OrderedCallsValidator(subtasks=[get_parameters_subtask])
check_spawnable_entities_val = OrderedCallsValidator(
    subtasks=[check_spawnable_entities_subtask]
)
spawn_entity_val = OrderedCallsValidator(subtasks=[spawn_entity_subtask])

set_grounded_sam_opt_val_1 = OptionalValidator(
    subtasks=[set_grounded_sam_subtask_1, set_grounded_sam_atomically_subtask_1]
)
set_grounded_dino_opt_val_1 = OptionalValidator(
    subtasks=[set_grounded_dino_subtask_1, set_grounding_dino_atomically_subtask_1]
)
set_o3de_fps_opt_val_1 = OptionalValidator(
    subtasks=[set_o3de_fps_subtask_1, set_o3de_fps_atomically_subtask_1]
)


set_grounded_sam_opt_val_2 = OptionalValidator(
    subtasks=[set_grounded_sam_subtask_2, set_grounded_sam_atomically_subtask_2]
)
set_grounded_dino_opt_val_2 = OptionalValidator(
    subtasks=[set_grounding_dino_subtask_2, set_grounding_dino_atomically_subtask_2]
)
set_o3de_fps_opt_val_2 = OptionalValidator(
    subtasks=[set_o3de_fps_subtask_2, set_o3de_fps_atomically_subtask_2]
)


def get_basic_tasks(
    extra_tool_calls: List[int] = [0],
    prompt_detail: List[Literal["brief", "descriptive"]] = ["brief", "descriptive"],
    n_shots: List[Literal[0, 2, 5]] = [0, 2, 5],
) -> List[Task]:
    """Get predefined basic tasks.

    Parameters
    ----------
    Parameters match :class:`~src.rai_bench.rai_bench.test_models.ToolCallingAgentBenchmarkConfig`.
    See the class documentation for parameter descriptions.

    Returns
    -------
    Returned list match :func:`~src.rai_bench.rai_bench.tool_calling_agent.predefined.tasks.get_tasks`.
    """
    tasks: List[Task] = []

    # Generate all combinations of prompt_detail and n_shots and extra tool calls
    for extra_calls in extra_tool_calls:
        for detail in prompt_detail:
            for shots in n_shots:
                task_args = TaskArgs(
                    extra_tool_calls=extra_calls,
                    prompt_detail=detail,
                    examples_in_system_prompt=shots,
                )

                tasks.extend(
                    [
                        GetROS2RGBCameraTask(
                            validators=[color_image_ord_val],
                            task_args=task_args,
                        ),
                        GetROS2TopicsTask(
                            validators=[topics_ord_val],
                            task_args=task_args,
                        ),
                        GetROS2DepthCameraTask(
                            validators=[depth_image_ord_val],
                            task_args=task_args,
                        ),
                        GetAllROS2CamerasTask(
                            validators=[all_camera_images_notord_val],
                            task_args=task_args,
                        ),
                        GetPointcloudTask(
                            validators=[get_pointcloud_ord_val], task_args=task_args
                        ),
                        GetRobotDescriptionTask(
                            validators=[get_robot_desc_ord_val], task_args=task_args
                        ),
                        GetROS2ServicesTask(
                            validators=[services_ord_val],
                            task_args=task_args,
                        ),
                        ListRobotParametersTask(
                            validators=[list_parameters_val],
                            task_args=task_args,
                        ),
                        GetSpecificParameterTask(
                            parameter="publish_frequency",
                            validators=[get_parameters_val],
                            task_args=task_args,
                        ),
                        CheckSpawnableEntitiesTask(
                            validators=[check_spawnable_entities_val],
                            task_args=task_args,
                        ),
                        SpawnEntityTask(
                            entity=TOMATO_ENTITY,
                            validators=[spawn_entity_val],
                            task_args=task_args,
                        ),
                        SetRobotParameterTask(
                            value=DEFAULT_PUBLISH_FREQUENCY,
                            validators=[set_param_val],
                            task_args=task_args,
                        ),
                        SetRobotParameterTask(
                            value=25.0, validators=[set_param_val], task_args=task_args
                        ),
                        ConfigureVisionPipelineTask(
                            sam_confidence_threshold=DEFAULT_SAM_CONFIDENCE,
                            dino_confidence_threshold=DEFAULT_DINO_CONFIDENCE,
                            fps=DEFAULT_FPS,
                            validators=[
                                set_grounded_sam_opt_val_1,
                                set_grounded_dino_opt_val_1,
                                set_o3de_fps_opt_val_1,
                            ],
                            task_args=task_args,
                        ),
                        ConfigureVisionPipelineTask(
                            sam_confidence_threshold=0.6,
                            dino_confidence_threshold=0.6,
                            fps=10,
                            validators=[
                                set_grounded_sam_opt_val_2,
                                set_grounded_dino_opt_val_2,
                                set_o3de_fps_opt_val_2,
                            ],
                            task_args=task_args,
                        ),
                        RespawnEntitiesTask(
                            names=[BOX1_ENTITY, BOX2_ENTITY],
                            coords=[BOX1_POSITION, BOX2_POSITION],
                            validators=[delete_both_val, spawn_both_val],
                            task_args=task_args,
                        ),
                    ]
                )

    return tasks

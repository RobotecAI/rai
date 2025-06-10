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
    OrderedCallsValidator,
)

########## SUBTASKS #################################################################

get_topics_subtask = CheckArgsToolCallSubTask(
    expected_tool_name="get_ros2_topics_names_and_types", expected_args={}
)

color_image5_subtask = CheckArgsToolCallSubTask(
    expected_tool_name="get_ros2_image",
    expected_args={"topic": "/color_image5"},
    expected_optional_args={"timeout_sec": int},
)
depth_image5_subtask = CheckArgsToolCallSubTask(
    expected_tool_name="get_ros2_image",
    expected_args={"topic": "/depth_image5"},
    expected_optional_args={"timeout_sec": int},
)

color_camera_info5_subtask = CheckArgsToolCallSubTask(
    expected_tool_name="receive_ros2_message",
    expected_args={"topic": "/color_camera_info5"},
    expected_optional_args={"timeout_sec": int},
)
depth_camera_info5_subtask = CheckArgsToolCallSubTask(
    expected_tool_name="receive_ros2_message",
    expected_args={"topic": "/depth_camera_info5"},
    expected_optional_args={"timeout_sec": int},
)

receive_robot_desc_subtask = CheckArgsToolCallSubTask(
    expected_tool_name="receive_ros2_message",
    expected_args={"topic": "/robot_description"},
    expected_optional_args={"timeout_sec": int},
)

receive_pointcloud_subtask = CheckArgsToolCallSubTask(
    expected_tool_name="receive_ros2_message",
    expected_args={"topic": "/pointcloud"},
    expected_optional_args={"timeout_sec": int},
)


robot_description_subtask = CheckArgsToolCallSubTask(
    expected_tool_name="receive_ros2_message",
    expected_args={"topic": "/robot_description"},
    expected_optional_args={"timeout_sec": int},
)


scan_subtask = CheckArgsToolCallSubTask(
    expected_tool_name="receive_ros2_message",
    expected_args={"topic": "/scan"},
    expected_optional_args={"timeout_sec": int},
)
pointcloud_subtask = CheckArgsToolCallSubTask(
    expected_tool_name="receive_ros2_message",
    expected_args={"topic": "/pointcloud"},
    expected_optional_args={"timeout_sec": int},
)


get_services_subtask = CheckArgsToolCallSubTask(
    expected_tool_name="get_ros2_services_names_and_types", expected_args={}
)

list_parameters_subtask = CheckServiceFieldsToolCallSubTask(
    expected_tool_name="call_ros2_service",
    expected_service="/robot_state_publisher/list_parameters",
    expected_service_type="rcl_interfaces/srv/ListParameters",
    expected_fields={"": {}},
)

get_parameters_subtask = CheckServiceFieldsToolCallSubTask(
    expected_tool_name="call_ros2_service",
    expected_service="/robot_state_publisher/get_parameters",
    expected_service_type="rcl_interfaces/srv/GetParameters",
    expected_fields={"names.0": "publish_frequency"},
)

check_spawnable_entities_subtask = CheckServiceFieldsToolCallSubTask(
    expected_tool_name="call_ros2_service",
    expected_service="/get_available_spawnable_names",
    expected_service_type="gazebo_msgs/srv/GetModelList",
    expected_fields={"": {}},
)

spawn_entity_subtask = CheckServiceFieldsToolCallSubTask(
    expected_tool_name="call_ros2_service",
    expected_service="/spawn_entity",
    expected_service_type="gazebo_msgs/srv/SpawnEntity",
    expected_fields={
        "name": "test_box",
        "xml": str,
    },
)

set_robot_state_params_subtask = CheckServiceFieldsToolCallSubTask(
    expected_tool_name="call_ros2_service",
    expected_service="/robot_state_publisher/set_parameters",
    expected_service_type="rcl_interfaces/srv/SetParameters",
    expected_fields={
        "parameters.0.name": "publish_frequency",
        "parameters.0.value.type": "3",
        "parameters.0.value.double_value": 30.0,
    },
)

set_param_val = OrderedCallsValidator(subtasks=[set_robot_state_params_subtask])

grounded_sam_config_subtask = CheckServiceFieldsToolCallSubTask(
    expected_tool_name="call_ros2_service",
    expected_service="/grounded_sam/set_parameters",
    expected_service_type="rcl_interfaces/srv/SetParameters",
    expected_fields={
        "parameters.0.name": "confidence_threshold",
        "parameters.0.value.type": 3,
        "parameters.0.value.double_value": 0.8,
    },
)

grounding_dino_config_subtask = CheckServiceFieldsToolCallSubTask(
    expected_tool_name="call_ros2_service",
    expected_service="/grounding_dino/set_parameters",
    expected_service_type="rcl_interfaces/srv/SetParameters",
    expected_fields={
        "parameters.0.name": "detection_threshold",
        "parameters.0.value.type": 3,
        "parameters.0.value.double_value": 0.7,
    },
)
o3de_fps_config_subtask = CheckServiceFieldsToolCallSubTask(
    expected_tool_name="call_ros2_service",
    expected_service="/o3de_ros2_node/set_parameters",
    expected_service_type="rcl_interfaces/srv/SetParameters",
    expected_fields={
        "parameters.0.name": "fps",
        "parameters.0.value.type": 2,
        "parameters.0.value.integer_value": 30,
    },
)

delete_entity_subtask1 = CheckServiceFieldsToolCallSubTask(
    expected_tool_name="call_ros2_service",
    expected_service="/delete_entity",
    expected_service_type="gazebo_msgs/srv/DeleteEntity",
    expected_fields={
        "name": "box1",
    },
)
delete_entity_subtask2 = CheckServiceFieldsToolCallSubTask(
    expected_tool_name="call_ros2_service",
    expected_service="/delete_entity",
    expected_service_type="gazebo_msgs/srv/DeleteEntity",
    expected_fields={
        "name": "box2",
    },
)
spawn_entity_subtask1 = CheckServiceFieldsToolCallSubTask(
    expected_tool_name="call_ros2_service",
    expected_service="/spawn_entity",
    expected_service_type="gazebo_msgs/srv/SpawnEntity",
    expected_fields={
        "name": "box1",
        "xml": str,
        "initial_pose.position.x": 0.2,
        "initial_pose.position.y": 0.2,
        "initial_pose.position.z": 0.2,
    },
)
spawn_entity_subtask2 = CheckServiceFieldsToolCallSubTask(
    expected_tool_name="call_ros2_service",
    expected_service="/spawn_entity",
    expected_service_type="gazebo_msgs/srv/SpawnEntity",
    expected_fields={
        "name": "box2",
        "xml": str,
        "initial_pose.position.x": 0.4,
        "initial_pose.position.y": 0.4,
        "initial_pose.position.z": 0.2,
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

# color_image_with_info_ord_val = NotOrderedCallsValidator(
#     subtasks=[color_image5_subtask, color_camera_info5_subtask]
# )
# depth_image_with_info_ord_val = NotOrderedCallsValidator(
#     subtasks=[depth_image5_subtask, color_camera_info5_subtask]
# )

all_camera_images_notord_val = NotOrderedCallsValidator(
    subtasks=[
        color_image5_subtask,
        depth_image5_subtask,
    ]
)


get_pointcloud_ord_val = OrderedCallsValidator(subtasks=[receive_pointcloud_subtask])
get_robot_desc_ord_val = OrderedCallsValidator(subtasks=[receive_robot_desc_subtask])


services_ord_val = OrderedCallsValidator(subtasks=[get_services_subtask])
list_parameters_val = OrderedCallsValidator(subtasks=[list_parameters_subtask])
get_parameters_val = OrderedCallsValidator(subtasks=[get_parameters_subtask])
check_spawnable_entities_val = OrderedCallsValidator(
    subtasks=[check_spawnable_entities_subtask]
)
spawn_entity_val = OrderedCallsValidator(subtasks=[spawn_entity_subtask])

vision_pipeline_config_val = NotOrderedCallsValidator(
    subtasks=[
        grounded_sam_config_subtask,
        grounding_dino_config_subtask,
        o3de_fps_config_subtask,
    ]
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
                            entity="tomato",
                            validators=[spawn_entity_val],
                            task_args=task_args,
                        ),
                        SetRobotParameterTask(
                            value=30.0, validators=[set_param_val], task_args=task_args
                        ),
                        SetRobotParameterTask(
                            value=25.0, validators=[set_param_val], task_args=task_args
                        ),
                        ConfigureVisionPipelineTask(
                            sam_confidence_threshold=0.8,
                            dino_confidence_threshold=0.5,
                            fps=30,
                            validators=[vision_pipeline_config_val],
                            task_args=task_args,
                        ),
                        ConfigureVisionPipelineTask(
                            sam_confidence_threshold=0.6,
                            dino_confidence_threshold=0.6,
                            fps=10,
                            validators=[vision_pipeline_config_val],
                            task_args=task_args,
                        ),
                        RespawnEntitiesTask(
                            names=["box1", "box2"],
                            coords=[(0.2, 0.2, 0.2), (0.4, 0.4, 0.2)],
                            validators=[delete_both_val, spawn_both_val],
                            task_args=task_args,
                        ),
                    ]
                )

    return tasks

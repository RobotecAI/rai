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
    COLOR_IMAGE_TOPIC,
    DEPTH_IMAGE_TOPIC,
    GET_SPAWNABLE_NAMES_SERVICE,
    GET_WORLD_PROPERTIES_TYPE,
    LIST_PARAMETERS_TYPE,
    POINTCLOUD_TOPIC,
    ROBOT_DESCRIPTION_TOPIC,
    ROBOT_STATE_PUBLISHER_LIST_PARAMS,
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

########## SUBTASKS FOR TASKS WITHOUT REFACTORED VALIDATORS #################################################################
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

receive_pointcloud_subtask = CheckArgsToolCallSubTask(
    expected_tool_name="receive_ros2_message",
    expected_args={"topic": POINTCLOUD_TOPIC},
    expected_optional_args={"timeout_sec": int},
)

receive_robot_desc_subtask = CheckArgsToolCallSubTask(
    expected_tool_name="receive_ros2_message",
    expected_args={"topic": ROBOT_DESCRIPTION_TOPIC},
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

check_spawnable_entities_subtask = CheckServiceFieldsToolCallSubTask(
    expected_tool_name="call_ros2_service",
    expected_service=GET_SPAWNABLE_NAMES_SERVICE,
    expected_service_type=GET_WORLD_PROPERTIES_TYPE,
    expected_fields={"": {}},
)

######### VALIDATORS FOR TASKS WITHOUT REFACTORED VALIDATORS #########################################################################################
topics_ord_val = OrderedCallsValidator(subtasks=[get_topics_subtask])

color_image_ord_val = OrderedCallsValidator(subtasks=[color_image5_subtask])
depth_image_ord_val = OrderedCallsValidator(subtasks=[depth_image5_subtask])

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
check_spawnable_entities_val = OrderedCallsValidator(
    subtasks=[check_spawnable_entities_subtask]
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
                            task_args=task_args,
                            validators=[color_image_ord_val],
                        ),
                        GetROS2TopicsTask(
                            task_args=task_args,
                            validators=[topics_ord_val],
                        ),
                        GetROS2DepthCameraTask(
                            task_args=task_args,
                            validators=[depth_image_ord_val],
                        ),
                        GetAllROS2CamerasTask(
                            task_args=task_args,
                            validators=[all_camera_images_notord_val],
                        ),
                        GetPointcloudTask(
                            task_args=task_args,
                            validators=[get_pointcloud_ord_val],
                        ),
                        GetRobotDescriptionTask(
                            task_args=task_args,
                            validators=[get_robot_desc_ord_val],
                        ),
                        GetROS2ServicesTask(
                            task_args=task_args,
                            validators=[services_ord_val],
                        ),
                        ListRobotParametersTask(
                            task_args=task_args,
                            validators=[list_parameters_val],
                        ),
                        CheckSpawnableEntitiesTask(
                            task_args=task_args,
                            validators=[check_spawnable_entities_val],
                        ),
                        # Tasks with refactored validators - now use defaults
                        GetSpecificParameterTask(
                            parameter="publish_frequency",
                            task_args=task_args,
                        ),
                        SpawnEntityTask(
                            entity=TOMATO_ENTITY,
                            task_args=task_args,
                        ),
                        SetRobotParameterTask(
                            value=DEFAULT_PUBLISH_FREQUENCY,
                            task_args=task_args,
                        ),
                        SetRobotParameterTask(
                            value=25.0,
                            task_args=task_args,
                        ),
                        ConfigureVisionPipelineTask(
                            sam_confidence_threshold=DEFAULT_SAM_CONFIDENCE,
                            dino_confidence_threshold=DEFAULT_DINO_CONFIDENCE,
                            fps=DEFAULT_FPS,
                            task_args=task_args,
                        ),
                        RespawnEntitiesTask(
                            names=[BOX1_ENTITY, BOX2_ENTITY],
                            coords=[BOX1_POSITION, BOX2_POSITION],
                            task_args=task_args,
                        ),
                    ]
                )

    return tasks

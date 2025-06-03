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
)
from rai_bench.tool_calling_agent.tasks.basic import (
    AssessSensorDataQualityTask,
    CheckRobotHealthTask,
    GetAllROS2CamerasTask,
    GetPointcloudTask,
    GetRobotDescriptionTask,
    GetROS2DepthCameraTask,
    GetROS2RGBCameraTask,
    GetROS2TopicsTask,
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
    expected_tool_name="get_ros2_image",
    expected_args={"topic": "/color_image5"},
    expected_optional_args={"timeout_sec": int},
)
depth_camera_info5_subtask = CheckArgsToolCallSubTask(
    expected_tool_name="get_ros2_image",
    expected_args={"topic": "/depth_image5"},
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

# System health subtasks
diagnostics_subtask = CheckArgsToolCallSubTask(
    expected_tool_name="receive_ros2_message",
    expected_args={"topic": "/diagnostics"},
    expected_optional_args={"timeout_sec": int},
)
rosout_subtask = CheckArgsToolCallSubTask(
    expected_tool_name="receive_ros2_message",
    expected_args={"topic": "/rosout"},
    expected_optional_args={"timeout_sec": int},
)
joint_states_subtask = CheckArgsToolCallSubTask(
    expected_tool_name="receive_ros2_message",
    expected_args={"topic": "/joint_states"},
    expected_optional_args={"timeout_sec": int},
)

# Odometry subtasks
odom_subtask = CheckArgsToolCallSubTask(
    expected_tool_name="receive_ros2_message",
    expected_args={"topic": "/odom"},
    expected_optional_args={"timeout_sec": int},
)
filtered_odom_subtask = CheckArgsToolCallSubTask(
    expected_tool_name="receive_ros2_message",
    expected_args={"topic": "/odometry/filtered"},
    expected_optional_args={"timeout_sec": int},
)

# Transform subtasks
tf_subtask = CheckArgsToolCallSubTask(
    expected_tool_name="receive_ros2_message",
    expected_args={"topic": "/tf"},
    expected_optional_args={"timeout_sec": int},
)
tf_static_subtask = CheckArgsToolCallSubTask(
    expected_tool_name="receive_ros2_message",
    expected_args={"topic": "/tf_static"},
    expected_optional_args={"timeout_sec": int},
)


# Robot description subtasks
robot_description_subtask = CheckArgsToolCallSubTask(
    expected_tool_name="receive_ros2_message",
    expected_args={"topic": "/robot_description"},
    expected_optional_args={"timeout_sec": int},
)
robot_description_semantic_subtask = CheckArgsToolCallSubTask(
    expected_tool_name="receive_ros2_message",
    expected_args={"topic": "/robot_description_semantic"},
    expected_optional_args={"timeout_sec": int},
)

# Sensor data subtasks
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


# Robot description subtasks
robot_description_subtask = CheckArgsToolCallSubTask(
    expected_tool_name="receive_ros2_message",
    expected_args={"topic": "/robot_description"},
    expected_optional_args={"timeout_sec": int},
)
robot_description_semantic_subtask = CheckArgsToolCallSubTask(
    expected_tool_name="receive_ros2_message",
    expected_args={"topic": "/robot_description_semantic"},
    expected_optional_args={"timeout_sec": int},
)

# Sensor data subtasks
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

# Robot description subtasks
robot_description_subtask = CheckArgsToolCallSubTask(
    expected_tool_name="receive_ros2_message",
    expected_args={"topic": "/robot_description"},
    expected_optional_args={"timeout_sec": int},
)
robot_description_semantic_subtask = CheckArgsToolCallSubTask(
    expected_tool_name="receive_ros2_message",
    expected_args={"topic": "/robot_description_semantic"},
    expected_optional_args={"timeout_sec": int},
)

######### VALIDATORS #########################################################################################
topics_ord_val = OrderedCallsValidator(subtasks=[get_topics_subtask])

color_image_ord_val = OrderedCallsValidator(subtasks=[color_image5_subtask])
depth_image_ord_val = OrderedCallsValidator(subtasks=[depth_image5_subtask])

color_camera_info_ord_val = OrderedCallsValidator(subtasks=[color_camera_info5_subtask])
depth_camera_info_ord_val = OrderedCallsValidator(subtasks=[depth_camera_info5_subtask])

color_image_with_info_ord_val = NotOrderedCallsValidator(
    subtasks=[color_image5_subtask, color_camera_info5_subtask]
)
depth_image_with_info_ord_val = NotOrderedCallsValidator(
    subtasks=[depth_image5_subtask, color_camera_info5_subtask]
)

all_camera_images_notord_val = NotOrderedCallsValidator(
    subtasks=[
        color_image5_subtask,
        depth_image5_subtask,
    ]
)
all_camera_info_notord_val = NotOrderedCallsValidator(
    subtasks=[
        color_camera_info5_subtask,
        depth_camera_info5_subtask,
    ]
)
all_camera_images_with_info_notord_val = NotOrderedCallsValidator(
    subtasks=[
        color_image5_subtask,
        depth_image5_subtask,
        color_camera_info5_subtask,
        depth_camera_info5_subtask,
    ]
)

joint_states_ord_val = OrderedCallsValidator(subtasks=[joint_states_subtask])
diagnostics_ord_val = OrderedCallsValidator(subtasks=[diagnostics_subtask])

get_pointcloud_ord_val = OrderedCallsValidator(subtasks=[receive_pointcloud_subtask])
get_robot_desc_ord_val = OrderedCallsValidator(subtasks=[receive_robot_desc_subtask])

diagnostics_ord_val = NotOrderedCallsValidator(subtasks=[diagnostics_subtask])
joint_states_ord_val = NotOrderedCallsValidator(subtasks=[joint_states_subtask])
rosout_ord_val = NotOrderedCallsValidator(subtasks=[rosout_subtask])
robot_health_val = NotOrderedCallsValidator(
    subtasks=[diagnostics_subtask, joint_states_subtask, rosout_subtask]
)

odometry_comparison_val = NotOrderedCallsValidator(
    subtasks=[odom_subtask, filtered_odom_subtask]
)
sensor_data_val = NotOrderedCallsValidator(
    subtasks=[
        scan_subtask,
        receive_pointcloud_subtask,
        color_image5_subtask,
        depth_image5_subtask,
        color_camera_info5_subtask,
        depth_camera_info5_subtask,
    ]
)


def get_basic_tasks(
    extra_tool_calls: int = 0,
    prompt_detail: List[Literal["brief", "moderate", "descriptive"]] = [
        "brief",
        "moderate",
        "descriptive",
    ],
    n_shots: List[Literal[0, 2, 5]] = [0, 2, 5],
) -> List[Task]:
    tasks: List[Task] = []

    # Generate all combinations of prompt_detail and n_shots
    for detail in prompt_detail:
        for shots in n_shots:
            task_args = TaskArgs(
                extra_tool_calls=extra_tool_calls,
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
                    CheckRobotHealthTask(
                        validators=[
                            diagnostics_ord_val,
                            rosout_ord_val,
                            joint_states_ord_val,
                        ],
                        task_args=task_args,
                    ),
                    AssessSensorDataQualityTask(
                        validators=[sensor_data_val],
                        task_args=task_args,
                    ),
                ]
            )

    return tasks

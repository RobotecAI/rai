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

import random
from typing import List, Literal, Sequence

from rai.tools.ros2 import MoveToPointToolInput

from rai_bench.tool_calling_agent.interfaces import (
    Task,
    TaskArgs,
)
from rai_bench.tool_calling_agent.subtasks import (
    CheckActionFieldsToolCallSubTask,
    CheckArgsToolCallSubTask,
    CheckTopicFieldsToolCallSubTask,
)
from rai_bench.tool_calling_agent.tasks.basic import (
    GetAllROS2CamerasTask,
    GetROS2DepthCameraTask,
    GetROS2RGBCameraTask,
    GetROS2TopicsTask,
)
from rai_bench.tool_calling_agent.tasks.custom_interfaces import (
    PublishROS2HRIMessageTextTask,
)
from rai_bench.tool_calling_agent.tasks.manipulation import (
    MoveToPointTask,
)
from rai_bench.tool_calling_agent.tasks.navigation import (
    MoveToBedTask,
    MoveToFrontTask,
    NavigateToPointTask,
    SpinAroundTask,
)
from rai_bench.tool_calling_agent.tasks.spatial import (
    BoolImageTaskEasy,
    BoolImageTaskHard,
    BoolImageTaskInput,
    BoolImageTaskMedium,
)
from rai_bench.tool_calling_agent.validators import (
    NotOrderedCallsValidator,
    OrderedCallsValidator,
)

IMG_PATH = "src/rai_bench/rai_bench/tool_calling_agent/predefined/images/"
true_response_inputs: List[BoolImageTaskInput] = [
    BoolImageTaskInput(
        question="Is the door on the left from the desk?",
        images_paths=[IMG_PATH + "image_1.jpg"],
    ),
    BoolImageTaskInput(
        question="Is the light on in the room?",
        images_paths=[IMG_PATH + "image_2.jpg"],
    ),
    BoolImageTaskInput(
        question="Do you see the plant?",
        images_paths=[IMG_PATH + "image_2.jpg"],
    ),
    BoolImageTaskInput(
        question="Are there any pictures on the wall?",
        images_paths=[IMG_PATH + "image_3.jpg"],
    ),
    BoolImageTaskInput(
        question="Are there 3 pictures on the wall?",
        images_paths=[IMG_PATH + "image_4.jpg"],
    ),
    BoolImageTaskInput(
        question="Is there a plant behind the rack?",
        images_paths=[IMG_PATH + "image_5.jpg"],
    ),
    BoolImageTaskInput(
        question="Is there a pillow on the armchain?",
        images_paths=[IMG_PATH + "image_7.jpg"],
    ),
]
false_response_inputs: List[BoolImageTaskInput] = [
    BoolImageTaskInput(
        question="Is the door open?",
        images_paths=[IMG_PATH + "image_1.jpg"],
    ),
    BoolImageTaskInput(
        question="Is someone in the room?",
        images_paths=[IMG_PATH + "image_1.jpg"],
    ),
    BoolImageTaskInput(
        question="Do you see the plant?",
        images_paths=[IMG_PATH + "image_3.jpg"],
    ),
    BoolImageTaskInput(
        question="Are there 4 pictures on the wall?",
        images_paths=[IMG_PATH + "image_4.jpg"],
    ),
    BoolImageTaskInput(
        question="Is there a rack on the left from the sofa?",
        images_paths=[IMG_PATH + "image_4.jpg"],
    ),
    BoolImageTaskInput(
        question="Is there a plant on the right from the window?",
        images_paths=[IMG_PATH + "image_6.jpg"],
    ),
    BoolImageTaskInput(
        question="Is there a red pillow on the armchair?",
        images_paths=[IMG_PATH + "image_7.jpg"],
    ),
]
########## SUBTASKS #######################################################################################
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

receive_robot_desc_subtask = CheckArgsToolCallSubTask(
    expected_tool_name="receive_ros2_message",
    expected_args={"topic": "/robot_description"},
    expected_optional_args={"timeout_sec": int},
)

move_to_point_subtask_grab = CheckArgsToolCallSubTask(
    expected_tool_name="move_to_point",
    expected_args={"x": 1.0, "y": 2.0, "z": 3.0, "task": "grab"},
)
move_to_point_subtask_drop = CheckArgsToolCallSubTask(
    expected_tool_name="move_to_point",
    expected_args={"x": 1.2, "y": 2.3, "z": 3.4, "task": "drop"},
)

pub_HRIMessage_text_subtask = CheckTopicFieldsToolCallSubTask(
    expected_tool_name="publish_ros2_message",
    expected_topic="/to_human",
    expected_message_type="rai_interfaces/msg/HRIMessage",
    expected_fields={"text": "Hello!"},
)

get_tohuman_interface_subtask = CheckArgsToolCallSubTask(
    expected_tool_name="get_ros2_message_interface",
    expected_args={"msg_type": "rai_interfaces/msg/HRIMessage"},
)


return_true_subtask = CheckArgsToolCallSubTask(
    expected_tool_name="return_bool_response", expected_args={"response": True}
)
return_false_subtask = CheckArgsToolCallSubTask(
    expected_tool_name="return_bool_response", expected_args={"response": False}
)

start_nav_action_subtask = CheckActionFieldsToolCallSubTask(
    expected_tool_name="start_ros2_action",
    expected_action="/navigate_to_pose",
    expected_action_type="nav2_msgs/action/NavigateToPose",
    expected_fields={
        "pose": {
            "header": {"frame_id": "map"},
            "pose": {
                "position": {"x": 2.0, "y": 2.0, "z": 0.0},
            },
        },
    },
)
start_spin_action_subtask = CheckActionFieldsToolCallSubTask(
    expected_tool_name="start_ros2_action",
    expected_action="/spin",
    expected_action_type="nav2_msgs/action/Spin",
    expected_fields={"target_yaw": 3},
)
start_move_front_action_subtask = CheckActionFieldsToolCallSubTask(
    expected_tool_name="start_ros2_action",
    expected_action="/drive_on_heading",
    expected_action_type="nav2_msgs/action/DriveOnHeading",
    expected_fields={
        "target": {"y": 0.0, "z": 0.0},
    },
)
######### VALIDATORS #########################################################################################
topics_ord_val = OrderedCallsValidator(subtasks=[get_topics_subtask])
topics_and_color_image_ord_val = OrderedCallsValidator(
    subtasks=[
        get_topics_subtask,
        color_image5_subtask,
    ]
)
color_image_ord_val = OrderedCallsValidator(subtasks=[color_image5_subtask])
depth_image_ord_val = OrderedCallsValidator(subtasks=[depth_image5_subtask])
all_camera_iamges_notord_val = NotOrderedCallsValidator(
    subtasks=[
        color_image5_subtask,
        color_image5_subtask,
        depth_image5_subtask,
        depth_image5_subtask,
    ]
)

move_to_point_ord_val_grab = OrderedCallsValidator(
    subtasks=[move_to_point_subtask_grab]
)
move_to_point_ord_val_drop = OrderedCallsValidator(
    subtasks=[move_to_point_subtask_drop]
)

pub_HRIMessage_text_ord_val = OrderedCallsValidator(
    subtasks=[pub_HRIMessage_text_subtask]
)

list_topic_get_interface_publish_ord_val = OrderedCallsValidator(
    subtasks=[
        get_topics_subtask,
        get_tohuman_interface_subtask,
        pub_HRIMessage_text_subtask,
    ]
)

ret_true_ord_val = OrderedCallsValidator(subtasks=[return_true_subtask])
ret_false_ord_val = OrderedCallsValidator(subtasks=[return_false_subtask])

start_navigate_action_ord_val = OrderedCallsValidator(
    subtasks=[start_nav_action_subtask]
)
start_spin_action_ord_val = OrderedCallsValidator(subtasks=[start_spin_action_subtask])
move_ahead_ord_val = OrderedCallsValidator(subtasks=[start_move_front_action_subtask])


######### TASKS ############################################################################################
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
                    # 3 options to validate same task:
                    # most strict, agent has to call both tool correctly to pass this validator
                    GetROS2RGBCameraTask(
                        validators=[topics_and_color_image_ord_val], task_args=task_args
                    ),
                    # verifying only if the GetCameraImage call was made properly
                    GetROS2RGBCameraTask(
                        validators=[color_image_ord_val],
                        task_args=task_args,
                    ),
                    # Soft verification. verifying in separate validators the list topic and get image.
                    #  agent can get 0.5 score by only calling list topics
                    GetROS2RGBCameraTask(
                        validators=[topics_ord_val, color_image_ord_val],
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
                        validators=[all_camera_iamges_notord_val],
                        task_args=task_args,
                    ),
                ]
            )

    return tasks


def get_navigation_tasks(
    extra_tool_calls: int = 0,
    prompt_detail: List[Literal["brief", "moderate", "descriptive"]] = [
        "brief",
        "moderate",
        "descriptive",
    ],
    n_shots: List[Literal[0, 2, 5]] = [0, 2, 5],
) -> List[Task]:
    tasks: List[Task] = []

    for detail in prompt_detail:
        for shots in n_shots:
            task_args = TaskArgs(
                extra_tool_calls=extra_tool_calls,
                prompt_detail=detail,
                examples_in_system_prompt=shots,
            )
            tasks.extend(
                [
                    NavigateToPointTask(
                        validators=[start_navigate_action_ord_val],
                        task_args=task_args,
                    ),
                    SpinAroundTask(
                        validators=[start_spin_action_ord_val],
                        task_args=task_args,
                    ),
                    MoveToBedTask(
                        validators=[move_ahead_ord_val],
                        task_args=task_args,
                    ),
                    MoveToFrontTask(
                        validators=[move_ahead_ord_val],
                        task_args=task_args,
                    ),
                ]
            )

    return tasks


def get_manipulation_tasks(
    extra_tool_calls: int = 0,
    prompt_detail: List[Literal["brief", "moderate", "descriptive"]] = [
        "brief",
        "moderate",
        "descriptive",
    ],
    n_shots: List[Literal[0, 2, 5]] = [0, 2, 5],
) -> List[Task]:
    tasks: List[Task] = []

    for detail in prompt_detail:
        for shots in n_shots:
            task_args = TaskArgs(
                extra_tool_calls=extra_tool_calls,
                prompt_detail=detail,
                examples_in_system_prompt=shots,
            )
            tasks.extend(
                [
                    MoveToPointTask(
                        move_to_tool_input=MoveToPointToolInput(
                            x=1.0, y=2.0, z=3.0, task="grab"
                        ),
                        validators=[move_to_point_ord_val_grab],
                        task_args=task_args,
                    ),
                    MoveToPointTask(
                        move_to_tool_input=MoveToPointToolInput(
                            x=1.2, y=2.3, z=3.4, task="drop"
                        ),
                        validators=[move_to_point_ord_val_drop],
                        task_args=task_args,
                    ),
                ]
            )

    return tasks


def get_custom_interfaces_tasks(
    extra_tool_calls: int = 0,
    prompt_detail: List[Literal["brief", "moderate", "descriptive"]] = [
        "brief",
        "moderate",
        "descriptive",
    ],
    n_shots: List[Literal[0, 2, 5]] = [0, 2, 5],
) -> List[Task]:
    tasks: List[Task] = []

    for detail in prompt_detail:
        for shots in n_shots:
            task_args = TaskArgs(
                extra_tool_calls=extra_tool_calls,
                prompt_detail=detail,
                examples_in_system_prompt=shots,
            )
            tasks.extend(
                [
                    PublishROS2HRIMessageTextTask(
                        topic="/to_human",
                        validators=[pub_HRIMessage_text_ord_val],
                        task_args=task_args,
                        text="Hello!",
                    ),
                    PublishROS2HRIMessageTextTask(
                        topic="/to_human",
                        validators=[list_topic_get_interface_publish_ord_val],
                        task_args=task_args,
                        text="Hello!",
                    ),
                ]
            )

    return tasks


def get_spatial_tasks(
    extra_tool_calls: int = 0,
    prompt_detail: List[Literal["brief", "moderate", "descriptive"]] = [
        "brief",
        "moderate",
        "descriptive",
    ],
    n_shots: List[Literal[0, 2, 5]] = [0, 2, 5],
) -> Sequence[Task]:
    tasks: List[Task] = []

    # Categorize tasks by complexity based on question difficulty
    easy_true_inputs = [
        # Single object presence/detection
        BoolImageTaskInput(
            question="Is the light on in the room?",
            images_paths=[IMG_PATH + "image_2.jpg"],
        ),
        BoolImageTaskInput(
            question="Do you see the plant?", images_paths=[IMG_PATH + "image_2.jpg"]
        ),
        BoolImageTaskInput(
            question="Are there any pictures on the wall?",
            images_paths=[IMG_PATH + "image_3.jpg"],
        ),
        BoolImageTaskInput(
            question="Is there a pillow on the armchain?",
            images_paths=[IMG_PATH + "image_7.jpg"],
        ),
    ]

    medium_true_inputs = [
        # Object state or counting
        BoolImageTaskInput(
            question="Are there 3 pictures on the wall?",
            images_paths=[IMG_PATH + "image_4.jpg"],
        ),
    ]

    hard_true_inputs = [
        # Spatial relationships between objects
        BoolImageTaskInput(
            question="Is the door on the left from the desk?",
            images_paths=[IMG_PATH + "image_1.jpg"],
        ),
        BoolImageTaskInput(
            question="Is there a plant behind the rack?",
            images_paths=[IMG_PATH + "image_5.jpg"],
        ),
    ]

    easy_false_inputs = [
        # Single object presence/detection
        BoolImageTaskInput(
            question="Is someone in the room?", images_paths=[IMG_PATH + "image_1.jpg"]
        ),
        BoolImageTaskInput(
            question="Do you see the plant?", images_paths=[IMG_PATH + "image_3.jpg"]
        ),
        BoolImageTaskInput(
            question="Is there a red pillow on the armchair?",
            images_paths=[IMG_PATH + "image_7.jpg"],
        ),
    ]

    medium_false_inputs = [
        # Object state or counting
        BoolImageTaskInput(
            question="Is the door open?", images_paths=[IMG_PATH + "image_1.jpg"]
        ),
        BoolImageTaskInput(
            question="Are there 4 pictures on the wall?",
            images_paths=[IMG_PATH + "image_4.jpg"],
        ),
    ]

    hard_false_inputs = [
        # Spatial relationships between objects
        BoolImageTaskInput(
            question="Is there a rack on the left from the sofa?",
            images_paths=[IMG_PATH + "image_4.jpg"],
        ),
        BoolImageTaskInput(
            question="Is there a plant on the right from the window?",
            images_paths=[IMG_PATH + "image_6.jpg"],
        ),
    ]

    for detail in prompt_detail:
        for shots in n_shots:
            task_args = TaskArgs(
                extra_tool_calls=extra_tool_calls,
                prompt_detail=detail,
                examples_in_system_prompt=shots,
            )

            [
                BoolImageTaskEasy(
                    task_input=input_item,
                    validators=[ret_true_ord_val],
                    task_args=task_args,
                )
                for input_item in easy_true_inputs
            ]

            tasks.extend(
                [
                    BoolImageTaskEasy(
                        task_input=input_item,
                        validators=[ret_true_ord_val],
                        task_args=task_args,
                    )
                    for input_item in easy_true_inputs
                ]
            )

            tasks.extend(
                [
                    BoolImageTaskEasy(
                        task_input=input_item,
                        validators=[ret_false_ord_val],
                        task_args=task_args,
                    )
                    for input_item in easy_false_inputs
                ]
            )

            tasks.extend(
                [
                    BoolImageTaskMedium(
                        task_input=input_item,
                        validators=[ret_true_ord_val],
                        task_args=task_args,
                    )
                    for input_item in medium_true_inputs
                ]
            )

            tasks.extend(
                [
                    BoolImageTaskMedium(
                        task_input=input_item,
                        validators=[ret_false_ord_val],
                        task_args=task_args,
                    )
                    for input_item in medium_false_inputs
                ]
            )

            tasks.extend(
                [
                    BoolImageTaskHard(
                        task_input=input_item,
                        validators=[ret_true_ord_val],
                        task_args=task_args,
                    )
                    for input_item in hard_true_inputs
                ]
            )

            tasks.extend(
                [
                    BoolImageTaskHard(
                        task_input=input_item,
                        validators=[ret_false_ord_val],
                        task_args=task_args,
                    )
                    for input_item in hard_false_inputs
                ]
            )

    return tasks


def get_tasks(
    extra_tool_calls: int = 0,
    complexities: List[Literal["easy", "medium", "hard"]] = ["easy", "medium", "hard"],
    prompt_detail: List[Literal["brief", "moderate", "descriptive"]] = [
        "brief",
        "moderate",
        "descriptive",
    ],
    n_shots: List[Literal[0, 2, 5]] = [0, 2, 5],
    task_types: List[
        Literal[
            "basic",
            "manipulation",
            "navigation",
            "custom_interfaces",
            "spatial_reasoning",
        ]
    ] = [
        "basic",
        "manipulation",
        "navigation",
        "custom_interfaces",
        "spatial_reasoning",
    ],
) -> List[Task]:
    all_tasks: List[Task] = []
    if "basic" in task_types:
        all_tasks += get_basic_tasks(
            extra_tool_calls=extra_tool_calls,
            prompt_detail=prompt_detail,
            n_shots=n_shots,
        )
    if "custom_interfaces" in task_types:
        all_tasks += get_custom_interfaces_tasks(extra_tool_calls=extra_tool_calls)
    if "manipulation" in task_types:
        all_tasks += get_manipulation_tasks(extra_tool_calls=extra_tool_calls)
    if "navigation" in task_types:
        all_tasks += get_navigation_tasks(extra_tool_calls=extra_tool_calls)
    if "spatial_reasoning" in task_types:
        all_tasks += get_spatial_tasks(extra_tool_calls=extra_tool_calls)

    filtered_tasks: List[Task] = []
    for task in all_tasks:
        if task.complexity not in complexities:
            continue

        filtered_tasks.append(task)

    random.shuffle(all_tasks)
    return all_tasks

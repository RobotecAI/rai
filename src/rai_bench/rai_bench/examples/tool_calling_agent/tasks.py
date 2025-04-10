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

from typing import Sequence

from rai_bench.tool_calling_agent.interfaces import (
    Task,
)
from rai_bench.tool_calling_agent.tasks.basic import (
    GetAllROS2DepthCamerasTask,
    GetAllROS2RGBCamerasTask,
    GetROS2DepthCameraTask,
    GetROS2RGBCameraTask,
    GetROS2TopicsTask,
)
from rai_bench.tool_calling_agent.tasks.subtasks import (
    CheckToolCallSubTask,
)
from rai_bench.tool_calling_agent.validators import (
    NotOrderedCallsValidator,
    OrderedCallsValidator,
)

########## subtasks
get_topics_subtask = CheckToolCallSubTask(
    expected_tool_name="get_ros2_topics_names_and_types"
)

color_image_subtask = CheckToolCallSubTask(
    expected_tool_name="get_ros2_image", expected_args={"topic": "/camera_image_color"}
)
depth_image_subtask = CheckToolCallSubTask(
    expected_tool_name="get_ros2_image", expected_args={"topic": "/camera_image_depth"}
)
color_image5_subtask = CheckToolCallSubTask(
    expected_tool_name="get_ros2_image", expected_args={"topic": "/color_image5"}
)
depth_image5_subtask = CheckToolCallSubTask(
    expected_tool_name="get_ros2_image", expected_args={"topic": "/depth_image5"}
)

receive_robot_desc_subtask = CheckToolCallSubTask(
    expected_tool_name="receive_ros2_message",
    expected_args={"topic": "/robot_description"},
)

move_to_point_subtask = CheckToolCallSubTask(
    expected_tool_name="move_to_point",
    expected_args={"x": 1.0, "y": 2.0, "z": 3.0, "task": "grab"},
)
######### validators
topics_ord_val = OrderedCallsValidator(subtasks=[get_topics_subtask])
topics_and_color_image_ord_val = OrderedCallsValidator(
    subtasks=[
        get_topics_subtask,
        color_image_subtask,
    ]
)
color_image_ord_val = OrderedCallsValidator(subtasks=[color_image_subtask])
depth_image_ord_val = OrderedCallsValidator(subtasks=[depth_image_subtask])
all_color_images_notord_val = NotOrderedCallsValidator(
    subtasks=[color_image_subtask, color_image5_subtask]
)
all_depth_images_notord_val = NotOrderedCallsValidator(
    subtasks=[depth_image_subtask, depth_image5_subtask]
)

# validators=
######### tasks
tasks: Sequence[Task] = [
    # 3 options to validate same task:
    # most strict, agent has to call both tool correctly to pass this validator
    GetROS2RGBCameraTask(validators=[topics_and_color_image_ord_val]),
    # verifing only if the GetCameraImage call was made properly
    GetROS2RGBCameraTask(validators=[color_image_ord_val]),
    # Soft verification. verifing in separate vaidators the list topic and get image.
    #  agent can get 0.5 score by only calling list topics
    GetROS2RGBCameraTask(validators=[topics_ord_val, color_image_ord_val]),
    # we can also add extra tool calls to allow model to correct itself
    GetROS2RGBCameraTask(
        validators=[topics_ord_val, color_image_ord_val], extra_tool_calls=3
    ),
    GetROS2TopicsTask(validators=[topics_ord_val]),
    GetROS2DepthCameraTask(validators=[depth_image_ord_val]),
    GetAllROS2RGBCamerasTask(validators=[all_color_images_notord_val]),
    GetAllROS2DepthCamerasTask(validators=[all_depth_images_notord_val]),
    # GetRobotDescriptionTask(validators=)
    # GetROS2TopicsTask2(),
    # GetROS2MessageTask(),
    # MoveToPointTask(
    #     move_to_tool_input=MoveToPointToolInput(x=1.0, y=2.0, z=3.0, task="grab"), validators=
    # ),
    # MoveToPointTask(args={"x": 1.0, "y": 2.0, "z": 3.0, "task": "grab"}),
    # MoveToPointTask(args={"x": 1.2, "y": 2.3, "z": 3.4, "task": "drop"}),
    # GetObjectPositionsTask(
    #     objects={
    #         "carrot": [{"x": 1.0, "y": 2.0, "z": 3.0}],
    #         "apple": [{"x": 4.0, "y": 5.0, "z": 6.0}],
    #         "banana": [
    #             {"x": 7.0, "y": 8.0, "z": 9.0},
    #             {"x": 10.0, "y": 11.0, "z": 12.0},
    #         ],
    #     },
    # ),
    # GrabExistingObjectTask(
    #     object_to_grab="banana",
    #     objects={
    #         "banana": [{"x": 7.0, "y": 8.0, "z": 9.0}],
    #         "apple": [
    #             {"x": 4.0, "y": 5.0, "z": 6.0},
    #             {"x": 10.0, "y": 11.0, "z": 12.0},
    #         ],
    #     },
    # ),
    # GrabNotExistingObjectTask(
    #     object_to_grab="apple",
    #     objects={
    #         "banana": [{"x": 7.0, "y": 8.0, "z": 9.0}],
    #         "cube": [
    #             {"x": 4.0, "y": 5.0, "z": 6.0},
    #             {"x": 10.0, "y": 11.0, "z": 12.0},
    #         ],
    #     },
    # ),
    # MoveExistingObjectLeftTask(
    #     object_to_grab="banana",
    #     objects={
    #         "banana": [{"x": 7.0, "y": 8.0, "z": 9.0}],
    #         "apple": [
    #             {"x": 4.0, "y": 5.0, "z": 6.0},
    #             {"x": 10.0, "y": 11.0, "z": 12.0},
    #         ],
    #     },
    # ),
    # MoveExistingObjectFrontTask(
    #     object_to_grab="banana",
    #     objects={
    #         "banana": [{"x": 7.0, "y": 8.0, "z": 9.0}],
    #         "apple": [
    #             {"x": 4.0, "y": 5.0, "z": 6.0},
    #             {"x": 10.0, "y": 11.0, "z": 12.0},
    #         ],
    #     },
    # ),
    # SwapObjectsTask(
    #     objects={
    #         "banana": [{"x": 1.0, "y": 2.0, "z": 3.0}],
    #         "apple": [{"x": 4.0, "y": 5.0, "z": 6.0}],
    #     },
    #     objects_to_swap=["banana", "apple"],
    # ),
]

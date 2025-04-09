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
    GetROS2RGBCameraTask,
    GetROS2TopicsTask,
)
from rai_bench.tool_calling_agent.tasks.subtasks import (
    GetROS2ImageSubTask,
    GetROS2TopicNamesAndTypesSubTask,
)
from rai_bench.tool_calling_agent.validators import OrderedCallsValidator

get_topics_ord_val = OrderedCallsValidator(
    subtasks=[GetROS2TopicNamesAndTypesSubTask()]
)
get_topics_and_image_ord_val = OrderedCallsValidator(
    subtasks=[GetROS2TopicNamesAndTypesSubTask(), GetROS2ImageSubTask()]
)
get_image_ord_val = OrderedCallsValidator(subtasks=[GetROS2ImageSubTask()])


tasks: Sequence[Task] = [
    # 3 options to validate same task:
    # most strict, agent has to call both tool correctly to pass this validator
    GetROS2RGBCameraTask(validators=[get_topics_and_image_ord_val]),
    # verifing only if the GetCameraImage call was made properly
    GetROS2RGBCameraTask(validators=[get_image_ord_val]),
    # Soft verification. verifing in separate vaidators the list topic and get image.
    #  agent can get 0.5 score by only calling list topics
    GetROS2RGBCameraTask(validators=[get_topics_ord_val, get_image_ord_val]),
    # we can also add extra tool calls to allow model to correct itself
    GetROS2RGBCameraTask(
        validators=[get_topics_ord_val, get_image_ord_val], extra_tool_calls=3
    ),
    GetROS2TopicsTask(validators=[get_topics_ord_val]),
    # GetROS2DepthCameraTask(),
    # GetAllROS2RGBCamerasTask(),
    # GetROS2TopicsTask2(),
    # GetROS2MessageTask(),
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

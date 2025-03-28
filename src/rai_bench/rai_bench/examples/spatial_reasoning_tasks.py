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

from typing import List, Sequence

from rai_bench.tool_calling_agent_bench.agent_tasks_interfaces import (
    SpatialReasoningAgentTask,
)
from rai_bench.tool_calling_agent_bench.spatial_reasoning_tasks import (
    BoolImageTask,
    BoolImageTaskInput,
)

inputs: List[BoolImageTaskInput] = [
    BoolImageTaskInput(
        question="Is the door on the left from the desk?",
        images_paths=["src/rai_bench/rai_bench/examples/images/image_1.jpg"],
        expected_response=True,
    ),
    BoolImageTaskInput(
        question="Is the door open?",
        images_paths=["src/rai_bench/rai_bench/examples/images/image_1.jpg"],
        expected_response=False,
    ),
    BoolImageTaskInput(
        question="Is someone in the room?",
        images_paths=["src/rai_bench/rai_bench/examples/images/image_1.jpg"],
        expected_response=False,
    ),
    BoolImageTaskInput(
        question="Is the light on in the room?",
        images_paths=["src/rai_bench/rai_bench/examples/images/image_2.jpg"],
        expected_response=True,
    ),
    BoolImageTaskInput(
        question="Do you see the plant?",
        images_paths=["src/rai_bench/rai_bench/examples/images/image_2.jpg"],
        expected_response=True,
    ),
    BoolImageTaskInput(
        question="Do you see the plant?",
        images_paths=["src/rai_bench/rai_bench/examples/images/image_3.jpg"],
        expected_response=False,
    ),
    BoolImageTaskInput(
        question="Are there any pictures on the wall?",
        images_paths=["src/rai_bench/rai_bench/examples/images/image_3.jpg"],
        expected_response=True,
    ),
    BoolImageTaskInput(
        question="Are there 3 pictures on the wall?",
        images_paths=["src/rai_bench/rai_bench/examples/images/image_4.jpg"],
        expected_response=True,
    ),
    BoolImageTaskInput(
        question="Are there 4 pictures on the wall?",
        images_paths=["src/rai_bench/rai_bench/examples/images/image_4.jpg"],
        expected_response=False,
    ),
    BoolImageTaskInput(
        question="Is there a rack on the left from the sofa?",
        images_paths=["src/rai_bench/rai_bench/examples/images/image_4.jpg"],
        expected_response=False,
    ),
    BoolImageTaskInput(
        question="Is there a plant behind the rack?",
        images_paths=["src/rai_bench/rai_bench/examples/images/image_5.jpg"],
        expected_response=True,
    ),
    BoolImageTaskInput(
        question="Is there a plant on the right from the window?",
        images_paths=["src/rai_bench/rai_bench/examples/images/image_6.jpg"],
        expected_response=False,
    ),
    BoolImageTaskInput(
        question="Is there a pillow on the armchain?",
        images_paths=["src/rai_bench/rai_bench/examples/images/image_7.jpg"],
        expected_response=True,
    ),
    BoolImageTaskInput(
        question="Is there a red pillow on the armchair?",
        images_paths=["src/rai_bench/rai_bench/examples/images/image_7.jpg"],
        expected_response=False,
    ),
]

tasks: Sequence[SpatialReasoningAgentTask] = [
    BoolImageTask(task_input=input_item) for input_item in inputs
]

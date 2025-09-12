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

from typing import Any, List

from rai_bench.vlm_benchmark.interfaces import ImageReasoningTask
from rai_bench.vlm_benchmark.tasks.tasks import (
    BoolImageTask,
    BoolImageTaskInput,
    MultipleChoiceImageTask,
    MultipleChoiceImageTaskInput,
    QuantityImageTask,
    QuantityImageTaskInput,
)

IMG_PATH = "src/rai_bench/rai_bench/vlm_benchmark/predefined/images/"


image_1 = IMG_PATH + "image_1.jpg"

image_1_tasks: List[ImageReasoningTask[Any]] = [
    BoolImageTask(
        task_input=BoolImageTaskInput(
            question="Is the door on the left from the desk?",
            images_paths=[image_1],
            expected_answer=True,
        )
    ),
    BoolImageTask(
        task_input=BoolImageTaskInput(
            question="Is the door open?",
            images_paths=[image_1],
            expected_answer=False,
        )
    ),
    BoolImageTask(
        task_input=BoolImageTaskInput(
            question="Is someone in the room?",
            images_paths=[image_1],
            expected_answer=False,
        )
    ),
    MultipleChoiceImageTask(
        task_input=MultipleChoiceImageTaskInput(
            question="What is in the image?",
            images_paths=[image_1],
            options=[
                "gauge",
                "bed",
                "barrels",
                "lamp with a shade",
                "door",
                "boxes",
                "human",
                "desk",
                "plant",
                "roll container",
            ],
            expected_answer=["bed", "lamp with a shade", "door", "desk"],
        )
    ),
]

image_2 = IMG_PATH + "image_2.jpg"

image_2_tasks: List[ImageReasoningTask[Any]] = [
    BoolImageTask(
        task_input=BoolImageTaskInput(
            question="Is the light on in the room?",
            images_paths=[image_2],
            expected_answer=True,
        )
    ),
    BoolImageTask(
        task_input=BoolImageTaskInput(
            question="Do you see the plant?",
            images_paths=[image_2],
            expected_answer=True,
        )
    ),
    QuantityImageTask(
        task_input=QuantityImageTaskInput(
            question="How many windows are visible in the image?",
            images_paths=[image_2],
            expected_answer=2,
        )
    ),
]

image_3 = IMG_PATH + "image_3.jpg"

image_3_tasks: List[ImageReasoningTask[Any]] = [
    BoolImageTask(
        task_input=BoolImageTaskInput(
            question="Are there any pictures on the wall?",
            images_paths=[image_3],
            expected_answer=True,
        )
    ),
    BoolImageTask(
        task_input=BoolImageTaskInput(
            question="Do you see the plant?",
            images_paths=[image_3],
            expected_answer=False,
        )
    ),
    QuantityImageTask(
        task_input=QuantityImageTaskInput(
            question="How many pictures are on the wall?",
            images_paths=[image_3],
            expected_answer=3,
        )
    ),
]

image_4 = IMG_PATH + "image_4.jpg"

image_4_tasks: List[ImageReasoningTask[Any]] = [
    BoolImageTask(
        task_input=BoolImageTaskInput(
            question="Are there 3 pictures on the wall?",
            images_paths=[image_4],
            expected_answer=True,
        )
    ),
    BoolImageTask(
        task_input=BoolImageTaskInput(
            question="Are there 4 pictures on the wall?",
            images_paths=[image_4],
            expected_answer=False,
        )
    ),
    BoolImageTask(
        task_input=BoolImageTaskInput(
            question="Is there a rack on the left from the sofa?",
            images_paths=[image_4],
            expected_answer=False,
        )
    ),
    QuantityImageTask(
        task_input=QuantityImageTaskInput(
            question="How many pictures are on the wall?",
            images_paths=[image_4],
            expected_answer=3,
        )
    ),
    MultipleChoiceImageTask(
        task_input=MultipleChoiceImageTaskInput(
            question="What is in the image?",
            images_paths=[image_4],
            options=[
                "sofa",
                "gauge",
                "bed",
                "armchair",
                "barrels",
                "lamp with a shade",
                "door",
                "boxes",
                "human",
                "desk",
                "plant",
                "roll container",
                "dresser",
                "TV",
            ],
            expected_answer=["sofa", "plant", "TV", "dresser"],
        )
    ),
]

image_5 = IMG_PATH + "image_5.jpg"

image_5_tasks: List[ImageReasoningTask[Any]] = [
    BoolImageTask(
        task_input=BoolImageTaskInput(
            question="Is there a plant behind the rack?",
            images_paths=[image_5],
            expected_answer=True,
        )
    ),
]

image_6 = IMG_PATH + "image_6.jpg"

image_6_tasks: List[ImageReasoningTask[Any]] = [
    BoolImageTask(
        task_input=BoolImageTaskInput(
            question="Is there a plant on the right from the window?",
            images_paths=[image_6],
            expected_answer=False,
        )
    ),
    QuantityImageTask(
        task_input=QuantityImageTaskInput(
            question="How many pictures are on the wall?",
            images_paths=[image_6],
            expected_answer=2,
        )
    ),
]

image_7 = IMG_PATH + "image_7.jpg"

image_7_tasks: List[ImageReasoningTask[Any]] = [
    BoolImageTask(
        task_input=BoolImageTaskInput(
            question="Is there a pillow on the armchair?",
            images_paths=[image_7],
            expected_answer=True,
        )
    ),
    BoolImageTask(
        task_input=BoolImageTaskInput(
            question="Is there a red pillow on the armchair?",
            images_paths=[image_7],
            expected_answer=False,
        )
    ),
    MultipleChoiceImageTask(
        task_input=MultipleChoiceImageTaskInput(
            question="What is in the image?",
            images_paths=[image_7],
            options=[
                "sofa",
                "gauge",
                "armchair",
                "barrels",
                "door",
                "boxes",
                "human",
                "desk",
                "plant",
                "roll container",
            ],
            expected_answer=["sofa", "armchair", "door", "desk", "plant"],
        )
    ),
]

image_8 = IMG_PATH + "image_8.png"

image_8_tasks: List[ImageReasoningTask[Any]] = [
    BoolImageTask(
        task_input=BoolImageTaskInput(
            question="Do you see something anomalous in the image?",
            images_paths=[image_8],
            expected_answer=True,
        )
    ),
    QuantityImageTask(
        task_input=QuantityImageTaskInput(
            question="How many people are in the image?",
            images_paths=[image_8],
            expected_answer=1,
        )
    ),
    MultipleChoiceImageTask(
        task_input=MultipleChoiceImageTaskInput(
            question="What is in the image?",
            images_paths=[image_8],
            options=[
                "sofa",
                "gauge",
                "bed",
                "armchair",
                "barrels",
                "lamp with a shade",
                "door",
                "boxes",
                "human",
                "desk",
                "plant",
                "roll container",
                "dresser",
            ],
            expected_answer=["human", "boxes"],
        )
    ),
]

image_9 = IMG_PATH + "image_9.png"

image_9_tasks: List[ImageReasoningTask[Any]] = [
    BoolImageTask(
        task_input=BoolImageTaskInput(
            question="Do you see something anomalous in the image?",
            images_paths=[image_9],
            expected_answer=True,
        )
    ),
    QuantityImageTask(
        task_input=QuantityImageTaskInput(
            question="How many people are in the image?",
            images_paths=[image_9],
            expected_answer=1,
        )
    ),
    MultipleChoiceImageTask(
        task_input=MultipleChoiceImageTaskInput(
            question="What is in the image?",
            images_paths=[image_9],
            options=[
                "sofa",
                "gauge",
                "bed",
                "armchair",
                "barrels",
                "lamp with a shade",
                "door",
                "boxes",
                "human",
                "desk",
                "plant",
                "roll container",
                "dresser",
            ],
            expected_answer=["boxes", "human", "roll container"],
        )
    ),
]

image_10 = IMG_PATH + "image_10.png"

image_10_tasks: List[ImageReasoningTask[Any]] = [
    BoolImageTask(
        task_input=BoolImageTaskInput(
            question="Do you see something anomalous in the image?",
            images_paths=[image_10],
            expected_answer=True,
        )
    ),
    QuantityImageTask(
        task_input=QuantityImageTaskInput(
            question="How many people are in the image?",
            images_paths=[image_10],
            expected_answer=1,
        )
    ),
]

image_11 = IMG_PATH + "image_11.png"

image_11_tasks: List[ImageReasoningTask[Any]] = [
    BoolImageTask(
        task_input=BoolImageTaskInput(
            question="Is any person in the image?",
            images_paths=[image_11],
            expected_answer=True,
        )
    ),
    QuantityImageTask(
        task_input=QuantityImageTaskInput(
            question="How many barrels are in the image?",
            images_paths=[image_11],
            expected_answer=6,
        )
    ),
    QuantityImageTask(
        task_input=QuantityImageTaskInput(
            question="How many fallen barrels are in the image?",
            images_paths=[image_11],
            expected_answer=2,
        )
    ),
]

image_12 = IMG_PATH + "image_12.png"

image_12_tasks: List[ImageReasoningTask[Any]] = [
    QuantityImageTask(
        task_input=QuantityImageTaskInput(
            question="How many fallen barrels are in the image?",
            images_paths=[image_12],
            expected_answer=1,
        ),
    ),
    QuantityImageTask(
        task_input=QuantityImageTaskInput(
            question="How many people are in the image?",
            images_paths=[image_12],
            expected_answer=0,
        )
    ),
    MultipleChoiceImageTask(
        task_input=MultipleChoiceImageTaskInput(
            question="What is in the image?",
            images_paths=[image_12],
            options=[
                "sofa",
                "gauge",
                "bed",
                "armchair",
                "barrels",
                "lamp with a shade",
                "door",
                "boxes",
                "human",
                "desk",
                "plant",
                "roll container",
                "dresser",
            ],
            expected_answer=["barrels", "boxes", "roll container"],
        ),
    ),
]

image_13 = IMG_PATH + "image_13.png"

image_13_tasks: List[ImageReasoningTask[Any]] = [
    QuantityImageTask(
        task_input=QuantityImageTaskInput(
            question="How many fallen barrels are in the image?",
            images_paths=[image_13],
            expected_answer=1,
        )
    ),
    QuantityImageTask(
        task_input=QuantityImageTaskInput(
            question="How many barrels are in the image?",
            images_paths=[image_13],
            expected_answer=6,
        )
    ),
]

image_14 = IMG_PATH + "image_14.png"

image_14_tasks: List[ImageReasoningTask[Any]] = [
    BoolImageTask(
        task_input=BoolImageTaskInput(
            question="Is gauge level high?",
            images_paths=[image_14],
            expected_answer=True,
        )
    ),
    BoolImageTask(
        task_input=BoolImageTaskInput(
            question="Is gauge level low?",
            images_paths=[image_14],
            expected_answer=False,
        ),
    ),
    QuantityImageTask(
        task_input=QuantityImageTaskInput(
            question="How many barrels are in the image?",
            images_paths=[image_14],
            expected_answer=4,
        ),
    ),
    MultipleChoiceImageTask(
        task_input=MultipleChoiceImageTaskInput(
            question="What is in the image?",
            images_paths=[image_14],
            options=[
                "sofa",
                "gauge",
                "bed",
                "armchair",
                "barrels",
                "lamp with a shade",
                "door",
                "boxes",
                "human",
                "desk",
                "plant",
                "roll container",
                "dresser",
            ],
            expected_answer=["gauge", "barrels"],
        )
    ),
]

image_15 = IMG_PATH + "image_15.png"
image_15_tasks: List[ImageReasoningTask[Any]] = [
    BoolImageTask(
        task_input=BoolImageTaskInput(
            question="Is worker wearing a vest?",
            images_paths=[image_15],
            expected_answer=True,
        )
    ),
    BoolImageTask(
        task_input=BoolImageTaskInput(
            question="Is worker wearing a helmet?",
            images_paths=[image_15],
            expected_answer=False,
        )
    ),
]

image_16 = IMG_PATH + "image_16.png"

image_16_tasks: List[ImageReasoningTask[Any]] = [
    BoolImageTask(
        task_input=BoolImageTaskInput(
            question="Is worker wearing a helmet?",
            images_paths=[image_16],
            expected_answer=False,
        ),
    )
]

image_17 = IMG_PATH + "image_17.png"


image_17_tasks: List[ImageReasoningTask[Any]] = [
    QuantityImageTask(
        task_input=QuantityImageTaskInput(
            question="How many damaged boxes are in the image?",
            images_paths=[image_17],
            expected_answer=2,
        )
    ),
]


def get_spatial_tasks() -> List[ImageReasoningTask[Any]]:
    """
    Return a flat list with all predefined spatial image reasoning tasks
    declared in this module.
    """
    all_tasks: List[ImageReasoningTask[Any]] = []
    for task_group in [
        image_1_tasks,
        image_2_tasks,
        image_3_tasks,
        image_4_tasks,
        image_5_tasks,
        image_6_tasks,
        image_7_tasks,
        image_8_tasks,
        image_9_tasks,
        image_10_tasks,
        image_11_tasks,
        image_12_tasks,
        image_13_tasks,
        image_14_tasks,
        image_15_tasks,
        image_16_tasks,
        image_17_tasks,
    ]:
        all_tasks.extend(task_group)
    return all_tasks

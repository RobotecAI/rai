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

from typing import List, cast

from pydantic import BaseModel

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
true_response_inputs: List[BoolImageTaskInput] = [
    BoolImageTaskInput(
        question="Is the door on the left from the desk?",
        images_paths=[IMG_PATH + "image_1.jpg"],
        expected_answer=True,
    ),
    BoolImageTaskInput(
        question="Is the light on in the room?",
        images_paths=[IMG_PATH + "image_2.jpg"],
        expected_answer=True,
    ),
    BoolImageTaskInput(
        question="Do you see the plant?",
        images_paths=[IMG_PATH + "image_2.jpg"],
        expected_answer=True,
    ),
    BoolImageTaskInput(
        question="Are there any pictures on the wall?",
        images_paths=[IMG_PATH + "image_3.jpg"],
        expected_answer=True,
    ),
    BoolImageTaskInput(
        question="Are there 3 pictures on the wall?",
        images_paths=[IMG_PATH + "image_4.jpg"],
        expected_answer=True,
    ),
    BoolImageTaskInput(
        question="Is there a plant behind the rack?",
        images_paths=[IMG_PATH + "image_5.jpg"],
        expected_answer=True,
    ),
    BoolImageTaskInput(
        question="Is there a pillow on the armchair?",
        images_paths=[IMG_PATH + "image_7.jpg"],
        expected_answer=True,
    ),
    BoolImageTaskInput(
        question="Do you see something anomalous in the image?",
        images_paths=[IMG_PATH + "fallen_human_3.png"],
        expected_answer=True,
    ),
    BoolImageTaskInput(
        question="Do you see something anomalous in the image?",
        images_paths=[IMG_PATH + "fallen_human_2.png"],
        expected_answer=True,
    ),
    BoolImageTaskInput(
        question="Do you see something anomalous in the image?",
        images_paths=[IMG_PATH + "fallen_human_1.png"],
        expected_answer=True,
    ),
    BoolImageTaskInput(
        question="Is gauge level high?",
        images_paths=[IMG_PATH + "gauge_level.png"],
        expected_answer=True,
    ),
    BoolImageTaskInput(
        question="Is worker wearing a vest?",
        images_paths=[IMG_PATH + "woman_wo_helmet_2.png"],
        expected_answer=True,
    ),
    BoolImageTaskInput(
        question="Is any person in the image?",
        images_paths=[IMG_PATH + "fallen_barrels_1.png"],
        expected_answer=True,
    ),
]

false_response_inputs: List[BoolImageTaskInput] = [
    BoolImageTaskInput(
        question="Is the door open?",
        images_paths=[IMG_PATH + "image_1.jpg"],
        expected_answer=False,
    ),
    BoolImageTaskInput(
        question="Is someone in the room?",
        images_paths=[IMG_PATH + "image_1.jpg"],
        expected_answer=False,
    ),
    BoolImageTaskInput(
        question="Do you see the plant?",
        images_paths=[IMG_PATH + "image_3.jpg"],
        expected_answer=False,
    ),
    BoolImageTaskInput(
        question="Are there 4 pictures on the wall?",
        images_paths=[IMG_PATH + "image_4.jpg"],
        expected_answer=False,
    ),
    BoolImageTaskInput(
        question="Is there a rack on the left from the sofa?",
        images_paths=[IMG_PATH + "image_4.jpg"],
        expected_answer=False,
    ),
    BoolImageTaskInput(
        question="Is there a plant on the right from the window?",
        images_paths=[IMG_PATH + "image_6.jpg"],
        expected_answer=False,
    ),
    BoolImageTaskInput(
        question="Is there a red pillow on the armchair?",
        images_paths=[IMG_PATH + "image_7.jpg"],
        expected_answer=False,
    ),
    BoolImageTaskInput(
        question="Is worker wearing a helmet?",
        images_paths=[IMG_PATH + "woman_wo_helmet_1.png"],
        expected_answer=False,
    ),
    BoolImageTaskInput(
        question="Is worker wearing a helmet?",
        images_paths=[IMG_PATH + "woman_wo_helmet_2.png"],
        expected_answer=False,
    ),
    BoolImageTaskInput(
        question="Is gauge level low?",
        images_paths=[IMG_PATH + "gauge_level.png"],
        expected_answer=False,
    ),
]

quantity_response_inputs: List[QuantityImageTaskInput] = [
    QuantityImageTaskInput(
        question="How many barrels are in the image?",
        images_paths=[IMG_PATH + "fallen_barrels_1.png"],
        expected_answer=6,
    ),
    QuantityImageTaskInput(
        question="How many fallen barrels are in the image?",
        images_paths=[IMG_PATH + "fallen_barrels_1.png"],
        expected_answer=2,
    ),
    QuantityImageTaskInput(
        question="How many fallen barrels are in the image?",
        images_paths=[IMG_PATH + "fallen_barrels_2.png"],
        expected_answer=1,
    ),
    QuantityImageTaskInput(
        question="How many fallen barrels are in the image?",
        images_paths=[IMG_PATH + "fallen_barrels_3.png"],
        expected_answer=1,
    ),
    QuantityImageTaskInput(
        question="How many barrels are in the image?",
        images_paths=[IMG_PATH + "fallen_barrels_3.png"],
        expected_answer=6,
    ),
    QuantityImageTaskInput(
        question="How many people are in the image?",
        images_paths=[IMG_PATH + "fallen_barrels_2.png"],
        expected_answer=0,
    ),
    QuantityImageTaskInput(
        question="How many people are in the image?",
        images_paths=[IMG_PATH + "fallen_human_1.png"],
        expected_answer=1,
    ),
    QuantityImageTaskInput(
        question="How many people are in the image?",
        images_paths=[IMG_PATH + "fallen_human_2.png"],
        expected_answer=1,
    ),
    QuantityImageTaskInput(
        question="How many people are in the image?",
        images_paths=[IMG_PATH + "fallen_human_3.png"],
        expected_answer=1,
    ),
    QuantityImageTaskInput(
        question="How many barrels are in the image?",
        images_paths=[IMG_PATH + "gauge_level.png"],
        expected_answer=4,
    ),
    QuantityImageTaskInput(
        question="How many damaged boxes are in the image?",
        images_paths=[IMG_PATH + "damaged_boxes.png"],
        expected_answer=2,
    ),
    QuantityImageTaskInput(
        question="How many pictures are on the wall?",
        images_paths=[IMG_PATH + "image_3.jpg"],
        expected_answer=3,
    ),
    QuantityImageTaskInput(
        question="How many pictures are on the wall?",
        images_paths=[IMG_PATH + "image_4.jpg"],
        expected_answer=3,
    ),
    QuantityImageTaskInput(
        question="How many pictures are on the wall?",
        images_paths=[IMG_PATH + "image_6.jpg"],
        expected_answer=2,
    ),
    QuantityImageTaskInput(
        question="How many windows are visible in the image?",
        images_paths=[IMG_PATH + "image_2.jpg"],
        expected_answer=2,
    ),
]

multiple_choice_response_inputs: List[MultipleChoiceImageTaskInput] = [
    MultipleChoiceImageTaskInput(
        question="What is in the image?",
        images_paths=[IMG_PATH + "fallen_barrels_2.png"],
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
    MultipleChoiceImageTaskInput(
        question="What is in the image?",
        images_paths=[IMG_PATH + "fallen_human_1.png"],
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
    ),
    MultipleChoiceImageTaskInput(
        question="What is in the image?",
        images_paths=[IMG_PATH + "gauge_level.png"],
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
    ),
    MultipleChoiceImageTaskInput(
        question="What is in the image?",
        images_paths=[IMG_PATH + "fallen_human_2.png"],
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
    ),
    MultipleChoiceImageTaskInput(
        question="What is in the image?",
        images_paths=[IMG_PATH + "image_1.jpg"],
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
    ),
    MultipleChoiceImageTaskInput(
        question="What is in the image?",
        images_paths=[IMG_PATH + "image_4.jpg"],
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
    ),
    MultipleChoiceImageTaskInput(
        question="What is in the image?",
        images_paths=[IMG_PATH + "image_7.jpg"],
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
    ),
]


def get_spatial_tasks() -> List[ImageReasoningTask[BaseModel]]:
    true_tasks = [
        BoolImageTask(
            task_input=input_item,
        )
        for input_item in true_response_inputs
    ]
    false_tasks = [
        BoolImageTask(
            task_input=input_item,
        )
        for input_item in false_response_inputs
    ]
    quantity_tasks = [
        QuantityImageTask(
            task_input=input_item,
        )
        for input_item in quantity_response_inputs
    ]
    multiple_choice_tasks = [
        MultipleChoiceImageTask(
            task_input=input_item,
        )
        for input_item in multiple_choice_response_inputs
    ]
    return cast(
        List[ImageReasoningTask[BaseModel]],
        true_tasks + false_tasks + quantity_tasks + multiple_choice_tasks,
    )

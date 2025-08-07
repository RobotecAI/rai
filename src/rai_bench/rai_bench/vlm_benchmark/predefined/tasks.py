from typing import List, Sequence

from rai_bench.vlm_benchmark.interfaces import ImageReasoningTask
from rai_bench.vlm_benchmark.tasks.tasks import BoolImageTask, BoolImageTaskInput

IMG_PATH = "src/rai_bench/rai_bench/tool_calling_agent/predefined/images/"
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
]


true_spatial_tasks: List[ImageReasoningTask] = [
    BoolImageTask(task_input=input_item) for input_item in true_response_inputs
]
false_spatial_tasks: List[ImageReasoningTask] = [
    BoolImageTask(task_input=input_item) for input_item in false_response_inputs
]


def get_spatial_tasks() -> Sequence[ImageReasoningTask]:
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
    return true_tasks + false_tasks

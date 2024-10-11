# Copyright (C) 2024 Robotec.AI
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

import logging

import coloredlogs
import numpy as np
import rclpy
import rclpy.callback_groups
import rclpy.executors
import rclpy.qos
import rclpy.subscription
import rclpy.task
from langchain.tools.render import render_text_description_and_args

from rai.agents.conversational_agent import create_conversational_agent
from rai.node import RaiNode
from rai.tools.ros.manipulation import GetObjectPositionsTool, MoveToPointTool
from rai.tools.ros.native import GetCameraImage
from rai.utils.model_initialization import get_llm_model, get_tracing_callbacks

coloredlogs.install(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    rclpy.init()
    node = RaiNode(
        llm=get_llm_model(
            model_type="simple_model"
        ),  # smaller model used to describe the environment
        system_prompt="",
    )
    tools = [
        MoveToPointTool(node=node),
        GetObjectPositionsTool(
            node=node,
            source_frame="RGBDCamera5",
            target_frame="world",
            camera_topic="/color_image5",
            depth_topic="/depth_image5",
            camera_info_topic="/color_camera_info5",
        ),
        GetCameraImage(node=node),
    ]

    SYSTEM_PROMPT = f"""You are an autonomous robotic arm connected to ros2 environment. Your main goal is to fulfill the user's requests.
    Do not make assumptions about the environment you are currently in. You may be asked to move objects around.

    Coordinates are in meters. System is as follows:
    x - forward
    y - right
    z - up

    Arm can move in the following ranges:
    x - [0.2, 0.5]
    y - [-0.6, 0.65]
    z - [0.0, 1.0]

    User specified positions such as next, on top, etc. Make sure to follow them. The positions gathered from get_objects_positions tool are the centroids of the objects. Sometimes you may have to do some calculations to properly place the objects.
    When you drop the object, and you want to pick it up again, you must first get the positions of the objects again.
    The gripper can hold one object at a time. (if you pick up an object, you must always place it before picking up another one)

    After finishing the task/subtask, use camera to confirm the result. If you've failed, repeat the task. Make sure to always grab the latest position of the objects.

    Make sure to properly define class which get_objects_positions tool shoud look for. For example, do not query item, query box/cube/apple/cup etc. Do not use vague names, but specific ones (make sure to specify color).
    Make sure to always grab the item, do not grab the air.
    Mind the space limitations (eg limits or existing objects).
    Do not place objects on the position of other object unless user explicitly tells you to do so- this is very important. You must always adhere to this rule.
    Remember where you have already placed the objects.
    Do not move things unless it is necessary for the task.

    Do not run the tools at once, but execute them one by one- this is very important and you should always adhere to this rule.
    Run only one tool per turn. You will be givent another turn every time you use the tool.
    Do not make out coordinates of the objects by yourself.
    Use the tooling provided to gather information about the positions of the objects.
    Do not modify the coordinates (especially z axis, if you change it you wont grab the object). The move_to_point_tool has necessary calibration values.

    Use the tooling provided to gather information about the environment:

    {render_text_description_and_args(tools)}

    You can use ros2 topics, services and actions to operate.
    You can use /color_image5 to see the environment.
    When you get the task, make sure to plan first. Do planning every step of the way.
    After finishing the task, use camera to confirm the result. If you've failed, repeat the missing steps.-this is very important.
    Be observant and careful. Make sure the task has been finished properly.- this is very important.

    Examples:
    User: "Pick up the carrot"
     - Use get_objects_positions tool with argument "carrot" to get the position of the carrot.
     - Use move_to_point tool with the returned position and argument "grab" to pick up the carrot.

    User: "Swap the position of the cube at the left bottom and the carrot"
     - Use get_objects_positions to find the cube, reason which one is the proper cube.
     - Use get_objects_positions to find the carrot.
     - move the cube to the temporary, empty location, at least 0.05m away from other objects.
     - move the carrot to the previous position of the cube.
     - move the cube to the previous position of the carrot.

    This are some of the examples of the tasks you can perform. When asked for harder task, make sure to plan the task properly and if applicable use the examples as a guide.
    """

    agent = create_conversational_agent(
        llm=get_llm_model("complex_model"),
        tools=tools,
        system_prompt=SYSTEM_PROMPT,
        logger=logger,
    )

    tasks = [
        "Pick up the cube in the bottom left corner of the table and swap it with the carrot.",
        "Pick up the cube in the bottom left corner of the table and place it on top of the cube in the bottom right corner of the table.",
        "Swap positions of the cubes with the vegetables.",
        "Build a tower with all of the cubes.",
        "Put one vegetable on top of every cube.",
        "Pick up the cube in the bottom left corner of the table and throw it away.",
        "Arrange the vegetables alternately with the cubes in a line along the the y axis.",
    ]

    task = np.random.choice(tasks)
    logger.info(f"Starting task: {task}")

    for output in agent.stream(
        {"messages": [task]},
        config={"recursion_limit": 100, "callbacks": get_tracing_callbacks()},
    ):
        if "thinker" in output:
            logger.info(output["thinker"]["messages"][-1].pretty_repr())


if __name__ == "__main__":
    main()

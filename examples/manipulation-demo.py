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
# See the License for the specific language goveself.rning permissions and
# limitations under the License.


import asyncio
import logging
from typing import List

import rclpy
import rclpy.qos
from langchain_core.tools import BaseTool
from rai import get_llm_model
from rai.agents.langchain.core.plan_and_execute import create_plan_and_execute_runnable
from rai.communication.ros2 import wait_for_ros2_services, wait_for_ros2_topics
from rai.communication.ros2.connectors import ROS2Connector
from rai.tools.ros2.manipulation import GetObjectPositionsTool, MoveToPointTool
from rai.tools.ros2.simple import GetROS2ImageConfiguredTool
from rai_open_set_vision.tools import GetGrabbingPointTool

from rai_whoami.models import EmbodimentInfo

logger = logging.getLogger(__name__)


def create_agent():
    rclpy.init()
    connector = ROS2Connector(executor_type="single_threaded")

    required_services = ["/grounded_sam_segment", "/grounding_dino_classify"]
    required_topics = ["/color_image5", "/depth_image5", "/color_camera_info5"]
    wait_for_ros2_services(connector, required_services)
    wait_for_ros2_topics(connector, required_topics)

    node = connector.node
    node.declare_parameter("conversion_ratio", 1.0)

    camera_tool = GetROS2ImageConfiguredTool(connector=connector, topic="/color_image5")
    tools: List[BaseTool] = [
        GetObjectPositionsTool(
            connector=connector,
            target_frame="panda_link0",
            source_frame="RGBDCamera5",
            camera_topic="/color_image5",
            depth_topic="/depth_image5",
            camera_info_topic="/color_camera_info5",
            get_grabbing_point_tool=GetGrabbingPointTool(connector=connector),
        ),
        MoveToPointTool(connector=connector, manipulator_frame="panda_link0"),
        # camera_tool
    ]

    llm = get_llm_model(model_type="complex_model", streaming=True)
    embodiment_info = EmbodimentInfo.from_file(
        "examples/embodiments/manipulation_embodiment.json"
    )

    agent = create_plan_and_execute_runnable(
        llm=llm,
        tools=tools,
        system_prompt=embodiment_info._to_markdown()
        + "left/right - x axis, front/back - y axis",
        camera_tool=camera_tool,
    )
    return agent


async def main():
    agent = create_agent()

    prompt = "Place the carrot close to the apple"
    messages = {"input": prompt}
    config = {"recursion_limit": 100}

    async for event in agent.astream(messages, config=config):
        for k, v in event.items():
            if k != "__end__":
                print(v)


asyncio.run(main())

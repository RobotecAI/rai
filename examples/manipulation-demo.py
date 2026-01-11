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


import logging
from typing import List

from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.tools import BaseTool
from rai import get_llm_model
from rai.agents.langchain.core import create_react_runnable
from rai.communication.ros2 import wait_for_ros2_services, wait_for_ros2_topics
from rai.communication.ros2.connectors import ROS2Connector
from rai.tools.ros2.manipulation import (
    GetObjectPositionsTool,
    MoveObjectFromToTool,
    ResetArmTool,
)
from rai.tools.ros2.simple import GetROS2ImageConfiguredTool
from rai_perception.tools import GetGrabbingPointTool

from rai_whoami.models import EmbodimentInfo

logger = logging.getLogger(__name__)


def create_agent():
    connector = ROS2Connector(executor_type="single_threaded")

    required_services = ["/segmentation", "/detection"]
    required_topics = ["/color_image5", "/depth_image5", "/color_camera_info5"]
    wait_for_ros2_services(connector, required_services)
    wait_for_ros2_topics(connector, required_topics)

    node = connector.node
    node.declare_parameter("conversion_ratio", 1.0)

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
        MoveObjectFromToTool(connector=connector, manipulator_frame="panda_link0"),
        ResetArmTool(connector=connector, manipulator_frame="panda_link0"),
        GetROS2ImageConfiguredTool(connector=connector, topic="/color_image5"),
    ]

    llm = get_llm_model(model_type="complex_model", streaming=True)
    embodiment_info = EmbodimentInfo.from_file(
        "examples/embodiments/manipulation_embodiment.json"
    )

    agent = create_react_runnable(
        llm=llm,
        tools=tools,
        system_prompt=embodiment_info.to_langchain(),
    )
    return agent


def main():
    agent = create_agent()
    messages: List[BaseMessage] = []

    while True:
        prompt = input("Enter a prompt: ")
        messages.append(HumanMessage(content=prompt))
        output = agent.invoke({"messages": messages})
        output["messages"][-1].pretty_print()


if __name__ == "__main__":
    main()

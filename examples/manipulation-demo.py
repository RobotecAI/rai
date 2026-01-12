# Copyright (C) 2025 Julia Jia
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
from typing import List

import rclpy
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.tools import BaseTool
from rai import get_llm_model
from rai.agents.langchain.core import create_conversational_agent
from rai.communication.ros2.connectors import ROS2Connector
from rai.tools.ros2.manipulation import (
    MoveObjectFromToTool,
    ResetArmTool,
)
from rai.tools.ros2.simple import GetROS2ImageConfiguredTool
from rai_perception import (
    GetObjectGrippingPointsTool,
    wait_for_perception_dependencies,
)
from rai_perception.tools.gripping_points_tools import GRIPPING_POINTS_TOOL_PARAM_PREFIX

from rai_whoami.models import EmbodimentInfo

logger = logging.getLogger(__name__)


def create_agent():
    """Create and configure the manipulation agent.

    GetObjectGrippingPointsTool auto-declares ROS2 parameters with defaults.
    To override parameters for your deployment, declare them before tool initialization:

        node.declare_parameter(f"{GRIPPING_POINTS_TOOL_PARAM_PREFIX}.camera_topic", "/your/topic")
        node.declare_parameter(f"{GRIPPING_POINTS_TOOL_PARAM_PREFIX}.target_frame", "your_frame")
        # ... etc

    Or set them in launch files/YAML configs. See tool logs for current parameter values.
    """
    rclpy.init()
    connector = ROS2Connector(executor_type="single_threaded")
    node = connector.node

    # Set ROS2 parameters to match robot/simulation topic and frame names
    # For O3DE simulation, these are the default values
    # To use different values, change these or set them in a launch file
    node.declare_parameter(
        f"{GRIPPING_POINTS_TOOL_PARAM_PREFIX}.camera_topic", "/color_image5"
    )
    node.declare_parameter(
        f"{GRIPPING_POINTS_TOOL_PARAM_PREFIX}.depth_topic", "/depth_image5"
    )
    node.declare_parameter(
        f"{GRIPPING_POINTS_TOOL_PARAM_PREFIX}.camera_info_topic", "/color_camera_info5"
    )
    node.declare_parameter(
        f"{GRIPPING_POINTS_TOOL_PARAM_PREFIX}.target_frame", "panda_link0"
    )
    node.declare_parameter(
        f"{GRIPPING_POINTS_TOOL_PARAM_PREFIX}.source_frame", "RGBDCamera5"
    )

    # Initialize tools - GetObjectGrippingPointsTool will use the parameters we just set
    gripping_points_tool = GetObjectGrippingPointsTool(connector=connector)
    config = gripping_points_tool.get_config()

    tools: List[BaseTool] = [
        gripping_points_tool,
        MoveObjectFromToTool(
            connector=connector, manipulator_frame=config["target_frame"]
        ),
        ResetArmTool(connector=connector, manipulator_frame=config["target_frame"]),
        GetROS2ImageConfiguredTool(connector=connector, topic=config["camera_topic"]),
    ]

    wait_for_perception_dependencies(connector, tools)

    llm = get_llm_model(model_type="complex_model", streaming=True)
    embodiment_info = EmbodimentInfo.from_file(
        "examples/embodiments/manipulation_embodiment.json"
    )
    agent = create_conversational_agent(
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

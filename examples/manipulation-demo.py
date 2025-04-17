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


import rclpy
import rclpy.qos
from langchain_core.messages import HumanMessage
from rai import get_llm_model
from rai.agents.conversational_agent import create_conversational_agent
from rai.communication.ros2.connectors import ROS2Connector
from rai.tools.ros2 import GetROS2ImageTool, GetROS2TopicsNamesAndTypesTool
from rai.tools.ros2.manipulation import GetObjectPositionsTool, MoveToPointTool
from rai_open_set_vision.tools import GetGrabbingPointTool


def create_agent():
    rclpy.init()
    connector = ROS2Connector()
    node = connector.node
    node.declare_parameter("conversion_ratio", 1.0)

    tools = [
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
        GetROS2ImageTool(connector=connector),
        GetROS2TopicsNamesAndTypesTool(connector=connector),
    ]

    llm = get_llm_model(model_type="complex_model", streaming=True)

    system_prompt = """
    You are a robotic arm with interfaces to detect and manipulate objects.
    Here are the coordinates information:
    x - front to back (positive is forward)
    y - left to right (positive is right)
    z - up to down (positive is up)

    Before starting the task, make sure to grab the camera image to understand the environment.
    """

    agent = create_conversational_agent(
        llm=llm,
        tools=tools,
        system_prompt=system_prompt,
    )
    return agent


def main():
    agent = create_agent()
    messages = []
    while True:
        prompt = input("Enter a prompt: ")
        messages.append(HumanMessage(content=prompt))
        output = agent.invoke({"messages": messages})
        output["messages"][-1].pretty_print()


if __name__ == "__main__":
    main()

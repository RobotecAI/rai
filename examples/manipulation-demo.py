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

from typing import Literal, cast

import rclpy
from langchain_core.messages import HumanMessage
from sensor_msgs.msg import JointState

from rai.agents.conversational_agent import create_conversational_agent
from rai.node import RaiBaseNode
from rai.tools.ros.manipulation import GetObjectPositionsTool, MoveToPointTool
from rai.tools.ros.native import GetCameraImage, Ros2GetTopicsNamesAndTypesTool
from rai.utils.model_initialization import get_llm_model, get_tracing_callbacks

rclpy.init()
node = RaiBaseNode(node_name="manipulation_demo")
node.declare_parameter("conversion_ratio", 1.0)


def get_finger_state() -> Literal["open", "closed", "holding", "unknown"]:
    try:
        msg = cast(JointState, node.get_raw_message_from_topic("/joint_states"))
        first_finger_index = msg.name.index("panda_finger_joint1")
        second_finger_index = msg.name.index("panda_finger_joint2")
        first_finger_position = msg.position[first_finger_index]
        second_finger_position = msg.position[second_finger_index]
        delta_position = first_finger_position + second_finger_position
        if delta_position < 0.01:
            return "closed"
        elif delta_position > 0.01 and delta_position < 0.07:
            return "holding"
        else:
            return "open"
    except Exception:
        return "unknown"


tools = [
    GetObjectPositionsTool(
        node=node,
        target_frame="panda_link0",
        source_frame="RGBDCamera5",
        camera_topic="/color_image5",
        depth_topic="/depth_image5",
        camera_info_topic="/color_camera_info5",
    ),
    MoveToPointTool(
        node=node, manipulator_frame="panda_link0", finger_state=get_finger_state
    ),
    GetCameraImage(node=node),
    Ros2GetTopicsNamesAndTypesTool(node=node),
]

llm = get_llm_model(model_type="complex_model")

system_prompt = """
You are an expert robotic arm operator with interfaces to detect and manipulate objects.
Here are the coordinates information:
x - front to back (positive is forward)
y - left to right (positive is right)
z - up to down (positive is up)

When moving the object please mind the following:
- Height of the objects. The cubes are 0.05m high. Do not place one object on the exact same position as another object. You can drop it higher.
- Distance between objects. The minimum safe distance between objects is 0.1m.
- You can only hold one object at a time.

Before starting the task, make sure to grab the camera image to understand the environment.
"""

agent = create_conversational_agent(
    llm=llm,
    tools=tools,
    system_prompt=system_prompt,
)

messages = []
while True:
    prompt = input("Enter a prompt: ")
    messages.append(HumanMessage(content=prompt))
    output = agent.invoke(
        {"messages": messages}, config={"callbacks": get_tracing_callbacks()}
    )
    output["messages"][-1].pretty_print()

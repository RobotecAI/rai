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
from typing import List, Literal, Tuple

from langchain_core.tools import BaseTool
from rai import get_llm_model
from rai.agents.langchain.core import create_react_runnable
from rai.communication.ros2 import wait_for_ros2_services, wait_for_ros2_topics
from rai.communication.ros2.connectors import ROS2Connector
from rai.tools.ros2.manipulation import (
    GetObjectPositionsTool as LegacyGetObjectPositionsTool,
)
from rai.tools.ros2.manipulation import (
    MoveObjectFromToTool,
    ResetArmTool,
)
from rai.tools.ros2.simple import GetROS2ImageConfiguredTool
from rai_perception import (
    GetObjectGrippingPointsTool,
    wait_for_perception_dependencies,
)
from rai_perception.tools import GetGrabbingPointTool, GetObjectPositionsTool
from rai_perception.tools.gripping_points_tools import GRIPPING_POINTS_TOOL_PARAM_PREFIX

from rai_whoami.models import EmbodimentInfo

logger = logging.getLogger(__name__)


def create_agent(
    version: Literal["v1", "v2"] = "v1",
) -> Tuple:
    """Create and configure the manipulation agent.

    Args:
        version: Agent version to create. "v1" uses legacy tools and create_react_runnable,
                 "v2" uses GetObjectGrippingPointsTool and create_conversational_agent.

    Returns:
        Tuple of (agent, camera_tool) for both versions

    For v2, GetObjectGrippingPointsTool auto-declares ROS2 parameters with defaults.
    To override parameters for your deployment, declare them before tool initialization:

        node.declare_parameter(f"{GRIPPING_POINTS_TOOL_PARAM_PREFIX}.camera_topic", "/your/topic")
        node.declare_parameter(f"{GRIPPING_POINTS_TOOL_PARAM_PREFIX}.target_frame", "your_frame")
        # ... etc

    Or set them in launch files/YAML configs. See tool logs for current parameter values.

    Note: v2 uses the new model-agnostic service names (/detection, /segmentation).
    Legacy service names are disabled in run_perception_services.py.
    """
    if version == "v1":
        return _create_agent_v1()
    elif version == "v2":
        return _create_agent_v2()
    else:
        raise ValueError(f"Unknown version: {version}. Must be 'v1' or 'v2'")


def _create_agent_v1():
    """Create v1 agent with legacy tools."""
    connector = ROS2Connector(executor_type="single_threaded")

    required_services = ["/grounded_sam_segment", "/grounding_dino_classify"]
    required_topics = ["/color_image5", "/depth_image5", "/color_camera_info5"]
    wait_for_ros2_services(connector, required_services)
    wait_for_ros2_topics(connector, required_topics)

    node = connector.node
    node.declare_parameter("conversion_ratio", 1.0)

    camera_tool = GetROS2ImageConfiguredTool(connector=connector, topic="/color_image5")
    tools: List[BaseTool] = [
        LegacyGetObjectPositionsTool(
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
        camera_tool,
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
    return agent, camera_tool


def _create_agent_v2():
    """Create v2 agent with new gripping points tool."""

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

    # GetObjectPositionsTool wraps GetObjectGrippingPointsTool with default_grasp preset
    # for backward compatibility and clearer tool selection
    positions_tool = GetObjectPositionsTool(connector=connector)

    tools: List[BaseTool] = [
        positions_tool,  # get_object_positions - for general position queries
        gripping_points_tool,  # get_object_gripping_points - for advanced gripping strategies
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
    agent = create_react_runnable(
        llm=llm,
        tools=tools,
        system_prompt=embodiment_info.to_langchain(),
    )
    # Extract camera_tool from tools for consistency with v1
    camera_tool = tools[-1]  # camera_tool is the last tool
    return agent, camera_tool

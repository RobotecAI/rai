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
# See the License for the specific language goveself.rning permissions and
# limitations under the License.


import logging
from typing import List

import rclpy
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.tools import BaseTool
from rai import get_llm_model
from rai.agents.langchain.core import create_conversational_agent
from rai.communication.ros2 import wait_for_ros2_services, wait_for_ros2_topics
from rai.communication.ros2.connectors import ROS2Connector
from rai.tools.ros2.manipulation import (
    MoveObjectFromToTool
)
from rai.tools.ros2.simple import GetROS2ImageConfiguredTool
from rai_open_set_vision import (
    GetObjectGrippingPointsTool,
    GrippingPointEstimatorConfig,
    PointCloudFilterConfig,
    PointCloudFromSegmentationConfig,
)

from rai_whoami.models import EmbodimentInfo

logger = logging.getLogger(__name__)
param_prefix = "pcl.detection.gripping_points"


def initialize_tools(connector: ROS2Connector, camera_tool: GetROS2ImageConfiguredTool) -> List[BaseTool]:
    """Initialize and configure all tools for the manipulation agent."""
    node = connector.node

    # Parameters for GetObjectGrippingPointsTool, these also can be set in the launch file or load from yaml file
    parameters_to_set = [
        (f"{param_prefix}.target_frame", "panda_link0"),
        (f"{param_prefix}.source_frame", "RGBDCamera5"),
        (f"{param_prefix}.camera_topic", "/color_image5"),
        (f"{param_prefix}.depth_topic", "/depth_image5"),
        (f"{param_prefix}.camera_info_topic", "/color_camera_info5"),
        (f"{param_prefix}.timeout_sec", 10.0),
        (f"{param_prefix}.conversion_ratio", 1.0),
    ]

    for param_name, param_value in parameters_to_set:
        node.declare_parameter(param_name, param_value)

    # Configure gripping point detection algorithms
    segmentation_config = PointCloudFromSegmentationConfig(
        box_threshold=0.35,
        text_threshold=0.45,
    )

    estimator_config = GrippingPointEstimatorConfig(
        strategy="centroid",  # Options: "centroid", "top_plane", "biggest_plane"
        top_percentile=0.05,
        plane_bin_size_m=0.01,
        ransac_iterations=200,
        distance_threshold_m=0.01,
        min_points=10,
    )

    filter_config = PointCloudFilterConfig(
        strategy="isolation_forest",  # Options: "dbscan", "kmeans_largest_cluster", "isolation_forest", "lof"
        if_max_samples="auto",
        if_contamination=0.05,
        min_points=20,
    )

    manipulator_frame = node.get_parameter(f"{param_prefix}.target_frame").value
    camera_topic = node.get_parameter(f"{param_prefix}.camera_topic").value

    tools: List[BaseTool] = [
        GetObjectGrippingPointsTool(
            connector=connector,
            segmentation_config=segmentation_config,
            estimator_config=estimator_config,
            filter_config=filter_config,
        ),
        MoveObjectFromToTool(connector=connector, manipulator_frame=manipulator_frame),
        camera_tool
    ]

    return tools


def wait_for_ros2_services_and_topics(connector: ROS2Connector):
    required_services = ["/grounded_sam_segment", "/grounding_dino_classify"]
    required_topics = [
        connector.node.get_parameter(f"{param_prefix}.camera_topic").value,
        connector.node.get_parameter(f"{param_prefix}.depth_topic").value,
        connector.node.get_parameter(f"{param_prefix}.camera_info_topic").value,
    ]

    wait_for_ros2_services(connector, required_services)
    wait_for_ros2_topics(connector, required_topics)


def create_agent():
    if not rclpy.ok():
        rclpy.init()
    connector = ROS2Connector(executor_type="single_threaded")

    camera_tool = GetROS2ImageConfiguredTool(connector=connector, topic="/color_image5")
    tools = initialize_tools(connector, camera_tool)
    wait_for_ros2_services_and_topics(connector)

    llm = get_llm_model(model_type="complex_model", streaming=True)
    embodiment_info = EmbodimentInfo.from_file(
        "examples/embodiments/manipulation_embodiment.json"
    )
    agent = create_conversational_agent(
        llm=llm,
        tools=tools,
        system_prompt=embodiment_info.to_langchain(),
        camera_tool=camera_tool,
        logger=logger,
        connector=connector,
        manipulator_frame="panda_link0",
    )
    return agent, camera_tool


def main():
    agent, camera_tool = create_agent()
    messages: List[BaseMessage] = []

    while True:
        prompt = input("Enter a prompt: ")
        messages.append(HumanMessage(content=prompt))
        output = agent.invoke({"messages": messages})
        output["messages"][-1].pretty_print()


if __name__ == "__main__":
    main()

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

import rclpy
import rclpy.qos
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.tools import BaseTool
from rai import get_llm_model
from rai.agents.langchain.core import create_conversational_agent
from rai.communication.ros2 import wait_for_ros2_services, wait_for_ros2_topics
from rai.communication.ros2.connectors import ROS2Connector
from rai.tools.ros2.detection.pcl import (
    GrippingPointEstimatorConfig,
    PointCloudFilterConfig,
    PointCloudFromSegmentationConfig,
)
from rai.tools.ros2.detection.tools import GetGrippingPointTool
from rai.tools.ros2.manipulation import (
    MoveObjectFromToTool,
    ResetArmTool,
)
from rai.tools.ros2.simple import GetROS2ImageConfiguredTool

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

    # Declare and set parameters for GetGrippingPointTool
    # These also can be set in the launch file or during runtime
    parameters_to_set = [
        ("conversion_ratio", 1.0),
        ("detection_tools.gripping_point.target_frame", "panda_link0"),
        ("detection_tools.gripping_point.source_frame", "RGBDCamera5"),
        ("detection_tools.gripping_point.camera_topic", "/color_image5"),
        ("detection_tools.gripping_point.depth_topic", "/depth_image5"),
        ("detection_tools.gripping_point.camera_info_topic", "/color_camera_info5"),
    ]

    # Declare and set each parameter (timeout_sec handled by tool internally)
    for param_name, param_value in parameters_to_set:
        node.declare_parameter(param_name, param_value)

    # Configure gripping point detection algorithms
    segmentation_config = PointCloudFromSegmentationConfig(
        box_threshold=0.35,
        text_threshold=0.45,
    )

    estimator_config = GrippingPointEstimatorConfig(
        strategy="biggest_plane",  # Options: "centroid", "top_plane", "biggest_plane"
        top_percentile=0.05,
        plane_bin_size_m=0.01,
        ransac_iterations=200,
        distance_threshold_m=0.01,
        min_points=10,
    )

    filter_config = PointCloudFilterConfig(
        strategy="dbscan",
        min_points=20,
        dbscan_eps=0.02,
        dbscan_min_samples=10,
    )

    tools: List[BaseTool] = [
        GetGrippingPointTool(
            connector=connector,
            segmentation_config=segmentation_config,
            estimator_config=estimator_config,
            filter_config=filter_config,
        ),
        MoveObjectFromToTool(connector=connector, manipulator_frame="panda_link0"),
        ResetArmTool(connector=connector, manipulator_frame="panda_link0"),
        GetROS2ImageConfiguredTool(connector=connector, topic="/color_image5"),
    ]

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

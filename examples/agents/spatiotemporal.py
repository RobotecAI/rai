# Copyright (C) 2025 Robotec.AI
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
from rai.agents.spatiotemporal import ROS2SpatioTemporalAgent, ROS2SpatioTemporalConfig
from rai.utils.model_initialization import get_llm_model


def create_agent():
    config = ROS2SpatioTemporalConfig(
        robot_frame="base_link",
        world_frame="world",
        db_url="mongodb://localhost:27017/",
        db_name="rai",
        collection_name="spatiotemporal_collection",
        image_to_text_model=get_llm_model("simple_model"),
        context_compression_model=get_llm_model("simple_model"),
        time_interval=10.0,
        camera_topics=["/camera/camera/color/image_raw"],
    )
    agent = ROS2SpatioTemporalAgent(config)
    return agent


def main():
    rclpy.init()
    agent = create_agent()
    agent.run()


if __name__ == "__main__":
    main()

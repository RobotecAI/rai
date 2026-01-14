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
# See the License for the specific language governing permissions and
# limitations under the License.

from rai.agents import wait_for_shutdown
from rai.agents.langchain import StateBasedConfig
from rai.agents.ros2 import ROS2StateBasedAgent
from rai.aggregators.ros2 import (
    ROS2ImgVLMDiffAggregator,
    ROS2LogsAggregator,
)
from rai.communication.ros2 import (
    ROS2Connector,
    ROS2Context,
    ROS2HRIConnector,
)
from rai.tools.ros2.generic.toolkit import ROS2Toolkit


@ROS2Context()
def main():
    hri_connector = ROS2HRIConnector()
    ros2_connector = ROS2Connector()

    config = StateBasedConfig(
        aggregators={
            ("/camera/camera/color/image_raw", "sensor_msgs/msg/Image"): [
                ROS2ImgVLMDiffAggregator()
            ],
            "/rosout": [
                ROS2LogsAggregator()
            ],  # if msg_type is not provided, topic has to exist
        }
    )

    agent = ROS2StateBasedAgent(
        config=config,
        target_connectors={"to_human": hri_connector},
        tools=ROS2Toolkit(connector=ros2_connector).get_tools(),
    )
    agent.subscribe_source("/from_human", hri_connector)
    agent.run()
    wait_for_shutdown([agent])


if __name__ == "__main__":
    main()

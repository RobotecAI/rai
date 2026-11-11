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

import rai
from rai.agents import HRIOutput, StateBasedAgent, hri_connector_sender
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

    camera, logs = ROS2ImgVLMDiffAggregator(), ROS2LogsAggregator()
    ros2_connector.register_callback(
        "/camera/camera/color/image_raw",
        camera,
        raw=True,
        msg_type="sensor_msgs/msg/Image",
    )
    # if msg_type is not provided, the topic has to exist
    ros2_connector.register_callback("/rosout", logs, raw=True)

    agent = StateBasedAgent(
        tools=ROS2Toolkit(connector=ros2_connector).get_tools(),
        state_sources={"camera": camera.observe, "/rosout": logs.observe},
    )
    agent.subscribe(HRIOutput(hri_connector_sender(hri_connector, "to_human")))
    hri_connector.register_callback("/from_human", agent.submit)
    rai.run(agent)


if __name__ == "__main__":
    main()

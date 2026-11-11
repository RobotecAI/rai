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
from rai.agents import HRIOutput, ReActAgent, hri_connector_sender
from rai.communication.ros2 import ROS2Connector, ROS2Context
from rai.communication.ros2.connectors.hri_connector import ROS2HRIConnector
from rai.tools.ros2 import ROS2Toolkit


@ROS2Context()
def main():
    ros2_connector = ROS2Connector()
    hri_connector = ROS2HRIConnector()

    agent = ReActAgent(tools=ROS2Toolkit(connector=ros2_connector).get_tools())
    agent.subscribe(HRIOutput(hri_connector_sender(hri_connector, "/to_human")))
    # Agent will wait for messages published to /from_human ros2 topic
    hri_connector.register_callback("/from_human", agent.submit)
    rai.run(agent)


if __name__ == "__main__":
    main()

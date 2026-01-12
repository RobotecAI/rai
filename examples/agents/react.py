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


from rai.agents.langchain import ReActAgent
from rai.communication.hri_connector import HRIMessage
from rai.communication.ros2 import ROS2Connector, ROS2Context, ROS2HRIConnector
from rai.tools.ros2 import ROS2Toolkit


@ROS2Context()
def main():
    ros2_connector = ROS2Connector()
    hri_connector = ROS2HRIConnector()

    agent = ReActAgent(
        target_connectors={
            "/to_human": hri_connector,
        },  # agent's output is sent to /to_human ros2 topic
        tools=ROS2Toolkit(connector=ros2_connector).get_tools(),
    )
    agent.run()
    agent(HRIMessage(text="What do you see?"))
    agent.wait()  # wait for agent to finish
    agent.stop()


if __name__ == "__main__":
    main()

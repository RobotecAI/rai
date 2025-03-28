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

from rai.agents import ReActAgent
from rai.communication import ROS2ARIConnector, ROS2HRIConnector
from rai.tools.ros2 import ROS2ActionToolkit, ROS2ServicesToolkit, ROS2TopicsToolkit
from rai.utils import ROS2Context, wait_for_shutdown


@ROS2Context()
def main():
    connector = ROS2HRIConnector(sources=["/from_human"], targets=["/to_human"])
    ari_connector = ROS2ARIConnector()
    agent = ReActAgent(
        connectors={"hri": connector},
        tools=[
            *ROS2TopicsToolkit(connector=ari_connector).get_tools(),
            *ROS2ServicesToolkit(connector=ari_connector).get_tools(),
            *ROS2ActionToolkit(connector=ari_connector).get_tools(),
        ],
    )  # type: ignore
    agent.run()
    wait_for_shutdown([agent])


if __name__ == "__main__":
    main()

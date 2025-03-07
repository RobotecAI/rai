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
from rai.agents.simple_agent import SimpleAgent
from rai.communication.ros2.connectors import ROS2HRIConnector


def main():
    rclpy.init()
    connector = ROS2HRIConnector(targets=["/to_human"], sources=["/from_human"])
    agent = SimpleAgent(connectors={"ros2": connector})  # type: ignore
    agent.run()
    rclpy.shutdown()


if __name__ == "__main__":
    main()

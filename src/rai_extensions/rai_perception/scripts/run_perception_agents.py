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


import rclpy
from rai.agents import wait_for_shutdown
from rai_perception.agents import GroundedSamAgent, GroundingDinoAgent


def main():
    rclpy.init()
    agent1 = GroundingDinoAgent()
    agent2 = GroundedSamAgent()
    agent1.run()
    agent2.run()
    wait_for_shutdown([agent1, agent2])
    rclpy.shutdown()


if __name__ == "__main__":
    main()

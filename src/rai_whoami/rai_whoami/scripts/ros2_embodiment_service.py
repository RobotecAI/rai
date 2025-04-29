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

import argparse

from rai.agents import wait_for_shutdown
from rai.communication.ros2 import ROS2Context

from rai_whoami.agents.ros2 import ROS2EmbodimentInfoAgent


@ROS2Context()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root_dir", type=str)
    args = parser.parse_args()

    agent = ROS2EmbodimentInfoAgent(
        service_name="rai_whoami_embodiment_info_service",
        root_dir=args.root_dir,
    )
    agent.run()
    wait_for_shutdown([agent])


if __name__ == "__main__":
    main()

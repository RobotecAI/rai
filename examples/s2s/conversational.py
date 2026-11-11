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
# See the License for the specific language governing permissions and
# limitations under the License.

import rai
from rai.agents import HRIOutput, ReActAgent, hri_connector_sender
from rai.communication.ros2 import ROS2Context, ROS2HRIConnector

# A voice conversation: the ASR agent publishes what the human says to
# /from_human, the TTS agent speaks what this agent publishes to /to_human.


@ROS2Context()
def main():
    hri_connector = ROS2HRIConnector()
    agent = ReActAgent(
        system_prompt="Pretend you are a robot. Answer as if you were a robot."
    )
    agent.subscribe(HRIOutput(hri_connector_sender(hri_connector, "/to_human")))
    hri_connector.subscribe("/from_human", agent.submit)
    rai.run(agent)


if __name__ == "__main__":
    main()

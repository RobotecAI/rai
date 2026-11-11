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

import argparse
import asyncio

import rai
from langchain_core.messages import HumanMessage
from rai import get_llm
from rai.agents import BaseAgent, ReActAgent
from rai.communication.ros2 import ROS2Connector, ROS2Context
from rai.tools.ros2 import GetROS2MessageInterfaceTool, ROS2ServicesToolkit
from rai.tools.ros2.simple import GetROS2ImageConfiguredTool
from rai.tools.time import WaitForSecondsTool
from std_srvs.srv import Trigger

from rai_whoami.models import EmbodimentInfo


class SafetyAgent(BaseAgent):
    def __init__(
        self,
        agent: ReActAgent,
        connector: ROS2Connector,
        tractor_number: int,
        interval: float = 1.0,
    ):
        super().__init__()
        self.agent = agent
        self.connector = connector
        self.tractor_number = tractor_number
        self.working = False
        self.loop_interval = interval

    async def run(self):
        while True:
            if await self.check_tractor_state():
                self.logger.info("---------- RAI Agent invoked ----------")
                self.working = True
                self.agent.history = []  # every anomaly is a fresh conversation
                await self.agent.ainvoke(
                    HumanMessage(
                        content="Anomaly has been detected. The tractor has stopped. Please decide what to do."
                    )
                )
                self.working = False
                self.logger.info("---------- RAI Agent done ----------")
            await asyncio.sleep(self.loop_interval)

    async def check_tractor_state(self) -> bool:
        """Check the current state of the tractor and call the RAI agent if the tractor has stopped."""
        self.logger.info("Checking tractor state...")
        response: Trigger.Response = await self.connector.call(
            f"/tractor{self.tractor_number}/current_state", Trigger.Request()
        )
        if not self.working:
            self.logger.info(f"Tractor {self.tractor_number} state: {response.message}")

        return "STOPPED" in response.message and not self.working


@ROS2Context()
def main():
    parser = argparse.ArgumentParser(description="Autonomous Tractor Demo")
    parser.add_argument(
        "--tractor_number",
        type=int,
        choices=[1, 2],
        help="Tractor number (1 or 2)",
        default=1,
    )
    args = parser.parse_args()

    # Load the system prompt
    embodiment_info = EmbodimentInfo.from_file(
        "examples/embodiments/agriculture_embodiment.json"
    )

    # Initialize ROS 2 Communication
    connector = ROS2Connector()

    # Initialize LangGraph Agent
    agent = ReActAgent(
        get_llm("complex"),
        system_prompt=embodiment_info.to_langchain(),
        tools=[
            GetROS2ImageConfiguredTool(
                connector=connector,
                topic=f"/tractor{args.tractor_number}/camera_image_color",
            ),
            *ROS2ServicesToolkit(
                connector=connector,
                writable=[
                    f"/tractor{args.tractor_number}/continue",
                    f"/tractor{args.tractor_number}/current_state",
                    f"/tractor{args.tractor_number}/flash",
                    f"/tractor{args.tractor_number}/replan",
                    f"/tractor{args.tractor_number}/stop",
                ],
            ).get_tools(),
            GetROS2MessageInterfaceTool(connector=connector),
            WaitForSecondsTool(),
        ],
    )

    # Run the safety agent
    safety_agent = SafetyAgent(agent, connector, args.tractor_number)
    rai.run(safety_agent)


if __name__ == "__main__":
    main()

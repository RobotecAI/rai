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
# See the License for the specific language goveself.rning permissions and
# limitations under the License.

import argparse
from threading import Timer

from langchain_core.messages import HumanMessage
from langchain_core.runnables import Runnable
from rai import get_llm_model
from rai.agents.langchain.core import (
    ConversationalAgentState,
    create_conversational_agent,
)
from rai.communication.ros2.connectors import ROS2Connector
from rai.tools.ros2 import ROS2ServicesToolkit
from langchain_core.runnables import Runnable, RunnableConfig
from rai import get_llm_model, get_tracing_callbacks
from rai.agents import BaseAgent, wait_for_shutdown
from rai.communication.ros2 import ROS2Connector, ROS2Context, ROS2Message
from rai.tools.ros2 import GetROS2MessageInterfaceTool, ROS2ServicesToolkit
from rai.tools.ros2.simple import GetROS2ImageConfiguredTool
from rai.tools.time import WaitForSecondsTool
from std_srvs.srv import Trigger

from rai_whoami.models import EmbodimentInfo


class SafetyAgent(BaseAgent):
    def __init__(
        self,
        agent: Runnable[State, State],
        connector: ROS2Connector,
        tractor_number: int,
    ):
        super().__init__()
        self.agent = agent
        self.connector = connector
        self.tractor_number = tractor_number
        self.working = False
        self.langchain_callbacks = get_tracing_callbacks()

        self.timer = Timer(interval=1.0, function=self.check_tractor_state)
        self.logger.info(f"{self.__class__.__name__} initialized")

    def run(self):
        self.logger.info(f"{self.__class__.__name__} running")
        self.timer.start()

    def stop(self):
        self.logger.info(f"{self.__class__.__name__} stopping")
        self.timer.cancel()
        self.logger.info(f"{self.__class__.__name__} stopped")

    def check_tractor_state(self):
        """Check the current state of the tractor and call the RAI agent if the tractor has stopped."""
        response: Trigger.Response = self.connector.service_call(
            msg_type="std_srvs/srv/Trigger",
            message=ROS2Message(payload={}),
            target=f"/tractor{self.tractor_number}/current_state",
        ).payload
        if not self.working:
            self.logger.info(f"Tractor {self.tractor_number} state: {response.message}")

        if "STOPPED" in response.message and not self.working:
            self.logger.info("---------- RAI Agent invoked ----------")
            self.working = True
            self.agent.invoke(
                ConversationalAgentState(
                    messages=[
                        HumanMessage(
                            content="Anomaly has been detected. The tractor has stopped. Please decide what to do."
                        )
                    ]
                ),
                config=RunnableConfig(callbacks=self.langchain_callbacks),
            )
            self.working = False
            self.logger.info("---------- RAI Agent done ----------")


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
    system_prompt = EmbodimentInfo.from_file(
        "examples/embodiments/agriculture_embodiment.json"
    ).to_langchain()

    # Initialize ROS 2 Communication
    connector = ROS2Connector()

    # Initialize LangGraph Agent
    agent = create_conversational_agent(
        llm=get_llm_model("complex_model"),
        system_prompt=system_prompt,
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
    safety_agent.run()
    wait_for_shutdown([safety_agent])


if __name__ == "__main__":
    main()

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
import threading
import time

from langchain_core.messages import HumanMessage
from langchain_core.runnables import Runnable, RunnableConfig
from rai import get_llm_model, get_tracing_callbacks
from rai.agents import BaseAgent, wait_for_shutdown
from rai.agents.langchain.core import (
    ConversationalAgentState,
    create_conversational_agent,
)
from rai.communication.ros2 import (
    ROS2Connector,
    ROS2Context,
    ROS2Message,
    wait_for_ros2_services,
    wait_for_ros2_topics,
)
from rai.tools.ros2 import GetROS2MessageInterfaceTool, ROS2ServicesToolkit
from rai.tools.ros2.simple import GetROS2ImageConfiguredTool
from rai.tools.time import WaitForSecondsTool
from std_srvs.srv import Trigger

from rai_whoami.models import EmbodimentInfo


class SafetyAgent(BaseAgent):
    def __init__(
        self,
        agent: Runnable[ConversationalAgentState, ConversationalAgentState],
        connector: ROS2Connector,
        tractor_number: int,
        interval: float = 1.0,
    ):
        super().__init__()
        self.agent = agent
        self.connector = connector
        self.tractor_number = tractor_number
        self.working = False
        self.langchain_callbacks = get_tracing_callbacks()

        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self.loop)
        self.loop_interval = interval
        self.logger.info(f"{self.__class__.__name__} initialized")

    def run(self):
        self.stop_event.clear()
        self.logger.info(f"{self.__class__.__name__} running")
        self.thread.start()

    def loop(self):
        while True:
            self.check_tractor_state()
            time.sleep(self.loop_interval)
            if self.stop_event.is_set():
                break

    def stop(self):
        self.logger.info(f"{self.__class__.__name__} stopping")
        self.stop_event.set()
        self.logger.info(f"{self.__class__.__name__} stopped")

    def check_tractor_state(self):
        """Check the current state of the tractor and call the RAI agent if the tractor has stopped."""
        self.logger.info("Checking tractor state...")
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
    embodiment_info = EmbodimentInfo.from_file(
        "examples/embodiments/agriculture_embodiment.json"
    )

    # Initialize ROS 2 Communication
    connector = ROS2Connector()

    # Initialize LangGraph Agent
    agent = create_conversational_agent(
        llm=get_llm_model("complex_model"),
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

    # [Optional] wait for ROS 2 topics and services to be available
    wait_for_ros2_topics(
        connector, [f"/tractor{args.tractor_number}/camera_image_color"]
    )
    wait_for_ros2_services(connector, [f"/tractor{args.tractor_number}/current_state"])

    # Run the safety agent
    safety_agent = SafetyAgent(agent, connector, args.tractor_number)
    safety_agent.run()
    wait_for_shutdown([safety_agent])


if __name__ == "__main__":
    main()

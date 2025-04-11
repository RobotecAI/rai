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
import logging
import signal
import time
from queue import Queue
from threading import Event, Thread
from typing import Dict, List

import rclpy
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from rai.agents.base import BaseAgent
from rai.communication import BaseConnector
from rai.communication.ros2 import IROS2Message, ROS2HRIConnector, TopicConfig
from rai.utils.model_initialization import get_llm_model

from rai_interfaces.msg import HRIMessage as InterfacesHRIMessage

# NOTE: the Agent code included here is temporary until a dedicated speech agent is created
# it can still serve as a reference for writing your own RAI agents


class LLMTextHandler(BaseCallbackHandler):
    def __init__(self, connector: ROS2HRIConnector, speech_id: str = ""):
        self.connector = connector
        self.token_buffer = ""
        self.speech_id = speech_id

    def on_llm_new_token(self, token: str, **kwargs):
        self.token_buffer += token
        if len(self.token_buffer) > 100 or token in [".", "?", "!", ",", ";", ":"]:
            logging.info(f"Sending token buffer: {self.token_buffer}")
            self.connector.send_all_targets(
                AIMessage(content=self.token_buffer), self.speech_id
            )
            self.token_buffer = ""

    def on_llm_end(
        self,
        response,
        *,
        run_id,
        parent_run_id=None,
        **kwargs,
    ):
        if self.token_buffer:
            logging.info(f"Sending token buffer: {self.token_buffer}")
            self.connector.send_all_targets(AIMessage(content=self.token_buffer))
            self.token_buffer = ""


class S2SConversationalAgent(BaseAgent):
    def __init__(self, connectors: Dict[str, BaseConnector]):  # type: ignore
        super().__init__()
        self.connectors = connectors
        self.message_history: List[HumanMessage | AIMessage | SystemMessage] = [
            SystemMessage(
                content="Pretend you are a robot. Answer as if you were a robot."
            )
        ]
        self.speech_queue: Queue[InterfacesHRIMessage] = Queue()

        self.llm = get_llm_model(model_type="complex_model", streaming=True)
        self._setup_ros_connector()
        self.main_thread = None
        self.stop_thread = Event()
        self.current_speech_id = ""

    def run(self):
        logging.info("Running S2SConversationalAgent")
        self.main_thread = Thread(target=self._main_loop)
        self.main_thread.start()

    def _main_loop(self):
        while not self.stop_thread.is_set():
            time.sleep(0.01)
            speech = ""
            while not self.speech_queue.empty():
                speech_message = self.speech_queue.get()
                speech += "".join(speech_message.text)
                logging.info(f"Received human speech {speech}!")
                self.current_speech_id = speech_message.conversation_id
            if speech != "":
                self.message_history.append(
                    HumanMessage(content=speech, conversation_id=self.current_speech_id)
                )
                assert isinstance(self.connectors["ros2"], ROS2HRIConnector)
                ai_answer = self.llm.invoke(
                    self.message_history,
                    config={
                        "callbacks": [
                            LLMTextHandler(
                                self.connectors["ros2"], self.current_speech_id
                            )
                        ]
                    },
                )
                self.message_history.append(ai_answer)  # type: ignore

    def _on_from_human(self, msg: IROS2Message):
        assert isinstance(msg, InterfacesHRIMessage)
        logging.info("Received message from human: %s", msg.text)
        self.speech_queue.put(msg)

    def _setup_ros_connector(self):
        self.connectors["ros2"] = ROS2HRIConnector(
            sources=[
                (
                    "/from_human",
                    TopicConfig(
                        "rai_interfaces/msg/HRIMessage",
                        is_subscriber=True,
                        source_author="human",
                        subscriber_callback=self._on_from_human,
                    ),
                )
            ],
            targets=[
                (
                    "/to_human",
                    TopicConfig(
                        "rai_interfaces/msg/HRIMessage",
                        source_author="ai",
                        is_subscriber=False,
                    ),
                )
            ],
        )

    def stop(self):
        assert isinstance(self.connectors["ros2"], ROS2HRIConnector)
        self.connectors["ros2"].shutdown()
        self.stop_thread.set()
        if self.main_thread is not None:
            self.main_thread.join()


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Text To Speech Configuration",
        allow_abbrev=True,
    )

    # Use parse_known_args to ignore unknown arguments
    args, unknown = parser.parse_known_args()

    if unknown:
        print(f"Ignoring unknown arguments: {unknown}")

    return args


if __name__ == "__main__":
    args = parse_arguments()
    rclpy.init()
    agent = S2SConversationalAgent(connectors={})
    agent.run()

    def cleanup(signum, frame):
        print("\nCustom handler: Caught SIGINT (Ctrl+C).")
        print("Performing cleanup")
        # Optionally exit the program
        agent.stop()
        rclpy.shutdown()
        exit(0)

    signal.signal(signal.SIGINT, cleanup)

    while True:
        time.sleep(1)

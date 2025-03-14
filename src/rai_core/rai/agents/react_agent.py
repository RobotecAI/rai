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

import logging
import threading
import time
from typing import Any, Dict, List, Optional, cast

from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool

from rai.agents.base import BaseAgent
from rai.agents.langchain import HRICallbackHandler, create_react_agent
from rai.agents.langchain.react_agent import ReActAgentState
from rai.communication.hri_connector import HRIConnector, HRIMessage, HRIPayload


class ReActAgent(BaseAgent):
    def __init__(
        self,
        connectors: dict[str, HRIConnector[HRIMessage]],
        llm: Optional[BaseChatModel] = None,
        tools: Optional[List[BaseTool]] = None,
        state: Optional[ReActAgentState] = None,
    ):
        super().__init__(connectors=connectors)
        self.logger = logging.getLogger(__name__)
        self.agent = create_react_agent(llm=llm, tools=tools)
        self.callback = HRICallbackHandler(
            connectors=connectors, aggregate_chunks=True, logger=self.logger
        )
        self.state = state or ReActAgentState(messages=[])
        self.thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    def run(self):
        if self.thread is not None:
            raise RuntimeError("Agent is already running")
        self.thread = threading.Thread(target=self._run_loop)
        self.thread.start()

    def _run_loop(self):
        while not self._stop_event.is_set():
            received_messages = {}
            try:
                received_messages = self.receive_all_connectors()
            except ValueError:
                self.logger.info("Waiting for messages...")
            if received_messages:
                self.logger.info("Received messages")
                reduced_message = self._reduce_messages(received_messages)
                langchain_message = reduced_message.to_langchain()
                self.state["messages"].append(langchain_message)
                # callback is used to send messages to the connectors
                self.agent.invoke(self.state, config={"callbacks": [self.callback]})
            time.sleep(0.3)

    def stop(self):
        self._stop_event.set()
        if self.thread is not None:
            self.logger.info("Stopping the agent. Please wait...")
            self.thread.join()
            self.thread = None
            self.logger.info("Agent stopped")

    def receive_all_connectors(self) -> Dict[str, Dict[str, HRIMessage]]:
        received_messages: Dict[str, Any] = {}
        for connector_name, connector in self.connectors.items():
            received_message = cast(
                HRIConnector[HRIMessage], connector
            ).receive_all_sources()
            received_message = {k: v for k, v in received_message.items() if v}
            if received_message:
                received_messages[connector_name] = received_message
        return received_messages

    def _reduce_messages(
        self, received_messages: Dict[str, Dict[str, HRIMessage]]
    ) -> HRIMessage:
        hri_payload = HRIPayload(text="", images=[], audios=[])
        for connector_name, connector_sources in received_messages.items():
            hri_payload.text += f"{connector_name}\n"
            for source_name, source_message in connector_sources.items():
                hri_payload.text += f"{source_name}: {source_message.text}\n"
                hri_payload.images.extend(source_message.images)
                hri_payload.audios.extend(source_message.audios)
        return HRIMessage(payload=hri_payload, message_author="human")

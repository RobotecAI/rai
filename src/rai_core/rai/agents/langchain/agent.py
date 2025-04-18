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
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from typing import Deque, Dict, List, Literal, Optional, Tuple, TypedDict

from langchain_core.messages import BaseMessage
from langchain_core.runnables import Runnable
from pydantic import BaseModel

from rai.agents.base import BaseAgent
from rai.agents.langchain import HRICallbackHandler
from rai.agents.langchain.runnables import ReActAgentState
from rai.communication.hri_connector import HRIConnector, HRIMessage
from rai.initialization import get_tracing_callbacks


class BaseState(TypedDict):
    messages: List[BaseMessage]


class HRIConfig(BaseModel):
    source: str
    targets: List[str]


class LangChainAgent(BaseAgent):
    MAX_WORKERS = 10

    def __init__(
        self,
        target_connectors: Dict[str, HRIConnector],
        source_connector: Tuple[str, HRIConnector],
        runnable: Runnable,
        state: BaseState | None = None,
        new_message_behavior: Literal[
            "take_all",
            "keep_last",
            "queue",
            "interuppt_take_all",
            "interuppt_keep_last",
        ] = "interuppt_keep_last",
        max_workers: int = MAX_WORKERS,
    ):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.agent = runnable
        self.new_message_behavior = new_message_behavior
        self.tracing_callbacks = get_tracing_callbacks()
        self.state = state or ReActAgentState(messages=[])
        self.thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self.source_connector = source_connector
        self.callback = HRICallbackHandler(
            connectors=target_connectors,
            aggregate_chunks=True,
            logger=self.logger,
        )

        self.source, self.source_connector = source_connector
        self.source_connector.register_callback(
            self.source, self.source_callback, msg_type="rai_interfaces/msg/HRIMessage"
        )
        self.received_messages: Deque[HRIMessage] = deque()
        self.max_size = 100
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.interupt_event = threading.Event()
        self.agent_ready_event = threading.Event()

    def run(self):
        if self.thread is not None:
            raise RuntimeError("Agent is already running")
        self.thread = threading.Thread(target=self._run_loop)
        self.thread.start()
        self.agent_ready_event.set()

    def source_callback(self, msg: HRIMessage):
        if self.max_size is not None and len(self.received_messages) >= self.max_size:
            self.logger.warning("Buffer overflow. Dropping olders message")
            self.received_messages.popleft()
        if "interuppt" in self.new_message_behavior:
            self.executor.submit(self.interuppt_agent_and_run)
        self.logger.info(f"Received message: {msg}, {type(msg)}")
        self.received_messages.append(msg)

    def interuppt_agent_and_run(self):
        self.logger.info("Interuppting agent...")
        self.interupt_event.set()
        self.agent_ready_event.wait()
        self.interupt_event.clear()
        self.logger.info("Interuppting agent: DONE")

    def run_agent(self):
        self.agent_ready_event.clear()
        try:
            if len(self.received_messages) == 0:
                self.logger.info("Waiting for messages...")
                time.sleep(0.5)
                return
            self.logger.info("Running agent...")
            reduced_message = self._reduce_messages()
            langchain_message = reduced_message.to_langchain()
            self.state["messages"].append(langchain_message)
            for _ in self.agent.stream(
                self.state,
                config={"callbacks": [self.callback, *self.tracing_callbacks]},
            ):
                if self.interupt_event.is_set():
                    break
        finally:
            self.agent_ready_event.set()

    def _run_loop(self):
        while not self._stop_event.is_set():
            time.sleep(0.01)
            if self.agent_ready_event.is_set():
                self.run_agent()

    def stop(self):
        self._stop_event.set()
        self.interupt_event.set()
        self.agent_ready_event.wait()
        if self.thread is not None:
            self.logger.info("Stopping the agent. Please wait...")
            self.thread.join()
            self.thread = None
            self.logger.info("Agent stopped")

    def _reduce_messages(self) -> HRIMessage:
        text = ""
        images = []
        audios = []
        if "take_all" in self.new_message_behavior:
            while len(self.received_messages) > 0:
                source_message = self.received_messages.popleft()
                text += f"{source_message.text}\n"
                images.extend(source_message.images)
                audios.extend(source_message.audios)
        elif "keep_last" in self.new_message_behavior:
            # Take the recently added message
            source_message = self.received_messages.pop()
            text += f"{source_message.text}\n"
            images.extend(source_message.images)
            audios.extend(source_message.audios)
            self.received_messages.clear()
        elif self.new_message_behavior == "queue":
            # Take the first message from the queue. Let other messages wait.
            source_message = self.received_messages.popleft()
            text += f"{source_message.text}\n"
            images.extend(source_message.images)
            audios.extend(source_message.audios)
        else:
            raise ValueError(
                f"Invalid new_message_behavior: {self.new_message_behavior}"
            )
        return HRIMessage(
            text=text,
            images=images,
            audios=audios,
            message_author="human",
        )

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
from typing import Deque, Dict, List, Literal, Optional, TypedDict

from langchain_core.messages import BaseMessage
from langchain_core.runnables import Runnable
from pydantic import BaseModel

from rai.agents.base import BaseAgent
from rai.agents.langchain import HRICallbackHandler
from rai.agents.langchain.runnables import ReActAgentState
from rai.communication.base_connector import BaseConnector
from rai.communication.hri_connector import HRIMessage
from rai.initialization import get_tracing_callbacks


class BaseState(TypedDict):
    messages: List[BaseMessage]


class HRIConfig(BaseModel):
    source: str
    targets: List[str]


class LangChainAgent(BaseAgent):
    def __init__(
        self,
        target_connectors: Dict[str, BaseConnector],
        runnable: Runnable,
        state: BaseState | None = None,
        new_message_behavior: Literal[
            "take_all",
            "keep_last",
            "queue",
            "interuppt_take_all",
            "interuppt_keep_last",
        ] = "interuppt_keep_last",
        max_size: int = 100,
    ):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.agent = runnable
        self.new_message_behavior = new_message_behavior
        self.tracing_callbacks = get_tracing_callbacks()
        self.state = state or ReActAgentState(messages=[])
        self.callback = HRICallbackHandler(
            connectors=target_connectors,
            aggregate_chunks=True,
            logger=self.logger,
        )

        self._received_messages: Deque[HRIMessage] = deque()
        self.max_size = max_size

        self.thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._interupt_event = threading.Event()
        self._agent_ready_event = threading.Event()

    def __call__(self, msg: HRIMessage):
        if self.max_size is not None and len(self._received_messages) >= self.max_size:
            self.logger.warning("Buffer overflow. Dropping olders message")
            self._received_messages.popleft()
        if "interuppt" in self.new_message_behavior:
            self._executor.submit(self.interuppt_agent_and_run)
        self.logger.info(f"Received message: {msg}, {type(msg)}")
        self._received_messages.append(msg)

    def run(self):
        if self.thread is not None:
            raise RuntimeError("Agent is already running")
        self.thread = threading.Thread(target=self._run_loop)
        self.thread.start()
        self._agent_ready_event.set()
        self.logger.info("Agent started")

    def interuppt_agent_and_run(self):
        if self._agent_ready_event.is_set():
            self.logger.info("Agent is ready. No need to interuppt it.")
            return
        self.logger.info("Interuppting agent...")
        self._interupt_event.set()
        self._agent_ready_event.wait()
        self._interupt_event.clear()
        self.logger.info("Interuppting agent: DONE")

    def run_agent(self):
        if len(self._received_messages) == 0:
            self._agent_ready_event.set()
            self.logger.info("Waiting for messages...")
            time.sleep(0.5)
            return
        self._agent_ready_event.clear()
        try:
            self.logger.info("Running agent...")
            reduced_message = self._reduce_messages()
            langchain_message = reduced_message.to_langchain()
            self.state["messages"].append(langchain_message)
            for _ in self.agent.stream(
                self.state,
                config={"callbacks": [self.callback, *self.tracing_callbacks]},
            ):
                if self._interupt_event.is_set():
                    break
        finally:
            self._agent_ready_event.set()

    def _run_loop(self):
        while not self._stop_event.is_set():
            time.sleep(0.01)
            if self._agent_ready_event.is_set():
                self.run_agent()

    def stop(self):
        self._stop_event.set()
        self._interupt_event.set()
        self._agent_ready_event.wait()
        if self.thread is not None:
            self.logger.info("Stopping the agent. Please wait...")
            self.thread.join()
            self.thread = None
            self.logger.info("Agent stopped")

    def _reduce_messages(self) -> HRIMessage:
        text = ""
        images = []
        audios = []
        source_messages = list()
        if "take_all" in self.new_message_behavior:
            # Take all starting from the oldest
            while len(self._received_messages) > 0:
                source_messages.append(self._received_messages.popleft())
        elif "keep_last" in self.new_message_behavior:
            # Take the recently added message
            source_messages.append(self._received_messages.pop())
            self._received_messages.clear()
        elif self.new_message_behavior == "queue":
            # Take the first message from the queue. Let other messages wait.
            source_messages.append(self._received_messages.popleft())
        else:
            raise ValueError(
                f"Invalid new_message_behavior: {self.new_message_behavior}"
            )
        for source_message in source_messages:
            text += f"{source_message.text}\n"
            images.extend(source_message.images)
            audios.extend(source_message.audios)
        return HRIMessage(
            text=text,
            images=images,
            audios=audios,
            message_author="human",
        )

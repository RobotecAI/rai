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

from rai.agents.base import BaseAgent
from rai.agents.langchain import HRICallbackHandler
from rai.agents.langchain.runnables import ReActAgentState
from rai.communication.hri_connector import HRIConnector, HRIMessage
from rai.initialization import get_tracing_callbacks


class BaseState(TypedDict):
    messages: List[BaseMessage]


newMessageBehaviorType = Literal[
    "take_all",
    "keep_last",
    "queue",
    "interrupt_take_all",
    "interrupt_keep_last",
]


class LangChainAgent(BaseAgent):
    """
    Agent pareametrized by LangGraph runnable that communicates with environment using
    `HRIConnector`.

    Parameters
    ----------
    target_connectors : Dict[str, HRIConnector[HRIMessage]]
        Dict of target_name: connector. Agent will send it's output to these targets using connectors.
    runnable : Runnable
        LangChain runnable that will be used to generate output.
    state : BaseState | None, optional
        State to seed the LangChain runnable. If None - empty state is used.
    new_message_behavior : newMessageBehaviorType, optional
        Describes how to handle new messages and interact with LangChain runnable. There are 2 main options:
        1. Agent waits for LangChain runnable to finish processing:
            - "take_all": all messages from the queue are concatenated and processed.
            - "keep_last": only the last received message is processed, others are dropped.
            - "queue": only the first message from the queue is processed, others are kept in the queue.
        2. Agent interrupts LangChain runnable:
            - "interrupt_take_all": same as "take_all"
            - "interrupt_keep_last": same as "keep_last"
    max_size : int, optional
        Maximum number of messages to keep in the agent's queue. If exceeded, oldest messages are dropped.

    """

    def __init__(
        self,
        target_connectors: Dict[str, HRIConnector[HRIMessage]],
        runnable: Runnable,
        state: BaseState | None = None,
        new_message_behavior: newMessageBehaviorType = "interrupt_keep_last",
        max_size: int = 100,
    ):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.agent = runnable
        self.new_message_behavior: newMessageBehaviorType = new_message_behavior
        self.tracing_callbacks = get_tracing_callbacks()
        self.state = state or ReActAgentState(messages=[])
        self._langchain_callback = HRICallbackHandler(
            connectors=target_connectors,
            aggregate_chunks=True,
            logger=self.logger,
        )

        self._received_messages: Deque[HRIMessage] = deque()
        self._buffer_lock = threading.Lock()
        self.max_size = max_size

        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._interrupt_event = threading.Event()
        self._agent_ready_event = threading.Event()

    def subscribe_source(self, source: str, connector: HRIConnector[HRIMessage]):
        connector.register_callback(
            source,
            self.source_callback,
        )

    def source_callback(self, msg: HRIMessage):
        with self._buffer_lock:
            if (
                self.max_size is not None
                and len(self._received_messages) >= self.max_size
            ):
                self.logger.warning("Buffer overflow. Dropping olders message")
                self._received_messages.popleft()
            if "interrupt" in self.new_message_behavior:
                self._executor.submit(self.interrupt_agent_and_run)
            self.logger.info(f"Received message: {msg}, {type(msg)}")
            self._received_messages.append(msg)

    def run(self):
        if self._thread is not None:
            raise RuntimeError("Agent is already running")
        self._thread = threading.Thread(target=self._run_loop)
        self._thread.start()
        self._agent_ready_event.set()
        self.logger.info("Agent started")

    def interrupt_agent_and_run(self):
        if self._agent_ready_event.is_set():
            self.logger.info("Agent is ready. No need to interrupt it.")
            return
        self.logger.info("Interrupting agent...")
        self._interrupt_event.set()
        self._agent_ready_event.wait()
        self._interrupt_event.clear()
        self.logger.info("Interrupting agent: DONE")

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
                config={
                    "callbacks": [self._langchain_callback, *self.tracing_callbacks]
                },
            ):
                if self._interrupt_event.is_set():
                    break
        finally:
            self._agent_ready_event.set()

    def _run_loop(self):
        while not self._stop_event.is_set():
            if self._agent_ready_event.wait(0.01):
                self.run_agent()

    def stop(self):
        self._stop_event.set()
        self._interrupt_event.set()
        self._agent_ready_event.wait()
        if self._thread is not None:
            self.logger.info("Stopping the agent. Please wait...")
            self._thread.join()
            self._thread = None
            self.logger.info("Agent stopped")

    @staticmethod
    def _apply_reduction_behavior(
        method: newMessageBehaviorType, buffer: Deque[HRIMessage]
    ) -> List[HRIMessage]:
        output = list()
        if "take_all" in method:
            # Take all starting from the oldest
            while len(buffer) > 0:
                output.append(buffer.popleft())
        elif "keep_last" in method:
            # Take the recently added message
            output.append(buffer.pop())
            buffer.clear()
        elif method == "queue":
            # Take the first message from the queue. Let other messages wait.
            output.append(buffer.popleft())
        else:
            raise ValueError(f"Invalid new_message_behavior: {method}")
        return output

    def _reduce_messages(self) -> HRIMessage:
        text = ""
        images = []
        audios = []
        with self._buffer_lock:
            source_messages = self._apply_reduction_behavior(
                self.new_message_behavior, self._received_messages
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

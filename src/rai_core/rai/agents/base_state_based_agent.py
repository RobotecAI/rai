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

import threading
import time
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.tools import BaseTool
from pydantic import BaseModel, ConfigDict, Field
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.subscription import Subscription

from rai.agents.langchain import create_state_based_runnable
from rai.aggregators import BaseAggregator
from rai.communication.base_connector import BaseConnector
from rai.communication.hri_connector import HRIConnector, HRIMessage
from rai.messages.multimodal import HumanMultimodalMessage

from .langchain import ReActAgent, ReActAgentState, create_state_based_runnable


class StateBasedConfig(BaseModel):
    aggregators: Dict[str, List[BaseAggregator]]
    time_interval: float = Field(default=5.0)
    max_workers: int = 8

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )


class BaseStateBasedAgent(ReActAgent, ABC):
    """
    Agent that runs aggregators (config.aggregators) every config.time_interval seconds.
    Aggregators are registered to their sources using
    :py:class:`~rai.communication.ros2.connectors.ROS2Connector`

    Output from aggragators is called `state`. Such state is saved and can be
    retrieved by `get_state` method.

    In `StateBaseAgent`, state is added to LLM history. For more details about the LLM
    agent see :py:func:`~rai.agents.langchain.runnables.create_state_based_runnable`
    """

    def __init__(
        self,
        connectors: dict[str, HRIConnector[HRIMessage]],
        config: StateBasedConfig,
        llm: Optional[BaseChatModel] = None,
        tools: Optional[List[BaseTool]] = None,
        state: Optional[ReActAgentState] = None,
        system_prompt: Optional[str] = None,
    ):
        runnable = create_state_based_runnable(
            llm=llm,
            tools=tools,
            system_prompt=system_prompt,
            state_retriever=self.get_state,
        )
        super().__init__(
            connectors, llm, tools, state, system_prompt, runnable=runnable
        )
        self.config = config

        self._callback_group = ReentrantCallbackGroup()
        self._subscriptions: Dict[str, Subscription] = dict()

        self._aggregation_results: Dict[str, HumanMessage | HumanMultimodalMessage] = (
            dict()
        )
        self._aggregation_thread: threading.Thread | None = None

        self._registered_callbacks = set()
        self._connector = self.setup_connector()
        self._configure_state_sources()

    @abstractmethod
    def setup_connector(self) -> BaseConnector:
        pass

    def _configure_state_sources(self):
        for source, aggregators in self.config.aggregators.items():
            for aggregator in aggregators:
                callback_id = self._connector.register_callback(
                    source, aggregator, raw=True
                )
                self._registered_callbacks.add(callback_id)

    def run(self):
        super().run()
        self._aggregation_thread = threading.Thread(target=self._run_state_loop)
        self._aggregation_thread.start()

    def get_state(self) -> Dict[str, HumanMessage | HumanMultimodalMessage]:
        """Returns output for all aggregators"""
        return self._aggregation_results

    def _run_state_loop(self):
        """Runs aggregation on collected data"""
        while not self._stop_event.is_set():
            ts = time.perf_counter()
            self.logger.debug("Starting aggregation interval")
            self._on_aggregation_interval()
            elapsed_time = time.perf_counter() - ts
            self.logger.debug(f"Aggregation done in: {elapsed_time:.2f}s")
            if elapsed_time > self.config.time_interval:
                self.logger.warning(
                    "State aggregation time interval exceeded. Expected "
                    f"{self.config.time_interval:.2f}s, got {elapsed_time:.2f}s. Consider "
                    f"increasing {self.__class__.__name__}.config.time_interval."
                )
            time.sleep(max(0, self.config.time_interval - (elapsed_time)))

    def _on_aggregation_interval(self):
        """Runs aggregation on collected data"""

        def process_aggregator(
            source: str, aggregator: BaseAggregator
        ) -> Tuple[str, BaseMessage | None]:
            self.logger.info(
                f"Running aggregator: {aggregator}(source={source}) on {len(aggregator.get_buffer())} messages"
            )
            ts = time.perf_counter()

            output = aggregator.get()

            self.logger.debug(
                f'Aggregator "{aggregator}(source={source})" done in {time.perf_counter() - ts:.2f}s'
            )
            return source, output

        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            futures = list()
            for source, aggregators in self.config.aggregators.items():
                for aggregator in aggregators:
                    future = executor.submit(process_aggregator, source, aggregator)
                    futures.append(future)

            for future in as_completed(futures):
                try:
                    source, output = future.result()
                except Exception as e:
                    self.logger.error(f"Aggregator crashed: {e}")
                    continue

                if output is None:
                    continue
                self._aggregation_results[source] = output

    def stop(self):
        """Stop the agent's execution loop."""
        self.logger.info("Stopping the agent. Please wait...")
        self._stop_event.set()
        if self.thread is not None:
            self.thread.join()
            self.thread = None
        if self._aggregation_thread is not None:
            self._aggregation_thread.join()
            self._aggregation_thread = None
        self._stop_event.clear()
        for callback_id in self._registered_callbacks:
            self._connector.unregister_callback(callback_id)
        self._connector.shutdown()
        self.logger.info("Agent stopped")

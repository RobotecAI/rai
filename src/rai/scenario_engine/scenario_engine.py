import datetime
import enum
import logging
import os
import pickle
import threading
from typing import Any, Callable, Dict, List, Sequence, Union

import coloredlogs
import xxhash

from rai.actions.executor import ConditionalExecutor, Executor
from rai.history_saver import HistorySaver
from rai.message import AssistantMessage, ConditionalMessage, ConstantMessage, Message
from rai.requirements import RequirementSeverity
from rai.vendors.vendors import AiVendor

__all__ = [
    "ScenarioRunner",
    "ScenarioPartType",
    "ScenarioType",
    "ConditionalScenario",
    "RequirementSeverity",
]

coloredlogs.install()  # type: ignore


class ConditionalScenario:
    def __init__(
        self,
        if_true: "ScenarioType",
        if_false: "ScenarioType",
        condition: Callable[[List[Message]], bool],
    ):
        self.if_true = if_true
        self.if_false = if_false
        self.condition = condition

    def __call__(self, messages: List[Message]):
        response = self.condition(messages)
        if response:
            return self.if_true
        return self.if_false


ScenarioPartType = Union[
    ConstantMessage,
    AssistantMessage,
    ConditionalMessage,
    ConditionalExecutor,
    ConditionalScenario,
    Executor,
]
ScenarioType = Sequence[ScenarioPartType]


class ScenarioRunner:
    """
    The ScenarioRunner class is responsible for running a given scenario. It iterates over the scenario and executes the
    actions defined in the scenario.
    """

    def __init__(
        self,
        scenario: ScenarioType,
        ai_vendor: AiVendor,
        logging_level: int = logging.WARNING,
        use_cache: bool = False,
    ):
        self.scenario = scenario
        self.ai_vendor = ai_vendor
        self.logging_level = logging_level
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(self.logging_level)
        self.history: List[Message] = []
        self.logs_dir = os.path.join(
            "logs",
            self.ai_vendor.__class__.__name__
            + datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S.%f"),
        )
        self.use_cache = use_cache
        self.cache: Dict[str, Dict[int, ConstantMessage]] = {}
        if self.use_cache:
            # check if exists
            try:
                with open("cache.pkl", "rb") as f:
                    self.cache = pickle.load(f)
            except FileNotFoundError:
                self.cache = {}

    def run(self):
        self.logger.info(f"Starting conversation.")
        self.threads: List[threading.Thread] = []
        self._run(self.scenario)

        self.logger.info(f"Conversation completed.")
        self._wait_for_threads()
        return self.history

    def _run(self, scenario: ScenarioType):
        """Recursively run the scenario."""

        for msg in scenario:
            if isinstance(msg, ConstantMessage):
                self.history.append(msg)
            elif isinstance(msg, AssistantMessage):
                response = self._handle_assistant_message(self.history, msg)
                self.history.append(response)
            elif isinstance(msg, ConditionalMessage):
                ruled_prompt = self._handle_conditional_message(self.history, msg)
                self.history.append(ruled_prompt)
            elif isinstance(msg, (Executor, ConditionalExecutor)):
                thread = msg(self)
                if thread is not None:
                    self.threads.append(thread)
            elif isinstance(msg, ConditionalScenario):
                new_scenario = msg(self.history)
                self._run(new_scenario)
            else:
                raise ValueError(f"Unknown message type: {type(msg)}")
            self.logger.debug(
                f"Last message: {self.history[-1].role}:{self.history[-1].content}"
            )

    def save_to_html(self, folder: str = "") -> str:
        saver = HistorySaver(self.ai_vendor, self.history, self.logs_dir)
        out_file = saver.save_to_html(folder=folder)
        self.logger.info(f"Conversation saved to: {out_file}")
        return out_file

    def save_to_markdown(self):
        saver = HistorySaver(self.ai_vendor, self.history, self.logs_dir)
        saver.save_to_markdown()

    @staticmethod
    def get_html(history: List[Message], ai_vendor: AiVendor):
        saver = HistorySaver(ai_vendor, history)
        return saver.get_html()

    def _wait_for_threads(self):
        if len(self.threads) > 0:
            self.logger.info(f"Waiting for {len(self.threads)} threads to finish.")
            for thread in self.threads:
                self.logger.info(f"Waiting for thread: {thread.name}")
                thread.join()
                self.logger.info(f"Thread {thread.name} finished.")

    def _handle_assistant_message(
        self, messages: List[Message], assistant_message: AssistantMessage
    ) -> ConstantMessage:
        def _call_api_and_validate(messages: List[Message]):
            response = self.ai_vendor.call_api(messages, assistant_message.max_tokens)
            requirements_status = assistant_message.check_requirements(response)
            return response, requirements_status

        def _calculate_hash(messages: List[Message]):
            hashes = str([message.__hash__ for message in messages])
            return xxhash.xxh32_intdigest(hashes)

        def _dump_to_cache(response: ConstantMessage):
            history_hash = _calculate_hash(messages)
            if self.ai_vendor.model not in self.cache:
                self.cache[self.ai_vendor.model] = {}
            self.cache[self.ai_vendor.model][history_hash] = response
            with open("cache.pkl", "wb") as f:
                pickle.dump(self.cache, f)

        # check if in cache
        if self.use_cache:
            history_hash = _calculate_hash(messages)
            if self.ai_vendor.model in self.cache:
                if history_hash in self.cache[self.ai_vendor.model]:
                    self.logger.info("Using cached response")
                    return self.cache[self.ai_vendor.model][history_hash]

        response: ConstantMessage
        requirements_status: List[Dict[str, Any]] = []
        for i in range(assistant_message.max_retries + 1):
            vendor_response, requirements_status = _call_api_and_validate(messages)
            response = ConstantMessage(
                role="assistant",
                content=vendor_response["content"],
                images=vendor_response.get("images", []),
            )
            if all([requirement.get("status") for requirement in requirements_status]):
                if self.use_cache:
                    _dump_to_cache(response)
                return response

            self.logger.info(
                f"Assistant response did not meet requirements. Retrying... Attempt {i + 1}"
            )

        unsatisfied_requirements: List[tuple[str, enum.Enum]] = []
        for requirement in requirements_status:
            self.logger.info(
                f"Requirement: {requirement} status: {requirement['status']} severity: {requirement['severity']}"
            )
            unsatisfied_requirements.append(
                (requirement["name"], requirement["severity"])
            )
            if (
                requirement["status"] == False
                and requirement["severity"] == RequirementSeverity.MANDATORY
            ):
                msg = f"Failed to get a valid response from the assistant. Unmet requirements: {requirement['name']}. Severity: {requirement['severity']}"
                self.logger.error(msg)
                raise ValueError(msg)

        self.logger.warning(
            f"Failed to get a valid response from the assistant. Unmet requirements: {[(name, severity.name) for name, severity in unsatisfied_requirements]}. continuing..."
        )
        return response  # last response, might be better to return best response

    def _handle_conditional_message(
        self, messages: List[Message], conditional_message: ConditionalMessage
    ):
        ruled_prompt = conditional_message(messages)
        return ruled_prompt

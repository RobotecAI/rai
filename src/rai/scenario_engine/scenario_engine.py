import datetime
import enum
import logging
import os
import pickle
import threading
from typing import Any, Callable, Dict, List, Literal, Sequence, Union, cast

import coloredlogs
import xxhash
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolCall,
    ToolMessage,
)
from langchain_core.tools import BaseTool

from rai.history_saver import HistorySaver
from rai.scenario_engine.messages import AgentLoop, FutureAiMessage
from rai.scenario_engine.tool_runner import run_requested_tools

__all__ = [
    "ScenarioRunner",
    "ScenarioPartType",
    "ScenarioType",
    "ConditionalScenario",
]

coloredlogs.install()  # type: ignore


class ConditionalScenario:
    def __init__(
        self,
        if_true: "ScenarioType",
        if_false: "ScenarioType",
        condition: Callable[[Sequence[BaseMessage]], bool],
    ):
        self.if_true = if_true
        self.if_false = if_false
        self.condition = condition

    def __call__(self, messages: Sequence[BaseMessage]):
        response = self.condition(messages)
        if response:
            return self.if_true
        return self.if_false


ScenarioPartType = Union[
    SystemMessage,
    HumanMessage,
    AIMessage,
    ToolMessage,
    BaseMessage,
    ConditionalScenario,
    FutureAiMessage,
    AgentLoop,
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
        llm: BaseChatModel,
        llm_type: Literal["openai", "bedrock"],
        tools: Sequence[BaseTool],
        logging_level: int = logging.WARNING,
        use_cache: bool = False,
    ):
        self.scenario = scenario
        self.tools = tools
        self.llm = llm
        self.llm_type = llm_type
        self.llm_with_tools = llm.bind_tools(tools)
        self.logging_level = logging_level
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(self.logging_level)
        self.history: List[BaseMessage] = []
        self.logs_dir = os.path.join(
            "logs",
            self.llm.__class__.__name__
            + datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S.%f"),
        )
        self.use_cache = use_cache
        self.cache: Dict[str, Dict[int, BaseMessage]] = {}
        if self.use_cache:
            # check if exists
            try:
                with open("cache.pkl", "rb") as f:
                    self.cache = pickle.load(f)
            except FileNotFoundError:
                self.cache = {}

    def run(self):
        self.logger.info(f"Starting conversation.")
        self._run(self.scenario)

        self.logger.info(f"Conversation completed.")
        return self.history

    def _run(self, scenario: ScenarioType):
        """Recursively run the scenario."""

        for msg in scenario:
            if isinstance(msg, (HumanMessage, AIMessage, ToolMessage, SystemMessage)):
                self.history.append(msg)
            elif isinstance(msg, FutureAiMessage):
                ai_msg = cast(AIMessage, self.llm_with_tools.invoke(self.history))
                self.history.append(ai_msg)
                self.history = run_requested_tools(
                    ai_msg, self.tools, self.history, llm_type=self.llm_type
                )
            elif isinstance(msg, AgentLoop):
                self.logger.info(
                    f"Looping agent actions until {msg.stop_action}. Max {msg.stop_iters} loops."
                )
                for _ in range(msg.stop_iters):
                    ai_msg = cast(AIMessage, self.llm_with_tools.invoke(self.history))
                    self.history.append(ai_msg)
                    self.history = run_requested_tools(
                        ai_msg, self.tools, self.history, llm_type=self.llm_type
                    )
                    for tool_call in ai_msg.tool_calls:
                        if tool_call["name"] == msg.stop_action:
                            break
            elif isinstance(msg, ConditionalScenario):
                new_scenario = msg(self.history)
                self._run(new_scenario)
            else:
                raise ValueError(f"Unknown message type: {type(msg)}")
            self.logger.debug(
                f"Last message: {self.history[-1].type}:{self.history[-1].content}"
            )

    def save_to_html(self, folder: str = "") -> str:
        saver = HistorySaver(self.history, self.logs_dir)
        out_file = saver.save_to_html(folder=folder)
        self.logger.info(f"Conversation saved to: {out_file}")
        return out_file

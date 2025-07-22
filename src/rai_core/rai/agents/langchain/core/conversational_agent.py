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


import logging
from functools import partial
from typing import List, Optional, TypedDict

from langchain.chat_models.base import BaseChatModel
from langchain_core.messages import (
    BaseMessage,
    SystemMessage,
)
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from langgraph.graph import START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt.tool_node import tools_condition

from rai.agents.langchain.core.tool_runner import ToolRunner
from rai.agents.langchain.utils import invoke_llm_with_tracing


class State(TypedDict):
    messages: List[BaseMessage]


def agent(
    llm: BaseChatModel,
    logger: logging.Logger,
    system_prompt: str | SystemMessage,
    state: State,
    config: RunnableConfig,
):
    logger.info("Running thinker")

    # If there are no messages, do nothing
    if len(state["messages"]) == 0:
        return state

    # Insert system message if not already present
    if not isinstance(state["messages"][0], SystemMessage):
        system_msg = (
            SystemMessage(content=system_prompt)
            if isinstance(system_prompt, str)
            else system_prompt
        )
        state["messages"].insert(0, system_msg)
    
    # Invoke LLM with tracing if it is configured and available
    ai_msg = invoke_llm_with_tracing(llm, state["messages"], config)
    state["messages"].append(ai_msg)
    return state


def create_conversational_agent(
    llm: BaseChatModel,
    tools: List[BaseTool],
    system_prompt: str | SystemMessage,
    logger: Optional[logging.Logger] = None,
    debug: bool = False,
) -> CompiledStateGraph:
    _logger = None
    if logger:
        _logger = logger
    else:
        _logger = logging.getLogger(__name__)

    _logger.info("Creating state based agent")

    llm_with_tools = llm.bind_tools(tools)
    tool_node = ToolRunner(tools=tools, logger=_logger)

    workflow = StateGraph(State)
    workflow.add_node("tools", tool_node)
    workflow.add_node("thinker", partial(agent, llm_with_tools, _logger, system_prompt))

    workflow.add_edge(START, "thinker")
    workflow.add_edge("tools", "thinker")

    workflow.add_conditional_edges(
        "thinker",
        tools_condition,
    )

    app = workflow.compile(debug=debug)
    _logger.info("State based agent created")
    return app

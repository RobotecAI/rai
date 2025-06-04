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
from typing import List, Optional

from langchain.chat_models.base import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
)
from langchain_core.tools import BaseTool
from langgraph.graph import START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt.tool_node import tools_condition
from rai.agents.langchain.core.conversational_agent import State, agent
from rai.agents.langchain.core.tool_runner import ToolRunner


def multimodal_to_tool_bridge(state: State):
    """Node of langchain workflow designed to bridge
    nodes with llms. Removing images for context
    """

    cleaned_messages: List[BaseMessage] = []
    for msg in state["messages"]:
        if isinstance(msg, HumanMessage):
            # Remove images but keep the direct request
            if isinstance(msg.content, list):
                # Extract text only
                text_parts = [
                    part.get("text", "")
                    for part in msg.content
                    if isinstance(part, dict) and part.get("type") == "text"
                ]
                if text_parts:
                    cleaned_messages.append(HumanMessage(content=" ".join(text_parts)))
            else:
                cleaned_messages.append(msg)
        elif isinstance(msg, AIMessage):
            # Keep AI messages for context
            cleaned_messages.append(msg)

    state["messages"] = cleaned_messages
    return state


def create_multimodal_to_tool_agent(
    multimodal_llm: BaseChatModel,
    tool_llm: BaseChatModel,
    tools: List[BaseTool],
    multimodal_system_prompt: str,
    tool_system_prompt: str,
    logger: Optional[logging.Logger] = None,
    debug: bool = False,
) -> CompiledStateGraph:
    """
    Creates an agent flow where inputs first go to a multimodal LLM,
    then its output is passed to a tool-calling LLM.
    Can be usefull when multimodal llm does not provide tool calling.

    Args:
        tools: List of tools available to the tool agent

    Returns:
        Compiled state graph
    """
    _logger = None
    if logger:
        _logger = logger
    else:
        _logger = logging.getLogger(__name__)

    _logger.info("Creating multimodal to tool agent flow")

    tool_llm_with_tools = tool_llm.bind_tools(tools)
    tool_node = ToolRunner(tools=tools, logger=_logger)

    workflow = StateGraph(State)
    workflow.add_node(
        "thinker",
        partial(agent, multimodal_llm, _logger, multimodal_system_prompt),
    )
    # context bridge for altering the
    workflow.add_node(
        "context_bridge",
        multimodal_to_tool_bridge,
    )
    workflow.add_node(
        "tool_agent",
        partial(agent, tool_llm_with_tools, _logger, tool_system_prompt),
    )
    workflow.add_node("tools", tool_node)

    workflow.add_edge(START, "thinker")
    workflow.add_edge("thinker", "context_bridge")
    workflow.add_edge("context_bridge", "tool_agent")

    workflow.add_conditional_edges(
        "tool_agent",
        tools_condition,
    )

    # Tool node goes back to tool agent
    workflow.add_edge("tools", "tool_agent")

    app = workflow.compile(debug=debug)
    _logger.info("Multimodal to tool agent flow created")
    return app

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
from langchain_core.tools import BaseTool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt.tool_node import tools_condition
from pydantic import BaseModel, Field

from rai.agents.langchain.core.tool_runner import ToolRunner
from rai.communication.ros2.connectors import ROS2Connector
from rai.tools.ros2.simple import GetROS2ImageConfiguredTool


class State(TypedDict):
    messages: List[BaseMessage]


def agent(
    llm: BaseChatModel,
    logger: logging.Logger,
    system_prompt: str | SystemMessage,
    state: State,
):
    logger.info("🤖 Running thinker agent")
    logger.info(f"📝 Current state has {len(state['messages'])} messages")

    # If there are no messages, do nothing
    if len(state["messages"]) == 0:
        logger.info("⚠️ No messages in state, returning unchanged")
        return state

    # Insert system message if not already present
    if not isinstance(state["messages"][0], SystemMessage):
        system_msg = (
            SystemMessage(content=system_prompt)
            if isinstance(system_prompt, str)
            else system_prompt
        )
        state["messages"].insert(0, system_msg)
        logger.info("📋 Added system message to conversation")

    logger.info("🧠 Invoking LLM with current messages")
    ai_msg = llm.invoke(state["messages"])
    state["messages"].append(ai_msg)

    # Log the AI response
    if hasattr(ai_msg, "content") and ai_msg.content:
        logger.info(
            f"💬 AI Response: {ai_msg.content[:100]}{'...' if len(ai_msg.content) > 100 else ''}"
        )

    # Log tool calls if any
    if hasattr(ai_msg, "tool_calls") and ai_msg.tool_calls:
        logger.info(f"🔧 AI requested {len(ai_msg.tool_calls)} tool calls")
        for i, tool_call in enumerate(ai_msg.tool_calls):
            logger.info(f"   Tool {i + 1}: {tool_call.get('name', 'unknown')}")

    return state


class BoolAnswerWithJustification(BaseModel):
    """A boolean answer to the user question along with justification for the answer."""

    answer: bool = Field(
        ..., description="Whether the task has been completed successfully."
    )
    justification: str = Field(..., description="Justification for the answer.")


def create_conversational_agent(
    llm: BaseChatModel,
    tools: List[BaseTool],
    system_prompt: str | SystemMessage,
    camera_tool: GetROS2ImageConfiguredTool,
    logger: Optional[logging.Logger] = None,
    connector: ROS2Connector = None,
    manipulator_frame: str = "manipulator_base_link",
    debug: bool = False,
) -> CompiledStateGraph:
    _logger = None
    if logger:
        _logger = logger
    else:
        _logger = logging.getLogger(__name__)

    _logger.info("🚀 Creating conversational agent")
    _logger.info(f"🔧 Available tools: {[tool.name for tool in tools]}")
    _logger.info(f"📷 Camera tool: {camera_tool.__class__.__name__}")

    llm_with_tools = llm.bind_tools(tools)
    _logger.info("🔗 LLM bound with tools")

    tool_node = ToolRunner(tools=tools, logger=_logger)
    _logger.info("🛠️ Tool runner created")

    workflow = StateGraph(State)
    workflow.add_node("tools", tool_node)
    workflow.add_node("thinker", partial(agent, llm_with_tools, _logger, system_prompt))
    _logger.info("📊 Added nodes: 'tools' and 'thinker'")

    workflow.add_edge(START, "thinker")
    workflow.add_edge("tools", "thinker")
    _logger.info("🔗 Added edges: START->thinker, tools->thinker")

    workflow.add_conditional_edges(
        "thinker",
        partial(
            tools_condition,
            logger=_logger,
            connector=connector,
            manipulator_frame=manipulator_frame,
        ),
        {"thinker": "thinker", "tools": "tools", "__end__": END},
    )
    _logger.info("🔀 Added conditional edges from thinker")

    app = workflow.compile(debug=debug)
    _logger.info("✅ Conversational agent created successfully")
    return app

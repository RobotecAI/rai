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
from functools import partial
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    TypedDict,
    cast,
)

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from langgraph.graph import START, StateGraph
from langgraph.prebuilt.tool_node import tools_condition

from rai.agents.tool_runner import ToolRunner
from rai.initialization import get_llm_model
from rai.messages import HumanMultimodalMessage, SystemMultimodalMessage


class ReActAgentState(TypedDict):
    """State type for the react agent.

    Parameters
    ----------
    messages : List[BaseMessage]
        List of messages in the conversation
    """

    messages: List[BaseMessage]


def llm_node(
    llm: BaseChatModel,
    system_prompt: Optional[str | SystemMultimodalMessage],
    state: ReActAgentState,
):
    """Process messages using the LLM.

    Parameters
    ----------
    llm : BaseChatModel
        The language model to use for processing
    state : ReActAgentState
        Current state containing messages

    Returns
    -------
    ReActAgentState
        Updated state with new AI message

    Raises
    ------
    ValueError
        If state is invalid or LLM processing fails
    """
    if isinstance(system_prompt, SystemMultimodalMessage):
        if not isinstance(state["messages"][0], SystemMessage):
            state["messages"].insert(0, system_prompt)
    elif system_prompt:
        # at this point, state['messages'] length should at least be 1
        if not isinstance(state["messages"][0], SystemMessage):
            state["messages"].insert(0, SystemMessage(content=system_prompt))
    ai_msg = llm.invoke(state["messages"])
    state["messages"].append(ai_msg)


def create_react_runnable(
    llm: Optional[BaseChatModel] = None,
    tools: Optional[List[BaseTool]] = None,
    system_prompt: Optional[str | SystemMultimodalMessage] = None,
) -> Runnable[ReActAgentState, ReActAgentState]:
    """Create a react agent that can process messages and optionally use tools.

    Parameters
    ----------
    llm : Optional[BaseChatModel], default=None
        Language model to use. If None, will use complex_model from config
    tools : Optional[List[BaseTool]], default=None
        List of tools the agent can use

    Returns
    -------
    Runnable[ReActAgentState, ReActAgentState]
        A runnable that processes messages and optionally uses tools

    Raises
    ------
    ValueError
        If tools are provided but invalid
    """
    if llm is None:
        llm = get_llm_model("complex_model", streaming=True)

    graph = StateGraph(ReActAgentState)
    graph.add_edge(START, "llm")

    if tools:
        tool_runner = ToolRunner(tools)
        graph.add_node("tools", tool_runner)
        graph.add_conditional_edges(
            "llm",
            tools_condition,
        )
        graph.add_edge("tools", "llm")
        # Bind tools to LLM
        bound_llm = cast(BaseChatModel, llm.bind_tools(tools))
        graph.add_node("llm", partial(llm_node, bound_llm, system_prompt))
    else:
        graph.add_node("llm", partial(llm_node, llm, system_prompt))

    # Compile the graph
    return graph.compile()


def retriever_wrapper(
    state_retriever: Callable[[], Dict[str, HumanMessage | HumanMultimodalMessage]],
    state: ReActAgentState,
):
    """This wrapper is used to put state messages into LLM context"""
    for source, message in state_retriever().items():
        message.content = f"{source}: {message.content}"
        logging.getLogger("state_retriever").debug(
            f"Adding state message:\n{message.pretty_repr()}"
        )
        state["messages"].append(message)
    return state


def create_state_based_runnable(
    llm: Optional[BaseChatModel] = None,
    tools: Optional[List[BaseTool]] = None,
    system_prompt: Optional[str] = None,
    state_retriever: Optional[Callable[[], Dict[str, Any]]] = None,
) -> Runnable[ReActAgentState, ReActAgentState]:
    if llm is None:
        llm = get_llm_model("complex_model", streaming=True)
    graph = StateGraph(ReActAgentState)
    graph.add_edge(START, "state_retriever")
    graph.add_edge("state_retriever", "llm")
    graph.add_conditional_edges(
        "llm",
        tools_condition,
    )
    graph.add_edge("tools", "state_retriever")

    if state_retriever is None:
        state_retriever = lambda: {}

    graph.add_node("state_retriever", partial(retriever_wrapper, state_retriever))

    if tools is None:
        tools = []
    bound_llm = cast(BaseChatModel, llm.bind_tools(tools))
    graph.add_node("llm", partial(llm_node, bound_llm, system_prompt))

    tool_runner = ToolRunner(tools)
    graph.add_node("tools", tool_runner)

    return graph.compile()

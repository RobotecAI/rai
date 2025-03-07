import operator
from functools import partial
from typing import Annotated, List, Optional, TypedDict, cast

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from langgraph.graph import START, StateGraph
from langgraph.prebuilt.tool_node import tools_condition
from rai.agents.tool_runner import ToolRunner
from rai.utils.model_initialization import get_llm_model


class SimpleAgentState(TypedDict):
    """State type for the simple agent.

    Parameters
    ----------
    messages : List[BaseMessage]
        List of messages in the conversation, with operator.add for combining states
    """

    messages: Annotated[List[BaseMessage], operator.add]


def llm_node(llm: BaseChatModel, state: SimpleAgentState):
    """Process messages using the LLM.

    Parameters
    ----------
    llm : BaseChatModel
        The language model to use for processing
    state : SimpleAgentState
        Current state containing messages

    Returns
    -------
    SimpleAgentState
        Updated state with new AI message

    Raises
    ------
    ValueError
        If state is invalid or LLM processing fails
    """

    ai_msg = llm.invoke(state["messages"])
    return {"messages": [ai_msg]}


def create_simple_agent(
    llm: Optional[BaseChatModel] = None, tools: Optional[List[BaseTool]] = None
) -> Runnable[SimpleAgentState, SimpleAgentState]:
    """Create a simple agent that can process messages and optionally use tools.

    Parameters
    ----------
    llm : Optional[BaseChatModel], default=None
        Language model to use. If None, will use complex_model from config
    tools : Optional[List[BaseTool]], default=None
        List of tools the agent can use

    Returns
    -------
    Runnable[SimpleAgentState, SimpleAgentState]
        A runnable that processes messages and optionally uses tools

    Raises
    ------
    ValueError
        If tools are provided but invalid
    """
    if llm is None:
        llm = get_llm_model("complex_model")

    graph = StateGraph(SimpleAgentState)
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
        graph.add_node("llm", partial(llm_node, bound_llm))
    else:
        graph.add_node("llm", partial(llm_node, llm))

    # Compile the graph
    return graph.compile()

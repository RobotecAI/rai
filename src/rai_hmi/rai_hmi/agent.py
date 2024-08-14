import logging
from functools import partial
from typing import List, Optional, TypedDict, Union

from langchain.chat_models.base import BaseChatModel
from langchain_core.messages import BaseMessage, SystemMessage
from langchain_core.tools import BaseTool
from langgraph.graph import START, StateGraph
from langgraph.prebuilt.tool_node import tools_condition
from rclpy.impl.rcutils_logger import RcutilsLogger

from rai.agents.state_based import ToolRunner

loggers_type = Union[RcutilsLogger, logging.Logger]


class State(TypedDict):
    messages: List[BaseMessage]


def agent(llm: BaseChatModel, logger: loggers_type, system_prompt: str, state: State):
    logger.info("Running thinker")
    prompt = (
        system_prompt
        + "\nYour main task is to converse with the user and fulfill their requests using available tooling. "
    )
    ai_msg = llm.invoke([SystemMessage(content=prompt)] + state["messages"])
    state["messages"].append(ai_msg)
    return state


def create_conversational_agent(
    llm: BaseChatModel,
    tools: List[BaseTool],
    system_prompt: str,
    logger: Optional[RcutilsLogger | logging.Logger] = None,
    debug=False,
):
    _logger = None
    if isinstance(logger, RcutilsLogger):
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

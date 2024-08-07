import logging
from functools import partial
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Sequence,
    TypedDict,
    Union,
)

import numpy as np
from langchain.chat_models.base import BaseChatModel
from langchain_core.language_models.base import LanguageModelInput
from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolCall,
    ToolMessage,
)
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_core.runnables.config import get_executor_for_config
from langchain_core.tools import BaseTool
from langchain_core.tools import tool as create_tool
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt.tool_node import str_output
from langgraph.utils import RunnableCallable
from rclpy.logging import RcutilsLogger

from rai.scenario_engine.messages import HumanMultimodalMessage, preprocess_image

loggers_type = Union[RcutilsLogger, logging.Logger]


class State(TypedDict):
    messages: List[BaseMessage]


class Report(BaseModel):
    problem: str = Field(..., title="Problem", description="The problem that occured")
    solution: str = Field(
        ..., title="Solution", description="The solution to the problem"
    )
    outcome: str = Field(
        ..., title="Outcome", description="The outcome of the solution"
    )
    steps: List[str] = Field(
        ..., title="Steps", description="The steps taken to solve the problem"
    )


class MyToolNode(RunnableCallable):
    def __init__(
        self,
        tools: Sequence[Union[BaseTool, Callable]],
        *,
        name: str = "tools",
        tags: Optional[list[str]] = None,
        logger: loggers_type,
    ) -> None:
        super().__init__(self._func, name=name, tags=tags, trace=False)
        self.tools_by_name: Dict[str, BaseTool] = {}
        for tool_ in tools:
            if not isinstance(tool_, BaseTool):
                tool_ = create_tool(tool_)
            self.tools_by_name[tool_.name] = tool_
        self.logger = logger

    def _func(self, input: dict[str, Any], config: RunnableConfig) -> Any:
        if messages := input.get("messages", []):
            message = messages[-1]
        else:
            raise ValueError("No message found in input")

        if not isinstance(message, AIMessage):
            raise ValueError("Last message is not an AIMessage")

        def run_one(call: ToolCall):
            self.logger.info(f"Running tool: {call['name']}")
            output = self.tools_by_name[call["name"]].invoke(call["args"], config)
            return ToolMessage(
                content=str_output(output), name=call["name"], tool_call_id=call["id"]
            )

        with get_executor_for_config(config) as executor:
            outputs = [*executor.map(run_one, message.tool_calls)]
            input["messages"].extend(outputs)
            return input


def tools_condition(
    state: Union[list[AnyMessage], dict[str, Any]],
) -> Literal["tools", "summarizer"]:
    if isinstance(state, list):
        ai_message = state[-1]
    elif messages := state.get("messages", []):
        ai_message = messages[-1]
    else:
        raise ValueError(f"No messages found in input state to tool_edge: {state}")
    if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
        return "tools"
    return "summarizer"


def thinker(llm: BaseChatModel, logger: loggers_type, state: State):
    logger.info("Running thinker")
    prompt = (
        "Based on the data provided, reason about the situation. "
        "Analyze the context, identify any problems or challenges, "
        "and consider potential implications."
    )
    ai_msg = llm.invoke([SystemMessage(content=prompt)] + state["messages"])
    state["messages"].append(ai_msg)
    return state


def decider(
    llm: Runnable[LanguageModelInput, BaseMessage], logger: loggers_type, state: State
):
    logger.info("Running decider")
    prompt = (
        "Based on the previous information, make a decision using tools if necessary. "
        "If you are sure the problem has been solved, do not request any tools. "
        "Request one tool at a time."
    )

    input = state["messages"] + [HumanMessage(prompt)]
    ai_msg = llm.invoke(input)
    state["messages"].append(ai_msg)
    if ai_msg.tool_calls:
        logger.info("Tools requested: {}".format(ai_msg.tool_calls))
    return state


def summarizer(llm: BaseChatModel, logger: loggers_type, state: State):
    logger.info("Summarizing the conversation")
    prompt = (
        "You are the reporter. Your task is to summarize what happened previously. "
        "Make sure to mention the problem, solution and the outcome."
    )
    ai_msg = llm.with_structured_output(Report).invoke(
        [SystemMessage(content=prompt)] + state["messages"]
    )
    state["messages"].append(ai_msg)
    return state


def retriever_wrapper(
    state_retriever: Callable[[], Dict[str, Any]], llm: BaseChatModel, state: State
):
    retrieved_info = state_retriever()
    processed_info = {}
    images = retrieved_info.pop("images", [])
    if "img" in retrieved_info:
        if isinstance(retrieved_info["img"], list):
            for image in images:
                assert isinstance(
                    image, (str, bytes, np.ndarray)
                ), f"Image must be a string, bytes, or numpy array. Got {type(image)}"
                processed_info["image_descriptions"].append(
                    llm.invoke(
                        [
                            SystemMessage(content="Please describe the image."),
                            HumanMultimodalMessage(
                                content="The image", images=[preprocess_image(image)]
                            ),
                        ]
                    ).content
                )
        elif isinstance(retrieved_info["img"], (str, bytes, np.ndarray)):
            processed_info["image_descriptions"] = [
                llm.invoke(
                    [
                        SystemMessage(content="Please describe the image."),
                        HumanMultimodalMessage(
                            content="The image",
                            images=[preprocess_image(retrieved_info["img"])],
                        ),
                    ]
                ).content
            ]

    processed_info.update(retrieved_info)
    info = str_output(processed_info)
    state["messages"].append(HumanMessage(content="Retrieved state: {}".format(info)))
    return state


def create_state_based_agent(
    llm: BaseChatModel,
    tools: List[BaseTool],
    state_retriever: Callable[[], Dict[str, Any]],
    logger: Optional[RcutilsLogger | logging.Logger] = None,
):
    _logger = None
    if isinstance(logger, RcutilsLogger):
        _logger = logger
    else:
        _logger = logging.getLogger(__name__)

    _logger.info("Creating state based agent")

    llm_with_tools = llm.bind_tools(tools)
    tool_node = MyToolNode(tools=tools, logger=_logger)

    workflow = StateGraph(State)
    workflow.add_node(
        "state_retriever", partial(retriever_wrapper, state_retriever, llm)
    )
    workflow.add_node("tools", tool_node)
    workflow.add_node("thinker", partial(thinker, llm, _logger))
    workflow.add_node("decider", partial(decider, llm_with_tools, _logger))
    workflow.add_node("summarizer", partial(summarizer, llm, _logger))

    workflow.add_edge(START, "state_retriever")
    workflow.add_edge("state_retriever", "thinker")
    workflow.add_edge("thinker", "decider")
    workflow.add_edge("tools", "state_retriever")
    workflow.add_edge("summarizer", END)
    workflow.add_conditional_edges(
        "decider",
        tools_condition,
    )

    app = workflow.compile()
    _logger.info("State based agent created")
    return app

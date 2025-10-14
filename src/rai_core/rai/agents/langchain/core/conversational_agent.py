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

from langchain_core.messages import HumanMessage
from rai.messages import HumanMultimodalMessage
from langchain.chat_models.base import BaseChatModel
from langchain_core.messages import (
    BaseMessage,
    SystemMessage,
)
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool
from langgraph.graph import START, StateGraph, END
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel, Field

from rai.agents.langchain.core.tool_runner import ToolRunner
from rai.agents.langchain.invocation_helpers import invoke_llm_with_tracing
from rai.communication.ros2.connectors import ROS2Connector
from rai.tools.ros2.manipulation import ResetArmTool


class State(TypedDict):
    messages: List[BaseMessage]


def agent(
    llm: BaseChatModel,
    logger: logging.Logger,
    system_prompt: str | SystemMessage,
    state: State,
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
    ai_msg = llm.invoke(state["messages"])
    state["messages"].append(ai_msg)
    return state


class BoolAnswerWithJustification(BaseModel):
    """A boolean answer to the user question along with justification for the answer."""

    answer: bool = Field(
        ..., description="Whether the task has been completed successfully."
    )
    justification: str = Field(..., description="Justification for the answer.")


def camera_tool_critic(
    state: State, vlm, camera_tool: GetROS2ImageConfiguredTool, logger: logging.Logger
) -> BoolAnswerWithJustification:
    """
    Takes the latest camera image and the latest human message
    to judge if the task has been completed. Returns True if completed, else False.

    This is a stub function; actual implementation should use vision/language models.
    """
    # Find the last HumanMessage
    human_msg = None
    for msg in reversed(state["messages"]):
        if (
            msg.__class__.__name__ == "HumanMessage"
            or msg.__class__.__name__ == "HumanMultimodalMessage"
        ):
            human_msg = msg
            break

    if human_msg is None:
        logger.warning("No human message found in state")
        return BoolAnswerWithJustification(
            answer=False, justification="No human message found"
        )

    task = human_msg.content if hasattr(human_msg, "content") else str(human_msg)

    # Get initial image if available
    initial_image = None
    if hasattr(human_msg, "images") and human_msg.images:
        initial_image = human_msg.images[0]

    # Get current camera image
    try:
        _, artifact = camera_tool._run()
        current_image = (
            artifact.images[0]
            if hasattr(artifact, "images") and artifact.images
            else None
        )
    except Exception as e:
        logger.error(f"Failed to get camera image: {e}")
        return BoolAnswerWithJustification(
            answer=False, justification=f"Failed to get camera image: {e}"
        )

    if initial_image is None or current_image is None:
        logger.warning("Missing images for comparison")
        return BoolAnswerWithJustification(
            answer=False, justification="Missing images for comparison"
        )

    prompt = """
    You are a helpful assistant that judges if a manipulation task has been completed successfully.
    Compare the initial and current images to determine if the requested task has been accomplished.
    Return True if the task appears to be completed successfully, False otherwise.
    """

    system_msg = SystemMessage(content=prompt)
    images = [initial_image, current_image]
    message = HumanMultimodalMessage(
        content=f"Using these 2 images (initial and current) please judge if the task: '{task}' has been completed successfully.",
        images=images,
    )
    msgs = [system_msg, message]

    try:
        vlm_with_structured_output = vlm.with_structured_output(
            BoolAnswerWithJustification
        )
        vlm_response = vlm_with_structured_output.invoke(msgs)
        return vlm_response
    except Exception as e:
        logger.error(f"Failed to get VLM response: {e}")
        return BoolAnswerWithJustification(
            answer=False, justification=f"VLM evaluation failed: {e}"
        )


def tools_condition(
    state: State,
    vlm,
    camera_tool: GetROS2ImageConfiguredTool,
    logger: logging.Logger,
    messages_key: str = "messages",
) -> str:
    if isinstance(state, list):
        ai_message = state[-1]
    elif isinstance(state, dict) and (messages := state.get(messages_key, [])):
        ai_message = messages[-1]
    elif messages := getattr(state, messages_key, []):
        ai_message = messages[-1]
    else:
        raise ValueError(f"No messages found in input state to tool_edge: {state}")

    # Check if the AI message has tool calls
    if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
        return "tools"

    # If no tool calls, check if task is completed using the critic
    critic_response = camera_tool_critic(state, vlm, camera_tool, logger)
    if critic_response.answer:
        # Task completed successfully - end the conversation
        return "__end__"
    else:
        # Task not completed - continue with feedback
        state["messages"].append(
            HumanMessage(
                content=f"The task has not been completed yet. Feedback: {critic_response.justification}. Please try again."
            )
        )
        return "thinker"


@deprecated(
    "Use rai.agents.langchain.core.create_react_runnable instead. "
    "Support for the conversational agent will be removed in the 3.0 release."
)
class BoolAnswerWithJustification(BaseModel):
    """A boolean answer to the user question along with justification for the answer."""

    answer: bool = Field(
        ..., description="Whether the task has been completed successfully."
    )
    justification: str = Field(..., description="Justification for the answer.")


def tools_condition(
    state: State,
    logger: logging.Logger,
    messages_key: str = "messages",
    connector: ROS2Connector = None,
    manipulator_frame: str = "manipulator_base_link",
) -> str:
    logger.info("🔀 Running tools_condition to determine next step")

    if isinstance(state, list):
        ai_message = state[-1]
        logger.info(f"📝 State is list with {len(state)} items")
    elif isinstance(state, dict) and (messages := state.get(messages_key, [])):
        ai_message = messages[-1]
        logger.info(f"📝 State is dict with {len(messages)} messages")
    elif messages := getattr(state, messages_key, []):
        ai_message = messages[-1]
        logger.info(f"📝 State has messages attribute with {len(messages)} messages")
    else:
        logger.error(f"❌ No messages found in input state: {type(state)} - {state}")
        raise ValueError(f"No messages found in input state to tool_edge: {state}")

    # Check if the AI message has tool calls
    if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
        logger.info(
            f"🔧 AI message has {len(ai_message.tool_calls)} tool calls - routing to tools"
        )
        return "tools"

    reset_arm_tool = ResetArmTool(
        connector=connector, manipulator_frame=manipulator_frame
    )
    reset_arm_tool._run()
    return "__end__"


def create_conversational_agent(
    llm: BaseChatModel,
    tools: List[BaseTool],
    system_prompt: str | SystemMessage,
    camera_tool: GetROS2ImageConfiguredTool,
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
    workflow.add_node("critic", partial(camera_tool_critic, llm, camera_tool, _logger))

    workflow.add_conditional_edges(
        "thinker",
        partial(tools_condition, llm, camera_tool, _logger),
        {"thinker": "thinker", "tools": "tools", "__end__": END},
    )

    app = workflow.compile(debug=debug)
    _logger.info("State based agent created")
    return app

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

from functools import partial
from typing import (
    Any,
    Dict,
    List,
    Optional,
    cast,
)

from langchain.chat_models.base import BaseChatModel
from langchain.prompts import ChatPromptTemplate
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    SystemMessage,
)
from langchain_core.tools import BaseTool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from rai.agents.langchain.core import ReActAgentState
from rai.agents.langchain.core.tool_runner import ToolRunner
from rai.initialization import get_llm_model
from rai.messages import HumanMultimodalMessage, SystemMultimodalMessage


class TaskVerificationState(ReActAgentState):
    original_task: str
    task_completed: bool
    verification_attempts: int
    max_verification_attempts: int
    verification_results: List[Dict[str, Any]]


def worker_node(
    llm: BaseChatModel,
    system_prompt: Optional[str | SystemMultimodalMessage],
    state: TaskVerificationState,
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
    return state


def verification_node(
    llm: BaseChatModel,
    state: TaskVerificationState,
) -> TaskVerificationState:
    """
    Verification node that uses tools to check if the original task has been accomplished.
    """

    verification_prompt = ChatPromptTemplate.from_template("""You are a task verification assistant.

Your job is to:
1. Use available tools to gather information about the current environment
2. Determine if the original task has been completed successfully
3. Make sure to grab camera images and object positions to understand the environment

Original task: {original_task}

Current verification attempt: {attempt}/{max_attempts}

After using tools to gather information, conclude with either:
- TASK_COMPLETED: if the original task has been fully accomplished
- TASK_INCOMPLETE: if more work is needed

If TASK_INCOMPLETE, specify:
- What is missing or incorrect
- What tools should be used next
- What specific actions should be taken

Be thorough but concise in your analysis.""")
    # (NOTE) jmatejcz maybe recent messages can be included here also

    state["verification_attempts"] += 1
    prompt_input = {
        "original_task": state["original_task"],
        "attempt": state["verification_attempts"],
        "max_attempts": state["max_verification_attempts"],
    }
    verification_messages = verification_prompt.invoke(prompt_input)

    verification_response = llm.invoke(verification_messages)
    verification_content = verification_response.content

    state["messages"].append(verification_response)

    if "TASK_COMPLETED" in verification_content.upper():
        state["task_completed"] = True
    else:
        state["task_completed"] = False

    verification_result = {
        "attempt": state["verification_attempts"],
        "result": verification_content,
        "completed": state["task_completed"],
    }
    state["verification_results"].append(verification_result)

    return state


def verification_condition(
    state: TaskVerificationState,
) -> str:
    """Check if the task verification node should be invoked."""
    last_message = state["messages"][-1]
    if state["task_completed"]:
        return END  # Task is done
    elif state["verification_attempts"] > state["max_verification_attempts"]:
        return END  # Stop trying, max attempts reached
    elif isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "tools"
    else:
        return "llm"  # Continue working on the task to verification node


def tool_condition(
    state: ReActAgentState,
) -> str:
    """Check if the tool calls present, if yes go to tools if not go verification"""
    last_message = state["messages"][-1]

    # Check if the last message is an AI message with tool calls
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        # If there are tool calls, route to the tools node
        return "tools"
    else:
        # If no tool calls, go to verification node
        return "verification"


def create_task_verification_agent(
    tools: List[BaseTool],
    work_llm: Optional[BaseChatModel] = None,
    verification_llm: Optional[BaseChatModel] = None,
    system_prompt: Optional[str | SystemMultimodalMessage] = None,
) -> CompiledStateGraph:
    """Create a react agent that can process messages and optionally use tools.

    Parameters
    ----------
    llm : Optional[BaseChatModel], default=None
        Language model to use. If None, will use complex_model from config
    tools : Optional[List[BaseTool]], default=None
        List of tools the agent can use



    Raises
    ------
    ValueError
        If tools are provided but invalid
    """
    if work_llm is None:
        work_llm = get_llm_model("complex_model", streaming=True)
    if verification_llm is None:
        verification_llm = get_llm_model("complex_model", streaming=True)

    graph = StateGraph(TaskVerificationState)
    graph.add_edge(START, "worker")

    if tools:
        tool_runner = ToolRunner(tools)
        work_llm_with_tools = cast(BaseChatModel, work_llm.bind_tools(tools))
        verification_llm_with_tools = cast(
            BaseChatModel, verification_llm.bind_tools(tools)
        )

        graph.add_node(
            "worker", partial(worker_node, work_llm_with_tools, system_prompt)
        )
        graph.add_node("tools", tool_runner)
        graph.add_node(
            "verification",
            partial(verification_node, verification_llm_with_tools),
        )

        # after worker either use tools or verify
        graph.add_conditional_edges(
            "worker", tool_condition, {"tools": "tools", "verification": "verification"}
        )
        # always go back to worker after tools
        graph.add_edge("tools", "worker")

        # after verification either continue working or end
        graph.add_conditional_edges(
            "verification",
            verification_condition,
            {"worker": "worker", "tools": "tools", END: END},
        )

    else:
        graph.add_node("worker", partial(worker_node, work_llm, system_prompt))
        graph.add_node(
            "verification",
            partial(verification_node, verification_llm),
        )

        graph.add_edge("worker", "verification")
        graph.add_conditional_edges(
            "verification", verification_condition, {"worker": "worker", END: END}
        )

    return graph.compile()


def create_initial_task_verification_state(
    original_task: str,
    messages: Optional[List[BaseMessage]] = None,
    max_verification_attempts: int = 1,
) -> TaskVerificationState:
    """Create initial state for the task verification agent."""
    if messages is None:
        messages = [HumanMultimodalMessage(content=original_task)]
    return TaskVerificationState(
        original_task=original_task,
        messages=messages,
        task_completed=False,
        verification_attempts=0,
        max_verification_attempts=max_verification_attempts,
        verification_results=[],
    )

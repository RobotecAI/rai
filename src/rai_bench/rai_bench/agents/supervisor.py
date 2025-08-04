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

from typing import Any, Dict, List, Optional, Literal
from enum import Enum

from langchain.chat_models.base import BaseChatModel
from langchain_core.messages import BaseMessage, SystemMessage
from langchain_core.tools import BaseTool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import create_react_agent

from pydantic import BaseModel, Field
from rai.agents.langchain.core import ReActAgentState
from rai.agents.langchain.core.react_agent import create_react_runnable
from langgraph_supervisor import create_supervisor
from rai.initialization import get_llm_model
from rai.messages import HumanMultimodalMessage


class AgentType(str, Enum):
    """Types of specialized agents."""

    MANIPULATION = "manipulation"
    NAVIGATION = "navigation"
    COMPLETE = "complete"

class Plan(BaseModel):
    steps: List[]

class SupervisorDecision(BaseModel):
    """Decision made by the supervisor."""

    next_agent: AgentType = Field(
        description="Which agent should handle the task next. Use 'manipulation' for "
        "object detection and manipulation tasks, 'navigation' for movement/orientation tasks, "
        "or 'complete' if the task is finished."
    )
    task_instruction: str = Field(
        description="Specific instruction for the chosen agent"
    )


class SupervisorState(ReActAgentState):
    """State for the supervisor agent."""

    original_task: str
    # current_objective: str
    # manipulation_history: List[str]
    # navigation_history: List[str]
    next_agent: str
    task_completed: bool
    final_response: str


def should_continue(state: SupervisorState) -> str:
    """Determine which node to execute next."""
    if state["task_completed"] or state["final_response"]:
        return END
    elif state["next_agent"] == AgentType.MANIPULATION:
        return "manipulation"
    elif state["next_agent"] == AgentType.NAVIGATION:
        return "navigation"
    else:
        return END


from typing import Annotated
from langchain_core.tools import tool, InjectedToolCallId
from langgraph.prebuilt import InjectedState
from langgraph.graph import StateGraph, START, MessagesState
from langgraph.types import Command


def create_supervisor_agent(
    manipulation_tools: List[BaseTool],
    navigation_tools: List[BaseTool],
    supervisor_llm: Optional[BaseChatModel] = None,
    executor_llm: Optional[BaseChatModel] = None,
    system_prompt: Optional[str] = None,
) -> CompiledStateGraph:
    """Create a supervisor agent with manipulation and navigation specialist nodes.

    Parameters
    ----------
    manipulation_tools : List[BaseTool]
        Tools available to the manipulation agent (e.g., grasp, pick, place)
    navigation_tools : List[BaseTool]
        Tools available to the navigation agent (e.g., move_to, navigate, avoid_obstacles)
    supervisor_llm : Optional[BaseChatModel], default=None
        Language model for the supervisor decision-making
    executor_llm : Optional[BaseChatModel], default=None
        Language model for the specialist agents

    Returns
    -------
    CompiledStateGraph
        Compiled state graph for the supervisor agent

    Raises
    ------
    ValueError
        If tools are not provided
    """
    if supervisor_llm is None:
        supervisor_llm = get_llm_model("complex_model", streaming=True)
    if executor_llm is None:
        executor_llm = get_llm_model("complex_model", streaming=True)

    if not manipulation_tools and not navigation_tools:
        raise ValueError("At least one set of tools must be provided")

    def create_handoff_tool(*, agent_name: str, description: str | None = None):
        name = f"transfer_to_{agent_name}"
        description = description or f"Ask {agent_name} for help."

        @tool(name, description=description)
        def handoff_tool(
            state: Annotated[MessagesState, InjectedState],
            tool_call_id: Annotated[str, InjectedToolCallId],
        ) -> Command:
            tool_message = {
                "role": "tool",
                "content": f"Successfully transferred to {agent_name}",
                "name": name,
                "tool_call_id": tool_call_id,
            }
            return Command(
                goto=agent_name,
                update={**state, "messages": state["messages"] + [tool_message]},
                graph=Command.PARENT,
            )

        return handoff_tool

    manipulation_system_prompt = """You are a manipulation specialist robot agent.
Your role is to handle object manipulation tasks including picking up and droping objects using provided tools.

Ask the VLM for objects detection and positions before perfomring any manipulation action."""

    navigation_system_prompt = """You are a navigation specialist robot agent.
Your role is to handle navigation tasks in space using provided tools.

After performing navigation action, check your current position to ensure success"""

    supervisor_system_prompt = """You are a supervisor agent that coordinates
between manipulation and navigation specialists. Analyze the current task
and decide which specialist should handle it next. Break down complex tasks
into appropriate subtasks for each specialist.
Assign work to one agent at a time, do not call agents in parallel"""

    if system_prompt:
        supervisor_system_prompt += "\n"
        supervisor_system_prompt += system_prompt

    # Create specialist agents
    manipulation_agent = create_react_agent(
        model=executor_llm,
        prompt=manipulation_system_prompt,
        tools=manipulation_tools,
        name="manipulation",
    )

    navigation_agent = create_react_agent(
        model=executor_llm,
        prompt=navigation_system_prompt,
        tools=navigation_tools,
        name="navigation",
    )
    # Handoffs
    assign_to_nav_agent = create_handoff_tool(
        agent_name="navigation",
        description="Assign task to a navigation agent.",
    )

    assign_to_manipulation_agent = create_handoff_tool(
        agent_name="manipulation",
        description="Assign task to a manipulation agent.",
    )

    supervisor_agent = create_react_agent(
        model="openai:gpt-4.1",
        tools=[assign_to_nav_agent, assign_to_manipulation_agent],
        prompt=supervisor_system_prompt,
        name="supervisor",
    )

    supervisor = (
        StateGraph(MessagesState)
        .add_node(supervisor_agent)
        .add_node(navigation_agent)
        .add_node(manipulation_agent)
        .add_edge(START, "supervisor")
        # always return back to the supervisor
        .add_edge("navigation", "supervisor")
        .add_edge("manipulation", "supervisor")
        .compile()
    )

    return supervisor

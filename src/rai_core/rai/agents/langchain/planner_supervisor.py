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


### NOTE (jmatejcz) this agent is still in process of testing and refining
from enum import Enum
from functools import partial
from typing import Annotated, List, Optional

from langchain.chat_models.base import BaseChatModel
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.tools import BaseTool, InjectedToolCallId, tool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import create_react_agent
from langgraph.types import Command
from pydantic import BaseModel, Field

from rai.agents.langchain.core import ReActAgentState, SubAgentToolRunner
from rai.messages import (
    HumanMultimodalMessage,
)


class AgentType(str, Enum):
    """Types of specialized agents."""

    MANIPULATION = "manipulation"
    NAVIGATION = "navigation"
    COMPLETE = "complete"


class Plan(BaseModel):
    """A plan to help solve a user request."""

    steps: List[str] = Field(
        description="different steps to follow, should be in sorted order"
    )


class ExecutionPlan(ReActAgentState):
    plan: List[str]
    step: str
    step_messages: List[BaseMessage]  # messages for subagents for given step
    fail_msg: str


class SubAgentResponse(BaseModel):
    success: bool = Field(description="Whether the task was completed successfully")
    explanation: Optional[str] = Field(
        default=None,
        description="Explanation of what happened, required if success=False",
    )


def llm_node(
    llm: BaseChatModel,
    system_prompt: Optional[str],
    state: ExecutionPlan,
) -> ExecutionPlan:
    """Process messages using the LLM - returns the agent's response."""
    messages = state["step_messages"].copy()
    if system_prompt:
        messages.insert(0, HumanMessage(state["step"]))
        messages.insert(0, SystemMessage(content=system_prompt))

    ai_msg = llm.invoke(messages)
    # append to both
    # step messages
    state["messages"].append(ai_msg)
    state["step_messages"].append(ai_msg)
    return state


def structured_output_node(
    llm: BaseChatModel,
    state: ExecutionPlan,
):
    """Analyze the conversation and return structured output."""
    final_response = (
        state["step_messages"][-1].content if state["step_messages"] else ""
    )

    # Analyze with structured output
    analyzer = llm.with_structured_output(SubAgentResponse)
    analysis = analyzer.invoke(
        [
            SystemMessage(
                content=f"""
Analyze if this task was completed successfully:

Task: {state["step"]}
Agent Response: {final_response}

Determine success and provide brief explanation."""
            ),
            *state["step_messages"],
        ]
    )
    if analysis.success:
        # remove the step from plan
        state["plan"].pop(0)
        return state
    else:
        state["fail_msg"] = f"""step: `{state["step"]}` failed.
{analysis.explanation}"""
        return state


def should_continue_or_structure(state: ExecutionPlan) -> str:
    """Decide whether to continue with tools or return structured output."""
    last_message = state["step_messages"][-1]

    # If AI message has tool calls, continue to tools
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"

    # Otherwise, return structured output
    return "structured_output"


def create_react_structured_agent(
    llm: BaseChatModel,
    tools: Optional[List[BaseTool]] = None,
    system_prompt: Optional[str] = None,
) -> CompiledStateGraph:
    """Create a react agent that returns structured output."""

    graph = StateGraph(ExecutionPlan)
    graph.add_edge(START, "llm")

    if tools:
        tool_runner = SubAgentToolRunner(tools)
        graph.add_node("tools", tool_runner)

        bound_llm = llm.bind_tools(tools)
        graph.add_node("llm", partial(llm_node, bound_llm, system_prompt))

        graph.add_node("structured_output", partial(structured_output_node, llm))

        graph.add_conditional_edges(
            "llm",
            should_continue_or_structure,
            {"tools": "tools", "structured_output": "structured_output"},
        )
        graph.add_edge("tools", "llm")
        graph.add_edge("structured_output", END)
    else:
        graph.add_node("llm", partial(llm_node, llm, system_prompt))
        graph.add_node("structured_output", partial(structured_output_node, llm))
        graph.add_edge("llm", "structured_output")
        graph.add_edge("structured_output", END)

    return graph.compile()


def create_supervisor_node(
    supervisor_llm: BaseChatModel,
    handoff_tools: List[BaseTool],
    supervisor_system_prompt: str,
):
    """Create a custom supervisor node that handles ExecutionPlan state."""

    def supervisor_node(state: ExecutionPlan) -> ExecutionPlan:
        if not state["plan"]:
            # No more steps - task complete
            completion_message = HumanMessage(
                content="All steps completed successfully. Task finished."
            )
            return {**state, "messages": state["messages"] + [completion_message]}

        remaining_steps = "\n".join([step for step in state["plan"]])

        plan = f"""Plan:
{remaining_steps}
"""

        # Prepare messages for supervisor
        supervisor_messages = [
            SystemMessage(content=supervisor_system_prompt),
            HumanMessage(content=plan),
        ]
        if "fail_msg" in state:
            supervisor_messages.append(HumanMessage(content=state["fail_msg"]))

        # Create supervisor agent and invoke
        supervisor_agent = create_react_agent(
            model=supervisor_llm,
            tools=handoff_tools,
            prompt=supervisor_system_prompt,
            name="supervisor",
        )

        # Run supervisor with custom input
        supervisor_result = supervisor_agent.invoke({"messages": supervisor_messages})

        # Return updated state with supervisor's response
        return {
            **state,
            "messages": state["messages"]
            + supervisor_result["messages"][-1:],  # Add supervisor's last message"step
        }

    return supervisor_node


def create_planner_supervisor(
    manipulation_tools: List[BaseTool],
    navigation_tools: List[BaseTool],
    planner_llm: BaseChatModel,
    supervisor_llm: BaseChatModel,
    executor_llm: BaseChatModel,
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
    if not manipulation_tools and not navigation_tools:
        raise ValueError("At least one set of tools must be provided")

    def create_handoff_tool(*, agent_name: str, description: str | None = None):
        name = f"transfer_to_{agent_name}"
        description = description or f"Ask {agent_name} for help."

        @tool(name, description=description)
        def handoff_tool(
            task_instruction: str,  # The specific task for the agent
            state: ExecutionPlan,
            tool_call_id: Annotated[str, InjectedToolCallId],
        ) -> Command:
            # Create a clean task message for the specialist agent

            return Command(
                goto=agent_name,
                # Send only the task message to the specialist agent, not the full history
                update={"step": task_instruction, "step_messages": []},
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
between manipulation and navigation specialists. Analyze the current plan
and decide which specialist should handle next step.

Assign next step to one agent at a time, do not call agents in parallel.
Specialists don't have acccess to environment description so make sure to include all neccessery values
to complete given task."""

    planner_system_prompt = """For the given objective, come up with a simple step by step plan.
The plan will be executed by manipulation and navigation specialists.

Manipulaiton specialist can pick up and drop objects and ask VLM about the nearby objects.
Navigation specialist can navigate to certain coordinates and determine the location of robot in the environment.

Break down complex tasks into appropriate subtasks for each specialist.
If a sequence of actions can be done by single specialist make it single step of a plan.
"""
    if system_prompt:
        supervisor_system_prompt += "\n"
        supervisor_system_prompt += system_prompt
        planner_system_prompt += "\n"
        planner_system_prompt += system_prompt

    # Create specialist agents
    manipulation_agent = create_react_structured_agent(
        llm=executor_llm,
        system_prompt=manipulation_system_prompt,
        tools=manipulation_tools,
    )

    navigation_agent = create_react_structured_agent(
        llm=executor_llm,
        system_prompt=navigation_system_prompt,
        tools=navigation_tools,
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

    supervisor_node = create_supervisor_node(
        supervisor_llm=supervisor_llm,
        handoff_tools=[assign_to_manipulation_agent, assign_to_nav_agent],
        supervisor_system_prompt=supervisor_system_prompt,
    )
    planner = planner_llm.with_structured_output(Plan)

    def plan_step(state: ExecutionPlan):
        """Initial planning step."""
        # TODO (jmatejcz) add structured output to plan ?
        # maybe dict where the agent assigned is specified
        # then squash steps that are in sequence assigned to the same agent

        task = None
        for message in reversed(state["messages"]):
            if hasattr(message, "content") and message.content:
                task = message.content[0]["text"]
                break
        if not task:
            raise ValueError("task can't be none")

        messages = [
            SystemMessage(content=planner_system_prompt),
            HumanMultimodalMessage(content=task),
        ]
        plan = planner.invoke(messages)
        # plan_content = "Execution Plan:\n"

        # for i, step in enumerate(plan.steps, 1):
        #     plan_content += f"{step}\n"
        return {
            "plan": plan.steps,
            # "messages": [
            #     HumanMessage(plan_content)
            # ]  # Only send the plan, not the original prompt
        }

    supervisor = (
        StateGraph(ExecutionPlan)
        .add_node("planner", plan_step)
        .add_node("supervisor", supervisor_node)
        .add_node("navigation", navigation_agent)
        .add_node("manipulation", manipulation_agent)
        .add_edge(START, "planner")
        .add_edge("planner", "supervisor")
        # always return back to the supervisor
        .add_edge("navigation", "supervisor")
        .add_edge("manipulation", "supervisor")
        .compile()
    )

    return supervisor

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
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import partial
from typing import (
    Annotated,
    List,
    Optional,
)

from langchain.chat_models.base import BaseChatModel
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.tools import BaseTool, InjectedToolCallId, tool
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import create_react_agent
from langgraph.types import Command
from pydantic import BaseModel, Field

from rai.agents.langchain.core.tool_runner import SubAgentToolRunner
from rai.messages import (
    HumanMultimodalMessage,
)


class StepSuccess(BaseModel):
    """Output of success attacher"""

    success: bool = Field(description="Whether the task was completed successfully")
    explanation: str = Field(description="Explanation of what happened")


class MegamindState(MessagesState):
    original_task: str
    steps_done: List[str]
    step: Optional[str]
    step_success: StepSuccess
    step_messages: List[BaseMessage]


def llm_node(
    llm: BaseChatModel,
    system_prompt: Optional[str],
    state: MegamindState,
) -> MegamindState:
    """Process messages using the LLM - returns the agent's response."""
    messages = state["step_messages"].copy()
    if not state["step"]:
        raise ValueError("Step should be defined at this point")
    if system_prompt:
        messages.insert(0, HumanMessage(state["step"]))
        messages.insert(0, SystemMessage(content=system_prompt))

    ai_msg = llm.invoke(messages)
    # append to both
    state["step_messages"].append(ai_msg)
    state["messages"].append(ai_msg)
    return state


def analyzer_node(
    llm: BaseChatModel,
    planning_prompt: Optional[str],
    state: MegamindState,
) -> MegamindState:
    """Analyze the conversation and return structured output."""
    if not planning_prompt:
        planning_prompt = ""
    analyzer = llm.with_structured_output(StepSuccess)
    analysis = analyzer.invoke(
        [
            SystemMessage(
                content=f"""
Analyze if this task was completed successfully:

Task: {state["step"]}

{planning_prompt}
Below you have messages of agent doing the task:"""
            ),
            *state["step_messages"],
        ]
    )
    state["step_success"] = StepSuccess(
        success=analysis.success, explanation=analysis.explanation
    )
    state["steps_done"].append(f"{state['step_success'].explanation}")
    return state


def should_continue_or_structure(state: MegamindState) -> str:
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
    planning_prompt: Optional[str] = None,
) -> CompiledStateGraph:
    """Create a react agent that returns structured output."""

    graph = StateGraph(MegamindState)
    graph.add_edge(START, "llm")

    if tools:
        tool_runner = SubAgentToolRunner(tools)
        graph.add_node("tools", tool_runner)

        bound_llm = llm.bind_tools(tools)
        graph.add_node("llm", partial(llm_node, bound_llm, system_prompt))

        graph.add_node(
            "structured_output", partial(analyzer_node, llm, planning_prompt)
        )

        graph.add_conditional_edges(
            "llm",
            should_continue_or_structure,
            {"tools": "tools", "structured_output": "structured_output"},
        )
        graph.add_edge("tools", "llm")
        graph.add_edge("structured_output", END)
    else:
        graph.add_node("llm", partial(llm_node, llm, system_prompt))
        graph.add_node(
            "structured_output", partial(analyzer_node, llm, planning_prompt)
        )
        graph.add_edge("llm", "structured_output")
        graph.add_edge("structured_output", END)

    return graph.compile()


def create_handoff_tool(agent_name: str, description: str = None):
    """Create a handoff tool for transferring tasks to specialist agents."""
    name = f"transfer_to_{agent_name}"
    description = description or f" {agent_name} for help."

    @tool(name, description=description)
    def handoff_tool(
        task_instruction: str,  # The specific task for the agent
        tool_call_id: Annotated[str, InjectedToolCallId],
    ) -> Command:
        return Command(
            goto=agent_name,
            # Send only the task message to the specialist agent, not the full history
            update={"step": task_instruction, "step_messages": []},
            graph=Command.PARENT,
        )

    return handoff_tool


@dataclass
class Executor:
    name: str
    llm: BaseChatModel
    tools: List[BaseTool]
    system_prompt: str


class ContextProvider(ABC):
    """Context provider are meant to inject exteral info to megamind prompt"""

    @abstractmethod
    def get_context(self) -> str:
        pass


def get_initial_megamind_state(task: str):
    return MegamindState(
        {
            "original_task": task,
            "messages": [HumanMultimodalMessage(content=task)],
            "step": "",
            "steps_done": [],
            "step_success": StepSuccess(success=False, explanation=""),
            "step_messages": [],
        }
    )


def plan_step(
    megamind_agent: BaseChatModel,
    state: MegamindState,
    context_providers: Optional[List[ContextProvider]] = None,
) -> MegamindState:
    """Initial planning step."""
    if "original_task" not in state:
        state["original_task"] = state["messages"][0].content[0]["text"]
    if "steps_done" not in state:
        state["steps_done"] = []
    if "step" not in state:
        state["step"] = None

    megamind_prompt = f"You are given objective to complete: {state['original_task']}"
    for provider in context_providers:
        megamind_prompt += provider.get_context()
        megamind_prompt += "\n"
    if state["steps_done"]:
        megamind_prompt += "\n\n"
        megamind_prompt += "Steps that were already done successfully:\n"
        steps_done = "\n".join(
            [f"{i + 1}. {step}" for i, step in enumerate(state["steps_done"])]
        )
        megamind_prompt += steps_done
        megamind_prompt += "\n"

    if state["step"]:
        if not state["step_success"]:
            raise ValueError("Step success should be specified at this point")

        megamind_prompt += "\nBased on that outcome and past steps come up with the next step and delegate it to selected agent."

    else:
        megamind_prompt += "\n"
        megamind_prompt += (
            "Come up with the fist step and delegate it to selected agent."
        )

    megamind_prompt += "\n\n"
    megamind_prompt += (
        "When you decide that the objective is completed return response to user."
    )
    messages = [
        HumanMultimodalMessage(content=megamind_prompt),
    ]
    # NOTE (jmatejcz) the response of megamind isnt appended to messages
    # as Command from handoff instantly transitions to next node
    megamind_agent.invoke({"messages": messages})
    return state


def create_megamind(
    megamind_llm: BaseChatModel,
    executors: List[Executor],
    megamind_system_prompt: Optional[str] = None,
    task_planning_prompt: Optional[str] = None,
    context_providers: List[ContextProvider] = [],
) -> CompiledStateGraph:
    """Create a megamind langchain agent

    Args:
        executors (List[Executor]): Subagents for megamind, each can be a specialist with
        its own tools llm and system prompt

        megamind_system_prompt (Optional[str]): Prompt for megamind node. If not provided
        it will default to informing agent of the avaialble executors and listing their tools.

        task_planning_prompt (Optional[str]): Prompt that helps summarize the step in a way
        that helps planning task

        context_providers (List[ContextProvider]): Each ContextProvider can inject external info
        to prompt during planning phase


    """
    executor_agents = {}
    handoff_tools = []
    for executor in executors:
        executor_agents[executor.name] = create_react_structured_agent(
            llm=executor.llm,
            tools=executor.tools,
            system_prompt=executor.system_prompt,
            planning_prompt=task_planning_prompt,
        )

        handoff_tools.append(
            create_handoff_tool(
                agent_name=executor.name,
                description=f"Assign task to {executor.name} agent.",
            )
        )
    if not megamind_system_prompt:
        # make a generic system prompt that list executors and their tools
        specialists_info = []
        for executor in executors:
            tool_names = [tool.name for tool in executor.tools]
            tool_list = ", ".join(tool_names)
            specialists_info.append(f"- {executor.name}: Available tools: {tool_list}")

        specialists_section = "\n".join(specialists_info)
        megamind_system_prompt = f"""You manage specialists to whom you will delegate tasks to complete objective.
Available specialists and their capabilities:
{specialists_section}

The single task should be delegated to only 1 agent and should be doable by only 1 agent."""

    megamind_agent = create_react_agent(
        megamind_llm,
        tools=handoff_tools,
        prompt=megamind_system_prompt,
        name="megamind",
    )

    graph = StateGraph(MegamindState).add_node(
        "megamind",
        partial(plan_step, megamind_agent, context_providers=context_providers),
    )
    for agent_name, agent in executor_agents.items():
        graph.add_node(agent_name, agent)
        graph.add_edge(agent_name, "megamind")

    graph.add_edge(START, "megamind")
    return graph.compile()

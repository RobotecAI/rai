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
from typing import Any, Dict, List, Optional, Tuple, Union

from langchain.chat_models.base import BaseChatModel
from langchain.prompts import ChatPromptTemplate
from langchain_core.messages import BaseMessage, SystemMessage
from langchain_core.tools import BaseTool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel, Field
from rai.agents.langchain.core import ReActAgentState
from rai.agents.langchain.core.react_agent import create_react_runnable
from rai.initialization import get_llm_model


class Plan(BaseModel):
    """A plan to help solve a user request."""

    steps: List[str]


class Response(BaseModel):
    """Response to user."""

    response: str


class Act(BaseModel):
    """Action to take."""

    action: Union[Response, Plan] = Field(
        description="Action to perform. If you want to respond to user, use Response. "
        "If you need to further use tools to get the answer, use Plan."
    )


class PlanExecuteState(ReActAgentState):
    """State for the plan and execute agent."""
    # TODO (jmatejcz) should original_task be replaced with 
    # passing first message? The message can contain images etc.
    original_task: str
    plan: List[str]
    past_steps: List[Tuple[str, str]]
    response: str


def should_end(state: PlanExecuteState) -> str:
    """Check if we should end or continue planning."""
    if state["response"]:
        return END
    else:
        return "agent"


def create_plan_execute_agent(
    tools: List[BaseTool],
    planner_llm: Optional[BaseChatModel] = None,
    executor_llm: Optional[BaseChatModel] = None,
    replanner_llm: Optional[BaseChatModel] = None,
    system_prompt: Optional[str] = None,
) -> CompiledStateGraph:
    """Create a plan and execute agent that can break down complex tasks into steps.

    Parameters
    ----------
    tools : List[BaseTool]
        List of tools the agent can use during execution
    llm : Optional[BaseChatModel], default=None
        Language model to use. If None, will use complex_model from config
    system_prompt : Optional[str | SystemMultimodalMessage], default=None
        System prompt to use (currently not used in this implementation)

    Returns
    -------
    CompiledStateGraph
        Compiled state graph for the plan and execute agent

    Raises
    ------
    ValueError
        If tools are not provided or invalid
    """
    if planner_llm is None:
        planner_llm = get_llm_model("complex_model", streaming=True)
    if executor_llm is None:
        executor_llm = get_llm_model("complex_model", streaming=True)
    if replanner_llm is None:
        replanner_llm = get_llm_model("complex_model", streaming=True)

    if not tools:
        raise ValueError("Tools must be provided for plan and execute agent")
    if system_prompt is None:
        system_prompt = ""

    planner_prompt = """For the given objective, come up with a simple step by step plan. \
This plan should involve individual tasks, that if executed correctly will yield the correct answer.
Do not add any superfluous steps. The result of the final step should be the final answer.
Make sure that each step has all the information needed - do not skip steps."""

    replanner_prompt = """For the given objective, come up with a simple step by step plan.
This plan should involve individual tasks, that if executed correctly will yield the correct answer.
Do not add any superfluous steps. The result of the final step should be the final answer.
Make sure that each step has all the information needed - do not skip steps.

Your objective was this:
{original_task}

Your original plan was this:
{plan}

You have currently done the following steps:
{past_steps}

Update your plan accordingly. If no more steps are needed and you can return to the user, then respond with that. Otherwise, fill out the plan. Only add steps to the plan that still NEED to be done. Do not return previously done steps as part of the plan."""

    agent_executor = create_react_runnable(
        llm=executor_llm, system_prompt=system_prompt, tools=tools
    )
    # the prompt will be filled with values when passed to invoke
    planner = planner_llm.with_structured_output(Plan)  # type: ignore
    replanner = replanner_llm.with_structured_output(Act)  # type: ignore

    def execute_step(state: PlanExecuteState):
        """Execute the current step of the plan."""
        # TODO (jmatejcz) should we pass whole plan or only single the to the executor?
        plan = state["plan"]
        plan_str = "\n".join(f"{i + 1}. {step}" for i, step in enumerate(plan))
        task = plan[0]
        task_formatted = f"""For the following plan:
{plan_str}\n\nYou are tasked with executing step: {task}."""
        if not state["plan"]:
            return {}

        agent_response = agent_executor.invoke(
            {"messages": [("user", task_formatted)]}, config={"recursion_limit": 20}
        )
        result = agent_response["messages"][-1].content

        new_past_steps = state["past_steps"] + [(task, result)]
        # remaining_plan = state["plan"][1:]  # Remove the executed step

        return {
            "past_steps": new_past_steps,
            # "plan": remaining_plan,
        }

    def plan_step(state: PlanExecuteState):
        """Initial planning step."""
        messages = [
            SystemMessage(content=system_prompt + "\n" + planner_prompt),
            HumanMessage(content=state["original_task"]),
        ]
        plan = planner.invoke({"messages": [("user", state["original_task"])]})
        return {"plan": plan.steps}

    def replan_step(state: PlanExecuteState):
        """Replan based on execution results."""
        # Format past steps for the prompt
        past_steps_str = "\n".join(
            [
                f"{i + 1}. {step}: {result}"
                for i, (step, result) in enumerate(state["past_steps"])
            ]
        )

        # Format remaining plan
        plan_str = "\n".join(
            [f"{i + 1}. {step}" for i, step in enumerate(state["plan"])]
        )

        output = replanner.invoke(
            {
                "original_task": state["original_task"],
                "plan": plan_str,
                "past_steps": past_steps_str,
            }
        )

        if isinstance(output.action, Response):
            return {"response": output.action.response}
        else:
            return {"plan": output.action.steps}

    workflow = StateGraph(PlanExecuteState)

    workflow.add_node("planner", plan_step)
    workflow.add_node("agent", execute_step)
    workflow.add_node("replan", replan_step)

    workflow.add_edge(START, "planner")
    # From plan we go to agent
    workflow.add_edge("planner", "agent")
    # From agent, we replan
    workflow.add_edge("agent", "replan")

    workflow.add_conditional_edges(
        "replan",
        should_end,
        ["agent", END],
    )

    return workflow.compile()


def create_initial_plan_execute_state(
    original_task: str,
    messages: Optional[List[BaseMessage]] = None,
) -> PlanExecuteState:
    """Create initial state for the plan and execute agent.

    Parameters
    ----------
    input_text : str
        The user's input/objective to accomplish
    messages : Optional[List[BaseMessage]], default=None
        Initial messages for the conversation

    Returns
    -------
    PlanExecuteState
        Initial state for the agent
    """
    if messages is None:
        messages = []

    return PlanExecuteState(
        messages=messages,
        original_task=original_task,
        plan=[],
        past_steps=[],
        response="",
    )


def run_plan_execute_agent(
    agent: CompiledStateGraph,
    original_task: str,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Run the plan and execute agent on a given input.

    Parameters
    ----------
    agent : CompiledStateGraph
        The compiled plan and execute agent
    input_text : str
        The user's input/objective
    config : Optional[Dict[str, Any]], default=None
        Configuration for the agent execution

    Returns
    -------
    Dict[str, Any]
        Final state after execution
    """
    initial_state = create_initial_plan_execute_state(original_task)

    # Execute the agent
    result = agent.invoke(initial_state, config=config)

    return result

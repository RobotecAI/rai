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


# Inspierd by https://github.com/langchain-ai/langgraph/blob/main/docs/docs/tutorials/plan-and-execute/plan-and-execute.ipynb


import logging
import operator
from re import M
from typing import Annotated, List, Tuple, TypedDict, Union
from functools import partial
    
from langgraph.store.memory import InMemoryStore

from langchain.chat_models.base import BaseChatModel
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.store.base import BaseStore
from pydantic import BaseModel, Field
from pydantic.v1.utils import get_discriminator_alias_and_values
from typing_extensions import TypedDict, cast

from rai.agents.langchain.core.react_agent import create_react_runnable
from rai.tools.ros2.simple import GetROS2ImageConfiguredTool


class PlanExecute(TypedDict):
    input: str
    plan: List[str]

    past_steps: Annotated[List[Tuple], operator.add]
    critic_response: str
    should_continue: str
    response: str


class Plan(BaseModel):
    """Plan to follow in future"""

    steps: List[str] = Field(
        description="different steps to follow, should be in sorted order"
    )


class Response(BaseModel):
    """Response to user."""

    response: str


class Act(BaseModel):
    """Action to perform."""

    action: Union[Response, Plan] = Field(
        description="Action to perform. If you want to respond to user, use Response. "
        "If you need to further use tools to get the answer, use Plan."
    )


class Criticism(BaseModel):
    response: str = Field(
        description="Judgment about the current status of the plan with description of possible problems"
    )
    should_continue: bool = Field(description="Is it feasible to continue the plan")

critic_prompt = ChatPromptTemplate.from_template(
    "Describe the input image and based on the plan {plan} judge if the plan is feasible to continue. Respond in max 3 sentences, max 200 chars."
)

def should_end(state: PlanExecute):
    from pprint import pformat

    logging.info(f"[DeCider]: {pformat(state)}")
    if "response" in state and state["response"]:
        return END
    else:
        return "agent"

async def critic(critic_llm, state: PlanExecute, *, store: BaseStore):
    multimodal_artifact = store.get(
            namespace=("images",),
            key="before_plan",
    )
    logging.info(f'[Critic] multimodal_artifact received')
    plan = state["plan"]

    input = {'plan': plan}
    critic_llm = critic_prompt | critic_llm

    output = await critic_llm.ainvoke(input)
    output = output["messages"][-1]
    logging.info(f"[Critic] responded with: {output}")
    return {
        "critic_response": output.response,
        "should_continue": output.should_continue,
    }

def create_plan_and_execute_runnable(
    llm: BaseChatModel,
    tools: List[BaseTool],
    system_prompt: str,
    camera_tool: GetROS2ImageConfiguredTool,
    debug=False,
) -> CompiledStateGraph:
    planner_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                system_prompt
                + """
    rule: Before making the plan check task status using the camera image.

    For the given objective, come up with a simple step by step plan. Every step HAS TO be connected with a tool call. \
    This plan should involve individual tasks, that if executed correctly will yield the correct answer. Do not add any superfluous steps. \
    The result of the final step should be the final answer. Make sure that each step has all the information needed - do not skip steps.""",
            ),
            ("placeholder", "{messages}"),
        ]
    )
    replanner_prompt = ChatPromptTemplate.from_template(
        system_prompt
        + """
    rule: Before making the plan check task status using the camera image.

    For the given objective, come up with a simple step by step plan. Every step HAS TO be connected with a tool call. \
    This plan should involve individual tasks, that if executed correctly will yield the correct answer. Do not add any superfluous or verification steps. \
    The result of the final step should be the final answer. Make sure that each step has all the information needed - do not skip steps.

    Your objective was this:
    {input}

    Your original plan was this:
    {plan}

    You have currently done the follow steps:
    {past_steps}

    You analysed current camera image and concluded that:
    {critic_response}

    After analysis of the image you judged the feasibility of the task (True=feasible, False=not feasible):
    {should_continue}
    If it's False, then the task might need to be replaned or suspended.

    Update your plan accordingly. If no more steps are needed and you can return to the user, then respond with that. Otherwise, fill out the plan. Only add steps to the plan that still NEED to be done. Do not return previously done steps as part of the plan."""
    )


    thinking = ChatOpenAI(model="gpt-4o")
    planner = planner_prompt | thinking.with_structured_output(Plan)
    agent_executor = create_react_runnable(
        llm=llm, tools=tools, system_prompt=system_prompt + "\nLeft / right is on y axis."
    )
    critic_llm = create_react_runnable(
        llm=llm, tools=[camera_tool], system_prompt=None, output_template=Criticism
    )
    replanner = replanner_prompt | thinking.with_structured_output(Act)

    async def execute_step(state: PlanExecute):
        plan = state["plan"]
        plan.append("Last step: Reset arm to the initial position.")
        plan_str = "\n".join(f"{i + 1}. {step}" for i, step in enumerate(plan))

        task = plan[0]
        task_formatted = f"Please execute the following plan: {plan_str}"

        agent_response = await agent_executor.ainvoke(
            {"messages": [("user", task_formatted)]}, config={"recursion_limit": 100}
        )
        logging.info(f"[ReAct] responded with: {agent_response['messages'][-1].content}")
        return {
            "past_steps": [(task, agent_response["messages"][-1].content)],
        }

    async def plan_step(state: PlanExecute, *, store: BaseStore):
        _, multimodal_artifact = camera_tool._run()
        logging.info('[Planner] putting image into the store')
        store.put(
                namespace=("images",),
                key="before_plan",
                value=multimodal_artifact
        )
        plan = await planner.ainvoke({"messages": [("user", state["input"])]})
        plan = cast(Plan, plan)
        return {"plan": plan.steps}

    async def replan_step(state: PlanExecute):
        _, multimodal_artifact = camera_tool._run()
        logging.info('[RePlan] putting image into the store')
        store.put(
                namespace=("images",),
                key="before_plan",
                value=multimodal_artifact
        )
        if len(state["past_steps"]) > 10:
            logging.info("Trimming past steps")
            state["past_steps"] = state["past_steps"][-9:]
        output = await replanner.ainvoke(state)
        logging.info(f"[RePlan] responded with: {output}")
        if isinstance(output.action, Response):
            logging.info("Returned Response")
            return {"response": output.action.response}
        else:
            return {"plan": output.action.steps}

    workflow = StateGraph(PlanExecute)

    # Add the plan node
    workflow.add_node("planner", plan_step)

    # Add the execution step
    workflow.add_node("agent", execute_step)

    workflow.add_node("critic", partial(critic, critic_llm))
    # Add a replan node
    workflow.add_node("replan", replan_step)

    workflow.add_edge(START, "planner")

    # From plan we go to agent
    workflow.add_edge("planner", "agent")

    # From agent, we replan
    workflow.add_edge("agent", "critic")

    workflow.add_edge("critic", "replan")

    workflow.add_conditional_edges(
        "replan",
        # Next, we pass in the function that will determine which node is called next.
        should_end,
        ["agent", END],
    )

    # Finally, we compile it!
    # This compiles it into a LangChain Runnable,
    # meaning you can use it as you would any other runnable
    store = InMemoryStore()

    app = workflow.compile(debug=debug, store=store)
    return app

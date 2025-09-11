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
import json
import logging
import stat
import time
from enum import Enum
from functools import partial
from typing import (
    Annotated,
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Union,
    cast,
    Tuple,
)

from altair import Step
from langchain.chat_models.base import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolCall,
    ToolMessage,
)
from langchain_core.runnables import RunnableConfig
from langchain_core.runnables.config import get_executor_for_config
from langchain_core.tools import BaseTool, InjectedToolCallId, tool
from langchain_core.tools import tool as create_tool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import create_react_agent
from langgraph.prebuilt.tool_node import msg_content_output
from langgraph.types import Command
from langgraph.utils.runnable import RunnableCallable
from pydantic import BaseModel, Field, ValidationError

from typing_extensions import TypedDict
from rai.agents.langchain.core import ReActAgentState
from rai.messages import (
    HumanMultimodalMessage,
    MultimodalArtifact,
    ToolMultimodalMessage,
    store_artifacts,
)


class AgentType(str, Enum):
    """Types of specialized agents."""

    MANIPULATION = "manipulation"
    NAVIGATION = "navigation"
    COMPLETE = "complete"


class StepSuccess(BaseModel):
    """Output of success attacher"""

    success: bool = Field(description="Whether the task was completed successfully")
    explanation: str = Field(description="Explanation of what happened")


class PlanStep(BaseModel):
    """Output of megamind"""

    task: str = Field(description="Description of the task to be completed")


class State(ReActAgentState):
    original_task: str
    steps_done: List[str]
    step: Optional[str]
    step_success: StepSuccess
    step_messages: List[BaseMessage]


class ToolRunner(RunnableCallable):
    def __init__(
        self,
        tools: Sequence[Union[BaseTool, Callable]],
        *,
        name: str = "tools",
        tags: Optional[list[str]] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        super().__init__(self._func, name=name, tags=tags, trace=False)
        self.logger = logger or logging.getLogger(__name__)
        self.tools_by_name: Dict[str, BaseTool] = {}
        for tool_ in tools:
            if not isinstance(tool_, BaseTool):
                tool_ = create_tool(tool_)
            self.tools_by_name[tool_.name] = tool_

    def _func(self, input: dict[str, Any], config: RunnableConfig) -> Any:
        config["max_concurrency"] = (
            1  # TODO(maciejmajek): use better mechanism for task queueing
        )
        if messages := input.get("step_messages", []):
            message = messages[-1]
        else:
            raise ValueError("No message found in input")

        if not isinstance(message, AIMessage):
            raise ValueError("Last message is not an AIMessage")

        def run_one(call: ToolCall):
            self.logger.info(f"Running tool: {call['name']}, args: {call['args']}")
            artifact = None

            try:
                ts = time.perf_counter()
                output = self.tools_by_name[call["name"]].invoke(call, config)  # type: ignore
                te = time.perf_counter() - ts
                self.logger.info(
                    f"Tool {call['name']} completed in {te:.2f} seconds. Tool output: {str(output.content)[:100]}{'...' if len(str(output.content)) > 100 else ''}"
                )
                self.logger.debug(
                    f"Tool {call['name']} output: \n\n{str(output.content)}"
                )
            except ValidationError as e:
                errors = e.errors()
                for error in errors:
                    error.pop(
                        "url"
                    )  # get rid of the  https://errors.pydantic.dev/... url

                error_message = f"""
                                    Validation error in tool {call["name"]}:
                                    {e.title}
                                    Number of errors: {e.error_count()}
                                    Errors:
                                    {json.dumps(errors, indent=2)}
                                """
                self.logger.info(error_message)
                output = ToolMessage(
                    content=error_message,
                    name=call["name"],
                    tool_call_id=call["id"],
                    status="error",
                )
            except Exception as e:
                self.logger.info(f'Error in "{call["name"]}", error: {e}')
                output = ToolMessage(
                    content=f"Failed to run tool. Error: {e}",
                    name=call["name"],
                    tool_call_id=call["id"],
                    status="error",
                )

            if output.artifact is not None:
                artifact = output.artifact
                if not isinstance(artifact, dict):
                    raise ValueError(
                        "Artifact must be a dictionary with optional keys: 'images', 'audios'"
                    )

                artifact = cast(MultimodalArtifact, artifact)
                store_artifacts(output.tool_call_id, [artifact])

            if artifact is not None and (
                len(artifact.get("images", [])) > 0
                or len(artifact.get("audios", [])) > 0
            ):  # multimodal case, we currently support images and audios artifacts
                return ToolMultimodalMessage(
                    content=msg_content_output(output.content),
                    name=call["name"],
                    tool_call_id=call["id"],
                    images=artifact.get("images", []),
                    audios=artifact.get("audios", []),
                )

            return output

        with get_executor_for_config(config) as executor:
            raw_outputs = [*executor.map(run_one, message.tool_calls)]
            outputs: List[Any] = []
            for raw_output in raw_outputs:
                if isinstance(raw_output, ToolMultimodalMessage):
                    outputs.extend(
                        raw_output.postprocess()
                    )  # openai please allow tool messages with images!
                else:
                    outputs.append(raw_output)

            # because we can't answer an aiMessage with an alternating sequence of tool and human messages
            # we sort the messages by type so that the tool messages are sent first
            # for more information see implementation of ToolMultimodalMessage.postprocess
            outputs.sort(key=lambda x: x.__class__.__name__, reverse=True)
            input["messages"].extend(outputs)
            input["step_messages"].extend(outputs)
            return input


def llm_node(
    llm: BaseChatModel,
    system_prompt: Optional[str],
    state: State,
) -> State:
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


def structured_output_node(
    llm: BaseChatModel,
    state: State,
) -> State:
    """Analyze the conversation and return structured output."""
    final_response = (
        state["step_messages"][-1].content if state["step_messages"] else ""
    )

    # Analyze with structured output
    analyzer = llm.with_structured_output(StepSuccess)
    analysis = analyzer.invoke(
        [
            SystemMessage(
                content=f"""
Analyze if this task was completed successfully:

Task: {state["step"]}


Determine success and provide brief explanation of what happened. 
Below you have messages of agent doing the task:"""
            ),
            *state["step_messages"],
        ]
    )
    state["step_success"] = StepSuccess(
        success=analysis.success, explanation=analysis.explanation
    )
    return state


def should_continue_or_structure(state: State) -> str:
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

    graph = StateGraph(State)
    graph.add_edge(START, "llm")

    if tools:
        tool_runner = ToolRunner(tools)
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


def create_megamind(
    manipulation_tools: List[BaseTool],
    navigation_tools: List[BaseTool],
    megamind_llm: BaseChatModel,
    executor_llm: BaseChatModel,
    system_prompt: str,
) -> CompiledStateGraph:

    if not manipulation_tools and not navigation_tools:
        raise ValueError("At least one set of tools must be provided")

    def create_handoff_tool(*, agent_name: str, description: str | None = None):
        name = f"transfer_to_{agent_name}"
        description = description or f"Ask {agent_name} for help."

        @tool(name, description=description)
        def handoff_tool(
            task_instruction: str,  # The specific task for the agent
            state: State,
            tool_call_id: Annotated[str, InjectedToolCallId],
        ) -> Command:
            state["step"] = task_instruction
            state["step_messages"] = []
            return Command(
                goto=agent_name,
                # Send only the task message to the specialist agent, not the full history
                update=state,
                graph=Command.PARENT,
            )

        return handoff_tool

    manipulation_system_prompt = """You are a manipulation specialist robot agent.
Your role is to handle object manipulation tasks including picking up and droping objects using provided tools.

Ask the VLM for objects detection and positions before perfomring any manipulation action."""

    navigation_system_prompt = """You are a navigation specialist robot agent.
Your role is to handle navigation tasks in space using provided tools.

After performing navigation action, always check your current position to ensure success"""

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

    megamind_system_prompt = f"""You manage specialists to whom you will delegate tasks:
- Manipulaiton specialist can ask VLM about the nearby objects and their coordinates, pick up and drop objects.
- Navigation specialist can navigate to certain coordinates and determine the location of robot in the environment.
Always include coordinates to navigate to.

The single task should be delegated to only 1 agent and should be doable by only 1 agent.

Examples of a WELL formulated task:
- Navigate to (5.0, 4.0).
----------------------------------
- Check for objects near you.
----------------------------------
- Pick up green object.
----------------------------------
- Drop an object to the box.

Examples of WRONGLY formulated task:
- Pick up object and navigate to box.
"""
    system_prompt += "\n"
    system_prompt += megamind_system_prompt

    megamind_agent = create_react_agent(
        megamind_llm,
        tools=[assign_to_manipulation_agent, assign_to_nav_agent],
        prompt=system_prompt,
        name="megamind",
    )

    def plan_step(state: State) -> State:
        """Initial planning step."""
        if "original_task" not in state:
            state["original_task"] = state["messages"][0].content[0]["text"]
        if "steps_done" not in state:
            state["steps_done"] = []
        if "step" not in state:
            state["step"] = None

        megamind_prompt = (
            f'You are given objective to complete: {state["original_task"]}'
        )
        if state["steps_done"]:
            megamind_prompt += "\n"
            megamind_prompt += "Steps that were already done:\n"
            steps_done = "\n".join(state["steps_done"])
            megamind_prompt += steps_done
            megamind_prompt += "\n"

        if state["step"]:
            if not state["step_success"]:
                raise ValueError("Step success should be specified at this point")

            megamind_prompt += (
                f"\nLatest step delegated to specialist was:\n {state['step']}\n"
            )
            success_str = "success" if state["step_success"].success else "failure"
            megamind_prompt += f"\nIt resulted in {success_str} because: {state['step_success'].explanation}"
            megamind_prompt += "\nBased on that outcome come up with the next step and delegate it to selected agent."

            # only successful steps are appended to steps_done
            if state["step_success"].success:
                state["steps_done"].append(state["step"])
        else:
            megamind_prompt += "\n"
            megamind_prompt += (
                "Come up with the fist step and delegate it to selected agent."
            )

        messages = [
            HumanMultimodalMessage(content=megamind_prompt),
        ]

        response = megamind_agent.invoke({"messages": messages})

        state["messages"].append(response)
        return state

    megamind = (
        StateGraph(State)
        .add_node("megamind", plan_step)
        .add_node("navigation", navigation_agent)
        .add_node("manipulation", manipulation_agent)
        .add_edge(START, "megamind")
        # always return back to the supervisor
        .add_edge("navigation", "megamind")
        .add_edge("manipulation", "megamind")
        .compile()
    )

    return megamind

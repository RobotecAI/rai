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

from unittest.mock import MagicMock

import pytest
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.tools import tool
from rai.agents.langchain.core.megamind import (
    ContextProvider,
    Executor,
    MegamindState,
    PlanPrompts,
    StepSuccess,
    analyzer_node,
    create_megamind,
    create_react_structured_agent,
    get_initial_megamind_state,
    llm_node,
    plan_step,
    should_continue_or_structure,
)
from rai.messages import HumanMultimodalMessage


@pytest.fixture
def mock_llm():
    llm = MagicMock()
    llm.bind_tools.return_value = llm
    return llm


@pytest.fixture
def mock_tool():
    @tool
    def sample_tool(query: str):
        """Smple tool"""
        return f"Tool called with {query}"

    return sample_tool


@pytest.fixture(scope="function")
def default_state():
    state: MegamindState = {
        "original_task": "task",
        "steps_done": [],
        "step": "current step",
        "step_success": StepSuccess(success=False, explanation=""),
        "step_messages": [HumanMessage(content="hello")],
        "messages": [],
    }
    return state


class MockProvider(ContextProvider):
    def get_context(self) -> str:
        return "Extra context"


def test_llm_node(mock_llm, default_state):
    mock_response = AIMessage(content="response")
    mock_llm.invoke.return_value = mock_response

    new_state = llm_node(mock_llm, "system prompt", default_state)

    assert len(new_state["step_messages"]) == 2
    assert new_state["step_messages"][-1] == mock_response

    # Check if invoke was called with correct messages (including system prompt)
    args, _ = mock_llm.invoke.call_args
    messages = args[0]
    assert isinstance(messages[0], SystemMessage)
    assert messages[0].content == "system prompt"
    assert isinstance(messages[1], HumanMessage)
    assert messages[1].content == "current step"


def test_analyzer_node(mock_llm, default_state):
    mock_analysis = StepSuccess(success=True, explanation="Great success")

    # Mock with_structured_output to return a mock analyzer
    mock_analyzer = MagicMock()
    mock_analyzer.invoke.return_value = mock_analysis
    mock_llm.with_structured_output.return_value = mock_analyzer

    new_state = analyzer_node(mock_llm, "plan prompt", default_state)

    assert new_state["step_success"] == mock_analysis
    assert len(new_state["steps_done"]) == 1
    assert "Great success" in new_state["steps_done"][0]


def test_should_continue_or_structure(default_state):
    # no tool calls
    assert should_continue_or_structure(default_state) == "structured_output"

    # with tool calls
    AIMessage(content="text", tool_calls=[{"name": "tool", "args": {}, "id": "1"}])
    default_state["step_messages"] = [
        AIMessage(content="text", tool_calls=[{"name": "tool", "args": {}, "id": "1"}])
    ]
    assert should_continue_or_structure(default_state) == "tools"


# def test_create_handoff_tool():
#     handoff = create_handoff_tool("agent_x", "desc")
#     assert handoff.name == "transfer_to_agent_x"
#     assert handoff.description == "desc"

#     # Verify the tool logic (returns a Command)
#     result = handoff.invoke({"task_instruction": "do subtask"})
#     assert isinstance(result, Command)
#     assert result.goto == "agent_x"
#     assert result.update["step"] == "do subtask"
#     assert result.graph == Command.PARENT


def test_plan_step(mock_llm, default_state):
    default_state.update(
        {
            "messages": [HumanMultimodalMessage(content="task")],
            "steps_done": ["step 1 done"],
            "step": "step 1",
            "step_success": StepSuccess(success=True, explanation="ok"),
            "step_messages": [],
        }
    )

    mock_llm.invoke.return_value = AIMessage(content="ignored")

    plan_step(mock_llm, default_state)

    # Check if invoke was called with prompt containing history
    call_args = mock_llm.invoke.call_args
    messages = call_args[0][0]["messages"]
    prompt_text = messages[0].content
    assert "StepSuccess" not in str(type(prompt_text))
    assert mock_llm.invoke.called


def test_plan_prompts():
    defaults = PlanPrompts.default()
    assert "objective" in defaults.objective_template

    custom = PlanPrompts.custom(objective_template="Custom {original_task}")
    assert custom.objective_template == "Custom {original_task}"
    assert custom.steps_done_header == defaults.steps_done_header


def test_plan_step_edge_cases(mock_llm, default_state):
    default_state["messages"] = [HumanMultimodalMessage(content="task")]
    default_state["step"] = None  # To trigger "first_step_prompt" path
    mock_llm.invoke.return_value = AIMessage(content="ok")

    plan_step(
        mock_llm,
        default_state,
        prompts=None,
        context_providers=[MockProvider()],
    )

    args, _ = mock_llm.invoke.call_args
    prompt_content = args[0]["messages"][0].content[0]["text"]
    assert "Extra context" in prompt_content

    # step defined but success missing
    default_state["step"] = "some step"
    default_state["step_success"] = None
    with pytest.raises(ValueError, match="Step success should be specified"):
        plan_step(mock_llm, default_state)

    # missing keys initialization
    empty_state: MegamindState = {"messages": [HumanMultimodalMessage(content="Start")]}
    plan_step(mock_llm, empty_state)
    assert empty_state["original_task"] == "Start"
    assert empty_state["steps_done"] == []
    assert empty_state["step"] is None


def test_get_initial_megamind_state():
    state = get_initial_megamind_state("My Task")
    assert state["original_task"] == "My Task"
    assert len(state["messages"]) == 1
    assert state["messages"][0].content == [{"text": "My Task", "type": "text"}]
    assert state["steps_done"] == []


def test_create_react_structured_agent_no_tools(mock_llm):
    # Test creation without tools
    graph = create_react_structured_agent(mock_llm, tools=[])
    assert graph is not None


def test_llm_node_error(mock_llm, default_state):
    # Test missing step raises ValueError
    default_state["step"] = None
    with pytest.raises(ValueError, match="Step should be defined"):
        llm_node(mock_llm, "sys", default_state)


def test_create_megamind(mock_llm, mock_tool):
    executor = Executor(
        name="specialist", llm=mock_llm, tools=[mock_tool], system_prompt="sys prompt"
    )

    graph = create_megamind(
        megamind_llm=mock_llm,
        executors=[executor],
        megamind_system_prompt="mega prompt",
    )

    assert graph is not None

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

"""The Streamlit chat: event folding without a browser, and one AppTest smoke run."""

import asyncio
import textwrap

import pytest
from langchain_core.language_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage
from rai.agents import ReActAgent
from rai.agents.events import AgentError, ModelStart, ToolStart
from rai.tools import Allow, Deny, Effect, ToolCallContext, global_tool_hooks, rai_tool
from rai.ui.streamlit import ToolCall, Turn


class FakeModel(FakeMessagesListChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


@pytest.fixture(autouse=True)
def _clean_global_hooks():
    global_tool_hooks.clear()
    yield
    global_tool_hooks.clear()


@rai_tool(Effect.READ)
async def camera() -> tuple[str, dict]:
    """Take a picture."""
    return "picture taken", {"images": ["aGVsbG8="]}


camera.response_format = "content_and_artifact"


@rai_tool(Effect.ACTUATE)
async def drive(speed: float) -> str:
    """Drive the base."""
    return f"driving {speed}"


@rai_tool(Effect.READ)
async def broken() -> str:
    """Always fails."""
    raise RuntimeError("sensor offline")


def test_turn_shows_text_tools_images_and_denials_from_a_real_run():
    async def deny_actuate(call: ToolCallContext) -> Allow | Deny:
        return Deny("e-stop") if call.effect is Effect.ACTUATE else Allow()

    global_tool_hooks.add(deny_actuate)
    calls = [
        {"name": n, "args": a, "id": n}
        for n, a in [("camera", {}), ("drive", {"speed": 1.0}), ("broken", {})]
    ]
    agent = ReActAgent(
        FakeModel(
            responses=[
                AIMessage("Let me look.", tool_calls=calls),
                AIMessage("I see a robot."),
            ]
        ),
        [camera, drive, broken],
    )
    turn = Turn("look")
    agent.subscribe(turn.apply)
    asyncio.run(agent.ainvoke("look"))

    text1, *calls, text2 = turn.parts  # tools run concurrently, in any order
    assert (text1, text2) == ("Let me look.", "I see a robot.")  # final text once
    tools = {c.name: c for c in calls if isinstance(c, ToolCall)}
    cam, drv, brk = tools["camera"], tools["drive"], tools["broken"]
    assert (cam.status, cam.images, cam.content) == (
        "success",
        [b"hello"],
        "picture taken",
    )
    assert (cam.state, cam.expanded) == ("complete", True)
    assert cam.label == "**camera** :gray[read]"
    assert drv.status == "denied" and drv.label.startswith(
        "🚫 **drive** :gray[actuate]"
    )
    assert ":orange[denied]" in drv.label and (drv.state, drv.expanded) == (
        "error",
        True,
    )
    assert drv.args == {"speed": 1.0}
    assert brk.status == "error" and ":red[failed]" in brk.label
    assert "sensor offline" in brk.content
    assert turn.done and not turn.thinking and turn.error is None


def test_error_ends_the_turn_and_cancels_running_tools():
    turn = Turn("go")
    assert turn.apply(ModelStart(agent="a", turn="t")) and turn.thinking
    turn.apply(
        ToolStart(
            agent="a", turn="t", call_id="1", name="arm", args={}, effect=Effect.ACTUATE
        )
    )
    turn.apply(AgentError(agent="a", turn="t", error=RuntimeError("boom")))
    (call,) = turn.parts
    assert isinstance(call, ToolCall)
    assert call.status == "cancelled" and ":gray[cancelled]" in call.label
    assert turn.error == "RuntimeError: boom" and turn.done and not turn.thinking


APP = textwrap.dedent(
    """
    import streamlit as st
    from langchain_core.language_models import FakeMessagesListChatModel
    from langchain_core.messages import AIMessage
    from rai.agents import ReActAgent
    from rai.tools import Effect, rai_tool
    from rai.ui.streamlit import chat

    class FakeModel(FakeMessagesListChatModel):
        def bind_tools(self, tools, **kwargs):
            return self

    @rai_tool(Effect.READ)
    async def get_status(topic: str) -> str:
        '''Status of a topic.'''
        return f"ok {topic}"

    @st.cache_resource
    def make_agent():
        call = {"name": "get_status", "args": {"topic": "/a"}, "id": "1"}
        responses = [AIMessage("", tool_calls=[call]), AIMessage("All good.")]
        return ReActAgent(FakeModel(responses=responses), [get_status])

    chat(make_agent(), title="Robot", greeting="Hi!")
    """
)


def test_chat_app_runs_a_turn_and_keeps_it_across_reruns():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_string(APP).run()
    assert not at.exception
    assert [m.markdown[0].value for m in at.chat_message] == ["Hi!"]

    at.chat_input[0].set_value("status?").run()
    assert not at.exception
    assert [m.name for m in at.chat_message] == ["assistant", "user", "assistant"]
    assert at.chat_message[1].markdown[0].value == "status?"
    reply = at.chat_message[2]
    assert reply.status[0].label == "**get_status** :gray[read]"
    assert reply.markdown[-1].value == "All good."

    at.run()  # a rerun shows the stored turn
    assert at.chat_message[2].markdown[-1].value == "All good."

    at.sidebar.button[0].click().run()  # "Clear conversation"
    assert len(at.chat_message) == 1

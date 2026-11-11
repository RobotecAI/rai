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

"""rai.mcp server and client in one process, over the SDK's in-memory transport (no ROS)."""

import asyncio
import warnings
from pathlib import Path

import pytest

pytest.importorskip("mcp")
pytest.importorskip("langchain_mcp_adapters")

from langchain_core.messages import ToolMessage
from langchain_core.tools import StructuredTool
from langchain_core.tools import tool as lc_tool
from langchain_mcp_adapters.tools import load_mcp_tools
from mcp.shared.memory import create_connected_server_and_client_session
from rai.agents import ReActAgent
from rai.cli import main
from rai.mcp import build_server, load_tools
from rai.mcp.server import load_setup, mcp_tool
from rai.mcp.settings import MCPSettings
from rai.tools import Allow, Deny, Effect, from_langchain, global_tool_hooks, rai_tool

from tests.rai_mcp.scripted import ScriptedChatModel, deny_motion

PNG = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGNgYGBgAAAABQABpfZFQAAAAABJRU5ErkJggg=="


@rai_tool(Effect.READ, targets=["/battery"])
async def get_status(topic: str) -> str:
    """Return the latest status on a topic."""
    return f"ok {topic}"


@rai_tool(Effect.WRITE)
async def set_mode(mode: str) -> str:
    """Set the robot mode."""
    return f"mode {mode}"


@rai_tool(Effect.ACTUATE, targets=["/cmd_vel"])
async def drive(speed: float) -> str:
    """Drive the base."""
    return f"driving {speed}"


@rai_tool(Effect.READ)
async def broken() -> str:
    """Always fails."""
    raise RuntimeError("sensor offline")


@lc_tool(response_format="content_and_artifact")
async def snapshot() -> tuple[str, dict]:
    """A plain LangChain tool returning an image the RAI way."""
    return "Image received successfully", {"images": [PNG]}


TOOLS = [get_status, set_mode, drive, broken, snapshot]


@pytest.fixture(autouse=True)
def _clean_global_hooks():
    global_tool_hooks.clear()
    yield
    global_tool_hooks.clear()


def session_run(fn):
    async def run():
        server = build_server(TOOLS, name="test")
        async with create_connected_server_and_client_session(server) as session:
            return await fn(session)

    return asyncio.run(run())


def test_annotations_follow_effects():
    async def listed(session):
        return {t.name: t for t in (await session.list_tools()).tools}

    tools = session_run(listed)
    hints = {
        name: (t.annotations.readOnlyHint, t.annotations.destructiveHint)
        for name, t in tools.items()
    }
    assert hints == {
        "get_status": (True, False),
        "set_mode": (False, False),
        "drive": (False, True),
        "broken": (True, False),
        "snapshot": (False, True),  # not a RaiTool: ACTUATE
    }
    assert tools["drive"].meta == {
        "rai": {"effect": "actuate", "targets": ["/cmd_vel"]}
    }
    assert tools["get_status"].inputSchema["required"] == ["topic"]


def test_pydantic_v1_args_schema():
    """LangChain still accepts pydantic.v1 args schemas; their JSON schema comes from .schema()."""
    with warnings.catch_warnings():  # pydantic.v1 warns on Python >= 3.14
        warnings.simplefilter("ignore")
        from pydantic.v1 import BaseModel as BaseModelV1

        class Args(BaseModelV1):
            x: int

    def double(x: int) -> int:
        return 2 * x

    legacy = StructuredTool.from_function(
        func=double, name="double", description="Double x.", args_schema=Args
    )
    schema = mcp_tool(legacy).parameters
    assert schema["properties"] == {"x": {"title": "X", "type": "integer"}}
    assert schema["required"] == ["x"]


def test_results_and_errors():
    async def calls(session):
        return (
            await session.call_tool("get_status", {"topic": "/battery"}),
            await session.call_tool("broken", {}),
            await session.call_tool("drive", {}),  # missing argument
            await session.call_tool("drive", {"speed": "fast"}),
        )

    ok, broken_, missing, invalid = session_run(calls)
    assert not ok.isError and ok.content[0].text == "ok /battery"
    assert broken_.isError and "sensor offline" in broken_.content[0].text
    assert missing.isError  # RaiTool reports "Tool input validation error"
    assert invalid.isError


def test_tool_hooks_run_on_the_server_path():
    seen = []

    async def no_actuation(call):
        seen.append((call.name, call.effect, call.targets, dict(call.args)))
        return Deny("no actuation") if call.effect is Effect.ACTUATE else Allow()

    global_tool_hooks.add(no_actuation)

    async def calls(session):
        return (
            await session.call_tool("drive", {"speed": 1.0}),
            await session.call_tool("set_mode", {"mode": "auto"}),
        )

    denied, allowed = session_run(calls)
    assert denied.isError and "no actuation" in denied.content[0].text
    assert not allowed.isError
    assert seen[0] == ("drive", Effect.ACTUATE, ("/cmd_vel",), {"speed": 1.0})


def test_image_artifacts_become_image_content():
    result = session_run(lambda s: s.call_tool("snapshot", {}))
    text, image = result.content
    assert text.text == "Image received successfully"
    assert (image.type, image.mimeType, image.data) == ("image", "image/png", PNG)


def test_client_tools_get_effects_from_annotations_and_run_in_an_agent():
    async def run(session):
        tools = [from_langchain(t) for t in await load_mcp_tools(session)]
        by_name = {t.name: t for t in tools}
        agent = ReActAgent(
            ScriptedChatModel(
                steps=[
                    [("get_status", {"topic": "/battery"}), ("drive", {"speed": 1.0})],
                    "battery ok",
                ]
            ),
            tools,
        )
        global_tool_hooks.add(deny_actuate)
        answer = await agent.ainvoke("check")
        return by_name, answer, agent.history

    async def deny_actuate(call):
        return Deny("client says no") if call.effect is Effect.ACTUATE else Allow()

    tools, answer, history = session_run(run)
    assert {n: t.effect for n, t in tools.items()} == {
        "get_status": Effect.READ,
        "set_mode": Effect.WRITE,
        "drive": Effect.ACTUATE,
        "broken": Effect.READ,
        "snapshot": Effect.ACTUATE,
    }
    assert answer.text == "battery ok"
    status, drove = [m for m in history if isinstance(m, ToolMessage)]
    assert status.status == "success" and "ok /battery" in str(status.content)
    # denied on the client by the tool hook: the server never saw the call
    assert drove.status == "error" and "client says no" in str(drove.content)


def test_effects_override_server_hints():
    async def run(session):
        from rai.mcp.client import _wrap

        return _wrap(await load_mcp_tools(session), {"get_status": Effect.ACTUATE})

    tools = {t.name: t.effect for t in session_run(run)}
    assert tools["get_status"] is Effect.ACTUATE and tools["set_mode"] is Effect.WRITE


def test_load_tools_without_servers_says_so(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("RAI_CONFIG", raising=False)
    with pytest.raises(ValueError, match="No MCP servers"):
        asyncio.run(load_tools())


def test_load_setup_from_module_and_file():
    assert load_setup("tests.rai_mcp.scripted:deny_motion") is deny_motion
    from_file = load_setup(
        str(Path(__file__).with_name("scripted.py")) + ":deny_motion"
    )
    assert from_file.__name__ == "deny_motion"
    with pytest.raises(ValueError):
        load_setup("no_function")


def test_cli_flags_override_the_settings_file(tmp_path, monkeypatch):
    config = tmp_path / "rai.toml"
    config.write_text(
        '[mcp]\nname = "robot"\ngroups = ["diagnostics"]\nport = 9000\nsetup = ["a:b"]\n'
        '[mcp.servers.robot]\ntransport = "stdio"\ncommand = "rai"\n'
    )
    served = []
    monkeypatch.setattr("rai.mcp.server.serve", served.append)
    main(["mcp", "serve", "--config", str(config)])
    main(
        ["mcp", "serve", "--config", str(config), "--tools", "topics,actions"]
        + ["--read-only", "--node-name", "n", "--setup", "c:d"]
    )
    file_only, flags = served
    assert file_only == MCPSettings(
        name="robot",
        groups=["diagnostics"],
        port=9000,
        setup=["a:b"],
        servers={"robot": {"transport": "stdio", "command": "rai"}},
    )
    assert flags.groups == ["topics", "actions"] and flags.read_only
    assert (flags.node_name, flags.port, flags.setup) == ("n", 9000, ["a:b", "c:d"])
    with pytest.raises(SystemExit):
        main(["mcp", "serve", "--tools", "teleport"])

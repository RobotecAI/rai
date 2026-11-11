"""D1 slice on LangChain 1.x: create_agent + middleware, langchain-mcp-adapters."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from typing import Any

from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware, HumanInTheLoopMiddleware
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import BaseTool, StructuredTool, ToolException
from langgraph.checkpoint.memory import InMemorySaver

from ..common import robot
from ..common.guard import SafetyHook, ToolCall, check
from ..common.robot import Effect, RobotIO

# ---- 1. tools: a RAI tool is a StructuredTool with effect/targets in metadata ----


def rai_tool(fn, effect: Effect, targets: tuple[str, ...] = ()) -> BaseTool:
    return StructuredTool.from_function(
        coroutine=fn, metadata={"effect": effect, "targets": targets}
    )


def effect_of(tool: BaseTool) -> Effect:
    meta = tool.metadata or {}
    if "effect" in meta:
        return Effect(meta["effect"])
    if meta.get("readOnlyHint"):  # MCP client tools: effect from MCP annotations
        return Effect.READ
    return Effect.ACTUATE if meta.get("destructiveHint") else Effect.WRITE


def make_tools(io: RobotIO) -> list[BaseTool]:
    async def get_latest_message(topic: str) -> str:
        """Return the latest message on a ROS 2 topic as JSON."""
        return await robot.get_latest_message(io, topic)

    async def publish_twist(
        linear_x: float, angular_z: float, duration_s: float = 0.0
    ) -> str:
        """Drive the robot base by publishing a Twist to /cmd_vel for duration_s seconds."""
        return await robot.publish_twist(io, linear_x, angular_z, duration_s)

    return [
        rai_tool(get_latest_message, Effect.READ),
        rai_tool(publish_twist, Effect.ACTUATE, (robot.CMD_VEL,)),
    ]


# ---- 3. StateBased context: added to each model request, never to the state ----


class StateContext(AgentMiddleware):
    def __init__(self, observe: Callable[[], Awaitable[str]]) -> None:
        super().__init__()
        self.observe = observe

    async def awrap_model_call(self, request, handler):
        obs = HumanMessage(await self.observe())
        return await handler(request.override(messages=[*request.messages, obs]))


# ---- 5. fail-closed adapter: the hook decides before the tool runs ----


class FailClosed(AgentMiddleware):
    def __init__(self, hooks: Sequence[SafetyHook], timeout_s: float = 1.0) -> None:
        super().__init__()
        self.hooks, self.timeout_s = hooks, timeout_s

    async def awrap_tool_call(self, request, handler):
        tc = request.tool_call
        denial = await _check(
            self.hooks, request.tool, tc["name"], tc["args"], self.timeout_s
        )
        if denial:
            return ToolMessage(
                denial, tool_call_id=tc["id"], name=tc["name"], status="error"
            )
        return await handler(request)


async def _check(hooks, tool: BaseTool | None, name: str, args: dict, timeout_s: float):
    effect = effect_of(tool) if tool else Effect.WRITE
    targets = tuple((tool.metadata or {}).get("targets", ())) if tool else ()
    return await check(hooks, ToolCall(name, args, effect, targets), timeout_s)


def guarded(
    tool: BaseTool,
    hooks: Sequence[SafetyHook],
    timeout_s: float = 1.0,
    *,
    raise_denial: bool = False,
) -> BaseTool:
    """Tool-level form of FailClosed: covers every path (agent, MCP server, direct calls).
    On the agent path a denial becomes ToolMessage(status="error"); with raise_denial=True
    it propagates (MCP server: isError=True)."""

    async def run(**kwargs: Any) -> str:
        if denial := await _check(hooks, tool, tool.name, kwargs, timeout_s):
            raise ToolException(denial)
        return await tool.ainvoke(kwargs)

    return StructuredTool(
        name=tool.name,
        description=tool.description,
        args_schema=tool.args_schema,
        coroutine=run,
        metadata=tool.metadata,
        handle_tool_error=not raise_denial,
    )


# ---- 2 + 4. ReAct agent with the approval gate on ACTUATE tools ----


def build_agent(
    model: BaseChatModel,
    tools: Sequence[BaseTool],
    *,
    hooks: Sequence[SafetyHook] = (),
    context: Callable[[], Awaitable[str]] | None = None,
    approve_actuation: bool = True,
    hook_timeout_s: float = 1.0,
):
    middleware: list[AgentMiddleware] = []
    if approve_actuation:
        gated = {
            t.name: {"allowed_decisions": ["approve", "reject"]}
            for t in tools
            if effect_of(t) is Effect.ACTUATE
        }
        middleware.append(HumanInTheLoopMiddleware(interrupt_on=gated))
    if context:
        middleware.append(StateContext(context))
    middleware.append(FailClosed(hooks, hook_timeout_s))
    return create_agent(
        model, list(tools), middleware=middleware, checkpointer=InMemorySaver()
    )

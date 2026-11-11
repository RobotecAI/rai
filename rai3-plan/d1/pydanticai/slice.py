"""D1 slice on PydanticAI: toolsets, capabilities, deferred approval, MCPToolset."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, replace

from pydantic_ai import (
    Agent,
    ApprovalRequiredToolset,
    CombinedToolset,
    DeferredToolRequests,
    FunctionToolset,
    Tool,
    ToolFailed,
    WrapperToolset,
)
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.messages import ModelRequest, UserPromptPart
from pydantic_ai.tools import ToolDefinition
from pydantic_ai.toolsets import AbstractToolset

from ..common import robot
from ..common.guard import SafetyHook, ToolCall, check
from ..common.robot import Effect, RobotIO

# ---- 1. tools: a RAI tool is a pydantic_ai Tool with effect/targets in metadata ----


def rai_tool(fn, effect: Effect, targets: tuple[str, ...] = ()) -> Tool:
    return Tool(fn, metadata={"effect": effect, "targets": targets})


def effect_of(tool_def: ToolDefinition) -> Effect:
    meta = tool_def.metadata or {}
    if "effect" in meta:
        return Effect(meta["effect"])
    hints = (
        meta.get("annotations") or {}
    )  # MCP client tools: effect from MCP annotations
    if hints.get("readOnlyHint"):
        return Effect.READ
    return Effect.ACTUATE if hints.get("destructiveHint") else Effect.WRITE


def make_tools(io: RobotIO) -> FunctionToolset:
    async def get_latest_message(topic: str) -> str:
        """Return the latest message on a ROS 2 topic as JSON."""
        return await robot.get_latest_message(io, topic)

    async def publish_twist(
        linear_x: float, angular_z: float, duration_s: float = 0.0
    ) -> str:
        """Drive the robot base by publishing a Twist to /cmd_vel for duration_s seconds."""
        return await robot.publish_twist(io, linear_x, angular_z, duration_s)

    return FunctionToolset(
        [
            rai_tool(get_latest_message, Effect.READ),
            rai_tool(publish_twist, Effect.ACTUATE, (robot.CMD_VEL,)),
        ]
    )


# ---- 3. StateBased context: added to each model request, never to the history ----


@dataclass
class StateContext(AbstractCapability):
    observe: Callable[[], Awaitable[str]]

    async def before_model_request(self, ctx, request_context):
        obs = ModelRequest(parts=[UserPromptPart(await self.observe())])
        return replace(request_context, messages=[*request_context.messages, obs])


# ---- 5. fail-closed adapter: the hook decides before the tool runs ----


@dataclass
class FailClosed(WrapperToolset):
    hooks: Sequence[SafetyHook] = ()
    timeout_s: float = 1.0

    async def call_tool(self, name, tool_args, ctx, tool):
        d = tool.tool_def
        call = ToolCall(
            name, tool_args, effect_of(d), tuple((d.metadata or {}).get("targets", ()))
        )
        if denial := await check(self.hooks, call, self.timeout_s):
            raise ToolFailed(denial)
        return await super().call_tool(name, tool_args, ctx, tool)


# ---- 2 + 4. ReAct agent with the approval gate on ACTUATE tools ----


def guard(
    toolsets: Sequence[AbstractToolset],
    hooks: Sequence[SafetyHook] = (),
    *,
    approve_actuation: bool = True,
    hook_timeout_s: float = 1.0,
) -> AbstractToolset:
    ts: AbstractToolset = CombinedToolset(list(toolsets))
    if approve_actuation:
        ts = ApprovalRequiredToolset(
            ts, lambda ctx, d, args: effect_of(d) is Effect.ACTUATE
        )
    return FailClosed(
        ts, hooks, hook_timeout_s
    )  # outermost: runs before asking a human


def build_agent(
    model,
    toolsets: Sequence[AbstractToolset],
    *,
    hooks: Sequence[SafetyHook] = (),
    context: Callable[[], Awaitable[str]] | None = None,
    approve_actuation: bool = True,
    hook_timeout_s: float = 1.0,
) -> Agent:
    ts = guard(
        toolsets,
        hooks,
        approve_actuation=approve_actuation,
        hook_timeout_s=hook_timeout_s,
    )
    caps = [StateContext(context)] if context else []
    return Agent(
        model, toolsets=[ts], capabilities=caps, output_type=[str, DeferredToolRequests]
    )

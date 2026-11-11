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

"""The RAI tool class: a LangChain `BaseTool` with an effect, ROS targets and hooks."""

import asyncio
import contextvars
import inspect
from collections.abc import Awaitable, Callable, Iterable, Mapping
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from types import MappingProxyType
from typing import Any, ClassVar

from langchain_core.tools import BaseTool, StructuredTool, ToolException
from langchain_core.tools.base import create_schema_from_function
from pydantic import Field

from .hooks import Effect, ToolCallContext, ToolHooks, global_tool_hooks, run_tool_hooks


class RaiTool(BaseTool):
    """Async-native tool. Subclasses implement `async _acall(**kwargs)`.

    Every call runs `global_tool_hooks`, then `hooks`, fail closed. Denials, errors and
    timeouts reach the LLM as error tool messages; cancellation propagates.
    """

    default_effect: ClassVar[Effect] = Effect.ACTUATE
    default_targets: ClassVar[tuple[str, ...]] = ()

    hooks: ToolHooks = Field(default_factory=ToolHooks, exclude=True)
    timeout_s: float | None = None

    def __init__(
        self,
        *,
        effect: Effect | None = None,
        targets: Iterable[str] | None = None,
        **kwargs: Any,
    ) -> None:
        kwargs["metadata"] = {
            **(kwargs.get("metadata") or {}),
            "effect": Effect(effect or self.default_effect),
            "targets": tuple(self.default_targets if targets is None else targets),
        }
        kwargs.setdefault("handle_tool_error", True)
        kwargs.setdefault("handle_validation_error", True)
        super().__init__(**kwargs)
        if self.args_schema is None:
            self.args_schema = create_schema_from_function(self.name, self._acall)

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if cls._arun is not RaiTool._arun or cls._run is not RaiTool._run:
            raise TypeError(f"{cls.__name__}: implement _acall, not _run/_arun")

    @property
    def effect(self) -> Effect:
        return self.metadata["effect"]  # type: ignore[index]

    @property
    def targets(self) -> tuple[str, ...]:
        return self.metadata["targets"]  # type: ignore[index]

    async def _acall(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    async def _arun(self, *args: Any, **kwargs: Any) -> Any:
        if args:
            kwargs = dict(zip(self.args, args)) | kwargs
        if global_tool_hooks or self.hooks:  # no hooks: no overhead
            call = ToolCallContext(
                self.name, self.effect, self.targets, MappingProxyType(kwargs)
            )
            if denial := await run_tool_hooks(call, (global_tool_hooks, self.hooks)):
                raise ToolException(denial.reason)
        timeout = asyncio.timeout(self.timeout_s)
        try:
            async with timeout:
                return await self._acall(**kwargs)
        except ToolException:
            raise
        except Exception as e:  # errors go back to the LLM, they never end the run
            if timeout.expired():
                raise ToolException(f"Timed out after {self.timeout_s}s") from e
            raise ToolException(f"{type(e).__name__}: {e}") from e

    def _run(self, *args: Any, **kwargs: Any) -> Any:
        coro = self._arun(*args, **kwargs)
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(coro)
        ctx = contextvars.copy_context()  # called from a loop thread: run on a new loop
        with ThreadPoolExecutor(1) as pool:
            return pool.submit(ctx.run, asyncio.run, coro).result()


class _FunctionTool(RaiTool):
    fn: Callable[..., Awaitable[Any]] = Field(exclude=True)

    async def _acall(self, **kwargs: Any) -> Any:
        return await self.fn(**kwargs)


def rai_tool(
    effect: Effect,
    targets: Iterable[str] = (),
    *,
    name: str | None = None,
    timeout_s: float | None = None,
) -> Callable[[Callable[..., Any]], RaiTool]:
    """Turn a function (async preferred) into a `RaiTool`; the docstring is the description."""

    def wrap(fn: Callable[..., Any]) -> RaiTool:
        is_async = inspect.iscoroutinefunction(fn)
        spec = StructuredTool.from_function(
            func=None if is_async else fn, coroutine=fn if is_async else None, name=name
        )
        return _FunctionTool(
            name=spec.name,
            description=spec.description,
            args_schema=spec.args_schema,
            fn=fn if is_async else partial(asyncio.to_thread, fn),
            effect=effect,
            targets=targets,
            timeout_s=timeout_s,
        )

    return wrap


def effect_from_mcp_annotations(annotations: Mapping[str, Any] | None) -> Effect:
    """MCP hints to an effect: readOnlyHint -> READ, destructiveHint=False -> WRITE,
    anything else -> ACTUATE (MCP's destructiveHint defaults to true)."""
    annotations = annotations or {}
    if annotations.get("readOnlyHint"):
        return Effect.READ
    if annotations.get("destructiveHint") is False:
        return Effect.WRITE
    return Effect.ACTUATE


def from_langchain(
    tool: BaseTool,
    *,
    effect: Effect | None = None,
    targets: Iterable[str] = (),
    timeout_s: float | None = None,
) -> RaiTool:
    """Wrap a third-party or MCP client tool so RAI hooks apply to it.

    Without `effect`, it comes from MCP annotations in `tool.metadata` (ACTUATE if none).
    """
    if isinstance(tool, StructuredTool) and (tool.coroutine or tool.func):
        fn = tool.coroutine or partial(asyncio.to_thread, tool.func)
        response_format = tool.response_format
    else:
        fn = partial(_ainvoke, tool)
        response_format = "content"
    return _FunctionTool(
        name=tool.name,
        description=tool.description,
        args_schema=tool.args_schema or tool.get_input_schema(),
        fn=fn,
        response_format=response_format,
        effect=effect or effect_from_mcp_annotations(tool.metadata),
        targets=targets,
        timeout_s=timeout_s,
        metadata=tool.metadata,
    )


async def _ainvoke(tool: BaseTool, **kwargs: Any) -> Any:
    return await tool.ainvoke(kwargs)

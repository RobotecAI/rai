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

"""What every built-in safety hook shares: one decision on both hook points."""

import inspect
from collections.abc import Callable, Iterable, Sequence
from typing import Any, Optional, Self, Union

from rai.communication.ros2 import egress as eg
from rai.communication.ros2.egress import EgressHooks, global_egress
from rai.tools import hooks as th
from rai.tools.hooks import Effect, ToolCallContext, ToolHooks, global_tool_hooks

Target = Any
"""A ROS2Connector (its runtime's chain), a ROS2Runtime or an EgressHooks chain."""

EgressResult = Union[eg.Allow, eg.Deny, Any]  # Any: an awaitable of either


def runtime_of(target: Target) -> Any:
    """The ROS2Runtime behind a connector or runtime target."""
    return getattr(target, "_runtime", target)


def egress_chains(targets: Sequence[Target]) -> list[EgressHooks]:
    """Every runtime with no targets (global_egress), else each target's chain."""
    if not targets:
        return [global_egress]
    chains = []
    for target in targets:
        chain = target if isinstance(target, EgressHooks) else runtime_of(target).egress
        if chain not in chains:
            chains.append(chain)
    return chains


def _named(fn: Callable[[Any], Any], name: str) -> Callable[[Any], Any]:
    """A plain function wrapping fn whose __name__ is name (hooks report it)."""
    hook: Callable[[Any], Any]
    if inspect.iscoroutinefunction(fn):

        async def async_hook(arg: Any) -> Any:
            return await fn(arg)

        hook = async_hook
    else:

        def sync_hook(arg: Any) -> Any:
            return fn(arg)

        hook = sync_hook
    hook.__name__ = hook.__qualname__ = name
    return hook


class SafetyHook:
    """One safety rule installed on both hook points.

    ``tool_hook`` gates RaiTool calls whose effect is in ``tool_effects``;
    ``egress_hook`` gates connector egress matching ``types``/``names``/``kinds``
    (rai.communication.ros2.egress). Both fail closed through their chains.
    """

    def __init__(
        self,
        name: str,
        *,
        types: Optional[Iterable[str]] = None,
        names: Optional[Iterable[str]] = None,
        kinds: Iterable[str] = eg.WRITE_KINDS,
        tool_effects: Iterable[Effect] = (Effect.ACTUATE,),
        timeout_s: float = eg.DEFAULT_EGRESS_TIMEOUT_S,
    ) -> None:
        self.name = name
        self.types = frozenset(types or ())
        self.names = tuple(names or ())
        self.kinds = frozenset(kinds)
        self.tool_effects = frozenset(tool_effects)
        self.timeout_s = timeout_s
        self.tool_hook = _named(self._tool, name)
        self.egress_hook = _named(self._egress, name)
        self._added: list[tuple[Any, Callable[[Any], Any]]] = []

    async def _tool(self, call: ToolCallContext) -> Union[th.Allow, th.Deny]:
        raise NotImplementedError

    def _egress(self, egress: eg.Egress) -> EgressResult:
        raise NotImplementedError

    def install(
        self, *targets: Target, tools: Optional[ToolHooks] = global_tool_hooks
    ) -> Self:
        """Add egress_hook to the chains of targets (every runtime if none) and
        tool_hook to tools (every RaiTool by default; None: connector side only).
        Hooks run in install order, so install an EStop first."""
        self.add_to(egress_chains(targets), tools)
        return self

    def add_to(self, chains: Iterable[EgressHooks], tools: Optional[ToolHooks]) -> None:
        """Register the hooks on these chains (again: a second check point)."""
        for chain in chains:
            chain.add(
                self.egress_hook,
                types=self.types,
                names=self.names,
                kinds=self.kinds,
                timeout_s=self.timeout_s,
            )
            self._added.append((chain, self.egress_hook))
        if tools is not None:
            tools.add(self.tool_hook, timeout_s=self.timeout_s)
            self._added.append((tools, self.tool_hook))

    def uninstall(self) -> None:
        """Remove every registration of this hook."""
        for chain, hook in self._added:
            chain.remove(hook)
        self._added.clear()

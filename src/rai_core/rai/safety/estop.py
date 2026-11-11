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

"""EStop: a software e-stop latch on both hook points. Not a hardware e-stop."""

import concurrent.futures
import logging
import threading
from collections.abc import Callable, Iterable
from typing import Any, Optional, Self, Union

from rai.communication.ros2 import egress as eg
from rai.communication.ros2 import runtime as rt
from rai.communication.ros2.egress import MOTION
from rai.tools import hooks as th
from rai.tools.hooks import ToolCallContext, ToolHooks, global_tool_hooks

from ._base import SafetyHook, Target, runtime_of

logger = logging.getLogger(__name__)

ESTOP_TOPIC = "/rai/estop"
"""std_msgs/msg/Bool: True engages, False releases."""
ESTOP_SERVICE = "/rai/estop/set"
"""std_srvs/srv/SetBool: data=True engages, data=False releases; success is True."""


class EStop(SafetyHook):
    """While engaged, denies every MOTION egress (publish, service call, goal)
    and every ACTUATE tool call. Engaging cancels every active MOTION goal on
    the guarded runtimes (every runtime unless installed on targets), so it
    stops a tool that is already running. Cancels always pass.

    Trigger it from code (``engage``/``release``) or from ROS (``listen``):
    ``ros2 topic pub --once /rai/estop std_msgs/msg/Bool "{data: true}"`` or
    ``ros2 service call /rai/estop/set std_srvs/srv/SetBool "{data: true}"``.
    """

    def __init__(self, *, types: Iterable[str] = MOTION, name: str = "EStop") -> None:
        super().__init__(name, types=types)
        self.reason = ""
        self._engaged = False
        self._lock = threading.Lock()
        self._runtimes: Optional[list[Any]] = None  # None: every live runtime
        self._callbacks: tuple[Callable[[str], Any], ...] = ()
        self._watching = False
        self._subscriptions: list[Any] = []

    # ------------------------------------------------------------ state

    @property
    def engaged(self) -> bool:
        return self._engaged

    def engage(
        self, reason: str = "engaged from code"
    ) -> list[concurrent.futures.Future]:
        """Latch the e-stop and cancel every active MOTION goal. Returns the
        cancel futures (each resolves to True if the server accepted the
        cancel); it does not wait for them. Thread-safe, never blocks."""
        with self._lock:
            was, self._engaged, self.reason = self._engaged, True, reason
        if not was:
            logger.warning("E-stop engaged: %s", reason)
        futures = self.cancel_goals()
        for callback in self._callbacks:
            try:
                callback(reason)
            except Exception:
                logger.exception("E-stop callback %r failed", callback)
        return futures

    def release(self) -> None:
        with self._lock:
            was, self._engaged, self.reason = self._engaged, False, ""
        if was:
            logger.warning("E-stop released")

    def on_engage(self, callback: Callable[[str], Any]) -> Callable[[str], Any]:
        """Call callback(reason) on every engage, e.g. to interrupt an agent:
        ``estop.on_engage(lambda _: agent.interrupt())``."""
        self._callbacks = (*self._callbacks, callback)
        return callback

    def cancel_goals(self) -> list[concurrent.futures.Future]:
        runtimes = self._runtimes if self._runtimes is not None else rt.live_runtimes()
        futures: list[concurrent.futures.Future] = []
        for runtime in runtimes:
            try:
                futures += runtime.cancel_goals(self.types)
            except Exception:  # a closed runtime has nothing to stop
                logger.exception("E-stop could not cancel the goals of %r", runtime)
        return futures

    # ------------------------------------------------------------ hooks

    def _decide(self) -> Optional[str]:
        return f"e-stop engaged ({self.reason})" if self._engaged else None

    async def _tool(self, call: ToolCallContext) -> Union[th.Allow, th.Deny]:
        if call.effect in self.tool_effects and (reason := self._decide()):
            return th.Deny(reason)
        return th.Allow()

    def _egress(self, egress: eg.Egress) -> Union[eg.Allow, eg.Deny]:
        reason = self._decide()
        return eg.Deny(reason) if reason else eg.Allow()

    def _on_goal(self, runtime: Any, handle: Any) -> None:
        """A goal accepted after its gate passed but while engaged: cancel it."""
        if not self._engaged or handle.type not in self.types:
            return
        if self._runtimes is not None and runtime not in self._runtimes:
            return
        runtime.cancel_goals((handle.type,))

    def install(
        self, *targets: Target, tools: Optional[ToolHooks] = global_tool_hooks
    ) -> Self:
        """Guard targets (connectors or runtimes; every runtime if none) and
        tools. Install it before other hooks so nothing asks while engaged."""
        super().install(*targets, tools=tools)
        runtimes = [runtime_of(t) for t in targets if hasattr(runtime_of(t), "egress")]
        if runtimes:  # chains only (e.g. global_egress): every runtime
            self._runtimes = [*(self._runtimes or []), *runtimes]
        if not self._watching:
            rt.watch_goals(self._on_goal)
            self._watching = True
        return self

    def uninstall(self) -> None:
        super().uninstall()
        rt.unwatch_goals(self._on_goal)
        self._watching = False
        self._runtimes = None
        for subscription in self._subscriptions:
            subscription.close()
        self._subscriptions.clear()

    # ------------------------------------------------------------ ROS trigger

    def listen(
        self,
        connector: Any,
        *,
        topic: Optional[str] = ESTOP_TOPIC,
        service: Optional[str] = ESTOP_SERVICE,
    ) -> Self:
        """Engage and release from ROS through connector (a ROS2Connector): a
        std_msgs/msg/Bool topic and a std_srvs/srv/SetBool service. None skips
        one. The service lives as long as the connector."""
        if topic is not None:
            source = f"ROS topic {topic}"

            def on_message(msg: Any) -> None:
                if msg.data:
                    self.engage(source)
                else:
                    self.release()

            self._subscriptions.append(
                connector.subscribe(topic, on_message, msg_type="std_msgs/msg/Bool")
            )
        if service is not None:
            caller = f"ROS service {service}"

            def on_request(request: Any, response: Any) -> Any:
                if request.data:
                    self.engage(caller)
                else:
                    self.release()
                response.success = True
                response.message = "engaged" if self._engaged else "released"
                return response

            connector.create_service(
                service, on_request, service_type="std_srvs/srv/SetBool"
            )
        return self

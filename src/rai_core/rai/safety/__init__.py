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

"""Built-in robot safety hooks on both hook points (rai3-plan/PLAN.md 2.3).

Each hook is installed on the agent side (``rai.tools.global_tool_hooks``: every
``RaiTool`` call, including MCP and direct calls) and on the connector side
(``rai.communication.ros2.egress``: every publish, service call and goal). The
connector side is the guarantee; the agent side gives the LLM an early,
readable denial. Both fail closed.

- ``EStop``: a latch that denies MOTION egress and ACTUATE tool calls and
  cancels active MOTION goals when engaged. ROS trigger: topic ``/rai/estop``
  (std_msgs/msg/Bool) and service ``/rai/estop/set`` (std_srvs/srv/SetBool);
  True engages, False releases. Cancels are never gated.
- ``ConfirmActuation`` (opt-in): asks a human (``cli_asker``, ``HRIAsker``)
  before ACTUATE tool calls and MOTION egress; an approval leases the stream.
- ``RateLimit``: a token bucket per tool name and per topic/service/action.
- ``AuditLog``: an observer (it can never deny) writing JSON lines and feeding
  subscribers such as the flight recorder.

One call installs a standard set::

    from rai import safety

    guard = safety.install(audit_path="audit.jsonl")
    guard.estop.engage("operator")      # or: ros2 topic pub /rai/estop ...
    guard.uninstall()

``install`` defaults: EStop on, listening on ``/rai/estop`` through a connector
it creates on the shared runtime (``listen=False`` to skip); no confirmation;
no rate limits; no audit file. With no targets the hooks cover every runtime
(``global_egress``) and every RaiTool (``global_tool_hooks``). Order: EStop,
rate limits, ConfirmActuation, then EStop again, so nobody is asked about a
denied action and an e-stop engaged during a question still wins.
"""

import logging
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Union

from rai.tools.hooks import ToolHooks, global_tool_hooks

from ._base import SafetyHook, Target, egress_chains
from .audit import AuditLog, AuditRecord
from .confirm import YES, Asker, ConfirmActuation, HRIAsker, cli_asker
from .estop import ESTOP_SERVICE, ESTOP_TOPIC, EStop
from .ratelimit import RateLimit

logger = logging.getLogger(__name__)


@dataclass
class Safety:
    """The hooks ``install`` added, for engaging, inspecting and removing them."""

    estop: Optional[EStop] = None
    confirm: Optional[ConfirmActuation] = None
    rate_limits: tuple[RateLimit, ...] = ()
    audit: Optional[AuditLog] = None
    _owned: list[Any] = field(default_factory=list, repr=False)

    def uninstall(self) -> None:
        """Remove every hook and observer, close the audit file and the
        connector ``install`` created."""
        hooks: list[Optional[SafetyHook]] = [
            self.estop,
            *self.rate_limits,
            self.confirm,
        ]
        for hook in hooks:
            if hook is not None:
                hook.uninstall()
        if self.audit is not None:
            self.audit.close()
        for connector in self._owned:
            connector.close()
        self._owned.clear()


def install(
    *targets: Target,
    estop: Union[bool, EStop] = True,
    listen: Union[bool, Any] = True,
    confirm: Union[bool, Asker, ConfirmActuation] = False,
    rate_limits: Iterable[RateLimit] = (),
    audit_path: Union[str, Path, None] = None,
    tools: Optional[ToolHooks] = global_tool_hooks,
) -> Safety:
    """Install a standard safety set and return its handles.

    Args:
        targets: connectors or runtimes to guard; none guards every runtime.
        estop: True for a new EStop, an EStop to use, or False.
        listen: let ROS engage the EStop: True through a new ROS2Connector,
            a ROS2Connector to use, or False.
        confirm: False; True asks on the terminal (cli_asker); an asker
            ``async (str) -> bool``; or a ConfirmActuation.
        rate_limits: RateLimit hooks to install.
        audit_path: write an AuditLog there (JSON lines); None: no AuditLog.
        tools: the agent-side chain (None: connector side only).
    """
    safety = Safety()
    chains = egress_chains(targets)
    try:
        if estop:
            safety.estop = estop if isinstance(estop, EStop) else EStop()
            safety.estop.install(*targets, tools=tools)
            if listen:
                connector = listen
                if listen is True:
                    from rai.communication.ros2 import ROS2Connector

                    connector = ROS2Connector()
                    safety._owned.append(connector)
                safety.estop.listen(connector)
        safety.rate_limits = tuple(rate_limits)
        for limit in safety.rate_limits:
            limit.install(*targets, tools=tools)
        if confirm:
            if isinstance(confirm, ConfirmActuation):
                safety.confirm = confirm
            else:
                safety.confirm = ConfirmActuation(
                    cli_asker() if confirm is True else confirm
                )
            safety.confirm.install(*targets, tools=tools)
            if safety.estop is not None:  # re-check after the human answered
                safety.estop.add_to(chains, tools)
        if audit_path is not None:
            safety.audit = AuditLog(audit_path).install(*targets, tools=tools)
    except BaseException:
        safety.uninstall()
        raise
    return safety


__all__ = [
    "ESTOP_SERVICE",
    "ESTOP_TOPIC",
    "YES",
    "Asker",
    "AuditLog",
    "AuditRecord",
    "ConfirmActuation",
    "EStop",
    "HRIAsker",
    "RateLimit",
    "Safety",
    "SafetyHook",
    "cli_asker",
    "install",
]

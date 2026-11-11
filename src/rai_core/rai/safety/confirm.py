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

"""ConfirmActuation: ask a human before anything moves the robot.

An asker is ``async (question: str) -> bool``; only ``True`` approves. Two are
provided: ``cli_asker`` (stdin) and ``HRIAsker`` (a ROS2HRIConnector).
"""

import asyncio
import logging
import sys
import threading
import time
from collections import deque
from collections.abc import Awaitable, Callable, Iterable
from typing import Any, Optional, TextIO, Union

from rai.communication.ros2 import egress as eg
from rai.communication.ros2.egress import MOTION
from rai.tools import hooks as th
from rai.tools.hooks import Effect, ToolCallContext

from ._base import SafetyHook

logger = logging.getLogger(__name__)

Asker = Callable[[str], Awaitable[bool]]

YES = frozenset({"y", "yes", "ok", "approve", "allow", "go", "tak"})


class ConfirmActuation(SafetyHook):
    """Asks before ACTUATE tool calls and MOTION egress; opt-in.

    - Approval of a tool call lets that tool's own egress through for
      ``lease_s`` (no second question for the same action).
    - Approval of an egress grants an egress lease of ``lease_s`` on that
      (kind, name, type), so a 30 Hz cmd_vel stream is approved once.
    - No answer within ``timeout_s``, a failing asker or any answer but True
      denies; the reason reaches the LLM as the tool's error.

    Install it after EStop and RateLimit, so a human is never asked about
    something that would be denied anyway (``rai.safety.install`` does, and
    re-checks the EStop after the answer).
    """

    def __init__(
        self,
        ask: Asker,
        *,
        timeout_s: float = 30.0,
        lease_s: float = 10.0,
        types: Iterable[str] = MOTION,
        names: Optional[Iterable[str]] = None,
        tool_effects: Iterable[Effect] = (Effect.ACTUATE,),
        name: str = "ConfirmActuation",
    ) -> None:
        # The chain's timeout is a backstop: our own timeout gives the reason.
        super().__init__(
            name,
            types=types,
            names=names,
            tool_effects=tool_effects,
            timeout_s=timeout_s + 1.0,
        )
        self.ask = ask
        self.answer_timeout_s = timeout_s
        self.lease_s = lease_s
        self._granted: dict[str, float] = {}  # tool name -> approved until

    async def _confirm(self, question: str) -> Optional[str]:
        """None if approved, else the reason for the denial."""
        try:
            async with asyncio.timeout(self.answer_timeout_s):
                answer = await self.ask(question)
        except TimeoutError:
            return f"no answer from the operator within {self.answer_timeout_s:g} s"
        except Exception as e:
            logger.warning("Confirmation asker failed: %r", e)
            return f"could not ask the operator ({type(e).__name__}: {e})"
        return None if answer is True else "the operator denied it"

    async def _tool(self, call: ToolCallContext) -> Union[th.Allow, th.Deny]:
        if call.effect not in self.tool_effects:
            return th.Allow()
        args = ", ".join(f"{k}={v!r}" for k, v in call.args.items())
        targets = f" on {', '.join(call.targets)}" if call.targets else ""
        question = f"Allow the robot to run {call.name}({args}){targets}? [y/N]"
        if reason := await self._confirm(question):
            return th.Deny(reason)
        self._granted[call.name] = time.monotonic() + self.lease_s
        return th.Allow()

    def _egress(self, egress: eg.Egress) -> Any:
        if egress.origin and self._granted.get(egress.origin, 0.0) > time.monotonic():
            return eg.Allow()  # this tool call was approved on the agent side
        return self._egress_ask(egress)

    async def _egress_ask(self, egress: eg.Egress) -> Union[eg.Allow, eg.Deny]:
        who = egress.origin or "code"
        question = (
            f"Allow {who} to {egress.kind} {egress.type} on {egress.name}"
            f" (approval holds {self.lease_s:g} s)? [y/N]"
        )
        if reason := await self._confirm(question):
            return eg.Deny(reason)
        return eg.Allow(lease_s=self.lease_s)


def cli_asker(
    *,
    stdin: Optional[TextIO] = None,
    stdout: Optional[TextIO] = None,
    yes: Iterable[str] = YES,
) -> Asker:
    """Ask on the terminal. One question at a time; anything but a yes word
    (y, yes, ok, ...) denies. A question that timed out still waits for its
    line, which is then discarded."""
    words = frozenset(w.lower() for w in yes)
    lock = threading.Lock()

    def ask_blocking(question: str) -> bool:
        with lock:
            out = stdout or sys.stdout
            out.write(f"{question} ")
            out.flush()
            line = (stdin or sys.stdin).readline()
        return line.strip().lower() in words

    async def ask(question: str) -> bool:
        return await asyncio.to_thread(ask_blocking, question)

    return ask


class HRIAsker:
    """Ask through an HRI connector: send the question to ``to_human`` and take
    the next message on ``from_human`` as the answer (a yes word approves,
    anything else denies). Questions are answered in order.

    Uses ``connector.build_message``, ``send_message``, ``register_callback``
    and ``unregister_callback``; any HRIConnector with them works. The
    connector is not closed by ``close()``.
    """

    def __init__(
        self,
        connector: Any,
        to_human: str,
        from_human: str,
        *,
        yes: Iterable[str] = YES,
    ) -> None:
        self.connector = connector
        self.to_human = to_human
        self.yes = frozenset(w.lower() for w in yes)
        self._pending: deque[tuple[asyncio.AbstractEventLoop, asyncio.Future]] = deque()
        self._lock = threading.Lock()
        self._callback_id = connector.register_callback(
            from_human, self._on_message, raw=True
        )

    def _on_message(self, msg: Any) -> None:
        text = str(getattr(msg, "text", "")).strip().lower().rstrip(".!")
        with self._lock:
            while self._pending:
                loop, future = self._pending.popleft()
                if not future.done():
                    break
            else:
                return  # nobody asked: not an answer
        approved = text in self.yes
        loop.call_soon_threadsafe(lambda: future.done() or future.set_result(approved))

    async def __call__(self, question: str) -> bool:
        from langchain_core.messages import AIMessage

        loop = asyncio.get_running_loop()
        future: asyncio.Future = loop.create_future()
        with self._lock:
            self._pending.append((loop, future))
        try:
            message = self.connector.build_message(AIMessage(question))
            await asyncio.to_thread(self.connector.send_message, message, self.to_human)
            return await future
        finally:
            future.cancel()  # a timed-out question takes no later answer

    def close(self) -> None:
        self.connector.unregister_callback(self._callback_id)


__all__ = ["YES", "Asker", "ConfirmActuation", "HRIAsker", "cli_asker"]

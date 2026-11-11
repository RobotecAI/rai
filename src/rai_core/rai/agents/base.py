# Copyright (C) 2024 Robotec.AI
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

"""Agent lifecycle: an explicit, async-first state machine.

::

    CREATED -> STARTING -> RUNNING -> STOPPING -> STOPPED
                  |           |          ^
                  +--------> FAILED -----+

STARTING may also go straight to STOPPING, and CREATED to STOPPED (stop() before
start() is a no-op). STOPPED is terminal: create a new agent to run again.

All coroutines of one agent must run on the same event loop. Use
:class:`rai.agents.ThreadedAgent` or :func:`rai.run` to drive agents from sync code.
"""

import asyncio
import logging
import threading
from collections.abc import AsyncIterator, Callable
from enum import StrEnum
from typing import Any, Self

from rai.agents.events import AgentEvent, StateChanged


class AgentState(StrEnum):
    CREATED = "created"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    FAILED = "failed"


_S = AgentState
_ALLOWED: dict[AgentState, frozenset[AgentState]] = {
    _S.CREATED: frozenset({_S.STARTING, _S.STOPPED}),
    _S.STARTING: frozenset({_S.RUNNING, _S.STOPPING, _S.FAILED}),
    _S.RUNNING: frozenset({_S.STOPPING, _S.FAILED}),
    _S.FAILED: frozenset({_S.STOPPING}),
    _S.STOPPING: frozenset({_S.STOPPED}),
    _S.STOPPED: frozenset(),
}


class InvalidTransitionError(RuntimeError):
    """Raised on an illegal lifecycle transition, e.g. starting a stopped agent."""


class BaseAgent:
    """Base class of every RAI agent.

    Subclasses override any of the three async hooks:

    - :meth:`on_start`: acquire resources (subscribe, create services).
    - :meth:`run`: the main body. The default waits until the agent is stopped.
      If it returns, the agent stops itself; if it raises, the agent goes FAILED.
    - :meth:`on_stop`: release resources. Runs once, also after a failure.

    :meth:`stop` cancels the task running :meth:`on_start` / :meth:`run`, so
    whatever it awaits (an LLM call, a tool) is cancelled too.

    Agents emit :mod:`rai.agents.events`; consume them with :meth:`subscribe`
    or :meth:`events`.
    """

    def __init__(self, *, name: str | None = None):
        self.name = name or type(self).__name__
        self.logger = logging.getLogger(self.name)
        self.error: BaseException | None = None
        self._state = AgentState.CREATED
        self._lock = threading.RLock()
        self._task: asyncio.Task[None] | None = None
        self._done = asyncio.Event()  # set on FAILED or STOPPED
        self._stopped = asyncio.Event()
        self._outputs: tuple[Callable[[AgentEvent], None], ...] = ()

    @property
    def state(self) -> AgentState:
        return self._state

    def _transition(self, new: AgentState) -> None:
        with self._lock:
            old = self._state
            if new not in _ALLOWED[old]:
                raise InvalidTransitionError(
                    f"{self.name}: cannot go from {old} to {new}"
                )
            self._state = new
        self.logger.debug("%s: %s -> %s", self.name, old, new)
        if new in (AgentState.FAILED, AgentState.STOPPED):
            self._done.set()
        if new is AgentState.STOPPED:
            self._stopped.set()
        self.emit(StateChanged(agent=self.name, old=old, new=new))

    def subscribe(self, output: Callable[[AgentEvent], None]) -> Callable[[], None]:
        """Call ``output`` with every event, in the emitting thread. Returns a
        function that unsubscribes it. Errors in ``output`` are logged."""
        with self._lock:
            self._outputs = (*self._outputs, output)

        def unsubscribe() -> None:
            with self._lock:
                self._outputs = tuple(o for o in self._outputs if o is not output)

        return unsubscribe

    def emit(self, event: AgentEvent) -> None:
        for output in self._outputs:
            try:
                output(event)
            except Exception:
                self.logger.exception("%s: event output %r failed", self.name, output)

    def events(self) -> AsyncIterator[AgentEvent]:
        """Events from now on, delivered on the calling event loop. The iterator
        ends after the agent is STOPPED."""
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[AgentEvent] = asyncio.Queue()
        unsubscribe = self.subscribe(
            lambda event: loop.call_soon_threadsafe(queue.put_nowait, event)
        )

        async def stream() -> AsyncIterator[AgentEvent]:
            try:
                while True:
                    event = await queue.get()
                    yield event
                    if isinstance(event, StateChanged) and event.new is _S.STOPPED:
                        return
            finally:
                unsubscribe()

        return stream()

    async def on_start(self) -> None:
        pass

    async def run(self) -> None:
        await asyncio.Event().wait()

    async def on_stop(self) -> None:
        pass

    async def start(self) -> None:
        """Run :meth:`on_start` and spawn :meth:`run`. Raises what on_start raised."""
        self._transition(AgentState.STARTING)
        started: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        self._task = asyncio.create_task(self._lifecycle(started), name=self.name)
        await asyncio.shield(started)

    async def _lifecycle(self, started: asyncio.Future[None]) -> None:
        try:
            await self.on_start()
            if self._state is not AgentState.STARTING:
                return  # stopped while starting
            self._transition(AgentState.RUNNING)
            started.set_result(None)
            await self.run()
        except Exception as e:
            self.error = e
            self.logger.exception("%s failed", self.name)
            if self._state in (AgentState.STARTING, AgentState.RUNNING):
                self._transition(AgentState.FAILED)
            if not started.done():
                started.set_exception(e)
            return
        finally:
            if not started.done():
                started.set_result(None)
        if self._state is AgentState.RUNNING:
            await self.stop()  # run() returned: the agent is done

    async def stop(self, timeout: float = 10.0) -> None:
        """Stop the agent. Idempotent; a no-op before start.

        Cancels the running task and waits up to ``timeout`` seconds for it, then
        runs :meth:`on_stop`, again for up to ``timeout`` seconds. Errors and
        timeouts are logged, and the agent always ends STOPPED.
        """
        with self._lock:
            prev = self._state
            if prev is AgentState.CREATED:
                self._transition(AgentState.STOPPED)
                return
            if prev not in (AgentState.STOPPING, AgentState.STOPPED):
                self._transition(AgentState.STOPPING)
        if prev in (AgentState.STOPPING, AgentState.STOPPED):
            await self._stopped.wait()
            return
        if self._task is not None and self._task is not asyncio.current_task():
            self._task.cancel()
            await _bounded(self._task, timeout, f"{self.name}.run", self.logger)
        await _bounded(self.on_stop(), timeout, f"{self.name}.on_stop", self.logger)
        self._transition(AgentState.STOPPED)

    async def wait(self) -> None:
        """Wait until the agent is STOPPED or FAILED."""
        await self._done.wait()

    async def __aenter__(self) -> Self:
        await self.start()
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.stop()


class SyncAgent(BaseAgent):
    """An agent whose start and stop hooks are blocking functions.

    Override :meth:`setup` and :meth:`teardown`. They run in a worker thread, so
    they never block the event loop and the stop timeout still applies.
    """

    _setup_done: asyncio.Future[None] | None = None

    def setup(self) -> None:
        pass

    def teardown(self) -> None:
        pass

    async def on_start(self) -> None:
        self._setup_done = _in_thread(self.setup, f"{self.name}-setup")
        await asyncio.shield(self._setup_done)

    async def on_stop(self) -> None:
        if self._setup_done is not None:  # never tear down while setup still runs
            await asyncio.wait({self._setup_done})
        await _in_thread(self.teardown, f"{self.name}-teardown")


async def _bounded(aw: Any, timeout: float, what: str, logger: logging.Logger) -> None:
    """Await ``aw`` for at most ``timeout`` s, logging errors. Never hangs on a task
    that ignores cancellation (unlike ``asyncio.wait_for``)."""
    task = asyncio.ensure_future(aw)
    done, _ = await asyncio.wait({task}, timeout=timeout)
    if not done:
        task.cancel()
        logger.warning("%s did not finish within %.1fs, abandoning it", what, timeout)
    elif not task.cancelled() and (exc := task.exception()) is not None:
        logger.error("%s raised", what, exc_info=exc)


def _in_thread(fn: Callable[[], None], name: str) -> asyncio.Future[None]:
    """Run ``fn`` in a daemon thread. Unlike ``asyncio.to_thread``, a hung call
    cannot block interpreter or event loop shutdown."""
    loop = asyncio.get_running_loop()
    fut: asyncio.Future[None] = loop.create_future()

    def resolve(exc: BaseException | None) -> None:
        if fut.done():
            return
        if exc is None:
            fut.set_result(None)
        else:
            fut.set_exception(exc)

    def target() -> None:
        exc = None
        try:
            fn()
        except BaseException as e:
            exc = e
        try:
            loop.call_soon_threadsafe(resolve, exc)
        except RuntimeError:  # loop already closed: the caller gave up on us
            pass

    threading.Thread(target=target, name=name, daemon=True).start()
    return fut

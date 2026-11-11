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

"""Sync entry points for async agents: :func:`run` and :class:`ThreadedAgent`."""

import asyncio
import logging
import signal
import threading
from concurrent.futures import Future
from typing import Any, Coroutine, Self

from rai.agents.base import AgentState, BaseAgent

logger = logging.getLogger(__name__)


def run(*agents: BaseAgent, stop_timeout: float = 10.0) -> None:
    """Start ``agents`` in order and block until SIGINT/SIGTERM or until all of
    them have finished, then stop them in reverse order.

    Signal handlers are installed only when called from the main thread.
    Raises an :class:`ExceptionGroup` of the errors of agents that FAILED.
    """
    if not agents:
        raise ValueError("rai.run() needs at least one agent")
    asyncio.run(_run(agents, stop_timeout))
    if errors := [a.error for a in agents if isinstance(a.error, Exception)]:
        raise ExceptionGroup("agents failed", errors)


async def _run(agents: tuple[BaseAgent, ...], stop_timeout: float) -> None:
    loop = asyncio.get_running_loop()
    shutdown = asyncio.Event()
    signals = (signal.SIGINT, signal.SIGTERM)
    use_signals = threading.current_thread() is threading.main_thread()
    if use_signals:
        for sig in signals:
            loop.add_signal_handler(sig, shutdown.set)
    started: list[BaseAgent] = []
    try:
        for agent in agents:
            started.append(agent)
            await agent.start()
        finished = asyncio.gather(*(a.wait() for a in agents))
        stop_requested = asyncio.ensure_future(shutdown.wait())
        await asyncio.wait(
            {finished, stop_requested}, return_when=asyncio.FIRST_COMPLETED
        )
        for task in (finished, stop_requested):
            task.cancel()
        if shutdown.is_set():
            logger.info("Shutdown signal received, stopping agents")
    finally:
        for agent in reversed(started):
            await agent.stop(stop_timeout)
        if use_signals:
            for sig in signals:
                loop.remove_signal_handler(sig)


class ThreadedAgent:
    """Runs one async agent on its own thread and event loop, for sync code.

    ::

        with ThreadedAgent(agent):  # start() ... stop()
            ...

    The thread ends when :meth:`stop` is called or when the agent finishes on its
    own (an agent that FAILED is stopped, so its :meth:`~BaseAgent.on_stop` runs).
    """

    def __init__(self, agent: BaseAgent, *, stop_timeout: float = 10.0):
        self.agent = agent
        self.stop_timeout = stop_timeout
        self._thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._ready = threading.Event()

    @property
    def state(self) -> AgentState:
        return self.agent.state

    def start(self, timeout: float | None = None) -> None:
        """Start the agent; returns once it is RUNNING. Raises what start raised."""
        if self._thread is not None:
            raise RuntimeError(f"{self.agent.name} was already started")
        self._thread = threading.Thread(
            target=asyncio.run,
            args=(self._main(),),
            name=f"rai-{self.agent.name}",
            daemon=True,
        )
        self._thread.start()
        self._ready.wait()
        try:
            self._submit(self.agent.start()).result(timeout)
        except BaseException:
            self.stop()
            raise

    async def _main(self) -> None:
        self._loop = asyncio.get_running_loop()
        self._ready.set()
        await self.agent.wait()
        await self.agent.stop(self.stop_timeout)

    def _submit(self, coro: Coroutine[Any, Any, None]) -> Future[None]:
        assert self._loop is not None
        return asyncio.run_coroutine_threadsafe(coro, self._loop)

    def stop(self, timeout: float | None = None) -> None:
        """Stop the agent and join its thread. Idempotent; a no-op before start."""
        timeout = self.stop_timeout if timeout is None else timeout
        if self._thread is None:
            with self.agent._lock:
                if self.agent.state is AgentState.CREATED:
                    self.agent._transition(AgentState.STOPPED)
            return
        coro = self.agent.stop(timeout)
        try:
            self._submit(coro)
        except RuntimeError:  # loop closed: the agent has already finished
            coro.close()
        # The thread ends once the agent is STOPPED. agent.stop() is bounded by
        # 2 * timeout, unless run() blocks the event loop with sync code.
        if not self.join(2 * timeout + 1):
            logger.warning("%s: thread did not end, abandoning it", self.agent.name)

    def join(self, timeout: float | None = None) -> bool:
        """Wait for the agent thread to end. Returns False on timeout."""
        if self._thread is None:
            return True
        self._thread.join(timeout)
        return not self._thread.is_alive()

    def __enter__(self) -> Self:
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.stop()

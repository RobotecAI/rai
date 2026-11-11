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

import asyncio
import os
import signal
import subprocess
import sys
import textwrap
import threading
import time

import pytest
import rai
from rai.agents import (
    AgentState,
    BaseAgent,
    InvalidTransitionError,
    SyncAgent,
    ThreadedAgent,
)

S = AgentState


class Recorder(BaseAgent):
    """Records hook calls. run() blocks on a long, cancellable step."""

    def __init__(self, name: str = "rec", log: list[str] | None = None):
        super().__init__(name=name)
        self.log = log if log is not None else []
        self.step_cancelled = False

    async def on_start(self):
        self.log.append(f"{self.name}.on_start")

    async def run(self):
        try:
            await asyncio.sleep(1000)  # e.g. a long tool call
        except asyncio.CancelledError:
            self.step_cancelled = True
            raise

    async def on_stop(self):
        self.log.append(f"{self.name}.on_stop")


class Crashing(BaseAgent):
    async def run(self):
        raise ValueError("boom")


class Finite(BaseAgent):
    async def run(self):
        await asyncio.sleep(0.01)


def _rai_threads() -> list[threading.Thread]:
    return [t for t in threading.enumerate() if t.name.startswith("rai-")]


def test_happy_path_transitions():
    async def main():
        agent = Recorder()
        assert agent.state is S.CREATED
        await agent.start()
        assert agent.state is S.RUNNING
        await agent.stop()
        assert agent.state is S.STOPPED
        assert agent.log == ["rec.on_start", "rec.on_stop"]

    asyncio.run(main())


@pytest.mark.timeout(5)
def test_stop_before_start_is_a_noop():
    agent = Recorder()
    asyncio.run(agent.stop())
    assert agent.state is S.STOPPED
    assert agent.log == []  # on_stop never runs for an agent that never started


def test_double_stop_is_idempotent():
    async def main():
        agent = Recorder()
        await agent.start()
        await asyncio.gather(agent.stop(), agent.stop())
        await agent.stop()
        assert agent.state is S.STOPPED
        assert agent.log.count("rec.on_stop") == 1

    asyncio.run(main())


@pytest.mark.parametrize("first", ["start", "stop"])
def test_restart_is_rejected(first: str):
    async def main():
        agent = Recorder()
        if first == "start":
            await agent.start()
            with pytest.raises(InvalidTransitionError, match="running to starting"):
                await agent.start()
        await agent.stop()
        with pytest.raises(InvalidTransitionError, match="stopped to starting"):
            await agent.start()

    asyncio.run(main())


@pytest.mark.timeout(5)
def test_stop_cancels_long_running_step():
    async def main():
        agent = Recorder()
        await agent.start()
        await asyncio.sleep(0.05)
        t0 = time.monotonic()
        await agent.stop()
        assert time.monotonic() - t0 < 1
        assert agent.step_cancelled

    asyncio.run(main())


@pytest.mark.timeout(5)
def test_stop_timeout_abandons_task_that_ignores_cancel():
    class Stubborn(BaseAgent):
        async def run(self):
            try:
                await asyncio.sleep(1000)
            except asyncio.CancelledError:
                await asyncio.sleep(2)  # slow cleanup that ignores the cancel

    async def main():
        agent = Stubborn()
        await agent.start()
        t0 = time.monotonic()
        await agent.stop(timeout=0.2)
        assert time.monotonic() - t0 < 1
        assert agent.state is S.STOPPED

    asyncio.run(main())


def test_exception_in_run_fails_then_stop_cleans_up():
    class CrashingRecorder(Recorder):
        async def run(self):
            raise ValueError("boom")

    async def main():
        agent = CrashingRecorder()
        await agent.start()
        await agent.wait()
        assert agent.state is S.FAILED
        assert isinstance(agent.error, ValueError)
        await agent.stop()
        assert agent.state is S.STOPPED
        assert agent.log == ["rec.on_start", "rec.on_stop"]

    asyncio.run(main())


def test_exception_in_on_start_is_raised_by_start():
    class BadStart(BaseAgent):
        async def on_start(self):
            raise RuntimeError("no device")

    async def main():
        agent = BadStart()
        with pytest.raises(RuntimeError, match="no device"):
            await agent.start()
        assert agent.state is S.FAILED
        await agent.stop()
        assert agent.state is S.STOPPED

    asyncio.run(main())


def test_agent_whose_run_returns_stops_itself():
    async def main():
        agent = Finite()
        await agent.start()
        await asyncio.wait_for(agent.wait(), 1)
        assert agent.state is S.STOPPED

    asyncio.run(main())


def test_stop_during_on_start():
    class SlowStart(Recorder):
        async def on_start(self):
            await asyncio.sleep(1000)

    async def main():
        agent = SlowStart()
        start = asyncio.create_task(agent.start())
        await asyncio.sleep(0.05)
        assert agent.state is S.STARTING
        await agent.stop()
        await start  # returns instead of raising CancelledError
        assert agent.state is S.STOPPED

    asyncio.run(main())


class Blocking(SyncAgent):
    def __init__(self):
        super().__init__()
        self.calls: list[str] = []

    def setup(self):
        self.calls.append("setup")

    def teardown(self):
        self.calls.append("teardown")


@pytest.mark.timeout(5)
def test_threaded_agent_start_stop_join_leaves_no_threads():
    before = set(threading.enumerate())
    agent = Blocking()
    with ThreadedAgent(agent) as threaded:
        assert threaded.state is S.RUNNING
        assert _rai_threads()
    assert agent.state is S.STOPPED
    assert agent.calls == ["setup", "teardown"]
    assert threaded.join(0)
    for t in set(threading.enumerate()) - before:
        t.join(1)  # the setup/teardown helper threads are already done
    assert set(threading.enumerate()) <= before
    threaded.stop()  # idempotent


@pytest.mark.timeout(5)
def test_threaded_agent_thread_ends_when_agent_fails():
    agent = Crashing()
    threaded = ThreadedAgent(agent)
    threaded.start()
    assert threaded.join(2)
    assert agent.state is S.STOPPED
    assert isinstance(agent.error, ValueError)


@pytest.mark.timeout(5)
def test_threaded_agent_start_failure_is_raised():
    class BadSetup(SyncAgent):
        def setup(self):
            raise OSError("no mic")

    threaded = ThreadedAgent(BadSetup())
    with pytest.raises(OSError, match="no mic"):
        threaded.start()
    assert threaded.join(0)
    assert threaded.state is S.STOPPED


@pytest.mark.timeout(5)
def test_run_returns_when_all_agents_finish():
    agents = [Finite(), Finite()]
    rai.run(*agents)
    assert [a.state for a in agents] == [S.STOPPED, S.STOPPED]


@pytest.mark.timeout(5)
def test_run_raises_agent_errors():
    with pytest.raises(ExceptionGroup) as info:
        rai.run(Crashing(), Finite())
    assert [type(e) for e in info.value.exceptions] == [ValueError]


def test_run_rejects_no_agents():
    with pytest.raises(ValueError, match="at least one agent"):
        rai.run()


@pytest.mark.timeout(30)
@pytest.mark.parametrize("sig", [signal.SIGINT, signal.SIGTERM])
def test_run_stops_agents_on_signal(sig: signal.Signals):
    script = textwrap.dedent(
        """
        import rai
        from rai.agents import BaseAgent

        class Agent(BaseAgent):
            async def on_start(self):
                print(f"{self.name} started", flush=True)

            async def on_stop(self):
                print(f"{self.name} stopped", flush=True)

        rai.run(Agent(name="a"), Agent(name="b"))
        print("exited", flush=True)
        """
    )
    proc = subprocess.Popen(
        [sys.executable, "-c", script],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=os.environ.copy(),
    )
    assert proc.stdout is not None
    lines = []
    while "b started" not in lines:
        line = proc.stdout.readline()
        assert line, f"process ended early: {lines}"
        lines.append(line.strip())
    proc.send_signal(sig)
    out, _ = proc.communicate(timeout=10)
    assert proc.returncode == 0, out
    events = [ln for ln in out.splitlines() if ln.endswith(("stopped", "exited"))]
    assert events == ["b stopped", "a stopped", "exited"]

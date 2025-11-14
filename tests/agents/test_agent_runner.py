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

import signal
import threading
import time

from rai.agents.base import BaseAgent
from rai.agents.runner import AgentRunner


class DummyAgent(BaseAgent):
    def __init__(self):
        super().__init__()
        self.stop_called = False

    def run(self):
        pass

    def stop(self):
        self.stop_called = True


def test_agent_runner_wait_for_shutdown_stops_agents(monkeypatch):
    agents = [DummyAgent(), DummyAgent()]
    runner = AgentRunner(agents)

    registered_handlers: dict[int, signal.Handlers] = {}

    def fake_signal(signum, handler):
        registered_handlers[signum] = handler

    monkeypatch.setattr(signal, "signal", fake_signal)

    thread = threading.Thread(target=runner.wait_for_shutdown)
    thread.start()

    for _ in range(50):
        if (
            signal.SIGINT in registered_handlers
            and signal.SIGTERM in registered_handlers
        ):
            break
        time.sleep(0.01)

    handler = registered_handlers[signal.SIGINT]
    handler(signal.SIGINT, None)

    thread.join(timeout=1)

    assert thread.is_alive() is False
    assert all(agent.stop_called for agent in agents)

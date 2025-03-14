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
from threading import Event
from typing import List

from rai.agents.base import BaseAgent


# Temporary solution until a Runner is implemented
def wait_for_shutdown(agents: List[BaseAgent]):
    """Blocks execution until shutdown signal (SIGINT/SIGTERM) is received.

    Args:
        agent: Agent instance implementing stop() method

    Note:
        Ensures graceful shutdown of the agent and ROS2 node on interrupt.
        Handles both SIGINT (Ctrl+C) and SIGTERM signals.
    """
    shutdown_event = Event()

    def signal_handler(signum, frame):
        shutdown_event.set()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        shutdown_event.wait()
    finally:
        for agent in agents:
            agent.stop()

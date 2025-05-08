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

import logging
import signal
from threading import Event
from typing import List

from rai.agents.base import BaseAgent

logger = logging.getLogger(__name__)


def wait_for_shutdown(agents: List[BaseAgent]):
    """Block until a shutdown signal (SIGINT or SIGTERM) is received, ensuring graceful termination.

    Parameters
    ----------
    agents : List[BaseAgent]
        List of running agents to be stopped on shutdown.

    Notes
    -----
    This method ensures a graceful shutdown of both the agent and the ROS2 node upon receiving
    an interrupt signal (SIGINT, e.g., Ctrl+C) or SIGTERM. It installs signal handlers to
    capture these events and invokes the agent's ``stop()`` method as part of the shutdown process.
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


def run_agents(agents: List[BaseAgent]):
    """Runs the agents in the background.

    Args:
        agents: List of agent instances
    """
    logger.info(
        "run_agents is an experimental function. \
                   If you believe that your agents are not running properly, \
                   please run them separately (in different processes)."
    )
    for agent in agents:
        agent.run()


class AgentRunner:
    """Runs the agents in the background.

    Parameters
    ----------
    agents : List[BaseAgent]
        List of agent instances
    """

    def __init__(self, agents: List[BaseAgent]):
        self.agents = agents
        self.logger = logging.getLogger(__name__)

    def run(self):
        self.logger.info(
            f"{self.__class__.__name__}.{self.run.__name__} is an experimental function. \
                            If you believe that your agents are not running properly, \
                            please run them separately (in different processes)."
        )
        for agent in self.agents:
            agent.run()

    def run_and_wait_for_shutdown(self):
        self.run()
        self.wait_for_shutdown()

    def wait_for_shutdown(self):
        shutdown_event = Event()

        def signal_handler(signum, frame):
            shutdown_event.set()

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        try:
            shutdown_event.wait()
        finally:
            for agent in self.agents:
                agent.stop()

    def stop(self):
        for agent in self.agents:
            agent.stop()

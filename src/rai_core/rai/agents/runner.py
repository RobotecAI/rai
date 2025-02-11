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
import logging
import threading

from rai.agents.base import BaseAgent


class AgentRunner:
    """
    Manages and runs a collection of agents.

    Parameters
    ----------
    agents : list of BaseAgent
        A list of agent instances that implement `run` and `stop` methods.

    Attributes
    ----------
    agents : list of BaseAgent
        The list of agents managed by this runner.
    """

    def __init__(self, agents: list[BaseAgent]):
        """
        Initializes the AgentRunner with a list of agents.

        Parameters
        ----------
        agents : list of BaseAgent
            A list of agent instances that will be managed and executed.
        """
        self.agents = agents

    def run_indefinitely(self):
        """
        Starts all agents and keeps them running indefinitely.

        This method runs each agent's `run` method and waits indefinitely for a stop signal.
        If a `KeyboardInterrupt` is received, it logs the interruption and stops all agents.
        """
        for agent in self.agents:
            agent.run()

        stop_signal = threading.Event()

        try:
            stop_signal.wait()
        except KeyboardInterrupt:
            logging.info("KeyboardInterrupt received! Shutting down...")

        for agent in self.agents:
            agent.stop()

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
from abc import ABC, abstractmethod


class BaseAgent(ABC):
    def __init__(self):
        """Initializes a new agent instance and sets up logging with the class name."""
        self.logger = logging.getLogger(self.__class__.__name__)

    @abstractmethod
    def run(self):
        """Starts the agent's main execution loop.
        In some cases, concrete run implementation may not be needed.
        In that case use pass as a placeholder."""
        pass

    @abstractmethod
    def stop(self):
        """Gracefully terminates the agent's execution and cleans up resources."""
        pass

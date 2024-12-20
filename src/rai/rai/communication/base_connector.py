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
from typing import Callable
from uuid import uuid4

class BaseMessage(ABC):
    ...

class BaseConnector(ABC):

    def _generate_handle(self) -> str:
        return str(uuid4())

    @abstractmethod
    def send_message(self, msg: BaseMessage, target: str) -> None:
        ...

    @abstractmethod
    def receive_message(self, source: str) -> BaseMessage:
        ...

    @abstractmethod
    def send_and_wait(self, target: str) -> BaseMessage:
        ...
    
    @abstractmethod
    def start_action(self, target: str, on_feedback: Callable, on_finish: Callable = lambda _: None)  -> str:
        ...
    
    @abstractmethod
    def terminate_action(self, action_handle: str):
        ...

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

from abc import abstractmethod
from typing import Any, Callable, Dict, Generic, Optional, TypeVar
from uuid import uuid4


class BaseMessage:
    payload: Any
    metadata: Dict[str, Any]

    def __init__(
        self,
        payload: Any,
        metadata: Optional[Dict[str, Any]] = None,
        *args: Any,
        **kwargs: Any,
    ):
        self.payload = payload
        if metadata is None:
            self.metadata = {}
        else:
            self.metadata = metadata


T = TypeVar("T", bound=BaseMessage)


class BaseConnector(Generic[T]):

    def _generate_handle(self) -> str:
        return str(uuid4())

    @abstractmethod
    def send_message(self, message: T, target: str):
        pass

    @abstractmethod
    def receive_message(self, source: str, timeout_sec: float = 1.0) -> T:
        pass

    @abstractmethod
    def service_call(self, message: T, target: str, timeout_sec: float = 1.0) -> T:
        pass

    @abstractmethod
    def start_action(
        self,
        action_data: Optional[T],
        target: str,
        on_feedback: Callable,
        on_done: Callable,
        timeout_sec: float = 1.0,
    ) -> str:
        pass

    @abstractmethod
    def terminate_action(self, action_handle: str):
        pass

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
from abc import abstractmethod
from typing import Any, Callable, Dict, Generic, Optional, TypeVar
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field


class BaseMessage(BaseModel):
    metadata: Dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(arbitrary_types_allowed=True)


T = TypeVar("T", bound=BaseMessage)


class BaseConnector(Generic[T]):
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

    def _generate_handle(self) -> str:
        return str(uuid4())

    @abstractmethod
    def send_message(self, message: T, target: str, **kwargs: Any):
        pass

    @abstractmethod
    def receive_message(self, source: str, timeout_sec: float, **kwargs: Any) -> T:
        pass

    @abstractmethod
    def service_call(
        self, message: T, target: str, timeout_sec: float, **kwargs: Any
    ) -> BaseMessage:
        pass

    @abstractmethod
    def create_service(
        self,
        service_name: str,
        on_request: Callable,
        on_done: Optional[Callable] = None,
        **kwargs: Any,
    ) -> str:
        pass

    @abstractmethod
    def create_action(
        self,
        action_name: str,
        generate_feedback_callback: Callable,
        **kwargs: Any,
    ) -> str:
        pass

    @abstractmethod
    def start_action(
        self,
        action_data: Optional[T],
        target: str,
        on_feedback: Callable,
        on_done: Callable,
        timeout_sec: float,
        **kwargs: Any,
    ) -> str:
        pass

    @abstractmethod
    def terminate_action(self, action_handle: str, **kwargs: Any):
        pass

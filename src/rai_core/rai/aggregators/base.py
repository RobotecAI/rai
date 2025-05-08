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

from abc import ABC, abstractmethod
from collections import deque
from typing import Deque, Generic, List, TypeVar

from langchain_core.messages import BaseMessage

T = TypeVar("T")


class BaseAggregator(ABC, Generic[T]):
    """
    Interface for aggregators.
    """

    def __init__(self, max_size: int | None = None) -> None:
        super().__init__()
        self._buffer: Deque[T] = deque()
        self.max_size = max_size

    def __call__(self, msg: T) -> None:
        if self.max_size is not None and len(self._buffer) >= self.max_size:
            self._buffer.popleft()
        self._buffer.append(msg)

    @abstractmethod
    def get(self) -> BaseMessage | None:
        """Returns the outcome of processing the aggregated message"""
        pass

    def clear_buffer(self) -> None:
        """Clears the buffer of messages"""
        self._buffer.clear()

    def get_buffer(self) -> List[T]:
        """Returns a copy of the buffer of messages"""
        return list(self._buffer)

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(len={len(self._buffer)})"

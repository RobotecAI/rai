# Copyright (C) 2025 Julia Jia
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
from typing import Any, Dict, List, Optional


class BaseMemory(ABC):
    """Abstract base class for agent memory systems."""

    @abstractmethod
    def store(
        self, key: str, value: Any, metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Store a value with optional metadata. Returns storage ID."""
        pass

    @abstractmethod
    def retrieve(
        self, query: str, filters: Optional[Dict[str, Any]] = None
    ) -> List[Any]:
        """Retrieve values matching query and filters.

        Designed for vector database use cases where query is text to embed
        for similarity search, and filters are metadata constraints.
        Not suitable for spatial databases which require concrete query methods
        (e.g., query_by_location, query_by_region).
        """
        pass

    @abstractmethod
    def delete(self, key: str) -> bool:
        """Delete a stored value. Returns success status."""
        pass

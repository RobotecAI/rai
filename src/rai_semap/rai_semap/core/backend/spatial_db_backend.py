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
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from geometry_msgs.msg import Point

if TYPE_CHECKING:
    from rai_semap.core.semantic_map_memory import SemanticAnnotation


class SpatialDBBackend(ABC):
    """Abstract backend for spatial database operations."""

    @abstractmethod
    def init_schema(self) -> None:
        """Initialize database schema with spatial extensions."""
        pass

    @abstractmethod
    def insert_annotation(self, annotation: "SemanticAnnotation") -> str:
        """Insert annotation with spatial indexing."""
        pass

    @abstractmethod
    def spatial_query(
        self, center: Point, radius: float, filters: Optional[Dict[str, Any]] = None
    ) -> List["SemanticAnnotation"]:
        """Execute spatial query with optional filters."""
        pass

    @abstractmethod
    def delete_annotation(self, annotation_id: str) -> bool:
        """Delete annotation by ID. Returns success status."""
        pass

    @abstractmethod
    def delete_all_annotations(self, location_id: Optional[str] = None) -> int:
        """Delete all annotations, optionally filtered by location_id.

        Args:
            location_id: If provided, only delete annotations for this location.
                         If None, delete all annotations.

        Returns:
            Number of annotations deleted.
        """
        pass

    @abstractmethod
    def update_annotation(self, annotation: "SemanticAnnotation") -> bool:
        """Update existing annotation by ID. Returns success status."""
        pass

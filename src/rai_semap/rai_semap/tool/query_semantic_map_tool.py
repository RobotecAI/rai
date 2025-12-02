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

from typing import Optional, Type

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from rai_semap.core.semantic_map_memory import SemanticMapMemory


class QuerySemanticMapToolInput(BaseModel):
    """Input schema for QuerySemanticMapTool."""

    query: str = Field(description="Natural language query about object locations")
    room: Optional[str] = Field(
        default=None, description="Optional room or region name"
    )


class QuerySemanticMapTool(BaseTool):
    """Tool for querying semantic map for object locations."""

    name: str = "query_semantic_map"
    description: str = "Query the semantic map for object locations"

    args_schema: Type[QuerySemanticMapToolInput] = QuerySemanticMapToolInput

    memory: SemanticMapMemory

    def _run(self, query: str, room: Optional[str] = None) -> str:
        """Execute semantic map query."""
        pass

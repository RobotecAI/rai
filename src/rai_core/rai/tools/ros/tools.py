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


import json
import logging
import time
from typing import Any, Dict, Type

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class AddDescribedWaypointToDatabaseToolInput(BaseModel):
    """Input for the add described waypoint to database tool."""

    x: float = Field(..., description="X coordinate of the waypoint")
    y: float = Field(..., description="Y coordinate of the waypoint")
    z: float = Field(0.0, description="Z coordinate of the waypoint")
    text: str = Field(
        ...,
        description="Text to display on the waypoint (very short, one or two words)",
    )


class AddDescribedWaypointToDatabaseTool(BaseTool):
    """Add described waypoint to the database tool."""

    name: str = "AddDescribedWaypointToDatabaseTool"
    description: str = (
        "A tool for adding a described waypoint to the database for later use. "
    )

    args_schema: Type[AddDescribedWaypointToDatabaseToolInput] = (
        AddDescribedWaypointToDatabaseToolInput
    )

    map_database: str = ""

    def _run(self, x: float, y: float, z: float = 0.0, text: str = ""):
        try:
            self.update_map_database(x, y, z, text)
        except FileNotFoundError:
            logger.warn(f"Database {self.map_database} not found.")
        return {"content": "Waypoint added successfully"}

    def update_map_database(
        self,
        x: float,
        y: float,
        z: float,
        text: str,
        frame_id: str = "map",
        child_frame_id: str = "base_link",
    ):
        with open(self.map_database, "r") as file:
            map_database = json.load(file)

        data: Dict[str, Any] = {
            "header": {"frame_id": frame_id, "stamp": time.time()},
            "child_frame_id": child_frame_id,
            "transform": {
                "translation": {"x": x, "y": y, "z": z},
                "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
            },
            "text": text,
        }
        map_database.append(data)

        with open(self.map_database, "w") as file:
            json.dump(map_database, file, indent=2)

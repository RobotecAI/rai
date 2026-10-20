# Copyright (C) 2026 Robotec.AI
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

from math import inf
from typing import Annotated, Optional, Tuple

from pydantic import BaseModel, Field, model_validator
from typing_extensions import Self

# a plain float tuple would accept NaN, which silently disables every comparison
Coordinate = Annotated[float, Field(allow_inf_nan=False)]
Bounds = Optional[Tuple[Coordinate, Coordinate, Coordinate]]


class WorkspaceBounds(BaseModel):
    """Optional (x, y, z) box a navigation goal has to fall inside."""

    workspace_bounds_min: Bounds = Field(
        default=None,
        description="Optional minimum (x, y, z) workspace bounds for navigation goals",
    )
    workspace_bounds_max: Bounds = Field(
        default=None,
        description="Optional maximum (x, y, z) workspace bounds for navigation goals",
    )

    @model_validator(mode="after")
    def _check_bounds_ordering(self) -> Self:
        lo, hi = self.workspace_bounds_min, self.workspace_bounds_max
        if (
            lo is not None
            and hi is not None
            and any(low > high for low, high in zip(lo, hi))
        ):
            raise ValueError(
                f"workspace_bounds_min {lo} exceeds workspace_bounds_max {hi}"
            )
        return self

    def reject_out_of_bounds(self, x: float, y: float, z: float) -> None:
        """Raise if the goal falls outside the configured workspace."""
        lo = self.workspace_bounds_min or (-inf, -inf, -inf)
        hi = self.workspace_bounds_max or (inf, inf, inf)
        for axis, value, low, high in zip("xyz", (x, y, z), lo, hi):
            if not low <= value <= high:
                raise ValueError(
                    f"Goal rejected: {axis}={value} outside the workspace "
                    f"[{low}, {high}]"
                )

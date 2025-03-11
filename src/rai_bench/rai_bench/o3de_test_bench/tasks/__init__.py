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
from rai_bench.o3de_test_bench.tasks.build_tower_task import (
    BuildBlueCubeTowerTask,
    BuildCubeTowerTask,
    BuildRedCubeTowerTask,
    BuildYellowCubeTowerTask,
)
from rai_bench.o3de_test_bench.tasks.group_vegetables_task import GroupVegetablesTask
from rai_bench.o3de_test_bench.tasks.move_object_to_left_task import (
    MoveObjectsToLeftTask,
)
from rai_bench.o3de_test_bench.tasks.place_cubes_task import PlaceCubesTask

__all__ = [
    "BuildBlueCubeTowerTask",
    "BuildCubeTowerTask",
    "BuildRedCubeTowerTask",
    "BuildYellowCubeTowerTask",
    "GroupVegetablesTask",
    "MoveObjectsToLeftTask",
    "PlaceCubesTask",
]

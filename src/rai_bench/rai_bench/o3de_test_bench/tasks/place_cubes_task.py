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

from rai_bench.benchmark_model import (
    EntitiesMismatchException,
    Task,
)
from rai_sim.o3de.o3de_bridge import SimulationBridge
from rai_sim.simulation_bridge import SimulationConfig, SimulationConfigT


class PlaceCubesTask(Task):
    def get_prompt(self) -> str:
        return "Manipulate objects, so that all cubes are adjacent to at least one cube"

    def validate_config(self, simulation_config: SimulationConfig) -> bool:
        cube_types = ["red_cube", "blue_cube", "yellow_cube"]
        cubes_num = 0
        for ent in simulation_config.entities:
            if ent.prefab_name in cube_types:
                cubes_num += 1
                if cubes_num > 1:
                    return True

        return False

    def calculate_result(
        self, simulation_bridge: SimulationBridge[SimulationConfigT]
    ) -> float:
        # TODO (jm) extract common logic to some parent manipulation task?
        initially_misplaced_now_correct = 0  # when the object which was in the incorrect place at the start, is in a correct place at the end
        initially_misplaced_still_incorrect = 0  # when the object which was in the incorrect place at the start, is in a incorrect place at the end
        initially_correct_still_correct = 0  # when the object which was in the correct place at the start, is in a correct place at the end
        initially_correct_now_incorrect = 0  # when the object which was in the correct place at the start, is in a incorrect place at the end

        cube_types = ["red_cube", "blue_cube", "yellow_cube"]
        scene_state = simulation_bridge.get_scene_state()

        initial_cubes = self.filter_entities_by_prefab_type(
            simulation_bridge.spawned_entities, prefab_types=cube_types
        )
        final_cubes = self.filter_entities_by_prefab_type(
            scene_state.entities, prefab_types=cube_types
        )
        num_of_objects = len(initial_cubes)

        if num_of_objects != len(final_cubes):
            raise EntitiesMismatchException(
                "Number of initially spawned entities does not match number of entities present at the end."
            )

        else:
            ini_poses = [cube.pose for cube in initial_cubes]
            final_poses = [cube.pose for cube in final_cubes]
            # NOTE the specific coords that refer to for example
            # middle of the table can differ across simulations,
            # take that into consideration
            self.logger.debug(f"initial positions: {initial_cubes}")
            self.logger.debug(f"current positions: {final_cubes}")
            for i, ini_cube in enumerate(initial_cubes):
                for j, final_cube in enumerate(final_cubes):
                    if ini_cube.name == final_cube.name:
                        was_adjacent_initially = self.is_adjacent_to_any(
                            ini_cube.pose,
                            [p for p in ini_poses if p != ini_cube.pose],
                            0.15,
                        )
                        is_adjacent_finally = self.is_adjacent_to_any(
                            final_cube.pose,
                            [p for p in final_poses if p != final_cube.pose],
                            0.15,
                        )
                        if not was_adjacent_initially and is_adjacent_finally:
                            initially_misplaced_now_correct += 1
                        elif not was_adjacent_initially and not is_adjacent_finally:
                            initially_misplaced_still_incorrect += 1
                        elif was_adjacent_initially and is_adjacent_finally:
                            initially_correct_still_correct += 1
                        elif was_adjacent_initially and not is_adjacent_finally:
                            initially_correct_now_incorrect += 1

                        break
                else:
                    raise EntitiesMismatchException(
                        f"Entity with name: {ini_cube.name} which was present in initial scene, not found in final scene."
                    )

            self.logger.info(
                f"initially_misplaced_now_correct: {initially_misplaced_now_correct}, initially_misplaced_still_incorrect: {initially_misplaced_still_incorrect}, initially_correct_still_correct: {initially_correct_still_correct}, initially_correct_now_incorrect: {initially_correct_now_incorrect}"
            )
            return (
                initially_misplaced_now_correct + initially_correct_still_correct
            ) / num_of_objects

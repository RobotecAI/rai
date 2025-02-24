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
from rai_sim.o3de.o3de_bridge import (
    SimulationBridge,
)
from rai_sim.simulation_bridge import SimulationConfig


class GrabCarrotTask(Task):
    def get_prompt(self) -> str:
        return "Manipulate objects, so that all carrots to the left side of the table (positive y)"

    def validate_config(self, simulation_config: SimulationConfig) -> bool:
        for ent in simulation_config.entities:
            if ent.prefab_name == "carrot":
                return True

        return False

    def calculate_result(self, simulation_bridge: SimulationBridge) -> float:
        # TODO (jm) extract common logic to some parent manipulation task?
        initially_misplaced_now_correct = 0  # when the object which was in the incorrect place at the start, is in a correct place at the end
        initially_misplaced_still_incorrect = 0  # when the object which was in the incorrect place at the start, is in a incorrect place at the end
        initially_correct_still_correct = 0  # when the object which was in the correct place at the start, is in a correct place at the end
        initially_correct_now_incorrect = 0  # when the object which was in the correct place at the start, is in a incorrect place at the end

        scene_state = simulation_bridge.get_scene_state()
        initial_carrots = self.filter_entities_by_prefab_type(
            simulation_bridge.spawned_entities, prefab_types=["carrot"]
        )
        final_carrots = self.filter_entities_by_prefab_type(
            scene_state.entities, prefab_types=["carrot"]
        )
        num_initial_carrots = len(initial_carrots)

        if num_initial_carrots != len(final_carrots):
            raise EntitiesMismatchException(
                "Number of initially spawned entities does not match number of entities present at the end."
            )

        else:
            self.logger.debug(f"initial positions: {initial_carrots}")
            self.logger.debug(f"current positions: {final_carrots}")
            for ini_carrot in initial_carrots:
                for final_carrot in final_carrots:
                    if ini_carrot.name == final_carrot.name:
                        initial_y = ini_carrot.pose.translation.y
                        final_y = final_carrot.pose.translation.y
                        # NOTE the specific coords that refer to for example
                        # middle of the table can differ across simulations,
                        # take that into consideration
                        if (
                            initial_y <= 0.0
                        ):  # Carrot started in the incorrect place (right side)
                            if final_y >= 0.0:
                                initially_misplaced_now_correct += (
                                    1  # Moved to correct side
                                )
                            else:
                                initially_misplaced_still_incorrect += (
                                    1  # Stayed on incorrect side
                                )
                        else:  # Carrot started in the correct place (left side)
                            if final_y >= 0.0:
                                initially_correct_still_correct += (
                                    1  # Stayed on correct side
                                )
                            else:
                                initially_correct_now_incorrect += (
                                    1  # Moved incorrectly to the wrong side
                                )
                        break
                else:
                    raise EntitiesMismatchException(
                        f"Entity with name: {ini_carrot.name} which was present in initial scene, not found in final scene."
                    )

            self.logger.info(
                f"initially_misplaced_now_correct: {initially_misplaced_now_correct}, initially_misplaced_still_incorrect: {initially_misplaced_still_incorrect}, initially_correct_still_correct: {initially_correct_still_correct}, initially_correct_now_incorrect: {initially_correct_now_incorrect}"
            )
            return (
                initially_misplaced_now_correct + initially_correct_still_correct
            ) / num_initial_carrots

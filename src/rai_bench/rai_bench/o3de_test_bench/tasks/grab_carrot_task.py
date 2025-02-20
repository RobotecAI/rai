from rai_bench.benchmark_model import (
    Task,
    EntitiesMismatchException,
)

from rai_sim.o3de.o3de_bridge import (
    SimulationBridge,
)


class GrabCarrotTask(Task):
    def get_prompt(self) -> str:
        return "Manipulate objects, so that all carrots to the left side of the table (positive y)"

    def calculate_result(self, simulation_bridge: SimulationBridge) -> float:
        corrected_objects = 0  # when the object which was in the incorrect place at the start, is in a correct place at the end
        misplaced_objects = 0  # when the object which was in the incorrect place at the start, is in a incorrect place at the end
        unchanged_correct = 0  # when the object which was in the correct place at the start, is in a correct place at the end
        displaced_objects = 0  # when the object which was in the correct place at the start, is in a incorrect place at the end

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

        if num_initial_carrots == 0:
            self.logger.info("No objects to manipulate, returning 1.0")
            return 1.0
        else:

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
                                corrected_objects += 1  # Moved to correct side
                            else:
                                misplaced_objects += 1  # Stayed on incorrect side
                        else:  # Carrot started in the correct place (left side)
                            if final_y >= 0.0:
                                unchanged_correct += 1  # Stayed on correct side
                            else:
                                displaced_objects += (
                                    1  # Moved incorrectly to the wrong side
                                )
                        break
                else:
                    raise EntitiesMismatchException(
                        f"Entity with name: {ini_carrot.name} which was present in initial scene, not found in final scene."
                    )

            self.logger.info(
                f"corrected_objects: {corrected_objects}, misplaced_objects: {misplaced_objects}, unchanged_correct: {unchanged_correct}, displaced_objects: {displaced_objects}"
            )
            return (corrected_objects + unchanged_correct) / num_initial_carrots

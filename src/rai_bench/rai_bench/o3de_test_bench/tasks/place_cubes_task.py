from rai_bench.benchmark_model import (
    Task,
    EntitiesMismatchException,
)

from rai_sim.o3de.o3de_bridge import SimulationBridge, SimulationConfig


class PlaceCubesTask(Task):
    def get_prompt(self) -> str:
        return "Manipulate objects, so that  all cubes are next to each other"

    def validate_scene(self, simulation_config: SimulationConfig) -> bool:
        cube_types = ["red_cube", "blue_cube", "yellow_cube"]
        cubes_num = 0
        for ent in simulation_config.entities:
            if ent.prefab_name in cube_types:
                cubes_num += 1
                if cubes_num > 1:
                    return True

        return False

    def calculate_result(self, simulation_bridge: SimulationBridge) -> float:
        corrected_objects = 0  # when the object which was in the incorrect place at the start, is in a correct place at the end
        misplaced_objects = 0  # when the object which was in the incorrect place at the start, is in a incorrect place at the end
        unchanged_correct = 0  # when the object which was in the correct place at the start, is in a correct place at the end
        displaced_objects = 0  # when the object which was in the correct place at the start, is in a incorrect place at the end

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
            for ini_cube in initial_cubes:
                for final_cube in final_cubes:
                    if ini_cube.name == final_cube.name:
                        was_adjacent_initially = self.is_adjacent_to_any(
                            ini_cube.pose, ini_poses, 0.1
                        )
                        is_adjacent_finally = self.is_adjacent_to_any(
                            final_cube.pose, final_poses, 0.1
                        )
                        if not was_adjacent_initially and is_adjacent_finally:
                            corrected_objects += 1
                        elif not was_adjacent_initially and not is_adjacent_finally:
                            misplaced_objects += 1
                        elif was_adjacent_initially and is_adjacent_finally:
                            unchanged_correct += 1
                        elif was_adjacent_initially and not is_adjacent_finally:
                            displaced_objects += 1

                        break
                else:
                    raise EntitiesMismatchException(
                        f"Entity with name: {ini_cube.name} which was present in initial scene, not found in final scene."
                    )

            self.logger.info(
                f"corrected_objects: {corrected_objects}, misplaced_objects: {misplaced_objects}, unchanged_correct: {unchanged_correct}, displaced_objects: {displaced_objects}"
            )
            return (corrected_objects + unchanged_correct) / num_of_objects

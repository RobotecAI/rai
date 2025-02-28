from typing import Dict, List, Set
from rai_bench.benchmark_model import EntitiesMismatchException  # type: ignore
from rai_sim.o3de.o3de_bridge import SimulationBridge  # type: ignore
from rai_sim.simulation_bridge import SimulationConfig, SpawnedEntity  # type: ignore
from rai_bench.o3de_test_bench.tasks.manipulation_task import ManipulationTask


class GroupVegetablesTask(ManipulationTask):
    obj_types = ["tomato", "apple", "corn", "carrot"]

    def get_prompt(self) -> str:
        return (
            "Manipulate objects, so that vegetables will be in separate clusters based on their types. "
            "Each cluster must: "
            "1. Contain ALL vegetables of the same type "
            "2. Contain ONLY vegetables of the same type "
            "3. Form a single connected group (all vegetables of the same type must be adjacent) "
            "4. Be completely separated from other clusters (objects of different types cannot be adjacent) "
        )

    def validate_config(self, simulation_config: SimulationConfig) -> bool:
        """Ensure that there are at least 2 types of vegetables"""
        veg_types = {
            ent.prefab_name
            for ent in simulation_config.entities
            if ent.prefab_name in self.obj_types
        }
        return len(veg_types) > 1

    def calculate_result(
        self, simulation_bridge: SimulationBridge[SimulationConfig]
    ) -> float:
        # NOTE for now if all veggies from same type don't form single cluster,
        # every veggie will be counted as misplaced
        # only when they form  a signle cluster and none of them is adjacent to other types
        # they will be counted as correctly placed
        # it can be modified in the future to count only part of veggies as correct
        self.reset_values()
        initial_veggies, current_veggies = self.get_initial_and_current_positions(
            simulation_bridge=simulation_bridge, object_types=self.obj_types
        )

        initial_veggies_by_type = self.group_entities_by_type(initial_veggies)
        current_veggies_by_type = self.group_entities_by_type(current_veggies)

        initially_properly_clustered: List[SpawnedEntity] = []
        currently_properly_clustered: List[SpawnedEntity] = []

        for veg_type, veggies in initial_veggies_by_type.items():
            neighbourhood_list = self.build_neighbourhood_list(veggies)
            clusters = self.find_clusters(neighbourhood_list)
            if len(clusters) == 1:
                # there is only 1 cluster so the 1st condition is matched
                # now check if every veggie from the cluster is ajacent only to
                # other veggies of same type
                if all(
                    self.check_neighbourhood_types(
                        neighbourhood=neighbourhood_list[veggie],
                        allowed_types=[veg_type],
                    )
                    for veggie in clusters[0]
                ):
                    initially_properly_clustered.extend(clusters[0])

        for veg_type, veggies in current_veggies_by_type.items():
            neighbourhood_list = self.build_neighbourhood_list(veggies)
            clusters = self.find_clusters(neighbourhood_list)
            if len(clusters) == 1:
                # there is only 1 cluster so the 1st condition is matched
                # now check if every veggie from the cluster is ajacent only to
                # other veggies of same type
                if all(
                    self.check_neighbourhood_types(
                        neighbourhood=neighbourhood_list[veggie],
                        allowed_types=[veg_type],
                    )
                    for veggie in clusters[0]
                ):
                    currently_properly_clustered.extend(clusters[0])

        self.initially_misplaced_now_correct = len(
            set(currently_properly_clustered) - set(initially_properly_clustered)
        )
        self.initially_misplaced_still_incorrect = len(
            set(initial_veggies) - set(initially_properly_clustered)
        )
        self.initially_correct_still_correct = len(
            set(initially_properly_clustered) & set(currently_properly_clustered)
        )
        self.initially_correct_now_incorrect = len(
            set(initially_properly_clustered) - set(currently_properly_clustered)
        )

        return (
            self.initially_misplaced_now_correct + self.initially_correct_still_correct
        ) / len(initial_veggies)

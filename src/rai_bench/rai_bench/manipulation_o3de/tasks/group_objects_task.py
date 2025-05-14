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
import logging
from typing import List, Tuple, Union

from rclpy.impl.rcutils_logger import RcutilsLogger

from rai_bench.manipulation_o3de.interfaces import (
    ManipulationTask,
)
from rai_sim.simulation_bridge import Entity, SceneConfig

loggers_type = Union[RcutilsLogger, logging.Logger]


class GroupObjectsTask(ManipulationTask):
    def __init__(
        self,
        obj_types: List[str],
        threshold_distance: float = 0.15,
        logger: loggers_type | None = None,
    ):
        """
        This task requires that objects of specified types form a single, well-defined cluster.

        Parameters
        ----------
        obj_types : List[str]
            A list of object types to be grouped into clusters. Only objects whose prefab names match
            one of these types will be evaluated.
        threshold_distance : float, optional
            The maximum distance between two objects (in meters) for them to be considered neighbours
            when building the neighbourhood list. Defaults to 0.15.
        """
        super().__init__(logger)
        self.obj_types = obj_types
        self.threshold_distance = threshold_distance

    @property
    def task_prompt(self) -> str:
        obj_names = ", ".join(obj + "s" for obj in self.obj_types).replace(
            "_", " "
        )  # create prompt, add s for plural form
        return (
            f"Manipulate objects so that {obj_names} form separate clusters based on their types. "
            "Each cluster must: "
            "1. Contain ALL objects of the same type "
            "2. Contain ONLY objects of the same type "
            "3. Form a single connected group "
            "4. Be completely separated from other clusters "
        )

    def check_if_required_objects_present(self, simulation_config: SceneConfig) -> bool:
        """
        Returns
        -------
        bool
            True if at least one entity of all specified object types are present, False otherwise.
        """
        object_types_present = {
            ent.prefab_name
            for ent in simulation_config.entities
            if ent.prefab_name in self.obj_types
        }

        return set(self.obj_types) <= object_types_present

    def calculate_correct(self, entities: List[Entity]) -> Tuple[int, int]:
        """
        Count correctly and incorrectly clustered objects based on clustering rules.

        Method first groups the entities by type.
        Then, using the specified threshold distance, it builds a neighbourhood list
        and identifies clusters using a DFS-based algorithm.
        A cluster is considered properly clustered if:
        1. Only one cluster is found for that type.
        2. All objects in the cluster have neighbours exclusively of the same type.
        If these conditions are met, the objects in that cluster are counted as correctly clustered.
        Otherwise, all objects of that type are counted as misclustered.

        Parameters
        ----------
        entities : List[Entity]
            List of all entities present in the simulation scene.

        Returns
        -------
        Tuple[int, int]
            A tuple where the first element is the number of correctly clustered objects
            and the second element is the number of misclustered objects.
        """
        properly_clustered: List[Entity] = []
        misclustered: List[Entity] = []

        neighbourhood_list = self.build_neighbourhood_list(
            entities, threshold_distance=self.threshold_distance
        )
        clusters = self.find_clusters(neighbourhood_list)
        selected_type_objects = self.filter_entities_by_object_type(
            entities=entities, object_types=self.obj_types
        )
        entities_by_type = self.group_entities_by_type(selected_type_objects)
        for obj_type, objects in entities_by_type.items():
            # Filter clusters that contain only entities of this type.
            clusters_of_type = [
                cluster
                for cluster in clusters
                if all(ent.prefab_name == obj_type for ent in cluster)
            ]
            # Check if exactly one cluster exists and it includes all objects of that type.
            if len(clusters_of_type) == 1 and len(objects) == len(clusters_of_type[0]):
                # Verify that every entity in this cluster has neighbours exclusively of the allowed type.
                properly_clustered.extend(clusters_of_type[0])
            else:
                # Either no cluster or more than one cluster means misclustering.
                misclustered.extend(objects)

        return len(properly_clustered), len(misclustered)

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
from pathlib import Path
from typing import List, Union

from rclpy.impl.rcutils_logger import RcutilsLogger

from rai_bench.benchmark_model import (  # type: ignore
    Benchmark,
    Scenario,
    Task,
)
from rai_bench.o3de_test_bench.tasks import (  # type: ignore
    MoveObjectsToLeftTask,
    PlaceCubesTask,
    PlaceObjectAtCoordTask,
)
from rai_sim.o3de.o3de_bridge import (  # type: ignore
    O3DExROS2SimulationConfig,
)

loggers_type = Union[RcutilsLogger, logging.Logger]


def trivial_scenarios(
    configs_dir: str, connector_path: str, logger: loggers_type | None
) -> List[Scenario[O3DExROS2SimulationConfig]]:
    """Packet of trivial scenarios. The grading is subjective.
    This packet contains easiest tasks with minimalistic scenes setups(1 object).
    In this packet:
        PlaceObjectAtCoordTask with large allowable_displacement
        MoveObjectsToLeftTask with only 1 object type

    This level of difficulty requires recognizing position of object and move it once

    Parameters
    ----------
    configs_dir : str
        path to directory with simulation configs
    connector_path : str
        path to connector config


    Returns
    -------
    List[Scenario[O3DExROS2SimulationConfig]]
        list of trivial scenarios
    """
    simulation_configs_paths: List[str] = [
        configs_dir + "1a.yaml",
        configs_dir + "1rc.yaml",
        configs_dir + "1t.yaml",
        configs_dir + "1yc.yaml",
        configs_dir + "1carrot.yaml",
    ]
    simulations_configs = [
        O3DExROS2SimulationConfig.load_config(Path(path), Path(connector_path))
        for path in simulation_configs_paths
    ]
    # place object at coords
    place_obj_types = [
        "apple",
        "carrot",
        "yellow_cube",
        "red_cube",
    ]
    target_coords = [(0.3, 0.3), (0.2, -0.4)]
    allowable_displacements = [0.1]  # large margin
    place_object_tasks: List[Task] = []
    for obj in place_obj_types:
        for coord in target_coords:
            for disp in allowable_displacements:
                place_object_tasks.append(
                    PlaceObjectAtCoordTask(obj, coord, disp, logger=logger)
                )
    easy_place_objects_scenarios = Benchmark.create_scenarios(
        tasks=place_object_tasks,
        simulation_configs=simulations_configs,
        simulation_configs_paths=simulation_configs_paths,
    )
    # move objects to the left
    object_groups = [["carrot"], ["red_cube"], ["tomato"], ["yellow_cube"]]

    move_to_left_tasks: List[Task] = [
        MoveObjectsToLeftTask(obj_types=objects, logger=logger)
        for objects in object_groups
    ]

    easy_move_to_left_scenarios = Benchmark.create_scenarios(
        tasks=move_to_left_tasks,
        simulation_configs=simulations_configs,
        simulation_configs_paths=simulation_configs_paths,
    )

    return [*easy_move_to_left_scenarios, *easy_place_objects_scenarios]


def easy_scenarios(
    configs_dir: str, connector_path: str, logger: loggers_type | None
) -> List[Scenario[O3DExROS2SimulationConfig]]:
    """Packet of easy scenarios. The grading is subjective.
    This packet contains easy tasks with scenes containg no more than 3 objects
    In this packet:
        PlaceObjectAtCoordTask with small allowable_displacement
        MoveObjectsToLeftTask with only 1 object type
        PlaceCubesTask with large threshold

    This level of difficulty requires recognizing proper type of object.
    Some scenarios will require moving more than 1 object or moving with more precision.

    Parameters
    ----------
    configs_dir : str
        path to directory with simulation configs
    connector_path : str
        path to connector config


    Returns
    -------
    List[Scenario[O3DExROS2SimulationConfig]]
        list of easy scenarios
    """
    simulation_configs_paths: List[str] = [
        configs_dir + "1a_1t.yaml",
        configs_dir + "1a_2bc.yaml",
        configs_dir + "1bc_1rc_1yc.yaml",
        configs_dir + "1carrot_1bc.yaml",
        configs_dir + "1carrot_1corn.yaml",
        configs_dir + "1yc_1rc.yaml",
        configs_dir + "2rc.yaml",
        configs_dir + "2t.yaml",
    ]
    simulations_configs = [
        O3DExROS2SimulationConfig.load_config(Path(path), Path(connector_path))
        for path in simulation_configs_paths
    ]
    # place object at coords
    place_obj_types = [
        "apple",
        "tomato",
        "carrot",
        "yellow_cube",
        "red_cube",
    ]
    target_coords = [(0.3, 0.3), (0.2, -0.4)]
    allowable_displacements = [0.1]  # large margin
    place_object_tasks: List[Task] = []
    for obj in place_obj_types:
        for coord in target_coords:
            for disp in allowable_displacements:
                place_object_tasks.append(
                    PlaceObjectAtCoordTask(obj, coord, disp, logger=logger)
                )
    easy_place_objects_scenarios = Benchmark.create_scenarios(
        tasks=place_object_tasks,
        simulation_configs=simulations_configs,
        simulation_configs_paths=simulation_configs_paths,
    )
    # move objects to the left
    object_groups = [
        ["carrot"],
        ["red_cube"],
        ["blue_cube"],
        ["yellow_cube"],
        ["tomato"],
    ]

    move_to_left_tasks: List[Task] = [
        MoveObjectsToLeftTask(obj_types=objects, logger=logger)
        for objects in object_groups
    ]

    easy_move_to_left_scenarios = Benchmark.create_scenarios(
        tasks=move_to_left_tasks,
        simulation_configs=simulations_configs,
        simulation_configs_paths=simulation_configs_paths,
    )

    # place cubes
    task = PlaceCubesTask(threshold_distance=0.2, logger=logger)
    easy_place_cubes_scenarios = Benchmark.create_scenarios(
        tasks=[task],
        simulation_configs=simulations_configs,
        simulation_configs_paths=simulation_configs_paths,
    )

    return [
        *easy_move_to_left_scenarios,
        *easy_place_objects_scenarios,
        *easy_place_cubes_scenarios,
    ]

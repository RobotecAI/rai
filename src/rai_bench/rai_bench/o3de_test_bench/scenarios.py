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

from rai_bench.benchmark_model import (
    Benchmark,
    Scenario,
    Task,
)
from rai_bench.o3de_test_bench.tasks import (
    BuildCubeTowerTask,
    GroupObjectsTask,
    MoveObjectsToLeftTask,
    PlaceCubesTask,
    PlaceObjectAtCoordTask,
)
from rai_sim.o3de.o3de_bridge import (
    O3DExROS2SimulationConfig,
)

loggers_type = Union[RcutilsLogger, logging.Logger]


def trivial_scenarios(
    configs_dir: str, bridge_config_path: str, logger: loggers_type | None
) -> List[Scenario[O3DExROS2SimulationConfig]]:
    """Packet of trivial scenarios. The grading is subjective.
    This packet contains easy variants of 'easy' tasks with minimalistic scenes setups(1 object).

    In this packet:
        PlaceObjectAtCoordTask with large allowable_displacement
        MoveObjectsToLeftTask with only 1 object type

    This level of difficulty requires recognizing position of object and moving it once

    Parameters
    ----------
    configs_dir : str
        path to directory with simulation configs
    bridge_config_path : str
        path to simulation bridge config


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
        O3DExROS2SimulationConfig.load_config(Path(path), Path(bridge_config_path))
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
    configs_dir: str, bridge_config_path: str, logger: loggers_type | None
) -> List[Scenario[O3DExROS2SimulationConfig]]:
    """Packet of easy scenarios. The grading is subjective.
    This packet contains easy variants of 'easy' tasks with scenes containg no more than 3 objects

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
    bridge_config_path : str
        path to simulation bridge config


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
        configs_dir + "2a_1bc.yaml",
        configs_dir + "1carrot_1t_1rc.yaml",
    ]
    simulations_configs = [
        O3DExROS2SimulationConfig.load_config(Path(path), Path(bridge_config_path))
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
        logger=logger,
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


def medium_scenarios(
    configs_dir: str, bridge_config_path: str, logger: loggers_type | None
) -> List[Scenario[O3DExROS2SimulationConfig]]:
    """Packet of medium scenarios. The grading is subjective.
    This packet contains harder variants of 'easy' tasks with scenes containg 4-7 objects
    and easy variants of 'hard' tasks with scenes contating 2-3 objects

    In this packet:
        MoveObjectsToLeftTask with multiple object types to move
        PlaceCubesTask with small threshold
        BuildTowerTask with only one type of objects to move
        GroupObjectsTask with only one type of objects to move

    This level of difficulty requires recognizing multiple proper type of objects.
    All scenarios will require moving more than 1 object.
    Some tasks will require good spacial awareness to make structures.

    Parameters
    ----------
    configs_dir : str
        path to directory with simulation configs
    bridge_config_path : str
        path to simulation bridge config


    Returns
    -------
    List[Scenario[O3DExROS2SimulationConfig]]
        list of easy scenarios
    """
    medium_simulation_configs_paths: List[str] = [
        configs_dir + "1rc_2bc_3yc.yaml",
        configs_dir + "2carrots_2a.yaml",
        configs_dir + "2yc_1bc_1rc.yaml",
        configs_dir + "4carrots.yaml",
        configs_dir + "1carrot_1a_1t_1bc_1corn.yaml",
        configs_dir + "4bc.yaml",
        configs_dir + "2a_1c_2rc.yaml",
    ]

    easy_simulation_configs_paths: List[str] = [
        configs_dir + "1a_1t.yaml",
        configs_dir + "1a_2bc.yaml",
        configs_dir + "1bc_1rc_1yc.yaml",
        configs_dir + "1carrot_1bc.yaml",
        configs_dir + "1carrot_1corn.yaml",
        configs_dir + "1yc_1rc.yaml",
        configs_dir + "2rc.yaml",
        configs_dir + "2t.yaml",
        configs_dir + "2a_1bc.yaml",
        configs_dir + "1carrot_1t_1rc.yaml",
    ]
    medium_simulations_configs = [
        O3DExROS2SimulationConfig.load_config(Path(path), Path(bridge_config_path))
        for path in medium_simulation_configs_paths
    ]
    easy_simulations_configs = [
        O3DExROS2SimulationConfig.load_config(Path(path), Path(bridge_config_path))
        for path in easy_simulation_configs_paths
    ]
    # move objects to the left
    object_groups = [
        ["red_cube", "blue_cube"],
        ["carrots"],
        ["carrots", "apple"],
        ["yellow_cube", "blue_cube"],
        ["tomato", "apple"],
        ["blue_cube"],
    ]

    move_to_left_tasks: List[Task] = [
        MoveObjectsToLeftTask(obj_types=objects, logger=logger)
        for objects in object_groups
    ]

    move_to_left_scenarios = Benchmark.create_scenarios(
        tasks=move_to_left_tasks,
        simulation_configs=medium_simulations_configs,
        simulation_configs_paths=medium_simulation_configs_paths,
        logger=logger,
    )

    # place cubes
    task = PlaceCubesTask(threshold_distance=0.1, logger=logger)
    easy_place_cubes_scenarios = Benchmark.create_scenarios(
        tasks=[task],
        simulation_configs=medium_simulations_configs,
        simulation_configs_paths=medium_simulation_configs_paths,
        logger=logger,
    )

    # build tower task
    object_groups = [
        ["red_cube", "blue_cube", "yellow_cube"],
    ]

    build_tower_tasks: List[Task] = [
        BuildCubeTowerTask(obj_types=objects, logger=logger)
        for objects in object_groups
    ]

    build_tower_scenarios = Benchmark.create_scenarios(
        tasks=build_tower_tasks,
        simulation_configs=easy_simulations_configs,
        simulation_configs_paths=easy_simulation_configs_paths,
    )

    # group object task
    object_groups = [
        ["apple"],
        ["carrot"],
        ["tomato"],
        ["red_cube"],
        ["tomato"],
        ["blue_cube"],
    ]

    group_object_tasks: List[Task] = [
        GroupObjectsTask(obj_types=objects, logger=logger) for objects in object_groups
    ]

    group_object_scenarios = Benchmark.create_scenarios(
        tasks=group_object_tasks,
        simulation_configs=easy_simulations_configs,
        simulation_configs_paths=easy_simulation_configs_paths,
    )
    return [
        *move_to_left_scenarios,
        *build_tower_scenarios,
        *easy_place_cubes_scenarios,
        *group_object_scenarios,
    ]


def hard_scenarios(
    configs_dir: str, bridge_config_path: str, logger: loggers_type | None
) -> List[Scenario[O3DExROS2SimulationConfig]]:
    """Packet of hard scenarios. The grading is subjective.
    This packet contains harder variants of 'easy' tasks with majority of scenes containg 8+ objects,
    Objects can be positioned in an unusual way, for example stacked.
    And easy variants of 'hard' tasks with scenes containing 4-7 objects

    In this packet:
        MoveObjectsToLeftTask with multiple object types to move
        PlaceCubesTask with small threshold
        BuildTowerTask with all cubes available
        GroupObjectsTask with 1-2 types of objects to be grouped

    This level of difficulty requires recognizing multiple proper type of objects.
    All scenarios will require moving multiple objects.
    Some tasks will require good spacial awareness to make structures.

    Parameters
    ----------
    configs_dir : str
        path to directory with simulation configs
    bridge_config_path : str
        path to simulation bridge config


    Returns
    -------
    List[Scenario[O3DExROS2SimulationConfig]]
        list of easy scenarios
    """
    medium_simulation_configs_paths: List[str] = [
        configs_dir + "1rc_2bc_3yc.yaml",
        configs_dir + "2carrots_2a.yaml",
        configs_dir + "2yc_1bc_1rc.yaml",
        configs_dir + "4carrots.yaml",
        configs_dir + "1carrot_1a_1t_1bc_1corn.yaml",
        configs_dir + "4bc.yaml",
        configs_dir + "2a_1c_2rc.yaml",
    ]

    hard_simulation_configs_paths: List[str] = [
        configs_dir + "3carrots_1a_1t_2bc_2yc.yaml",
        configs_dir + "1carrot_1a_2t_1bc_1rc_3yc_stacked.yaml",
        configs_dir + "2carrots_1a_1t_1bc_1rc_1yc_1corn.yaml",
        configs_dir + "2rc_3bc_4yc_stacked.yaml",
        configs_dir + "2t_3a_1corn_2rc.yaml",
        configs_dir + "3a_4t_2bc.yaml",
        configs_dir + "2rc.yaml",
        configs_dir + "3carrots_1a_2bc_1rc_1yc_1corn.yaml",
        configs_dir + "3rc_3bc_stacked.yaml",
    ]
    medium_simulations_configs = [
        O3DExROS2SimulationConfig.load_config(Path(path), Path(bridge_config_path))
        for path in medium_simulation_configs_paths
    ]
    hard_simulations_configs = [
        O3DExROS2SimulationConfig.load_config(Path(path), Path(bridge_config_path))
        for path in hard_simulation_configs_paths
    ]
    # move objects to the left
    object_groups = [
        ["red_cube", "blue_cube"],
        ["carrots", "apple", "yellow_cube"],
        ["carrots", "apple"],
        ["yellow_cube", "blue_cube"],
        ["tomato", "apple"],
        ["blue_cube", "red_cube"],
    ]

    move_to_left_tasks: List[Task] = [
        MoveObjectsToLeftTask(obj_types=objects, logger=logger)
        for objects in object_groups
    ]

    move_to_left_scenarios = Benchmark.create_scenarios(
        tasks=move_to_left_tasks,
        simulation_configs=hard_simulations_configs,
        simulation_configs_paths=hard_simulation_configs_paths,
    )

    # place cubes
    task = PlaceCubesTask(threshold_distance=0.1, logger=logger)
    easy_place_cubes_scenarios = Benchmark.create_scenarios(
        tasks=[task],
        simulation_configs=hard_simulations_configs,
        simulation_configs_paths=hard_simulation_configs_paths,
    )

    # build tower task
    object_groups = [
        ["red_cube", "blue_cube", "yellow_cube"],
    ]

    build_tower_tasks: List[Task] = [
        BuildCubeTowerTask(obj_types=objects, logger=logger)
        for objects in object_groups
    ]

    build_tower_scenarios = Benchmark.create_scenarios(
        tasks=build_tower_tasks,
        simulation_configs=medium_simulations_configs,
        simulation_configs_paths=medium_simulation_configs_paths,
        logger=logger,
    )

    # group object task
    object_groups = [
        ["apple", "carrot"],
        ["carrot", "tomato"],
        ["tomato", "blue_cube"],
        ["red_cube"],
        ["carrot"],
        ["blue_cube"],
        ["yellow_cube", "red_cube"],
    ]

    group_object_tasks: List[Task] = [
        GroupObjectsTask(obj_types=objects, logger=logger) for objects in object_groups
    ]

    group_object_scenarios = Benchmark.create_scenarios(
        tasks=group_object_tasks,
        simulation_configs=medium_simulations_configs,
        simulation_configs_paths=medium_simulation_configs_paths,
    )
    return [
        *move_to_left_scenarios,
        *build_tower_scenarios,
        *easy_place_cubes_scenarios,
        *group_object_scenarios,
    ]


def very_hard_scenarios(
    configs_dir: str, bridge_config_path: str, logger: loggers_type | None
) -> List[Scenario[O3DExROS2SimulationConfig]]:
    """Packet of very_hard scenarios. The grading is subjective.
    This packet contains harder variants of 'hard' tasks with majority of scenes containg 8+ objects,
    Objects can be positioned in an unusual way, for example stacked.
    In this packet:
        BuildTowerTask with only ceratin type of cubes
        GroupObjectsTask with multiple objects to be grouped

    This level of difficulty requires recognizing multiple proper type of objects.
    All scenarios will require moving multiple objects.
    All tasks will require very good spacial awareness to make structures.

    Parameters
    ----------
    configs_dir : str
        path to directory with simulation configs
    bridge_config_path : str
        path to simulation bridge config


    Returns
    -------
    List[Scenario[O3DExROS2SimulationConfig]]
        list of easy scenarios
    """
    hard_simulation_configs_paths: List[str] = [
        configs_dir + "3carrots_1a_1t_2bc_2yc.yaml",
        configs_dir + "1carrot_1a_2t_1bc_1rc_3yc_stacked.yaml",
        configs_dir + "2carrots_1a_1t_1bc_1rc_1yc_1corn.yaml",
        configs_dir + "2rc_3bc_4yc_stacked.yaml",
        configs_dir + "2t_3a_1corn_2rc.yaml",
        configs_dir + "3a_4t_2bc.yaml",
        configs_dir + "2rc.yaml",
        configs_dir + "3carrots_1a_2bc_1rc_1yc_1corn.yaml",
        configs_dir + "3rc_3bc_stacked.yaml",
    ]
    hard_simulations_configs = [
        O3DExROS2SimulationConfig.load_config(Path(path), Path(bridge_config_path))
        for path in hard_simulation_configs_paths
    ]
    # build tower task
    object_groups = [
        ["red_cube", "blue_cube"],
        ["red_cube"],
        ["blue_cube"],
        ["yellow_cube"],
        ["yellow_cube"],
        ["blue_cube"],
    ]

    build_tower_tasks: List[Task] = [
        BuildCubeTowerTask(obj_types=objects, logger=logger)
        for objects in object_groups
    ]

    build_tower_scenarios = Benchmark.create_scenarios(
        tasks=build_tower_tasks,
        simulation_configs=hard_simulations_configs,
        simulation_configs_paths=hard_simulation_configs_paths,
        logger=logger,
    )

    # group object task
    object_groups = [
        ["apple", "carrot"],
        ["carrot", "tomato"],
        ["tomato", "blue_cube", "yellow_cube"],
        ["red_cube", "blue_cube"],
        ["tomato", "apple", "carrot"],
        ["blue_cube", "carrot"],
    ]

    group_object_tasks: List[Task] = [
        GroupObjectsTask(obj_types=objects, logger=logger) for objects in object_groups
    ]

    group_object_scenarios = Benchmark.create_scenarios(
        tasks=group_object_tasks,
        simulation_configs=hard_simulations_configs,
        simulation_configs_paths=hard_simulation_configs_paths,
    )
    return [
        *build_tower_scenarios,
        *group_object_scenarios,
    ]

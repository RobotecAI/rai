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
from typing import List, Literal

from rai_bench.manipulation_o3de.benchmark import ManipulationO3DEBenchmark, Scenario
from rai_bench.manipulation_o3de.interfaces import Task
from rai_bench.manipulation_o3de.tasks import (
    BuildCubeTowerTask,
    GroupObjectsTask,
    MoveObjectsToLeftTask,
    PlaceCubesTask,
    PlaceObjectAtCoordTask,
)
from rai_sim.simulation_bridge import SceneConfig

CONFIGS_DIR = "src/rai_bench/rai_bench/manipulation_o3de/predefined/configs/"


def trivial_scenarios(logger: logging.Logger | None) -> List[Scenario]:
    """Packet of trivial scenarios. The grading is subjective.
    This packet contains easy variants of 'easy' tasks with minimalistic scenes setups(1 object).

    In this packet:
        PlaceObjectAtCoordTask with large allowable_displacement
        MoveObjectsToLeftTask with only 1 object type

    This level of difficulty requires recognizing position of object and moving it once


    Returns
    -------
    List[Scenario[O3DExROS2SimulationConfig]]
        list of trivial scenarios
    """
    scene_configs_paths: List[str] = [
        CONFIGS_DIR + "1a.yaml",
        CONFIGS_DIR + "1rc.yaml",
        CONFIGS_DIR + "1t.yaml",
        CONFIGS_DIR + "1yc.yaml",
        CONFIGS_DIR + "1carrot.yaml",
    ]
    scene_configs = [
        SceneConfig.load_base_config(Path(path)) for path in scene_configs_paths
    ]
    # place object at coordss
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
    place_objects_scenarios = ManipulationO3DEBenchmark.create_scenarios(
        tasks=place_object_tasks,
        scene_configs=scene_configs,
        scene_configs_paths=scene_configs_paths,
        level="trivial",
        logger=logger,
    )
    # move objects to the left
    object_groups = [["carrot"], ["red_cube"], ["tomato"], ["yellow_cube"]]

    move_to_left_tasks: List[Task] = [
        MoveObjectsToLeftTask(obj_types=objects, logger=logger)
        for objects in object_groups
    ]

    move_to_left_scenarios = ManipulationO3DEBenchmark.create_scenarios(
        tasks=move_to_left_tasks,
        scene_configs=scene_configs,
        scene_configs_paths=scene_configs_paths,
        level="trivial",
        logger=logger,
    )

    return [*move_to_left_scenarios, *place_objects_scenarios]


def easy_scenarios(logger: logging.Logger | None) -> List[Scenario]:
    """Packet of easy scenarios. The grading is subjective.
    This packet contains easy variants of 'easy' tasks with scenes containg no more than 3 objects

    In this packet:
        PlaceObjectAtCoordTask with small allowable_displacement
        MoveObjectsToLeftTask with only 1 object type
        PlaceCubesTask with large threshold

    This level of difficulty requires recognizing proper type of object.
    Some scenarios will require moving more than 1 object or moving with more precision.

    Returns
    -------
    List[Scenario[O3DExROS2SimulationConfig]]
        list of easy scenarios
    """
    scene_configs_paths: List[str] = [
        CONFIGS_DIR + "1a_1t.yaml",
        CONFIGS_DIR + "1a_2bc.yaml",
        CONFIGS_DIR + "1bc_1rc_1yc.yaml",
        CONFIGS_DIR + "1carrot_1bc.yaml",
        CONFIGS_DIR + "1carrot_1corn.yaml",
        CONFIGS_DIR + "1yc_1rc.yaml",
        CONFIGS_DIR + "2rc.yaml",
        CONFIGS_DIR + "2t.yaml",
        CONFIGS_DIR + "2a_1bc.yaml",
        CONFIGS_DIR + "1carrot_1t_1rc.yaml",
    ]
    scene_configs = [
        SceneConfig.load_base_config(Path(path)) for path in scene_configs_paths
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
    place_objects_scenarios = ManipulationO3DEBenchmark.create_scenarios(
        tasks=place_object_tasks,
        scene_configs=scene_configs,
        scene_configs_paths=scene_configs_paths,
        level="easy",
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

    move_to_left_scenarios = ManipulationO3DEBenchmark.create_scenarios(
        tasks=move_to_left_tasks,
        scene_configs=scene_configs,
        scene_configs_paths=scene_configs_paths,
        level="easy",
        logger=logger,
    )

    # place cubes
    task = PlaceCubesTask(threshold_distance=0.2, logger=logger)
    place_cubes_scenarios = ManipulationO3DEBenchmark.create_scenarios(
        tasks=[task],
        scene_configs=scene_configs,
        scene_configs_paths=scene_configs_paths,
        level="easy",
        logger=logger,
    )

    return [
        *move_to_left_scenarios,
        *place_objects_scenarios,
        *place_cubes_scenarios,
    ]


def medium_scenarios(logger: logging.Logger | None) -> List[Scenario]:
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


    Returns
    -------
    List[Scenario[O3DExROS2SimulationConfig]]
        list of easy scenarios
    """
    medium_scene_configs_paths: List[str] = [
        CONFIGS_DIR + "1rc_2bc_3yc.yaml",
        CONFIGS_DIR + "2carrots_2a.yaml",
        CONFIGS_DIR + "2yc_1bc_1rc.yaml",
        CONFIGS_DIR + "4carrots.yaml",
        CONFIGS_DIR + "1carrot_1a_1t_1bc_1corn.yaml",
        CONFIGS_DIR + "4bc.yaml",
        CONFIGS_DIR + "2a_1c_2rc.yaml",
    ]

    easy_scene_configs_paths: List[str] = [
        CONFIGS_DIR + "1a_1t.yaml",
        CONFIGS_DIR + "1a_2bc.yaml",
        CONFIGS_DIR + "1bc_1rc_1yc.yaml",
        CONFIGS_DIR + "1carrot_1bc.yaml",
        CONFIGS_DIR + "1carrot_1corn.yaml",
        CONFIGS_DIR + "1yc_1rc.yaml",
        CONFIGS_DIR + "2rc.yaml",
        CONFIGS_DIR + "2t.yaml",
        CONFIGS_DIR + "2a_1bc.yaml",
        CONFIGS_DIR + "1carrot_1t_1rc.yaml",
    ]
    medium_scene_configs = [
        SceneConfig.load_base_config(Path(path)) for path in medium_scene_configs_paths
    ]
    easy_scene_configs = [
        SceneConfig.load_base_config(Path(path)) for path in easy_scene_configs_paths
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

    move_to_left_scenarios = ManipulationO3DEBenchmark.create_scenarios(
        tasks=move_to_left_tasks,
        scene_configs=medium_scene_configs,
        scene_configs_paths=medium_scene_configs_paths,
        logger=logger,
        level="medium",
    )

    # place cubes
    task = PlaceCubesTask(threshold_distance=0.1, logger=logger)
    place_cubes_scenarios = ManipulationO3DEBenchmark.create_scenarios(
        tasks=[task],
        scene_configs=medium_scene_configs,
        scene_configs_paths=medium_scene_configs_paths,
        logger=logger,
        level="medium",
    )

    # build tower task
    object_groups = [
        ["red_cube", "blue_cube", "yellow_cube"],
    ]

    build_tower_tasks: List[Task] = [
        BuildCubeTowerTask(obj_types=objects, logger=logger)
        for objects in object_groups
    ]

    build_tower_scenarios = ManipulationO3DEBenchmark.create_scenarios(
        tasks=build_tower_tasks,
        scene_configs=easy_scene_configs,
        scene_configs_paths=easy_scene_configs_paths,
        level="medium",
        logger=logger,
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

    group_object_scenarios = ManipulationO3DEBenchmark.create_scenarios(
        tasks=group_object_tasks,
        scene_configs=easy_scene_configs,
        scene_configs_paths=easy_scene_configs_paths,
        level="medium",
        logger=logger,
    )
    return [
        *move_to_left_scenarios,
        *build_tower_scenarios,
        *place_cubes_scenarios,
        *group_object_scenarios,
    ]


def hard_scenarios(logger: logging.Logger | None) -> List[Scenario]:
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



    Returns
    -------
    List[Scenario[O3DExROS2SimulationConfig]]
        list of easy scenarios
    """
    medium_scene_configs_paths: List[str] = [
        CONFIGS_DIR + "1rc_2bc_3yc.yaml",
        CONFIGS_DIR + "2carrots_2a.yaml",
        CONFIGS_DIR + "2yc_1bc_1rc.yaml",
        CONFIGS_DIR + "4carrots.yaml",
        CONFIGS_DIR + "1carrot_1a_1t_1bc_1corn.yaml",
        CONFIGS_DIR + "4bc.yaml",
        CONFIGS_DIR + "2a_1c_2rc.yaml",
    ]

    hard_scene_configs_paths: List[str] = [
        CONFIGS_DIR + "3carrots_1a_1t_2bc_2yc.yaml",
        CONFIGS_DIR + "1carrot_1a_2t_1bc_1rc_3yc_stacked.yaml",
        CONFIGS_DIR + "2carrots_1a_1t_1bc_1rc_1yc_1corn.yaml",
        CONFIGS_DIR + "2rc_3bc_4yc_stacked.yaml",
        CONFIGS_DIR + "2t_3a_1corn_2rc.yaml",
        CONFIGS_DIR + "3a_4t_2bc.yaml",
        CONFIGS_DIR + "2rc.yaml",
        CONFIGS_DIR + "3carrots_1a_2bc_1rc_1yc_1corn.yaml",
        CONFIGS_DIR + "3rc_3bc_stacked.yaml",
    ]
    medium_scene_configs = [
        SceneConfig.load_base_config(Path(path)) for path in medium_scene_configs_paths
    ]
    hard_scene_configs = [
        SceneConfig.load_base_config(Path(path)) for path in hard_scene_configs_paths
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

    move_to_left_scenarios = ManipulationO3DEBenchmark.create_scenarios(
        tasks=move_to_left_tasks,
        scene_configs=hard_scene_configs,
        scene_configs_paths=hard_scene_configs_paths,
        level="hard",
        logger=logger,
    )

    # place cubes
    task = PlaceCubesTask(threshold_distance=0.1, logger=logger)
    place_cubes_scenarios = ManipulationO3DEBenchmark.create_scenarios(
        tasks=[task],
        scene_configs=hard_scene_configs,
        scene_configs_paths=hard_scene_configs_paths,
        level="hard",
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

    build_tower_scenarios = ManipulationO3DEBenchmark.create_scenarios(
        tasks=build_tower_tasks,
        scene_configs=medium_scene_configs,
        scene_configs_paths=medium_scene_configs_paths,
        level="hard",
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

    group_object_scenarios = ManipulationO3DEBenchmark.create_scenarios(
        tasks=group_object_tasks,
        scene_configs=medium_scene_configs,
        scene_configs_paths=medium_scene_configs_paths,
        level="hard",
        logger=logger,
    )
    return [
        *move_to_left_scenarios,
        *build_tower_scenarios,
        *place_cubes_scenarios,
        *group_object_scenarios,
    ]


def very_hard_scenarios(logger: logging.Logger | None) -> List[Scenario]:
    """Packet of very_hard scenarios. The grading is subjective.
    This packet contains harder variants of 'hard' tasks with majority of scenes containg 8+ objects,
    Objects can be positioned in an unusual way, for example stacked.
    In this packet:
        BuildTowerTask with only ceratin type of cubes
        GroupObjectsTask with multiple objects to be grouped

    This level of difficulty requires recognizing multiple proper type of objects.
    All scenarios will require moving multiple objects.
    All tasks will require very good spacial awareness to make structures.



    Returns
    -------
    List[Scenario[O3DExROS2SimulationConfig]]
        list of easy scenarios
    """
    hard_scene_configs_paths: List[str] = [
        CONFIGS_DIR + "3carrots_1a_1t_2bc_2yc.yaml",
        CONFIGS_DIR + "1carrot_1a_2t_1bc_1rc_3yc_stacked.yaml",
        CONFIGS_DIR + "2carrots_1a_1t_1bc_1rc_1yc_1corn.yaml",
        CONFIGS_DIR + "2rc_3bc_4yc_stacked.yaml",
        CONFIGS_DIR + "2t_3a_1corn_2rc.yaml",
        CONFIGS_DIR + "3a_4t_2bc.yaml",
        CONFIGS_DIR + "2rc.yaml",
        CONFIGS_DIR + "3carrots_1a_2bc_1rc_1yc_1corn.yaml",
        CONFIGS_DIR + "3rc_3bc_stacked.yaml",
    ]
    hard_scene_configs = [
        SceneConfig.load_base_config(Path(path)) for path in hard_scene_configs_paths
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

    build_tower_scenarios = ManipulationO3DEBenchmark.create_scenarios(
        tasks=build_tower_tasks,
        scene_configs=hard_scene_configs,
        scene_configs_paths=hard_scene_configs_paths,
        logger=logger,
        level="very_hard",
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

    group_object_scenarios = ManipulationO3DEBenchmark.create_scenarios(
        tasks=group_object_tasks,
        scene_configs=hard_scene_configs,
        scene_configs_paths=hard_scene_configs_paths,
        level="very_hard",
        logger=logger,
    )
    return [
        *build_tower_scenarios,
        *group_object_scenarios,
    ]


def get_scenarios(
    levels: List[Literal["trivial", "easy", "medium", "hard", "very_hard"]] = [
        "trivial",
        "easy",
        "medium",
        "hard",
        "very_hard",
    ],
    logger: logging.Logger | None = None,
):
    scenarios: List[Scenario] = []
    if "trivial" in levels:
        scenarios.extend(trivial_scenarios(logger=logger))
    if "easy" in levels:
        scenarios.extend(easy_scenarios(logger=logger))

    if "medium" in levels:
        scenarios.extend(medium_scenarios(logger=logger))

    if "hard" in levels:
        scenarios.extend(hard_scenarios(logger=logger))

    if "very_hard" in levels:
        scenarios.extend(very_hard_scenarios(logger=logger))

    return scenarios

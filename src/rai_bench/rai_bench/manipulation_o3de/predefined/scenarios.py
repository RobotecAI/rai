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
from typing import Dict, List, Literal

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

# Configurations for scenario paths by "difficulty" and "purpose"
SCENE_CONFIG_PATHS: Dict[str, List[str]] = {
    # Base sets
    "trivial": [
        CONFIGS_DIR + "1a.yaml",
        CONFIGS_DIR + "1rc.yaml",
        CONFIGS_DIR + "1t.yaml",
        CONFIGS_DIR + "1yc.yaml",
        CONFIGS_DIR + "1carrot.yaml",
    ],
    "easy": [
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
    ],
    "medium": [
        CONFIGS_DIR + "1rc_2bc_3yc.yaml",
        CONFIGS_DIR + "2carrots_2a.yaml",
        CONFIGS_DIR + "2yc_1bc_1rc.yaml",
        CONFIGS_DIR + "4carrots.yaml",
        CONFIGS_DIR + "4bc.yaml",
        CONFIGS_DIR + "2a_1c_2rc.yaml",
        CONFIGS_DIR + "2rc_2a.yaml",
        CONFIGS_DIR + "3rc_2a_1carrot.yaml",
    ],
    "hard": [
        CONFIGS_DIR + "3carrots_1a_1t_2bc_2yc.yaml",
        CONFIGS_DIR + "1carrot_1a_2t_1bc_1rc_3yc_stacked.yaml",
        CONFIGS_DIR + "2carrots_1a_1t_1bc_1rc_1yc_1corn.yaml",
        CONFIGS_DIR + "2rc_3bc_4yc_stacked.yaml",
        CONFIGS_DIR + "2t_3a_1corn_2rc.yaml",
        CONFIGS_DIR + "3a_4corn_2bc.yaml",
        CONFIGS_DIR + "3a_4corn_2rc.yaml",
        CONFIGS_DIR + "2rc.yaml",
        CONFIGS_DIR + "3carrots_1a_2bc_1rc_1yc_1corn.yaml",
        CONFIGS_DIR + "3rc_3bc_stacked.yaml",
        CONFIGS_DIR + "3carrots_3a_2rc.yaml",
    ],
}

# For sets reused for scenario generation (e.g., easy_scene_configs_paths for medium-level tasks)
SCENE_CONFIG_ALIASES: Dict[str, List[str]] = {
    "easy_for_medium": SCENE_CONFIG_PATHS["easy"]
    + [
        CONFIGS_DIR + "3rc.yaml",  # Only in medium, not in 'easy' base tasks
    ]
}


def get_scene_configs(paths: List[str]) -> List[SceneConfig]:
    """Helper to load base config objects from given paths."""
    return [SceneConfig.load_base_config(Path(path)) for path in paths]


def trivial_scenarios(logger: logging.Logger | None) -> List[Scenario]:
    scene_configs_paths = SCENE_CONFIG_PATHS["trivial"]
    scene_configs = get_scene_configs(scene_configs_paths)
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
    scene_configs_paths = SCENE_CONFIG_PATHS["easy"]
    scene_configs = get_scene_configs(scene_configs_paths)
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
    medium_scene_configs_paths = SCENE_CONFIG_PATHS["medium"]
    # Extend easy config for medium-level grouping/tower tasks needing smaller scenes
    easy_scene_configs_paths = SCENE_CONFIG_ALIASES["easy_for_medium"]
    medium_scene_configs = get_scene_configs(medium_scene_configs_paths)
    easy_scene_configs = get_scene_configs(easy_scene_configs_paths)
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
    medium_scene_configs_paths = SCENE_CONFIG_PATHS["medium"] + [
        CONFIGS_DIR + "1a_1t_1bc_2corn.yaml"  # Only in hard, not medium set before
    ]
    # All the hard configs
    hard_scene_configs_paths = SCENE_CONFIG_PATHS["hard"]
    # Hard config also needs medium configs for 'build tower' and 'group objects' tasks
    medium_scene_configs = get_scene_configs(medium_scene_configs_paths)
    hard_scene_configs = get_scene_configs(hard_scene_configs_paths)
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
    hard_scene_configs_paths = SCENE_CONFIG_PATHS["hard"]
    hard_scene_configs = get_scene_configs(hard_scene_configs_paths)
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

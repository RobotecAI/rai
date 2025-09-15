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
from typing import List, Sequence, Tuple, Union

from rclpy.impl.rcutils_logger import RcutilsLogger

from rai_bench.manipulation_o3de.benchmark import Scenario
from rai_bench.manipulation_o3de.interfaces import (
    ManipulationTask,
)
from rai_bench.manipulation_o3de.tasks import PlaceObjectAtCoordTask
from rai_sim.simulation_bridge import Entity, SceneConfig

loggers_type = Union[RcutilsLogger, logging.Logger]

### Define your scene setup ####################3
path_to_your_config = (
    "src/rai_bench/rai_bench/manipulation_o3de/predefined/configs/1a.yaml"
)
scene_config = SceneConfig.load_base_config(Path(path_to_your_config))

# Configure predefined task to place an apple on the table
target_coords = (0.1, 0.1)
disp = 0.1
task = PlaceObjectAtCoordTask(
    obj_type="apple",
    target_position=target_coords,
    allowable_displacement=disp,
)

# Create scene with apple on the table
Scenario(task=task, scene_config=scene_config, scene_config_path=path_to_your_config)


######### Define your task ###################
class ThrowObjectsOffTableTask(ManipulationTask):
    def __init__(self, obj_types: List[str], logger: loggers_type | None = None):
        super().__init__(logger=logger)
        # obj_types is a list of objects that are subject of the task
        # In this case, it will mean which objects should be thrown off the table
        # can be any objects
        self.obj_types = obj_types

    @property
    def task_prompt(self) -> str:
        # define prompt
        obj_names = ", ".join(obj + "s" for obj in self.obj_types).replace("_", " ")
        # 0.0 z is the level of table, so any coord below that means it is off the table
        return f"Manipulate objects, so that all of the {obj_names} are dropped outside of the table (for example y<-0.75)."

    def check_if_required_objects_present(self, simulation_config: SceneConfig) -> bool:
        # Validate if any required objects are present in sim config
        # if there is not a single object of provided type, there is no point in running
        # this task of given scene config
        count = sum(
            1 for ent in simulation_config.entities if ent.prefab_name in self.obj_types
        )
        return count > 1

    def calculate_correct(self, entities: Sequence[Entity]) -> Tuple[int, int]:
        selected_type_objects = self.filter_entities_by_object_type(
            entities=entities, object_types=self.obj_types
        )

        # check how many objects are below table, that will be our metric
        correct = sum(
            1 for ent in selected_type_objects if ent.pose.pose.position.z < 0.0
        )

        incorrect: int = len(selected_type_objects) - correct
        return correct, incorrect


# Task, throw apple off the table
remove_obj_from_table_task = ThrowObjectsOffTableTask(
    obj_types=["apple"],
)
super_scenario = Scenario(
    task=task, scene_config=scene_config, scene_config_path=path_to_your_config
)


super_scenario = Scenario(
    task=remove_obj_from_table_task,
    scene_config=scene_config,
    scene_config_path=path_to_your_config,
)

if __name__ == "__main__":
    from pathlib import Path

    from rai_bench import (
        define_benchmark_logger,
    )
    from rai_bench.manipulation_o3de import run_benchmark
    from rai_bench.utils import get_llm_for_benchmark

    experiment_dir = Path("src/rai_bench/rai_bench/experiments/custom_task/")

    experiment_dir.mkdir(parents=True, exist_ok=True)
    bench_logger = define_benchmark_logger(out_dir=experiment_dir)

    llm = get_llm_for_benchmark(
        model_name="gpt-4o",
        vendor="openai",
    )

    run_benchmark(
        llm=llm,
        out_dir=experiment_dir,
        # use your scenario
        scenarios=[super_scenario],
        bench_logger=bench_logger,
    )

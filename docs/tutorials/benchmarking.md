# Benchmarking

!!! note

    If you aren't familiar with our benchmark package, please read [RAI Bench](../simulation_and_benchmarking/rai_bench.md) first.

Currently we offer 2 predefined benchmarks:

-   [Manipulation_O3DE](#manipulation-o3de)
-   [Tool_Calling_Agent](#tool-calling-agent)

If you want to test multiple models across diffrent benchmark configurations to go [Testing Models](#testing-models)

If your goal is creating custom tasks and scenarios visit [Creating Custom Tasks](#creating-custom-tasks).

## Manipulation O3DE

-   Follow setup from [Manipulation demo Setup](../demos/manipulation.md#setup)
-   Create the o3de config:
    ```yaml
    binary_path: /path/to/binary/RAIManipulationDemo.GameLauncher
    level: RoboticManipulationBenchmark
    robotic_stack_command: ros2 launch examples/manipulation-demo-no-binary.launch.py
    required_simulation_ros2_interfaces:
    services:
        - /spawn_entity
        - /delete_entity
    topics:
        - /color_image5
        - /depth_image5
        - /color_camera_info5
    actions: []
    required_robotic_ros2_interfaces:
    services:
        - /grounding_dino_classify
        - /grounded_sam_segment
        - /manipulator_move_to
    topics: []
    actions: []
    ```
-   Run the benchmark with:

    ```bash
    python src/rai_bench/rai_bench/examples/manipulation_o3de.py --model-name <your-model> --vendor <your-vendor> --o3de-config-path <path/to/your/config> --levels <trivial, easy>
    ```

    !!! warning

        Running all scenarios will take a while. If you want to just try it out, we recommend chosing just one level of difficulty.

## Tool Calling Agent

This benchmark does not require any additinal setup besides the main one [Basic Setup](../setup/install.md), just run:

```bash
python src/rai_bench/rai_bench/examples/tool_calling_agent.py --model-name <model-name> --vendor <vendor> --extra-tool-calls <5> --task-types <basic> --out-dir <out_dir>
```

!!! note

    This Benchmark is significantly faster, but still if just trying out, we recommned chosing just one task-type

## Testing Models

The best way of benchmarking your models is using `rai_bench.test_models` function with benchmark configs.

??? info "test_models function definition"

    ::: rai_bench.test_models.test_models

Example usage

```python
from rai_bench import (
    ManipulationO3DEBenchmarkConfig,
    ToolCallingAgentBenchmarkConfig,
    test_models,
)

if __name__ == "__main__":
    # Define models you want to benchmark
    model_names = ["qwen2.5:7b", "llama3.2:3b"]
    vendors = ["ollama", "ollama"]

    # Define benchmarks that will be used
    man_conf = ManipulationO3DEBenchmarkConfig(
        o3de_config_path="path/to/your/o3de_config.yaml",  # path to your o3de config
        levels=[  # define what difficulty of tasks to include in benchmark
            "trivial",
        ],
        repeats=1,  # how many times to repeat
    )
    tool_conf = ToolCallingAgentBenchmarkConfig(
        extra_tool_calls=5,  # how many extra tool calls allowed to still pass
        task_types=[  # what types of tasks to include
            "basic",
            "spatial_reasoning",
            "manipulation",
        ],
        repeats=1,
    )

    out_dir = "src/rai_bench/rai_bench/experiments"
    test_models(
        model_names=model_names,
        vendors=vendors,
        benchmark_configs=[man_conf, tool_conf],
        out_dir=out_dir,
    )

```

## Viewing Results

From every benchmark run there will be results saved in provided output directory:

-   Logs - in `benchmark.log` file
-   results_summary.csv - for overall metrics
-   results.csv - for detailed results of ever task/scenario

When using `test_models` the output directories will be saved as `<run_datetime>/<benchmark_name>/<models>/<repeat>/...` and this format can be visualised with our streamlit script:

```bash
streamlit run  src/rai_bench/rai_bench/examples/visualise_streamlit.py
```

## Creating Custom Tasks

### Manipualtion O3DE Scenarios

To create your own Scenarios for you will need Scene Config and Task. You can combine already existing Scene and exisitng Task to create new Scenario like:

```python
from pathlib import Path
from rai_bench.manipulation_o3de.tasks import PlaceObjectAtCoordTask
from rai_sim.simulation_bridge import SceneConfig
from rai_bench.manipulation_o3de.benchmark import Scenario


path_to_your_config = "src/rai_bench/rai_bench/manipulation_o3de/predefined/configs/1a.yaml"
scene_config = SceneConfig.load_base_config(Path(path_to_your_config))

# configure exisitng Task with different params
target_coords = (0.1, 0.1)
disp = 0.1
task = PlaceObjectAtCoordTask(
    obj_type="apple",
    target_position=target_coords,
    allowable_displacement=disp,
)

Scenario(task=task, scene_config=scene_config, scene_config_path=path_to_your_config)
```

But you also can create them from scratch.
Creating Scene Config is very easy, just declare entities in yaml file like:

```yaml
entities:
  - name: apple1
    prefab_name: apple # make sure that this prefab exisits in simulaiton
      pose:
          translation:
              x: 0.0
              y: 0.5
              z: 0.05
          rotation:
              x: 0.0
              y: 0.0
              z: 0.0
              w: 1.0
```

Creating your own Task will require slightly more effort. Let's start with something simple - Task will require throwing given objects off the table:

```python
import logging
from typing import List, Tuple, Union
from rclpy.impl.rcutils_logger import RcutilsLogger
from rai_bench.manipulation_o3de.interfaces import (
    ManipulationTask,
)
from rai_sim.simulation_bridge import Entity, SimulationConfig

loggers_type = Union[RcutilsLogger, logging.Logger]


class ThrowObjectsOffTableTask(ManipulationTask):
    def __init__(self, obj_types: List[str], logger: loggers_type | None = None):
        super().__init__(logger=logger)
        # obj_types is a list of objects that are subject of the task
        # In this case it will mean which objects should be thrown off the table
        # can be any objects
        self.obj_types = obj_types

    @property
    def task_prompt(self) -> str:
        # define prompt
        obj_names = ", ".join(obj + "s" for obj in self.obj_types).replace("_", " ")
        # 0.0 z is the level of table, so any coord below that means it is off the table
        return f"Manipulate objects, so that all of the {obj_names} are thrown off the table (negative z)"

    def check_if_required_objects_present(
        self, simulation_config: SimulationConfig
    ) -> bool:
        # Validate if any required objects are present in sim config
        # if there is not a single object of provided type, there is no point in running
        # this task of given scene config
        count = sum(
            1 for ent in simulation_config.entities if ent.prefab_name in self.obj_types
        )
        return count > 1

    def calculate_correct(self, entities: List[Entity]) -> Tuple[int, int]:
        selected_type_objects = self.filter_entities_by_object_type(
            entities=entities, object_types=self.obj_types
        )

        # check how many objects are below table, that will be our metric
        correct = sum(
            1 for ent in selected_type_objects if ent.pose.pose.position.z < 0.0
        )

        incorrect: int = len(selected_type_objects) - correct
        return correct, incorrect
```

As `obj_types` is parameterizable it enables various variants of this Task. In combination with a lot of simulation configs available it means that a single Task can provide dozens of scenarios.

Congratulations, you just created your first Scenario from scratch!

### Tool Calling Tasks

To create a Tool Calling Task, you will need to define Subtasks, Validators, and Task itself.
Let's create a basic task that require using tool to receive message from specific topic.

```python
from rai_bench.tool_calling_agent.subtasks import (
    CheckArgsToolCallSubTask,
)
from rai_bench.tool_calling_agent.validators import (
    OrderedCallsValidator,
)
from rai_bench.tool_calling_agent.tasks.basic import BasicTask
from rai_bench.tool_calling_agent.mocked_tools import (
    MockGetROS2TopicsNamesAndTypesTool,
)
from langchain_core.tools import BaseTool
from typing import List

# configure exisitng Task with different params
target_coords = (0.1, 0.1)
disp = 0.1
task = PlaceObjectAtCoordTask(
    obj_type="apple",
    target_position=target_coords,
    allowable_displacement=disp,
)

Scenario(task=task, scene_config=scene_config, scene_config_path=path_to_your_config)

# define subtask that require
receive_robot_pos_subtask = CheckArgsToolCallSubTask(
    expected_tool_name="receive_ros2_message",
    expected_args={"topic": "/robot_position"},
    expected_optional_args={
        "timeout_sec": int
    },  # if there is not exact value exepected, you can pass type
)
# use OrderedCallValidator as there is only 1 subtask
topics_ord_val = OrderedCallsValidator(subtasks=[receive_robot_pos_subtask])


class GetROS2RobotPositionTask(BasicTask):
    complexity = "easy"

    @property
    def available_tools(self) -> List[BaseTool]:
        return [
            # define which topics will be seen by agent
            MockGetROS2TopicsNamesAndTypesTool(
                mock_topics_names_and_types=[
                    "topic: /attached_collision_object\ntype: moveit_msgs/msg/AttachedCollisionObject\n",
                    "topic: /clock\ntype: rosgraph_msgs/msg/Clock\n",
                    "topic: /collision_object\ntype: moveit_msgs/msg/CollisionObject\n",
                    "topic: /robot_position\n type: sensor_msgs/msg/RobotPosition",
                ]
            ),
        ]

    def get_prompt(self) -> str:
        return "Get the position of the robot."

# optionally pass number of extra tool calls
task = GetROS2RobotPositionTask(validators=[topics_ord_val], extra_tool_calls=1)
```

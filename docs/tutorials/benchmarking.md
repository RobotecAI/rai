# Benchmarking

!!! note

    If you aren't familiar with our benchmark package, please read [RAI Bench](../simulation_and_benchmarking/rai_bench.md) first.

Currently, we offer 2 predefined benchmarks:

-   [Manipulation_O3DE](#manipulation-o3de)
-   [Tool_Calling_Agent](#tool-calling-agent)

If you want to test multiple models across different benchmark configurations, go to [Testing Models](#testing-models).

If your goal is creating custom tasks and scenarios, visit [Creating Custom Tasks](#creating-custom-tasks).

## Manipulation O3DE

-   Follow the main setup [Basic Setup](../setup/install.md) and setup from [Manipulation demo Setup](../demos/manipulation.md#setup)
-   To see available options run:
    ```bash
    python src/rai_bench/rai_bench/examples/manipulation_o3de.py --help
    ```
-   Example usage:

    ```bash
    python src/rai_bench/rai_bench/examples/manipulation_o3de.py --model-name qwen2.5:7b --vendor ollama --levels trivial
    ```

    !!! note

          When using Ollama, be sure to pull the model first.

    !!! warning

          Running all scenarios will take a while. If you want to just try it out, we recommend choosing just one level of difficulty.

## Tool Calling Agent

-   This benchmark does not require any additional setup besides the main one [Basic Setup](../setup/install.md)
-   To see available options run:
    ```bash
    python src/rai_bench/rai_bench/examples/tool_calling_agent.py --help
    ```
-   Example usage:

```bash
python src/rai_bench/rai_bench/examples/tool_calling_agent.py --model-name qwen2.5:7b --vendor ollama --extra-tool-calls 5 --task-types basic  --n-shots 5 --prompt-detail descriptive --complexities easy
```

## Testing Models

The best way of benchmarking your models is using the `src/rai_bench/rai_bench/examples/benchmarking_models.py`

Feel free to modify the benchmark configs to suit your needs, you can choose every possible set of params
and the benchmark will be run tasks with every combination:

```python
if __name__ == "__main__":
    # Define models you want to benchmark
    model_names = ["qwen3:4b", "llama3.2:3b"]
    vendors = ["ollama", "ollama"]

    # Define benchmarks that will be used
    mani_conf = ManipulationO3DEBenchmarkConfig(
        o3de_config_path="src/rai_bench/rai_bench/manipulation_o3de/predefined/configs/o3de_config.yaml",
        levels=[  # define what difficulty of tasks to include in benchmark
            "trivial",
            "easy",
        ],
        repeats=1,  # how many times to repeat
    )
    tool_conf = ToolCallingAgentBenchmarkConfig(
        extra_tool_calls=[0, 5],  # how many extra tool calls allowed to still pass
        task_types=[  # what types of tasks to include
            "basic",
            "manipulation",
        ],
        N_shots=[0, 2],  # examples in system prompt
        prompt_detail=["brief", "descriptive"],  # how descriptive should task prompt be
        repeats=1,
    )

    out_dir = "src/rai_bench/rai_bench/experiments"
    test_models(
        model_names=model_names,
        vendors=vendors,
        benchmark_configs=[mani_conf, tool_conf],
        out_dir=out_dir,
        # if you want to pass any additinal args to model
        additional_model_args=[
            {"reasoning": False},
            {},
        ],
    )
```

Based on the example above the `Tool Calling` benchmark will run basic, spatial_reasoning and custom_interfaces tasks with every configuration of [extra_tool_calls x N_shots x prompt_detail] provided which will result in almost 500 tasks. Manipulation benchmark will run all specified task level once as there is no additional params. Reapeat is set to 1 in both configs so there will be no additional runs.

!!! note

    When using ollama vendor make sure to pull used models first

## Viewing Results

From every benchmark run, there will be results saved in the provided output directory:

-   Logs - in `benchmark.log` file
-   results_summary.csv - for overall metrics
-   results.csv - for detailed results of every task/scenario

When using `test_models`, the output directories will be saved as `<run_datetime>/<benchmark_name>/<model>/<repeat>/...` and this format can be visualized with our Streamlit script:

```bash
streamlit run src/rai_bench/rai_bench/examples/visualise_streamlit.py
```

## Creating Custom Tasks

### Manipulation O3DE Scenarios

To create your own Scenarios, you will need a Scene Config and Task - check out example `src/rai_bench/rai_bench/examples/custom_scenario.py`.
You can combine already existing Scene and existing Task to create a new Scenario like:

```python
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

# configure existing Task with different params
target_coords = (0.1, 0.1)
disp = 0.1
task = PlaceObjectAtCoordTask(
    obj_type="apple",
    target_position=target_coords,
    allowable_displacement=disp,
)

Scenario(task=task, scene_config=scene_config, scene_config_path=path_to_your_config)
```

But you can also create them from scratch.
Creating a Scene Config is very easy, just declare entities in a YAML file like:

```yaml
entities:
  - name: apple1
    prefab_name: apple # make sure that this prefab exists in simulation
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

Creating your own Task will require slightly more effort. Let's start with something simple - a Task that will require throwing given objects off the table:

```python
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


# configure existing Task with different params
target_coords = (0.1, 0.1)
disp = 0.1
task = ThrowObjectsOffTableTask(
    obj_types=["apple"],
)

super_scenario = Scenario(
    task=task, scene_config=scene_config, scene_config_path=path_to_your_config
)
```

As `obj_types` is parameterizable, it enables various variants of this Task. In combination with a lot of simulation configs available, it means that a single Task can provide dozens of scenarios.

Then yo test it simply run:

```python
##### Now you can run it in benchmark ##################
if __name__ == "__main__":
    from pathlib import Path

    from rai_bench import (
        define_benchmark_logger,
    )
    from rai_bench.manipulation_o3de import run_benchmark
    from rai_bench.utils import get_llm_for_benchmark

    experiment_dir = Path(out_dir="src/rai_bench/experiments/custom_task/")

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

```

Congratulations, you just created and launched your first Scenario from scratch!

### Tool Calling Tasks

To create a Tool Calling Task, you will need to define Subtasks, Validators, and Task itself.
Check the example `src/rai_bench/rai_bench/examples/custom_task.py`.
Let's create a basic task that requires using a tool to receive a message from a specific topic.

```python
from typing import List

from langchain_core.tools import BaseTool

from rai_bench.tool_calling_agent.interfaces import Task, TaskArgs
from rai_bench.tool_calling_agent.mocked_tools import (
    MockGetROS2TopicsNamesAndTypesTool,
    MockReceiveROS2MessageTool,
)
from rai_bench.tool_calling_agent.subtasks import (
    CheckArgsToolCallSubTask,
)
from rai_bench.tool_calling_agent.validators import (
    OrderedCallsValidator,
)


# This Task will check if robot can receive msessage from specified topic
class GetROS2RobotPositionTask(Task):
    complexity = "easy"
    type = "custom"

    @property
    def available_tools(self) -> List[BaseTool]:
        # define topics that will be seen by agent
        TOPICS = [
            "/robot_position",
            "/attached_collision_object",
            "/clock",
            "/collision_object",
        ]

        TOPICS_STRING = [
            "topic: /attached_collision_object\ntype: moveit_msgs/msg/AttachedCollisionObject\n",
            "topic: /clock\ntype: rosgraph_msgs/msg/Clock\n",
            "topic: /collision_object\ntype: moveit_msgs/msg/CollisionObject\n",
            "topic: /robot_position\n type: sensor_msgs/msg/RobotPosition",
        ]
        # define which tools will be available for agent
        return [
            MockGetROS2TopicsNamesAndTypesTool(
                mock_topics_names_and_types=TOPICS_STRING
            ),
            MockReceiveROS2MessageTool(available_topics=TOPICS),
        ]

    def get_system_prompt(self) -> str:
        return "You are a ROS 2 expert that want to solve tasks. You have access to various tools that allow you to query the ROS 2 system."

    def get_base_prompt(self) -> str:
        return "Get the position of the robot."

    def get_prompt(self) -> str:
        # Create versions for different levels
        if self.prompt_detail == "brief":
            return self.get_base_prompt()
        else:
            return (
                f"{self.get_base_prompt()} "
                "You can discover what topics are currently active."
            )

    @property
    def optional_tool_calls_number(self) -> int:
        # Listing topics before getting any message is fine
        return 1


# define subtask
receive_robot_pos_subtask = CheckArgsToolCallSubTask(
    expected_tool_name="receive_ros2_message",
    expected_args={"topic": "/robot_position"},
    expected_optional_args={
        "timeout_sec": int  # if there is not exact value expected, you can pass type
    },
)
# use OrderedCallValidator as there is only 1 subtask to check
topics_ord_val = OrderedCallsValidator(subtasks=[receive_robot_pos_subtask])


# optionally pass number of extra tool calls
args = TaskArgs(extra_tool_calls=0)
super_task = GetROS2RobotPositionTask(validators=[topics_ord_val], task_args=args)
```

Then run it with:

```python
##### Now you can run it in benchmark ##################
if __name__ == "__main__":
    from pathlib import Path

    from rai_bench import (
        define_benchmark_logger,
    )
    from rai_bench.tool_calling_agent import (
        run_benchmark,
    )
    from rai_bench.utils import get_llm_for_benchmark

    experiment_dir = Path("src/rai_bench/rai_bench/experiments/custom_task")
    experiment_dir.mkdir(parents=True, exist_ok=True)
    bench_logger = define_benchmark_logger(out_dir=experiment_dir)

    super_task.set_logger(bench_logger)

    llm = get_llm_for_benchmark(
        model_name="gpt-4o",
        vendor="openai",
    )

    run_benchmark(
        llm=llm,
        out_dir=experiment_dir,
        tasks=[super_task],
        bench_logger=bench_logger,
    )
```

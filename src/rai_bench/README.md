# RAI Benchmarks

The RAI Bench is a package including benchmarks and providing frame for creating new benchmarks

## Manipulation O3DE Benchmark

The Manipulation O3DE Benchmark [manipulation_o3de_benchmark_module](./rai_bench//manipulation_o3de/) provides tasks and scene configurations for robotic arm manipulation simulation in O3DE. The tasks use a common `ManipulationTask` logic and can be parameterized, which allows for many task variants. The current tasks include:

-   **MoveObjectToLeftTask**
-   **GroupObjectsTask**
-   **BuildCubeTowerTask**
-   **PlaceObjectAtCoordTask**
-   **RotateObjectTask** (currently not applicable due to limitations in the ManipulatorMoveTo tool)

The result of a task is a value between 0 and 1, calculated like initially_misplaced_now_correct / initially_misplaced. This score is calculated at the end of each scenario.

### Frame Components

-   `Task`
-   `Scenario`
-   `Benchmark`

For more information about these classes go to -> [benchmark](./rai_bench//manipulation_o3de/benchmark.py) and [Task](./rai_bench//manipulation_o3de//interfaces.py) and

### Example usage

Example of how to load scenes, define scenarios and run benchmark can be found in [manipulation_o3de_benchmark_example](rai_bench/examples/manipulation_o3de/main.py)

Scenarios can be loaded manually like:

```python
one_carrot_simulation_config = O3DExROS2SimulationConfig.load_config(
        base_config_path=Path("path_to_scene.yaml"),
        connector_config_path=Path("path_to_o3de_config.yaml"),
    )

Scenario(task=GrabCarrotTask(logger=some_logger), simulation_config=one_carrot_simulation_config)
```

or automatically like:

```python
scenarios = Benchmark.create_scenarios(
        tasks=tasks, simulation_configs=simulations_configs
    )
```

which will result in list of scenarios with combination of every possible task and scene(task decides if scene config is suitable for it).

or can be imported from exisitng packets [scenarios_packets](rai_bench/examples/manipulation_o3de/scenarios.py):

```python
t_scenarios = trivial_scenarios(
        configs_dir=configs_dir, connector_path=connector_path, logger=bench_logger
    )
e_scenarios = easy_scenarios(
    configs_dir=configs_dir, connector_path=connector_path, logger=bench_logger
)
m_scenarios = medium_scenarios(
    configs_dir=configs_dir, connector_path=connector_path, logger=bench_logger
)
h_scenarios = hard_scenarios(
    configs_dir=configs_dir, connector_path=connector_path, logger=bench_logger
)
vh_scenarios = very_hard_scenarios(
    configs_dir=configs_dir, connector_path=connector_path, logger=bench_logger
)
```

which are grouped by their subjective difficulty. For now there are 10 trivial, 42 easy, 23 medium, 38 hard and 47 very hard scenarios.
Check docstrings and code in [scenarios_packets](rai_bench/examples/manipulation_o3de/scenarios.py) if you want to know how scenarios are assigned to difficulty level.

### Running

1. Download O3DE simulation binary and unzip it.

    - [ros2-humble](https://robotec-ml-rai-public.s3.eu-north-1.amazonaws.com/RAIManipulationDemo_jammyhumble.zip)
    - [ros2-jazzy](https://robotec-ml-rai-public.s3.eu-north-1.amazonaws.com/RAIManipulationDemo_noblejazzy.zip)

2. Follow step 2 from [Manipulation demo Setup section](../../docs/demos/manipulation.md#setup)

3. Adjust the path to the binary in: [o3de_config.yaml](./rai_bench/examples/manipulation_o3de/configs/o3de_config.yaml)
4. Choose the model you want to run and a vendor.
    > [!NOTE]
    > The configs of vendors are defined in [config.toml](../../config.toml) Change ithem if needed.
5. Run benchmark with:

```bash
cd rai
source setup_shell.sh
python src/rai_bench/rai_bench/examples/manipulation_o3de/main.py --model-name llama3.2  --vendor ollama
```

> [!NOTE]
> For now benchmark runs all available scenarios (~160). See [Examples](#example-usege)
> section for details.

### Development

When creating new task or changing existing ones, make sure to add unit tests for score calculation in [rai_bench_tests](../../tests/rai_bench/manipulation_o3de/tasks/).
This applies also when you are adding or changing the helper methods in `Task` or `ManipulationTask`.

The number of scenarios can be easily extened without writing new tasks, by increasing number of variants of the same task and adding more simulation configs but it won't improve variety of scenarios as much as creating new tasks.

## Tool Calling Agent Benchmark

The Tool Calling Agent Benchmark is the benchmark for LangChain tool calling agents. It includes a set of tasks and a benchmark that evaluates the performance of the agent on those tasks by verifying the correctness of the tool calls requested by the agent. The benchmark is integrated with LangSmith and Langfuse tracing backends to easily track the performance of the agents.

### Frame Components

-   [Tool Calling Agent Benchmark](rai_bench//tool_calling_agent/benchmark.py) - Benchmark for LangChain tool calling agents
-   [Scores tracing](rai_bench/tool_calling_agent_bench/scores_tracing.py) - Component handling sending scores to tracing backends
-   [Interfaces](rai_bench//tool_calling_agent/interfaces.py) - Interfaces for validation classes - Task, Validator, SubTask
    For detailed description of validation visit -> [Validation](.//rai_bench/docs/tool_calling_agent_benchmark.md)

[tool_calling_agent_test_bench.py](rai_bench/examples/tool_calling_agent/main.py) - Script providing benchmark on tasks based on the ROS2 tools usage.

### Example Usage

Validators can be constructed from any SubTasks, Tasks can be validated by any numer of Validators, which makes whole validation process incredibly versital.

```python
# subtasks
get_topics_subtask = CheckArgsToolCallSubTask(
    expected_tool_name="get_ros2_topics_names_and_types"
)
color_image_subtask = CheckArgsToolCallSubTask(
    expected_tool_name="get_ros2_image", expected_args={"topic": "/camera_image_color"}
)
# validators - consist of subtasks
topics_ord_val = OrderedCallsValidator(subtasks=[get_topics_subtask])
color_image_ord_val = OrderedCallsValidator(subtasks=[color_image_subtask])
topics_and_color_image_ord_val = OrderedCallsValidator(
    subtasks=[
        get_topics_subtask,
        color_image_subtask,
    ]
)
# tasks - validated by list of validators
GetROS2TopicsTask(validators=[topics_ord_val])
GetROS2RGBCameraTask(validators=[topics_and_color_image_ord_val]),
GetROS2RGBCameraTask(validators=[topics_ord_val, color_image_ord_val]),
```

### Running

To set up tracing backends, please follow the instructions in the [tracing.md](../../docs/tracing.md) document.

To run the benchmark:

```bash
cd rai
source setup_shell.sh
python src/rai_bench/rai_bench/examples/tool_calling_agent/main.py
```

There is also flags to declare model type and vendor:

```bash
python src/rai_bench/rai_bench/examples/tool_calling_agent/main.py --model-name llama3.2 --vendor ollama
```

> [!NOTE]
> The configs of vendors are defined in [config.toml](../../config.toml) Change ithem if needed.

## Testing Models

To test multiple models, different benchamrks or couple repeats in one go - use script [test_models](./rai_bench/examples/test_models.py)

Modify these params:

```python
models_name = ["llama3.2", "qwen2.5:7b"]
vendors = ["ollama", "ollama"]
benchmarks = ["tool_calling_agent"]
repeats = 1
```

to your liking and run the script!

```bash
python src/rai_bench/rai_bench/examples/test_models.py
```

### Results and Visualization

All results from running benchmarks will be saved to folder [experiments](./rai_bench/experiments/)

If you run single benchmark test like:

```bash
python src/rai_bench/rai_bench/examples/<benchmark_name>/main.py
```

Results will be saved to dedicated directory named `<benchmark_name>`

When you run a test via:

```bash
python src/rai_bench/rai_bench/examples/test_models.py
```

results will be saved to separate folder in [results](./rai_bench/experiments/), with prefix `run_`

To visualise the results run:

```bash
streamlit run src/rai_bench/rai_bench/results_processing/visualise.py
```

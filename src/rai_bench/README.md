## RAI Benchmark

### Description

The RAI Bench is a package including benchmarks and providing frame for creating new benchmarks

### Frame Components

- `Task`
- `Scenario`
- `Benchmark`

For more information about these classes go to -> [benchmark_model](./rai_bench/benchmark_model.py)

### O3DE TEST BENCHMARK

The O3DE Test Benchmark [o3de_test_benchmark_module](./rai_bench/o3de_test_bench/) provides tasks and scene configurations for robotic arm manipulation task. The tasks use a common `ManipulationTask` logic and can be parameterized, which allows for many task variants. The current tasks include:

- **MoveObjectToLeftTask**
- **GroupObjectsTask**
- **BuildCubeTowerTask**
- **PlaceObjectAtCoordTask**
- **RotateObjectTask** (currently not applicable due to limitations in the ManipulatorMoveTo tool)

The result of a task is a value between 0 and 1, calculated like initially_misplaced_now_correct / initially_misplaced. This score is calculated at the end of each scenario.

### Example usage

Example of how to load scenes, define scenarios and run benchmark can be found in [o3de_test_benchmark_example](./rai_bench/examples/o3de_test_benchmark.py)

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

or can be imported from exisitng packets [scenarios_packets](./rai_bench/o3de_test_bench/scenarios.py):

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
Check docstrings and code in [scenarios_packets](./rai_bench/o3de_test_bench/scenarios.py) if you want to know how scenarios are assigned to difficulty level.

### Development

When creating new task or changing existing ones, make sure to add unit tests for score calculation in [rai_bench_tests](../../tests/rai_bench/).
This applies also when you are adding or changing the helper methods in `Task` or `ManipulationTask`.

The number of scenarios can be easily extened without writing new tasks, by increasing number of variants of the same task and adding more simulation configs but it won't improve variety of scenarios as much as creating new tasks.

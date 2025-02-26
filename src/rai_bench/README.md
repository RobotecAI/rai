## RAI Benchmark

## Description

The RAI Bench is a package including benchmarks and providing frame for creating new benchmarks

## Frame Components

Frame components can be found in `src/rai_bench/rai_bench/benchmark_model.py`

- `Task`
- `Scenario`
- `Benchmark`

For more information about these classes go to -> `src/rai_bench/rai_bench/benchmark_model.py`

### O3DE TEST BENCHMARK

O3DE Test Benchmark (`src/rai_bench/rai_bench/o3de_test_bench/`), contains 2 Tasks(`tasks/`) - GrabCarrotTask and PlaceCubesTask (these tasks implement calculating scores) and 4 scene_configs(`configs/`) for O3DE robotic arm simulation.

Both tasks calculate score, taking into consideration 4 values:

- initially_misplaced_now_correct
- initially_misplaced_still_incorrect
- initially_correct_still_correct
- initially_correct_now_incorrect

The result is a value between 0 and 1, calculated like (initially_misplaced_now_correct + initially_correct_still_correct) / number_of_initial_objects.
This score is calculated at the beggining and at the end of each scenario.

### Example usage

Example of how to load scenes, define scenarios and run benchmark can be found in `src/rai_bench/rai_bench/examples/o3de_test_benchmark.py`

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

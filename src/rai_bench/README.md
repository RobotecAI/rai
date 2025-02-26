## RAI Benchmark

## Description

The RAI Bench is a package including benchmarks and providing frame for creating new benchmarks

## Frame Components

Frame components can be found in `src/rai_bench/rai_bench/benchmark_model.py`

- `Task` - abstract class for creating specific task. It introduces helper funtions that make it easier to calculate metrics/scores. Your custom tasks must implement a prompt got agent to do, a way to calculate a result and a validation if given scene config suits the task.
- `Scenario` - class defined by a Scene and Task.
- `Benchmark` - class responsible for running and logging scenarios.

### O3DE TEST BENCHMARK

O3DE Test Benchmark (src/rai_bench/rai_bench/o3de_test_bench/), contains 2 Tasks(tasks/) - GrabCarrotTask and PlaceCubesTask (these tasks implement calculating scores) and 4 scene_configs(configs/) for O3DE robotic arm simulation.

Both tasks calculate score, taking into consideration 4 values:

- initially_misplaced_now_correct - when the object which was in the incorrect place at the start, is in a correct place at the end
- initially_misplaced_still_incorrect - when the object which was in the incorrect place at the start, is in a incorrect place at the end
- initially_correct_still_correct - when the object which was in the correct place at the start, is in a correct place at the end
- initially_correct_now_incorrect - when the object which was in the correct place at the start, is in a incorrect place at the end

The result is a value between 0 and 1, calculated like (initially_misplaced_now_correct + initially_correct_still_correct) / number_of_initial_objects.
This score is calculated at the beggining and at the end of each scenario.

### Example usage

Example of how to load scenes, define scenarios and run benchmark can be found in `src/rai_bench/rai_bench/benchmark_main.py`

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

Both approaches can be found in `main.py`

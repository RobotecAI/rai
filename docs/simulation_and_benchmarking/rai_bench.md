# RAI Bench

RAI Bench is a framework for creating and running benchmarks in simulation environments. It builds on top of RAI Sim to provide a structured way to define tasks, scenarios, and evaluate performance.

## Core Components

### Task

The `Task` class is an abstract base class that defines the interface for benchmark tasks. Each task must implement:

-   `get_prompt()`: Returns the task instruction for the agent
-   `validate_config()`: Verifies if a simulation configuration is suitable for the task
-   `calculate_result()`: Computes the task score (0.0 to 1.0)

### ManipulationTask

A specialized `Task` class for manipulation tasks that provides common functionality for:

-   Object type filtering
-   Placement validation
-   Score calculation based on object positions

### Scenario

A `Scenario` represents a specific test case combining:

-   A task to be executed
-   A simulation configuration
-   The path to the configuration file

### Benchmark

The `Benchmark` class manages the execution of scenarios and collects results. It provides:

-   Scenario execution management
-   Performance metrics tracking
-   Results logging and export

## Available Tasks

The framework includes several predefined manipulation tasks:

1. **MoveObjectsToLeftTask**

    - Moves specified objects to the left side of the table
    - Success measured by objects' y-coordinate being positive

2. **PlaceObjectAtCoordTask**

    - Places an object at specific coordinates
    - Success measured by distance from target position

3. **PlaceCubesTask**

    - Places cubes adjacent to each other
    - Success measured by proximity to other cubes

4. **BuildCubeTowerTask**

    - Stacks cubes to form a tower
    - Success measured by height and stability

5. **GroupObjectsTask**

    - Groups objects of specified types together
    - Success measured by object proximity

## Usage

### Creating Scenarios

Scenarios can be created manually:

```python
scenario = Scenario(
    task=MoveObjectsToLeftTask(obj_types=["cube"]),
    simulation_config=simulation_config,
    simulation_config_path="path/to/config.yaml"
)
```

Or automatically using the `Benchmark.create_scenarios()` method:

```python
scenarios = Benchmark.create_scenarios(
    tasks=tasks,
    simulation_configs=configs,
    simulation_configs_paths=config_paths
)
```

### Running Benchmarks

```python
benchmark = Benchmark(
    simulation_bridge=bridge,
    scenarios=scenarios,
    results_filename="results.csv"
)
```

## Scoring System

Tasks are scored on a scale from 0.0 to 1.0, where:

-   0.0 indicates no improvement or worse performance
-   1.0 indicates perfect completion

The score is typically calculated as:

```
score = (correctly_placed_now - correctly_placed_initially) / initially_incorrect
```

## Integration with RAI Sim

RAI Bench leverages RAI Sim's simulator-agnostic interface to:

-   Execute tasks in any supported simulation environment
-   Access and manipulate simulation entities
-   Monitor scene state and object positions
-   Manage simulation configurations

This integration allows for:

-   Consistent task execution across different simulators
-   Reliable performance measurement
-   Flexible task definition
-   Comprehensive result analysis

## Configuration

Simulation configurations are defined in YAML files that specify:

-   Scene setup
-   Object types and positions
-   Task-specific parameters

## Error Handling

The framework includes comprehensive error handling for:

-   Invalid configurations
-   Task validation failures
-   Simulation errors
-   Performance tracking

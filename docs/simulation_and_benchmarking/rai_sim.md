# RAI Sim

RAI Sim is a package that provides an interface for connecting with various simulation environments. It is designed to be simulator-agnostic, allowing RAI to work with any simulation environment that implements the required interface.

## Core Components

### SimulationBridge

The `SimulationBridge` is an abstract base class that defines the interface for communicating with different simulation environments. It provides the following key functionalities:

-   Scene setup and management
-   Entity spawning and despawning
-   Object pose retrieval
-   Scene state monitoring

### SimulationConfig

The `SimulationConfig` is a base configuration class that specifies the entities to be spawned in the simulation. Each simulation bridge can extend this with additional parameters specific to its implementation.

Key features:

-   Entity list management
-   Unique name validation
-   YAML configuration loading
-   Frame ID specification

### SceneState

The `SceneState` class maintains information about the current state of the simulation scene, including:

-   List of currently spawned entities
-   Current poses of all entities
-   Entity tracking and management

## Implementation Details

### Entity Management

The package provides two main entity classes:

-   `Entity`: Represents an entity that can be spawned in the simulation
-   `SpawnedEntity`: Represents an entity that has been successfully spawned

### Tools

RAI Sim includes utility tools for working with simulations:

-   `GetObjectPositionsGroundTruthTool`: Retrieves accurate positional data for objects in the simulation

## Usage

To use RAI Sim with a specific simulation environment:

1. Create a custom `SimulationBridge` implementation for your simulator
2. Extend `SimulationConfig` with simulator-specific parameters
3. Implement the required abstract methods:
    - `setup_scene`
    - `_spawn_entity`
    - `_despawn_entity`
    - `get_object_pose`
    - `get_scene_state`

## Configuration

Simulation configurations are typically loaded from YAML files with the following structure:

```yaml
frame_id: <reference_frame>
entities:
    - name: <unique_entity_name>
      prefab_name: <resource_name>
      pose:
          translation:
              x: <x_coordinate>
              y: <y_coordinate>
              z: <z_coordinate>
          rotation:
              x: <x_rotation>
              y: <y_rotation>
              z: <z_rotation>
              w: <w_rotation>
```

## Error Handling

The package includes comprehensive error handling for:

-   Duplicate entity names
-   Failed entity spawning/despawning
-   Invalid configurations
-   Simulation process management

## Integration with RAI Bench

RAI Sim serves as the foundation for RAI Bench by providing:

-   A consistent interface for all simulation environments
-   Entity management and tracking
-   Scene state monitoring
-   Configuration management

This allows RAI Bench to focus on task definition and evaluation while remaining simulator-agnostic.

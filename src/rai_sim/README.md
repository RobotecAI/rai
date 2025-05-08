## RAI Sim

## Description

The RAI Sim is a package providing interface to implement connection with a specific simulation.

### Components

-   `SimulationBridge` - An interface for connecting with a specific simulation. It manages scene setup, spawning, despawning objects, getting current state of the scene.

-   `SimulationConfig` - base config class to specify the entities to be spawned. For each simulation bridge there should be specified custom simulation config specifying additional parameters needed to run and connect with the simulation.

-   `SceneState` - stores the current info about spawned entities

### Example implementation

-   `O3DExROS2Bridge` - An implementation of SimulationBridge for working with simulation based on O3DE and ROS2.
-   `O3DExROS2SimulationConfig` - config class for `O3DExROS2Bridge`

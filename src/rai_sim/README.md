## RAI Sim

## Description

The RAI Sim is a package providing interface to implement connection with a specific simulation.

### Components

- `SimulationBridge` - An interface for connecting with a specific simulation. It manages scene setup, spawning, despawning objects, getting current state of the scene.

- `SimulationConfig` - base config class to specify the entities to be spawned. For each simulation bridge there should be specified custom simulation config specifying additional parameters needed to run and connect with the simulation.

- `SceneState` - stores the current info about spawned entities

### Example O3DE with ROS2 implementation

- `O3DExROS2Bridge` - An implementation of SimulationBridge for working with simulation based on O3DE and ROS2.
- `O3DExROS2SimulationConfig` - config class for `O3DExROS2Bridge`

#### Example usage with rai-manipulation-demo

1. Setup RAI - follow [README.md](https://github.com/RobotecAI/rai)
2. Setup rai-manipulation-demo - follow [manipulation.md](https://github.com/RobotecAI/rai/blob/main/docs/demos/manipulation.md)
3. Click to download GameLauncher binary from s3 bucket: [humble](https://robotec-ml-rai-public.s3.eu-north-1.amazonaws.com/RAIManipulationDemo_jammyhumble.zip) or [jazzy](https://robotec-ml-rai-public.s3.eu-north-1.amazonaws.com/RAIManipulationDemo_noblejazzy.zip).
4. Populate the .yaml config with the following content:

```
binary_path: /path/to/your/GameLauncher
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

5. Run example script:

# TODO (test it)

# TODO (change connector to bridge everywhere, check it)

```
import rclpy
import logging
import time
from pathlib import Path

from rai_sim.o3de.o3de_bridge import O3DExROS2Bridge, O3DExROS2SimulationConfig, O3DExROS2BridgeProcessesMonitor
from rai.communication.ros2.connectors import ROS2ARIConnector


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )
    logger = logging.getLogger("O3DExROS2-Sim")

    o3de = None
    connector = None
    monitor = None

    try:
        rclpy.init()

        connector = ROS2ARIConnector()
        o3de = O3DExROS2Bridge(connector, logger=logger)

        monitor = O3DExROS2BridgeProcessesMonitor(o3de, logger=logger)

        scene_config = O3DExROS2SimulationConfig.load_config(
            base_config_path=Path("src/rai_bench/rai_bench/o3de_test_bench/configs/scene1.yaml"),
            connector_config_path=Path("src/rai_bench/rai_bench/o3de_test_bench/configs/o3de_config.yaml")
        )

        o3de.setup_scene(scene_config)

        time.sleep(5)

        scene_config = O3DExROS2SimulationConfig.load_config(
            base_config_path=Path("src/rai_bench/rai_bench/o3de_test_bench/configs/scene2.yaml"),
            connector_config_path=Path("src/rai_bench/rai_bench/o3de_test_bench/configs/o3de_config.yaml")
        )
        o3de.setup_scene(scene_config)
        time.sleep(5)

    except KeyboardInterrupt:
        logger.info("Received KeyboardInterrupt.")
    except Exception as e:
        logger.exception(f"Error occurred: {e}")
        raise e

    finally:
        logger.info("Shutting down script...")

        if monitor:
            logger.info("Shutting down processes monitor")
            monitor.shutdown()

        if o3de:
            logger.info("Shutting down O3DE bridge")
            o3de.shutdown()

        if connector:
            logger.info("Shutting down connector")
            connector.shutdown()

        logger.info("Shutting down ROS2")
        rclpy.try_shutdown()

        logger.info("Cleanup complete")

```

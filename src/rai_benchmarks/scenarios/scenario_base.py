import random

import rclpy
from gazebo_msgs.srv import DeleteEntity, SpawnEntity
from geometry_msgs.msg import Pose, PoseStamped, Quaternion
from rclpy.client import Client
from rclpy.node import Node
from rclpy.task import Future


class ScenarioBase:
    """
    Base class for a scenario. A scenario is a task that the agent has to perform.
    """

    def __init__(
        self,
        spawn_client: Client,
        delete_client: Client,
        manipulator_client: Node,
        node: Node,
    ):
        self.spawn_client = spawn_client
        self.delete_client = delete_client
        self.manipulator_client = manipulator_client
        self.node = node
        self.entities: dict[str, str] = {}

    def __del__(self):
        for name in self.entities:
            self.delete_entity(name)

    def get_prompt(self) -> str:
        """
        Returns a prompt that describes the task that the agent has to perform.
        """
        return "Please do something interesting"

    def reset(self) -> None:
        """
        Resets the scenario to its initial state.

        This method is called before the scenario is started.
        """

        for name in self.entities:
            self.delete_entity(name)
        self.entities = {}

    def spawn_entity(self, prefab_name: str, name: str, pose: Pose) -> None:
        """
        Spawns an entity in the simulation.

        Args:
            prefab_name: The name of the prefab to spawn.
            name: The name of the entity, by which it can be later referenced.
            pose: The pose of the entity.
        """
        req = SpawnEntity.Request()
        req.name = prefab_name
        req.xml = ""
        req.robot_namespace = name
        req.initial_pose = pose

        self.spawn_client.call_async(req).add_done_callback(
            lambda future: self.entity_spawned_callback(future, name)
        )

    def spawn_entities_in_random_positions(
        self, prefab_names: list[str], names: list[str]
    ) -> None:
        """
        Spawns entities randomly positione around the table.

        Args:
            prefab_names: A list of prefab names to spawn.
            names: A list of names for the entities, by which they can be later referenced.
        """
        grid = [(x / 10.0 + 0.4, y / 10.0) for x in range(-1, 2) for y in range(-3, 4)]

        positions = random.sample(grid, k=len(prefab_names))
        for prefab_name, name, position in zip(prefab_names, names, positions):
            pose = PoseStamped()
            pose.header.frame_id = "world"
            pose.pose.position.x = position[0]
            pose.pose.position.y = position[1]
            pose.pose.position.z = 0.05
            pose.pose.orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
            pose_transformed = self.node.tf2_buffer.transform(
                pose, "odom", timeout=rclpy.time.Duration(seconds=5.0)
            )
            self.spawn_entity(prefab_name, name, pose_transformed.pose)

    def delete_entity(self, name: str) -> None:
        req = DeleteEntity.Request()
        req.name = self.entities[name]

        self.delete_client.call_async(req)

    def get_entity_pose(self, name: str) -> Pose:
        """
        Returns the pose of an entity.
        """
        pose = PoseStamped()
        entity_frame = name + "/"
        pose.header.frame_id = entity_frame
        pose = self.node.tf2_buffer.transform(
            pose, entity_frame + "odom", timeout=rclpy.time.Duration(seconds=5.0)
        )
        return pose.pose

    def pose_transformed(self, pose: Pose) -> Pose:
        """
        Transforms the pose into the frame of the panda robot.
        """
        pose_stamped = PoseStamped()
        pose_stamped.pose = pose
        pose_stamped.header.frame_id = "odom"
        pose = self.node.tf2_buffer.transform(
            pose_stamped, "world", timeout=rclpy.time.Duration(seconds=5.0)
        ).pose
        pose.orientation = Quaternion(x=0.923880, y=-0.382683, z=0.0, w=0.0)
        return pose

    def step(self) -> tuple[float, bool]:
        """
        Performs a step in the scenario.

        Returns:
            progress: A float between 0.0 and 1.0 that represents the progress of the task.
            terminated: A boolean that indicates whether the task is terminated.
        """
        return 0.0, False

    def entity_spawned_callback(self, future: Future, name: str):
        result = future.result()
        if result.success:
            self.node.get_logger().info(
                f"Entity spawned: {name} ({result.status_message})"
            )
            self.entities[name] = result.status_message
        else:
            self.node.get_logger().error(
                f"Failed to spawn entity: {result.status_message}"
            )

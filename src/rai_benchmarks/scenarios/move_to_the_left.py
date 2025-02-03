import rclpy
from geometry_msgs.msg import Point, PoseStamped, Quaternion
from rclpy.client import Client
from rclpy.node import Node
from rclpy.task import Future
from scenarios.scenario_base import ScenarioBase

from rai_interfaces.srv import ManipulatorMoveTo


class MoveToTheLeft(ScenarioBase):
    def __init__(
        self,
        spawn_client: Client,
        delete_client: Client,
        manipulator_client: Client,
        node: Node,
    ):
        super().__init__(spawn_client, delete_client, manipulator_client, node)

    def get_prompt(self):
        return "There are 5 apples on the right half of the table. Their Y position is positive. Move each of them to the left half of the table, such that they dont collide with each other. The Y position of the apples should be negative after the task is completed. First grab one apple using the 'grab' tool, and then drop it in the appropriate position using the 'drop' tool. Repeat this procedure for each apple."

    def reset(self):
        super().reset()

        for i in range(5):
            pose = PoseStamped()
            pose.header.frame_id = "world"
            pose.pose.position = Point(x=0.4, y=float(i) / 10 + 0.15, z=0.1)
            pose.pose.orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)

            pose_transformed = self.node.tf2_buffer.transform(
                pose, "odom", timeout=rclpy.time.Duration(seconds=5.0)
            )

            prefab = "apple"
            self.spawn_entity(prefab, f"{prefab}{i}", pose_transformed.pose)

    def calculate_progress(self):
        num_apples = sum(1 for name in self.entities if name.startswith("apple"))
        num_good_apples = sum(
            1
            for name in self.entities
            if name.startswith("apple")
            and self.pose_transformed(self.get_entity_pose(name)).position.y < 0.0
        )
        return num_good_apples / num_apples

    def step(self):
        if len(self.entities) == 0:  # The entities are not spawned yet
            return 0.0, False

        progress = self.calculate_progress()

        return progress, progress >= 1.0


class MoveToTheLeftAuto(MoveToTheLeft):
    def __init__(
        self,
        spawn_client: Client,
        delete_client: Client,
        manipulator_client: Client,
        node: Node,
    ):
        super().__init__(spawn_client, delete_client, manipulator_client, node)
        self.manipulator_busy = False

    def reset(self):
        super().reset()

        self.manipulator_busy = False
        self.manipulator_queue = []

    def move_to_the_left(self, name: str):
        pose = self.get_entity_pose(name)
        pose.position.z += 0.1

        req = ManipulatorMoveTo.Request()
        req.initial_gripper_state = True
        req.target_pose.pose = self.pose_transformed(pose)
        req.final_gripper_state = False
        self.manipulator_queue.append(req)

        pose.position.y -= 0.6
        req = ManipulatorMoveTo.Request()
        req.initial_gripper_state = False
        req.target_pose.pose = self.pose_transformed(pose)
        req.final_gripper_state = True
        self.manipulator_queue.append(req)

    def move_callback(self, future: Future):
        result = future.result()
        if result.success:
            self.node.get_logger().debug("Move performed")
        else:
            self.node.get_logger().error("Failed to perform move")
        self.manipulator_busy = False

    def step(self):
        if len(self.entities) == 0:  # The entities are not spawned yet
            return 0.0, False

        progress = self.calculate_progress()

        if not self.manipulator_busy:
            if len(self.manipulator_queue) == 0:
                for name in self.entities:
                    pose = self.pose_transformed(self.get_entity_pose(name))
                    if pose.position.y > 0.0:
                        self.node.get_logger().info(
                            f"Moving {name} to the left from {pose.position.y}"
                        )
                        self.move_to_the_left(name)
                        break

            if len(self.manipulator_queue) > 0:
                req = self.manipulator_queue.pop(0)
                self.manipulator_busy = True
                self.manipulator_client.call_async(req).add_done_callback(
                    self.move_callback
                )

        return progress, progress >= 1.0 and not self.manipulator_busy

import rclpy
from geometry_msgs.msg import Point, PoseStamped, Quaternion
from rclpy.client import Client
from rclpy.node import Node
from rclpy.task import Future
from scenarios.scenario_base import ScenarioBase

from rai_interfaces.srv import ManipulatorMoveTo


class LongestObject(ScenarioBase):
    def __init__(
        self,
        spawn_client: Client,
        delete_client: Client,
        manipulator_client: Client,
        node: Node,
    ):
        super().__init__(spawn_client, delete_client, manipulator_client, node)

    def get_prompt(self):
        return "Put the longest object from the table into the toy box."

    def reset(self):
        super().reset()

        prefabs = ["apple", "yellow_cube", "blue_cube", "carrot"]
        self.spawn_entities_in_random_positions(prefabs, prefabs)

        pose = PoseStamped()
        pose.header.frame_id = "world"
        pose.pose.position = Point(x=0.4, y=-0.5, z=0.1)
        pose.pose.orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
        pose_transformed = self.node.tf2_buffer.transform(
            pose, "odom", timeout=rclpy.time.Duration(seconds=5.0)
        ).pose
        self.spawn_entity("toy_box", "toy_box", pose_transformed)

    def calculate_progress(self):
        if len(self.entities) == 0:
            return 0.0

        carrot_position = self.pose_transformed(self.get_entity_pose("carrot")).position
        toy_box_position = self.pose_transformed(
            self.get_entity_pose("toy_box")
        ).position

        def distance(a, b):
            return ((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2) ** 0.5

        return max(0.0, 1.0 - 3.0 * distance(carrot_position, toy_box_position))

    def step(self):
        if len(self.entities) == 0:
            return 0.0, False

        progress = self.calculate_progress()

        return progress, progress >= 0.5


class LongestObjectAuto(LongestObject):
    def __init__(
        self,
        spawn_client: Client,
        delete_client: Client,
        manipulator_client: Client,
        node: Node,
    ):
        self.manipulator_busy = False

        super().__init__(spawn_client, delete_client, manipulator_client, node)

    def reset(self):
        super().reset()

        self.manipulator_busy = False
        self.manipulator_queue = []

    def place_on_top(self, bot_object: str, top_object: str):
        pose = self.get_entity_pose(top_object)
        pose.position.z += 0.1

        req = ManipulatorMoveTo.Request()
        req.initial_gripper_state = True
        req.target_pose.pose = self.pose_transformed(pose)
        req.final_gripper_state = False
        self.manipulator_queue.append(req)

        pose = self.get_entity_pose(bot_object)
        pose.position.z += 0.2
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
        if len(self.entities) == 0:
            return 0.0, False

        progress = self.calculate_progress()

        if not self.manipulator_busy:
            if len(self.manipulator_queue) == 0 and progress < 0.8:
                self.place_on_top("toy_box", "carrot")

            if len(self.manipulator_queue) > 0:
                req = self.manipulator_queue.pop(0)
                self.manipulator_busy = True
                self.manipulator_client.call_async(req).add_done_callback(
                    self.move_callback
                )

        return progress, progress >= 0.8 and not self.manipulator_busy

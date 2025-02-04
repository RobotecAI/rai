from rclpy.client import Client
from rclpy.node import Node
from rclpy.task import Future
from scenarios.scenario_base import ScenarioBase

from rai_interfaces.srv import ManipulatorMoveTo


class ReplaceTypes(ScenarioBase):
    def __init__(
        self,
        spawn_client: Client,
        delete_client: Client,
        manipulator_client: Client,
        node: Node,
    ):
        self.vegetable_poses = []
        self.toy_poses = []
        self.current_index = 0

        super().__init__(spawn_client, delete_client, manipulator_client, node)

    def get_prompt(self):
        return "Replace the objects in such a way that all the toys are in the places of vegetables and vice versa."

    def reset(self):
        super().reset()

        self.vegetable_poses = []
        self.toy_poses = []
        self.current_index = 0
        prefabs = ["apple", "yellow_cube", "blue_cube", "carrot"]
        self.spawn_entities_in_random_positions(prefabs, prefabs)
        for entity in prefabs:
            if self._get_type(entity) == "vegetable":
                self.vegetable_poses.append(
                    self.pose_transformed(self.get_entity_pose(entity))
                )
            elif self._get_type(entity) == "toy":
                self.toy_poses.append(
                    self.pose_transformed(self.get_entity_pose(entity))
                )
        return

    def _get_type(self, object_name: str):
        if object_name.startswith("apple") or object_name.startswith("carrot"):
            return "vegetable"
        elif object_name.startswith("yellow_cube") or object_name.startswith(
            "blue_cube"
        ):
            return "toy"
        else:
            return None

    def _min_distance(self, entity: str):
        def distance(a, b):
            return ((a.x - b.x) ** 2 + (a.y - b.y) ** 2) ** 0.5

        pose = self.pose_transformed(self.get_entity_pose(entity))
        min_distance = float("inf")
        if self._get_type(entity) == "vegetable":
            for toy_pose in self.toy_poses:
                min_distance = min(
                    min_distance, distance(pose.position, toy_pose.position)
                )
        elif self._get_type(entity) == "toy":
            for vegetable_pose in self.vegetable_poses:
                min_distance = min(
                    min_distance, distance(pose.position, vegetable_pose.position)
                )
        return min_distance

    def calculate_progress(self):
        if self.vegetable_poses == [] or self.toy_poses == []:
            return 0.0

        progress = 0.0

        for entity in self.entities:
            progress += max(0.0, 1.0 - self._min_distance(entity) * 10.0)

        return progress / len(self.entities)

    def step(self):
        if len(self.entities) == 0:
            return 0.0, False

        progress = self.calculate_progress()

        return progress, progress >= 0.8


class ReplaceTypesAuto(ReplaceTypes):
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

    def replace(self, a: str, b: str):
        from copy import deepcopy

        pose_a = self.pose_transformed(self.get_entity_pose(a))
        pose_a.position.z += 0.1

        buffer_pose = deepcopy(pose_a)
        buffer_pose.position.x = 0.4
        buffer_pose.position.y = -0.5

        req = ManipulatorMoveTo.Request()
        req.initial_gripper_state = True
        req.target_pose.pose = pose_a
        req.final_gripper_state = False
        self.manipulator_queue.append(req)

        req = ManipulatorMoveTo.Request()
        req.initial_gripper_state = False
        req.target_pose.pose = buffer_pose
        req.final_gripper_state = True
        self.manipulator_queue.append(req)

        pose_b = self.pose_transformed(self.get_entity_pose(b))
        pose_b.position.z += 0.1

        req = ManipulatorMoveTo.Request()
        req.initial_gripper_state = True
        req.target_pose.pose = pose_b
        req.final_gripper_state = False
        self.manipulator_queue.append(req)

        req = ManipulatorMoveTo.Request()
        req.initial_gripper_state = False
        req.target_pose.pose = pose_a
        req.final_gripper_state = True
        self.manipulator_queue.append(req)

        req = ManipulatorMoveTo.Request()
        req.initial_gripper_state = True
        req.target_pose.pose = buffer_pose
        req.final_gripper_state = False
        self.manipulator_queue.append(req)

        req = ManipulatorMoveTo.Request()
        req.initial_gripper_state = False
        req.target_pose.pose = pose_b
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
        if len(self.entities) < 4:
            return 0.0, False

        progress = self.calculate_progress()

        if not self.manipulator_busy:
            vegetables = []
            toys = []
            for entity in self.entities:
                if self._get_type(entity) == "vegetable":
                    vegetables.append(entity)
                elif self._get_type(entity) == "toy":
                    toys.append(entity)

            if len(self.manipulator_queue) == 0 and progress < 0.8:
                self.replace(vegetables[self.current_index], toys[self.current_index])
                self.current_index = (self.current_index + 1) % len(vegetables)

            if len(self.manipulator_queue) > 0:
                req = self.manipulator_queue.pop(0)
                self.manipulator_busy = True
                self.manipulator_client.call_async(req).add_done_callback(
                    self.move_callback
                )

        return progress, progress >= 0.8 and not self.manipulator_busy

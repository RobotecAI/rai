# Copyright (C) 2024 Robotec.AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#


from typing import Literal, Type

import numpy as np
import rclpy
import rclpy.callback_groups
import rclpy.executors
import rclpy.qos
import rclpy.subscription
import rclpy.task
from geometry_msgs.msg import Point, Pose, PoseStamped, Quaternion
from langchain.tools.render import render_text_description_and_args
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field
from rai_open_set_vision.tools import GetGrabbingPointTool
from rclpy.client import Client
from rclpy.node import Node
from tf2_geometry_msgs import do_transform_pose

from rai.agents.conversational_agent import create_conversational_agent
from rai.node import RaiNode
from rai.tools.ros.native import GetCameraImage
from rai.tools.utils import TF2TransformFetcher
from rai.utils.model_initialization import get_llm_model, get_tracing_callbacks
from rai_interfaces.srv import ManipulatorMoveTo


class MoveToPointToolInput(BaseModel):
    x: float = Field(description="The x coordinate of the point to move to")
    y: float = Field(description="The y coordinate of the point to move to")
    z: float = Field(description="The z coordinate of the point to move to")
    # yaw: float = Field(description="The yaw of the robot's end effector")
    task: Literal["grab", "place"] = Field(
        description="The task to be performed. If you want to pick up an object, use grab, if you want to place it, use place."
    )


class MoveToPointTool(BaseTool):
    name: str = "move_to_point"
    description: str = "Move the robot's end effector to a point in the world."

    node: Node
    client: Client
    args_schema: Type[MoveToPointToolInput] = MoveToPointToolInput

    def __init__(self, node: Node):
        super().__init__(
            node=node,
            client=node.create_client(
                ManipulatorMoveTo,
                "/manipulator_move_to",
            ),
        )

    def _run(
        self,
        x: float,
        y: float,
        z: float,
        # yaw: float,
        task: Literal["grab", "place"],
    ) -> str:
        pose_stamped = PoseStamped()
        pose_stamped.header.frame_id = "panda_link0"
        pose_stamped.pose = Pose(
            position=Point(x=x, y=y, z=z),
            orientation=Quaternion(
                x=0.9238795325112867, y=-0.3826834323650898, z=0.0, w=0.0
            ),
        )

        # calibration values for the manipulator
        calibration_x = 0.0285  # do not touch this value
        calibration_y = 0.0  # do not touch this value
        calibration_z = 0.138  # do not touch this value

        # additional fine-tuning calibration values for the manipulator
        additional_calibration_x = 0.0425
        additional_calibration_y = -0.025
        additional_calibration_z = 0.01

        # Apply calibration values

        if task == "place":
            pose_stamped.pose.position.z += 0.05

        pose_stamped.pose.position.x += calibration_x + additional_calibration_x
        pose_stamped.pose.position.y += calibration_y + additional_calibration_y
        pose_stamped.pose.position.z += calibration_z + additional_calibration_z

        pose_stamped.pose.position.z = np.max(
            [pose_stamped.pose.position.z, 0.135]
        )  # avoid hitting the table

        request = ManipulatorMoveTo.Request()
        request.target_pose = pose_stamped

        if task == "grab":
            request.initial_gripper_state = True  # open
            request.final_gripper_state = False  # closed
        else:
            request.initial_gripper_state = False  # closed
            request.final_gripper_state = True  # open

        future = self.client.call_async(request)
        self.node.get_logger().info(
            f"Calling ManipulatorMoveTo service with request: x={request.target_pose.pose.position.x:.2f}, y={request.target_pose.pose.position.y:.2f}, z={request.target_pose.pose.position.z:.2f}"
        )

        rclpy.spin_until_future_complete(self.node, future)

        if future.result() is not None:
            response = future.result()
            if response.success:
                return f"Successfully moved to point ({x}, {y}, {z})"
            else:
                return f"Failed to move to point ({x}, {y}, {z})"
        else:
            return f"Service call failed for point ({x}, {y}, {z})"


class GetObjectPositionsToolInput(BaseModel):
    object_name: str = Field(
        ..., description="The name of the object to get the positions of"
    )


class GetObjectPositionsTool(BaseTool):
    name: str = "get_object_positions"
    description: str = (
        "Get the positions of all objects of a given type in manipulator frame"
    )

    target_frame: str  # frame of the manipulator
    source_frame: str  # frame of the camera
    camera_topic: str  # rgb camera topic
    depth_topic: str
    camera_info_topic: str  # rgb camera info topic
    node: Node
    get_grabbing_point_tool: GetGrabbingPointTool

    def __init__(self, node: Node, **kwargs):
        super(GetObjectPositionsTool, self).__init__(
            node=node, get_grabbing_point_tool=GetGrabbingPointTool(node=node), **kwargs
        )

    args_schema: Type[GetObjectPositionsToolInput] = GetObjectPositionsToolInput

    def _run(self, object_name: str):
        transform = TF2TransformFetcher(
            target_frame=self.target_frame, source_frame=self.source_frame
        ).get_data()

        results = self.get_grabbing_point_tool._run(
            camera_topic=self.camera_topic,
            depth_topic=self.depth_topic,
            camera_info_topic=self.camera_info_topic,
            object_name=object_name,
        )

        poses = []
        for result in results:
            cam_pose = result[0]
            poses.append(
                Pose(position=Point(x=cam_pose[0], y=cam_pose[1], z=cam_pose[2]))
            )

        world_poses = []
        for pose in poses:
            world_pose = do_transform_pose(pose, transform)
            world_poses.append(world_pose)

        return world_poses


def main():
    rclpy.init()
    node = RaiNode(
        llm=get_llm_model(
            model_type="simple_model"
        ),  # smaller model used to describe the environment
        system_prompt="",
    )
    tools = [
        MoveToPointTool(node=node),
        GetObjectPositionsTool(
            node=node,
            source_frame="RGBDCamera5",
            target_frame="world",
            camera_topic="/color_image5",
            depth_topic="/depth_image5",
            camera_info_topic="/color_camera_info5",
        ),
        GetCameraImage(node=node),
    ]
    SYSTEM_PROMPT = f"""You are an autonomous robotic arm connected to ros2 environment. Your main goal is to fulfill the user's requests.
    Do not make assumptions about the environment you are currently in. You may be asked to move objects around.

    Coordinates are in meters. System is as follows:
    x - forward
    y - right
    z - up

    Arm can move in the following ranges:
    x - [0.2, 0.5]
    y - [-0.6, 0.65]
    z - [0.0, 1.0]

    User specified positions such as next, on top, etc. Make sure to follow them. The positions gathered from get_objects_positions tool are the centroids of the objects. Sometimes you may have to do some calculations to properly place the objects.
    When you drop the object, and you want to pick it up again, you must first get the positions of the objects again.
    The gripper can hold one object at a time. (if you pick up an object, you must always place it before picking up another one)

    After finishing the task/subtask, use camera to confirm the result. If you've failed, repeat the task. Make sure to always grab the latest position of the objects.

    Make sure to properly define class which get_objects_positions tool shoud look for. For example, do not query item, query box/cube/apple/cup etc. Do not use vague names, but specific ones (make sure to specify color).
    Make sure to always grab the item, do not grab the air.
    Mind the space limitations (eg limits or existing objects).
    Do not place objects on the position of other object unless user explicitly tells you to do so- this is very important. You must always adhere to this rule.
    Remember where you have already placed the objects.
    Do not move things unless it is necessary for the task.

    Do not run the tools at once, but execute them one by one- this is very important and you should always adhere to this rule.
    Run only one tool per turn. You will be givent another turn every time you use the tool.
    Do not make out coordinates of the objects by yourself.
    Use the tooling provided to gather information about the positions of the objects.
    Do not modify the coordinates (especially z axis, if you change it you wont grab the object). The move_to_point_tool has necessary calibration values.

    Use the tooling provided to gather information about the environment:

    {render_text_description_and_args(tools)}

    You can use ros2 topics, services and actions to operate.
    You can use /color_image5 to see the environment.
    When you get the task, make sure to plan first. Do planning every step of the way.
    After finishing the task, use camera to confirm the result. If you've failed, repeat the missing steps.-this is very important.
    Be observant and careful. Make sure the task has been finished properly.- this is very important.

    Examples:
    User: "Pick up the carrot"
     - Use get_objects_positions tool with argument "carrot" to get the position of the carrot.
     - Use move_to_point tool with the returned position and argument "grab" to pick up the carrot.

    User: "Swap the position of the cube at the left bottom and the carrot"
     - Use get_objects_positions to find the cube, reason which one is the proper cube.
     - Use get_objects_positions to find the carrot.
     - move the cube to the temporary, empty location, at least 0.05m away from other objects.
     - move the carrot to the previous position of the cube.
     - move the cube to the previous position of the carrot.

    This are some of the examples of the tasks you can perform. When asked for harder task, make sure to plan the task properly and if applicable use the examples as a guide.
    """

    SYSTEM_PROMPTS = """
    You are an autonomous robotic arm operating within a ROS2 environment. Your primary goal is to fulfill user requests, particularly in tasks involving object manipulation.

    Key Operational Guidelines:
    Environment Setup:
    You do not have prior knowledge of the environment, so do not make assumptions about the layout or object positions. Always gather up-to-date information using the tools and sensors provided.

    Coordinate System (in meters):

    x: Forward
    y: Right
    z: Up
    Arm movement ranges:

    x: [0.2, 0.5] # limited by the size of the table
    y: [-0.6, 0.7] # limited by the size of the table
    z: [0.0, 1.0]

    Point 0.3, 0.0, 0.0 is the center point of the table.

    The gripper can hold only one object at a time. Before picking up another object, you must always place the current object in a valid location.

    Gripper and Object Interaction:
    The gripper can hold only one object at a time. Before picking up another object, you must always place the current object in a valid location.
    Always ensure proper grabbing and placing of objects.
    Task Execution:
    Gather Information:
    Always gather the latest position and status of objects using the provided tools. Never assume positions or modify object coordinates on your own. Particularly, do not adjust the z-axis as the arm’s movement is calibrated for accurate gripping and placing.

    Object Identification:
    Use specific and clear names when querying objects (e.g., box, apple, cup). Always include distinguishing features like color to avoid confusion. Avoid vague terms like “item.”

    Placement Rules:

    Avoid placing objects on top of each other unless explicitly instructed by the user.
    If space is limited, the arm should intelligently adjust placement height or position to ensure objects are placed safely and do not interfere with existing ones.
    If a task involves stacking objects, ensure that the placement takes into account the total height, and do not attempt to place an object lower than a previous one in the stack.
    Task Validation:
    After completing a task or subtask:

    Use the camera (/color_image5) to visually confirm the result.
    If the result is unsatisfactory (e.g., wrong position, object not grabbed or placed correctly), repeat the task until it is successful.
    Task Planning:
    Before executing a task, always create a plan for every step. This includes:

    Identifying the correct object and its position.
    Determining a valid destination for placement, considering space limitations.
    Ensuring the task can be completed without interference from other objects or the environment.
    Planning should be revisited after each step to ensure ongoing accuracy and success.

    Operational Limits:

    Do not move objects unless necessary for completing the task.
    Mind the environment's limitations, including existing objects and movement constraints.
    Always remember where you have already placed objects to avoid collisions or misplacements.
    Tooling and Sensors:
    You can access the following to gather environment data:
    ROS2 topics, services, and actions.
    Use /color_image5 to view the environment and confirm placements, especially when you think you have finished the task.
    Do not execute multiple tools at once. Use tools sequentially, ensuring each action provides useful information before moving forward.
    Autonomous Operation:
    Act independently. Ask for user input only when absolutely necessary (e.g., when critical information is missing or ambiguous).
    Repeated failures should lead to re-evaluation of the plan, ensuring success after each iteration.
    """

    llm = get_llm_model("complex_model")

    agent = create_conversational_agent(
        llm=llm,
        tools=tools,
        system_prompt=SYSTEM_PROMPT,
        logger=node.get_logger(),
    )
    output = agent.invoke(
        {
            "messages": [
                # "Build a tower with all of the cubes."
                # "Form a circle with all of the cubes around the corn. Move all other vegetables to the left side of the table first."
                # "Move all the vegetables alongside left edge of the table, then move all the cubes alongside the right edge of the table."
                # "Move the cubes to the left side of the table, then move the vegetables to the right side of the table."
                # "Pick up the carrot"
                # "Pick up the carrot and place it next to the cube one the right side of the table."
                # "Swap the position of the cube at the left bottom and the carrot."
                # "Swap positions of the cubes with the vegetables."
                # "Deconstruct the tower of cubes and place them on the left side of the table. Make sure to start with the highest cube and end with the lowest one."
                # "Put one vegetable on top of every cube."
                # "Do something creative with the present objects."
                # "Deconstruct the tower of objects. Mind the hight of the tower. Make sure to start with the highest object- this is very very important. Do additional calculations and reasoning in order to complete this task properly."
                # "Place all the cubes and vegetables in a circle of radius 0.15m around position (0.35, 0.0, 0.0)." # doesnt work
                # "Arrange the vegetables alternately with the cubes in a line along the the y axis. During previous runs, you failed to grab the cubes and vegetables. You've grabbed the air and placed it somewhere. Make sure to avoid this mistake."
                # "Pick up autonomous vehicle platform (black and white robot placed on the table)."
                # "Place the cubes in the top left part of the table." # top of the black and white toy vehicle."
                "Pick up the object closest to the center of the table and throw it away."
            ]
        },
        config={"recursion_limit": 100, "callbacks": get_tracing_callbacks()},
    )
    output["messages"][-1].pretty_print()


if __name__ == "__main__":
    main()

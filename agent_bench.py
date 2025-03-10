from typing import List, Literal, Tuple, Type
from unittest.mock import Mock

import numpy as np
import rclpy
from geometry_msgs.msg import Point, Pose, Quaternion
from langchain.tools import BaseTool
from langsmith import Client
from pydantic import BaseModel, Field
from rai.agents.conversational_agent import create_conversational_agent
from rai.communication import ROS2ARIConnector
from rai.messages import MultimodalArtifact, preprocess_image
from rai.messages.multimodal import HumanMultimodalMessage
from rai.tools.ros.manipulation import (  # type: ignore
    # GetObjectPositionsTool,
    MoveToPointToolInput,
)

# from rai.tools.ros2 import GetROS2TopicsNamesAndTypesTool, GetROS2ImageTool
from rai.tools.ros2.topics import GetROS2ImageToolInput
from rai.utils.model_initialization import get_llm_model, get_tracing_callbacks
from rai_open_set_vision.tools import GetGrabbingPointTool

client = Client()
get_tracing_callbacks()
rclpy.init()
connector = ROS2ARIConnector()


class GetROS2TopicsNamesAndTypesTool(BaseTool):
    connector: ROS2ARIConnector
    name: str = "get_ros2_topics_names_and_types"
    description: str = "Get the names and types of all ROS2 topics"

    def _run(self) -> str:
        response = [
            "topic: /attached_collision_object\ntype: moveit_msgs/msg/AttachedCollisionObject\n",
            "topic: /camera_image_color\ntype: sensor_msgs/msg/Image\n",
            "topic: /camera_image_depth\ntype: sensor_msgs/msg/Image\n",
            "topic: /clock\ntype: rosgraph_msgs/msg/Clock\n",
            "topic: /collision_object\ntype: moveit_msgs/msg/CollisionObject\n",
            "topic: /color_camera_info\ntype: sensor_msgs/msg/CameraInfo\n",
            "topic: /color_camera_info5\ntype: sensor_msgs/msg/CameraInfo\n",
            "topic: /color_image5\ntype: sensor_msgs/msg/Image\n",
            "topic: /depth_camera_info5\ntype: sensor_msgs/msg/CameraInfo\n",
            "topic: /depth_image5\ntype: sensor_msgs/msg/Image\n",
            "topic: /display_contacts\ntype: visualization_msgs/msg/MarkerArray\n",
            "topic: /display_planned_path\ntype: moveit_msgs/msg/DisplayTrajectory\n",
            "topic: /execute_trajectory/_action/feedback\ntype: moveit_msgs/action/ExecuteTrajectory_FeedbackMessage\n",
            "topic: /execute_trajectory/_action/status\ntype: action_msgs/msg/GoalStatusArray\n",
            "topic: /joint_states\ntype: sensor_msgs/msg/JointState\n",
            "topic: /monitored_planning_scene\ntype: moveit_msgs/msg/PlanningScene\n",
            "topic: /motion_plan_request\ntype: moveit_msgs/msg/MotionPlanRequest\n",
            "topic: /move_action/_action/feedback\ntype: moveit_msgs/action/MoveGroup_FeedbackMessage\n",
            "topic: /move_action/_action/status\ntype: action_msgs/msg/GoalStatusArray\n",
            "topic: /panda_arm_controller/follow_joint_trajectory/_action/feedback\ntype: control_msgs/action/FollowJointTrajectory_FeedbackMessage\n",
            "topic: /panda_arm_controller/follow_joint_trajectory/_action/status\ntype: action_msgs/msg/GoalStatusArray\n",
            "topic: /panda_hand_controller/gripper_cmd/_action/feedback\ntype: control_msgs/action/GripperCommand_FeedbackMessage\n",
            "topic: /panda_hand_controller/gripper_cmd/_action/status\ntype: action_msgs/msg/GoalStatusArray\n",
            "topic: /parameter_events\ntype: rcl_interfaces/msg/ParameterEvent\n",
            "topic: /planning_scene\ntype: moveit_msgs/msg/PlanningScene\n",
            "topic: /planning_scene_world\ntype: moveit_msgs/msg/PlanningSceneWorld\n",
            "topic: /pointcloud\ntype: sensor_msgs/msg/PointCloud2\n",
            "topic: /robot_description\ntype: std_msgs/msg/String\n",
            "topic: /robot_description_semantic\ntype: std_msgs/msg/String\n",
            "topic: /rosout\ntype: rcl_interfaces/msg/Log\n",
            "topic: /tf\ntype: tf2_msgs/msg/TFMessage\n",
            "topic: /tf_static\ntype: tf2_msgs/msg/TFMessage\n",
            "topic: /trajectory_execution_event\ntype: std_msgs/msg/String\n",
        ]
        return "\n".join(response)


class GetObjectPositionsToolInput(BaseModel):
    object_name: str = Field(
        ..., description="The name of the object to get the positions of"
    )


class GetObjectPositionsTool(BaseTool):
    name: str = "get_object_positions"
    description: str = (
        "Retrieve the positions of all objects of a specified type in the target frame. "
        "This tool provides accurate positional data but does not distinguish between different colors of the same object type. "
        "While position detection is reliable, please note that object classification may occasionally be inaccurate."
    )

    target_frame: str
    source_frame: str
    camera_topic: str  # rgb camera topic
    depth_topic: str
    camera_info_topic: str  # rgb camera info topic
    connector: ROS2ARIConnector = Field(..., exclude=True)
    get_grabbing_point_tool: GetGrabbingPointTool

    args_schema: Type[GetObjectPositionsToolInput] = GetObjectPositionsToolInput

    @staticmethod
    def format_pose(pose: Pose):
        return f"Centroid(x={pose.position.x:.2f}, y={pose.position.y:2f}, z={pose.position.z:2f})"

    def _run(self, object_name: str):
        mock_positions = {
            "apple": [
                Pose(position=Point(x=0.4, y=0.2, z=0.15)),
                Pose(position=Point(x=0.5, y=-0.3, z=0.2)),
            ],
            "banana": [
                Pose(position=Point(x=0.3, y=0.1, z=0.12)),
            ],
            "cup": [
                Pose(position=Point(x=0.4, y=-0.2, z=0.25)),
                Pose(position=Point(x=0.6, y=0.0, z=0.3)),
            ],
        }

        poses: List[Pose] = mock_positions.get(object_name, [])

        if len(poses) == 0:
            return f"No {object_name}s detected."
        else:
            return f"Centroids of detected {object_name}s in manipulator frame: [{', '.join(map(self.format_pose, poses))}]. Sizes of the detected objects are unknown."


class MoveToPointTool(BaseTool):
    name: str = "move_to_point"
    description: str = (
        "Guide the robot's end effector to a specific point within the manipulator's operational space. "
        "This tool ensures precise movement to the desired location. "
        "While it confirms successful positioning, please note that it doesn't provide feedback on the "
        "success of grabbing or releasing objects. Use additional sensors or tools for that information."
    )

    connector: ROS2ARIConnector = Field(..., exclude=True)

    manipulator_frame: str = Field(..., description="Manipulator frame")
    min_z: float = Field(default=0.135, description="Minimum z coordinate [m]")
    calibration_x: float = Field(default=0.0, description="Calibration x [m]")
    calibration_y: float = Field(default=0.0, description="Calibration y [m]")
    calibration_z: float = Field(default=0.0, description="Calibration z [m]")
    additional_height: float = Field(
        default=0.05, description="Additional height for the place task [m]"
    )

    # constant quaternion
    quaternion: Quaternion = Field(
        default=Quaternion(x=0.9238795325112867, y=-0.3826834323650898, z=0.0, w=0.0),
        description="Constant quaternion",
    )

    args_schema: Type[MoveToPointToolInput] = MoveToPointToolInput

    def _run(
        self,
        x: float,
        y: float,
        z: float,
        task: Literal["grab", "drop"],
    ) -> str:
        response = Mock(success=True)

        if response.success:
            return f"End effector successfully positioned at coordinates ({x:.2f}, {y:.2f}, {z:.2f}). Note: The status of object interaction (grab/drop) is not confirmed by this movement."
        else:
            return f"Failed to position end effector at coordinates ({x:.2f}, {y:.2f}, {z:.2f})."


class GetROS2ImageTool(BaseTool):
    connector: ROS2ARIConnector
    name: str = "get_ros2_image"
    description: str = "Get an image from a ROS2 topic"
    args_schema: Type[GetROS2ImageToolInput] = GetROS2ImageToolInput
    response_format: Literal["content", "content_and_artifact"] = "content_and_artifact"

    def _run(self, topic: str) -> Tuple[str, MultimodalArtifact]:
        image = self.generate_mock_image()
        return "Image received successfully", MultimodalArtifact(
            images=[preprocess_image(image)]
        )

    @staticmethod
    def generate_mock_image():
        """Generate a blank black image (480x640, RGB)."""
        height, width = 480, 640
        blank_image = np.zeros((height, width, 3), dtype=np.uint8)
        return blank_image  # Returning image bytes


# tool = GetROS2TopicsNamesAndTypesTool(connector=connector)
tools: List[BaseTool] = [
    GetObjectPositionsTool(
        connector=connector,
        target_frame="panda_link0",
        source_frame="RGBDCamera5",
        camera_topic="/color_image5",
        depth_topic="/depth_image5",
        camera_info_topic="/color_camera_info5",
        get_grabbing_point_tool=GetGrabbingPointTool(connector=connector),
    ),
    MoveToPointTool(connector=connector, manipulator_frame="panda_link0"),
    GetROS2ImageTool(connector=connector),
    GetROS2TopicsNamesAndTypesTool(connector=connector),
]
agent = create_conversational_agent(
    llm=get_llm_model(model_type="complex_model"),
    tools=tools,
    system_prompt="""
    You are a robotic arm with interfaces to detect and manipulate objects.
    Here are the coordinates information:
    x - front to back (positive is forward)
    y - left to right (positive is right)
    z - up to down (positive is up)
    Before starting the task, make sure to grab the camera image to understand the environment.
    """,
)
response = agent.invoke({"messages": [HumanMultimodalMessage(content="Grab banana.")]})
response["messages"]
pass

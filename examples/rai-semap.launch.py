# Copyright (C) 2025 Julia Jia
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

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    # Declare launch arguments
    database_path_arg = DeclareLaunchArgument(
        "database_path",
        default_value="semantic_map.db",
        description="Path to SQLite database file for semantic map",
    )

    location_id_arg = DeclareLaunchArgument(
        "location_id",
        default_value="default_location",
        description="Identifier for the physical location",
    )

    camera_topic_arg = DeclareLaunchArgument(
        "camera_topic",
        default_value="/camera/camera/color/image_raw",
        description="Camera image topic",
    )

    depth_topic_arg = DeclareLaunchArgument(
        "depth_topic",
        default_value="/camera/camera/depth/image_rect_raw",
        description="Depth image topic",
    )

    camera_info_topic_arg = DeclareLaunchArgument(
        "camera_info_topic",
        default_value="/camera/camera/color/camera_info",
        description="Camera info topic",
    )

    # Launch detection publisher (bridges DINO service to /detection_array topic)
    camera_topic_param = LaunchConfiguration("camera_topic")
    depth_topic_param = LaunchConfiguration("depth_topic")
    camera_info_topic_param = LaunchConfiguration("camera_info_topic")

    launch_detection_publisher = ExecuteProcess(
        cmd=[
            "python",
            "-m",
            "rai_semap.ros2.detection_publisher",
            "--ros-args",
            "-p",
            ["camera_topic:=", camera_topic_param],
            "-p",
            ["depth_topic:=", depth_topic_param],
            "-p",
            ["camera_info_topic:=", camera_info_topic_param],
            "-p",
            "detection_topic:=/detection_array",
            "-p",
            "dino_service:=/grounding_dino_classify",
            "-p",
            "detection_interval:=2.0",
            "-p",
            "box_threshold:=0.4",
            "-p",
            "text_threshold:=0.3",
        ],
        output="screen",
    )

    # Launch semantic map node
    db_path_param = LaunchConfiguration("database_path")
    loc_id_param = LaunchConfiguration("location_id")

    launch_semap = ExecuteProcess(
        cmd=[
            "python",
            "-m",
            "rai_semap.ros2.node",
            "--ros-args",
            "-p",
            ["database_path:=", db_path_param],
            "-p",
            ["location_id:=", loc_id_param],
            "-p",
            "confidence_threshold:=0.5",
            "-p",
            "class_confidence_thresholds:=person:0.7,window:0.6,door:0.5",
            "-p",
            "class_merge_thresholds:=couch:2.5,table:1.5,shelf:1.5,chair:0.8",
            "-p",
            "min_bbox_area:=500.0",
            "-p",
            "use_pointcloud_dedup:=true",
            "-p",
            ["depth_topic:=", depth_topic_param],
            "-p",
            ["camera_info_topic:=", camera_info_topic_param],
            "-p",
            "detection_topic:=/detection_array",
            "-p",
            "map_topic:=/map",
            "-p",
            "map_frame_id:=map",
            "-p",
            "map_resolution:=0.05",
        ],
        output="screen",
    )

    return LaunchDescription(
        [
            database_path_arg,
            location_id_arg,
            camera_topic_arg,
            depth_topic_arg,
            camera_info_topic_arg,
            launch_detection_publisher,
            launch_semap,
        ]
    )

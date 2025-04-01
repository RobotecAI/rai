import logging
from abc import abstractmethod
from typing import Any, Dict, List, Sequence

from langchain_core.messages import AIMessage
from langchain_core.tools import BaseTool

from rai_bench.tool_calling_agent_bench.agent_tasks_interfaces import (
    ROS2ToolCallingAgentTask,
)
from rai_bench.tool_calling_agent_bench.mocked_tools import (
    MockCallROS2ServiceTool,
    MockGetROS2ActionsNamesAndTypesTool,
    MockGetROS2MessageInterfaceTool,
    MockGetROS2ServicesNamesAndTypesTool,
    MockGetROS2TopicsNamesAndTypesTool,
    MockPublishROS2MessageTool,
    MockStartROS2ActionTool,
)

loggers_type = logging.Logger


PROACTIVE_ROS2_EXPERT_SYSTEM_PROMPT = """You are a ROS 2 expert helping a user with their ROS 2 questions. You have access to various tools that allow you to query the ROS 2 system.
                Be proactive and use the tools to answer questions.
                """

# dict of interfaces where keys are interfaces types and values are output
# of GetROS2MessageInterfaceTool which are same as ros2 interface show outputs
# the dict contains custom as well as couple other common interfaces
INTERFACES: Dict[str, str] = {
    "sensor_msgs/msg/CameraInfo": """
# This message defines meta information for a camera. It should be in a
# camera namespace on topic "camera_info" and accompanied by up to five
# image topics named:
#
#   image_raw - raw data from the camera driver, possibly Bayer encoded
#   image            - monochrome, distorted
#   image_color      - color, distorted
#   image_rect       - monochrome, rectified
#   image_rect_color - color, rectified
#
# The image_pipeline contains packages (image_proc, stereo_image_proc)
# for producing the four processed image topics from image_raw and
# camera_info. The meaning of the camera parameters are described in
# detail at http://www.ros.org/wiki/image_pipeline/CameraInfo.
#
# The image_geometry package provides a user-friendly interface to
# common operations using this meta information. If you want to, e.g.,
# project a 3d point into image coordinates, we strongly recommend
# using image_geometry.
#
# If the camera is uncalibrated, the matrices D, K, R, P should be left
# zeroed out. In particular, clients may assume that K[0] == 0.0
# indicates an uncalibrated camera.

#######################################################################
#                     Image acquisition info                          #
#######################################################################

# Time of image acquisition, camera coordinate frame ID
std_msgs/Header header # Header timestamp should be acquisition time of image
	builtin_interfaces/Time stamp
		int32 sec
		uint32 nanosec
	string frame_id
                             # Header frame_id should be optical frame of camera
                             # origin of frame should be optical center of camera
                             # +x should point to the right in the image
                             # +y should point down in the image
                             # +z should point into the plane of the image


#######################################################################
#                      Calibration Parameters                         #
#######################################################################
# These are fixed during camera calibration. Their values will be the #
# same in all messages until the camera is recalibrated. Note that    #
# self-calibrating systems may "recalibrate" frequently.              #
#                                                                     #
# The internal parameters can be used to warp a raw (distorted) image #
# to:                                                                 #
#   1. An undistorted image (requires D and K)                        #
#   2. A rectified image (requires D, K, R)                           #
# The projection matrix P projects 3D points into the rectified image.#
#######################################################################

# The image dimensions with which the camera was calibrated.
# Normally this will be the full camera resolution in pixels.
uint32 height
uint32 width

# The distortion model used. Supported models are listed in
# sensor_msgs/distortion_models.hpp. For most cameras, "plumb_bob" - a
# simple model of radial and tangential distortion - is sufficent.
string distortion_model

# The distortion parameters, size depending on the distortion model.
# For "plumb_bob", the 5 parameters are: (k1, k2, t1, t2, k3).
float64[] d

# Intrinsic camera matrix for the raw (distorted) images.
#     [fx  0 cx]
# K = [ 0 fy cy]
#     [ 0  0  1]
# Projects 3D points in the camera coordinate frame to 2D pixel
# coordinates using the focal lengths (fx, fy) and principal point
# (cx, cy).
float64[9]  k # 3x3 row-major matrix

# Rectification matrix (stereo cameras only)
# A rotation matrix aligning the camera coordinate system to the ideal
# stereo image plane so that epipolar lines in both stereo images are
# parallel.
float64[9]  r # 3x3 row-major matrix

# Projection/camera matrix
#     [fx'  0  cx' Tx]
# P = [ 0  fy' cy' Ty]
#     [ 0   0   1   0]
# By convention, this matrix specifies the intrinsic (camera) matrix
#  of the processed (rectified) image. That is, the left 3x3 portion
#  is the normal camera intrinsic matrix for the rectified image.
# It projects 3D points in the camera coordinate frame to 2D pixel
#  coordinates using the focal lengths (fx', fy') and principal point
#  (cx', cy') - these may differ from the values in K.
# For monocular cameras, Tx = Ty = 0. Normally, monocular cameras will
#  also have R = the identity and P[1:3,1:3] = K.
# For a stereo pair, the fourth column [Tx Ty 0]' is related to the
#  position of the optical center of the second camera in the first
#  camera's frame. We assume Tz = 0 so both cameras are in the same
#  stereo image plane. The first camera always has Tx = Ty = 0. For
#  the right (second) camera of a horizontal stereo pair, Ty = 0 and
#  Tx = -fx' * B, where B is the baseline between the cameras.
# Given a 3D point [X Y Z]', the projection (x, y) of the point onto
#  the rectified image is given by:
#  [u v w]' = P * [X Y Z 1]'
#         x = u / w
#         y = v / w
#  This holds for both images of a stereo pair.
float64[12] p # 3x4 row-major matrix


#######################################################################
#                      Operational Parameters                         #
#######################################################################
# These define the image region actually captured by the camera       #
# driver. Although they affect the geometry of the output image, they #
# may be changed freely without recalibrating the camera.             #
#######################################################################

# Binning refers here to any camera setting which combines rectangular
#  neighborhoods of pixels into larger "super-pixels." It reduces the
#  resolution of the output image to
#  (width / binning_x) x (height / binning_y).
# The default values binning_x = binning_y = 0 is considered the same
#  as binning_x = binning_y = 1 (no subsampling).
uint32 binning_x
uint32 binning_y

# Region of interest (subwindow of full camera resolution), given in
#  full resolution (unbinned) image coordinates. A particular ROI
#  always denotes the same window of pixels on the camera sensor,
#  regardless of binning settings.
# The default setting of roi (all values 0) is considered the same as
#  full resolution (roi.width = width, roi.height = height).
RegionOfInterest roi
	#
	uint32 x_offset  #
	                 # (0 if the ROI includes the left edge of the image)
	uint32 y_offset  #
	                 # (0 if the ROI includes the top edge of the image)
	uint32 height    #
	uint32 width     #
	bool do_rectify
""",
    "moveit_msgs/msg/AttachedCollisionObject": """
# The CollisionObject will be attached with a fixed joint to this link
string link_name

#This contains the actual shapes and poses for the CollisionObject
#to be attached to the link
#If action is remove and no object.id is set, all objects
#attached to the link indicated by link_name will be removed
CollisionObject object
    std_msgs/Header header
        builtin_interfaces/Time stamp
            int32 sec
            uint32 nanosec
        string frame_id
    geometry_msgs/Pose pose
        Point position
            float64 x
            float64 y
            float64 z
        Quaternion orientation
            float64 x 0
            float64 y 0
            float64 z 0
            float64 w 1
    string id
    object_recognition_msgs/ObjectType type
        string key
        string db
    shape_msgs/SolidPrimitive[] primitives
        uint8 BOX=1
        uint8 SPHERE=2
        uint8 CYLINDER=3
        uint8 CONE=4
        uint8 PRISM=5
        uint8 type
        float64[<=3] dimensions  #
        uint8 BOX_X=0
        uint8 BOX_Y=1
        uint8 BOX_Z=2
        uint8 SPHERE_RADIUS=0
        uint8 CYLINDER_HEIGHT=0
        uint8 CYLINDER_RADIUS=1
        uint8 CONE_HEIGHT=0
        uint8 CONE_RADIUS=1
        uint8 PRISM_HEIGHT=0
        geometry_msgs/Polygon polygon
            Point32[] points
                #
                #
                float32 x
                float32 y
                float32 z
    geometry_msgs/Pose[] primitive_poses
        Point position
            float64 x
            float64 y
            float64 z
        Quaternion orientation
            float64 x 0
            float64 y 0
            float64 z 0
            float64 w 1
    shape_msgs/Mesh[] meshes
        MeshTriangle[] triangles
            uint32[3] vertex_indices
        geometry_msgs/Point[] vertices
            float64 x
            float64 y
            float64 z
    geometry_msgs/Pose[] mesh_poses
        Point position
            float64 x
            float64 y
            float64 z
        Quaternion orientation
            float64 x 0
            float64 y 0
            float64 z 0
            float64 w 1
    shape_msgs/Plane[] planes
        #
        float64[4] coef
    geometry_msgs/Pose[] plane_poses
        Point position
            float64 x
            float64 y
            float64 z
        Quaternion orientation
            float64 x 0
            float64 y 0
            float64 z 0
            float64 w 1
    string[] subframe_names
    geometry_msgs/Pose[] subframe_poses
        Point position
            float64 x
            float64 y
            float64 z
        Quaternion orientation
            float64 x 0
            float64 y 0
            float64 z 0
            float64 w 1
    byte ADD=0
    byte REMOVE=1
    byte APPEND=2
    byte MOVE=3
    byte operation

# The set of links that the attached objects are allowed to touch
# by default - the link_name is already considered by default
string[] touch_links

# If certain links were placed in a particular posture for this object to remain attached
# (e.g., an end effector closing around an object), the posture necessary for releasing
# the object is stored here
trajectory_msgs/JointTrajectory detach_posture
    std_msgs/Header header
        builtin_interfaces/Time stamp
            int32 sec
            uint32 nanosec
        string frame_id
    string[] joint_names
    JointTrajectoryPoint[] points
        float64[] positions
        float64[] velocities
        float64[] accelerations
        float64[] effort
        builtin_interfaces/Duration time_from_start
            int32 sec
            uint32 nanosec

# The weight of the attached object, if known
float64 weight

""",
    "sensor_msgs/msg/Image": """
# This message contains an uncompressed image
# (0, 0) is at top-left corner of image

std_msgs/Header header # Header timestamp should be acquisition time of image
	builtin_interfaces/Time stamp
		int32 sec
		uint32 nanosec
	string frame_id
                             # Header frame_id should be optical frame of camera
                             # origin of frame should be optical center of cameara
                             # +x should point to the right in the image
                             # +y should point down in the image
                             # +z should point into to plane of the image
                             # If the frame_id here and the frame_id of the CameraInfo
                             # message associated with the image conflict
                             # the behavior is undefined

uint32 height                # image height, that is, number of rows
uint32 width                 # image width, that is, number of columns

# The legal values for encoding are in file src/image_encodings.cpp
# If you want to standardize a new string format, join
# ros-users@lists.ros.org and send an email proposing a new encoding.

string encoding       # Encoding of pixels -- channel meaning, ordering, size
                      # taken from the list of strings in include/sensor_msgs/image_encodings.hpp

uint8 is_bigendian    # is this data bigendian?
uint32 step           # Full row length in bytes
uint8[] data          # actual matrix data, size is (step * rows)
""",
    "rosgraph_msgs/msg/Clock": """
# This message communicates the current time.
#
# For more information, see https://design.ros2.org/articles/clock_and_time.html.
builtin_interfaces/Time clock
	int32 sec
	uint32 nanosec
""",
    "rai_interfaces/msg/HRIMessage": """
#
# Copyright (C) 2024 Robotec.AI
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#          http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#

std_msgs/Header header
	builtin_interfaces/Time stamp
		int32 sec
		uint32 nanosec
	string frame_id
string text
sensor_msgs/Image[] images
	std_msgs/Header header #
		builtin_interfaces/Time stamp
			int32 sec
			uint32 nanosec
		string frame_id
	                             # Header frame_id should be optical frame of camera
	                             # origin of frame should be optical center of cameara
	                             # +x should point to the right in the image
	                             # +y should point down in the image
	                             # +z should point into to plane of the image
	                             # If the frame_id here and the frame_id of the CameraInfo
	                             # message associated with the image conflict
	                             # the behavior is undefined
	uint32 height                #
	uint32 width                 #
	string encoding       #
	                      # taken from the list of strings in include/sensor_msgs/image_encodings.hpp
	uint8 is_bigendian    #
	uint32 step           #
	uint8[] data          #
rai_interfaces/AudioMessage[] audios
	#
	#
	#
	#
	#
	int16[] audio
	uint16 sample_rate
	uint16 channels
""",
    "rai_interfaces/msg/AudioMessage": """
#
# Copyright (C) 2024 Robotec.AI
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#          http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#

int16[] audio
uint16 sample_rate
uint16 channels
""",
    "rai_interfaces/msg/RAIDetectionArray": """
#
# Copyright (C) 2024 Robotec.AI
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#          http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#

# A list of 2D detections, for a multi-object 2D detector.
std_msgs/Header header
	builtin_interfaces/Time stamp
		int32 sec
		uint32 nanosec
	string frame_id

# A list of the detected proposals. A multi-proposal detector might generate
#   this list with many candidate detections generated from a single input.
vision_msgs/Detection2D[] detections
	#
	std_msgs/Header header
		builtin_interfaces/Time stamp
			int32 sec
			uint32 nanosec
		string frame_id
	ObjectHypothesisWithPose[] results
		ObjectHypothesis hypothesis
			string class_id
			float64 score
		geometry_msgs/PoseWithCovariance pose
			Pose pose
				Point position
					float64 x
					float64 y
					float64 z
				Quaternion orientation
					float64 x 0
					float64 y 0
					float64 z 0
					float64 w 1
			float64[36] covariance
	BoundingBox2D bbox
		vision_msgs/Pose2D center
			vision_msgs/Point2D position
				float64 x
				float64 y
			float64 theta
		float64 size_x
		float64 size_y
	string id
# a list of classes being detected
string[] detection_classes
""",
    "rai_interfaces/srv/ManipulatorMoveTo": """
#
# Copyright (C) 2024 Robotec.AI
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#          http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

# A simplified approach with binary states for the gripper
bool initial_gripper_state
bool final_gripper_state
geometry_msgs/PoseStamped target_pose
	std_msgs/Header header
		builtin_interfaces/Time stamp
			int32 sec
			uint32 nanosec
		string frame_id
	Pose pose
		Point position
			float64 x
			float64 y
			float64 z
		Quaternion orientation
			float64 x 0
			float64 y 0
			float64 z 0
			float64 w 1
---
bool success
""",
    "rai_interfaces/srv/RAIGroundedSam": """
#
# Copyright (C) 2024 Robotec.AI
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#          http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
RAIDetectionArray detections
	#
	#
	#
	#
	#
	std_msgs/Header header
		builtin_interfaces/Time stamp
			int32 sec
			uint32 nanosec
		string frame_id
	vision_msgs/Detection2D[] detections
		#
		std_msgs/Header header
			builtin_interfaces/Time stamp
				int32 sec
				uint32 nanosec
			string frame_id
		ObjectHypothesisWithPose[] results
			ObjectHypothesis hypothesis
				string class_id
				float64 score
			geometry_msgs/PoseWithCovariance pose
				Pose pose
					Point position
						float64 x
						float64 y
						float64 z
					Quaternion orientation
						float64 x 0
						float64 y 0
						float64 z 0
						float64 w 1
				float64[36] covariance
		BoundingBox2D bbox
			vision_msgs/Pose2D center
				vision_msgs/Point2D position
					float64 x
					float64 y
				float64 theta
			float64 size_x
			float64 size_y
		string id
	string[] detection_classes
sensor_msgs/Image source_img
	std_msgs/Header header #
		builtin_interfaces/Time stamp
			int32 sec
			uint32 nanosec
		string frame_id
	                             # Header frame_id should be optical frame of camera
	                             # origin of frame should be optical center of cameara
	                             # +x should point to the right in the image
	                             # +y should point down in the image
	                             # +z should point into to plane of the image
	                             # If the frame_id here and the frame_id of the CameraInfo
	                             # message associated with the image conflict
	                             # the behavior is undefined
	uint32 height                #
	uint32 width                 #
	string encoding       #
	                      # taken from the list of strings in include/sensor_msgs/image_encodings.hpp
	uint8 is_bigendian    #
	uint32 step           #
	uint8[] data          #
---
sensor_msgs/Image[] masks
	std_msgs/Header header #
		builtin_interfaces/Time stamp
			int32 sec
			uint32 nanosec
		string frame_id
	                             # Header frame_id should be optical frame of camera
	                             # origin of frame should be optical center of cameara
	                             # +x should point to the right in the image
	                             # +y should point down in the image
	                             # +z should point into to plane of the image
	                             # If the frame_id here and the frame_id of the CameraInfo
	                             # message associated with the image conflict
	                             # the behavior is undefined
	uint32 height                #
	uint32 width                 #
	string encoding       #
	                      # taken from the list of strings in include/sensor_msgs/image_encodings.hpp
	uint8 is_bigendian    #
	uint32 step           #
	uint8[] data          #
""",
    "rai_interfaces/srv/RAIGroundingDino": """
#
# Copyright (C) 2024 Robotec.AI
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#          http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
string classes
float64 box_threshold
float64 text_threshold
sensor_msgs/Image source_img
	std_msgs/Header header #
		builtin_interfaces/Time stamp
			int32 sec
			uint32 nanosec
		string frame_id
	                             # Header frame_id should be optical frame of camera
	                             # origin of frame should be optical center of cameara
	                             # +x should point to the right in the image
	                             # +y should point down in the image
	                             # +z should point into to plane of the image
	                             # If the frame_id here and the frame_id of the CameraInfo
	                             # message associated with the image conflict
	                             # the behavior is undefined
	uint32 height                #
	uint32 width                 #
	string encoding       #
	                      # taken from the list of strings in include/sensor_msgs/image_encodings.hpp
	uint8 is_bigendian    #
	uint32 step           #
	uint8[] data          #
---
RAIDetectionArray detections
	#
	#
	#
	#
	#
	std_msgs/Header header
		builtin_interfaces/Time stamp
			int32 sec
			uint32 nanosec
		string frame_id
	vision_msgs/Detection2D[] detections
		#
		std_msgs/Header header
			builtin_interfaces/Time stamp
				int32 sec
				uint32 nanosec
			string frame_id
		ObjectHypothesisWithPose[] results
			ObjectHypothesis hypothesis
				string class_id
				float64 score
			geometry_msgs/PoseWithCovariance pose
				Pose pose
					Point position
						float64 x
						float64 y
						float64 z
					Quaternion orientation
						float64 x 0
						float64 y 0
						float64 z 0
						float64 w 1
				float64[36] covariance
		BoundingBox2D bbox
			vision_msgs/Pose2D center
				vision_msgs/Point2D position
					float64 x
					float64 y
				float64 theta
			float64 size_x
			float64 size_y
		string id
	string[] detection_classes
""",
    "rai_interfaces/srv/StringList": """
# Copyright (C) 2024 Robotec.AI
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#          http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#

# Request - empty
---
# Response
bool success
string[] string_list
""",
    "rai_interfaces/srv/VectorStoreRetrieval": """
# Copyright (C) 2024 Robotec.AI
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#          http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#

# Request
string query

---
# Response
bool success
string message
string[] documents
float32[] scores
""",
    "rai_interfaces/srv/WhatISee": """
# Copyright (C) 2024 Robotec.AI
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#          http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#

# Request (empty)

---
# Response, timed with image timestamp
string[] observations
string perception_source
sensor_msgs/Image image
	std_msgs/Header header #
		builtin_interfaces/Time stamp
			int32 sec
			uint32 nanosec
		string frame_id
	                             # Header frame_id should be optical frame of camera
	                             # origin of frame should be optical center of cameara
	                             # +x should point to the right in the image
	                             # +y should point down in the image
	                             # +z should point into to plane of the image
	                             # If the frame_id here and the frame_id of the CameraInfo
	                             # message associated with the image conflict
	                             # the behavior is undefined
	uint32 height                #
	uint32 width                 #
	string encoding       #
	                      # taken from the list of strings in include/sensor_msgs/image_encodings.hpp
	uint8 is_bigendian    #
	uint32 step           #
	uint8[] data          #
geometry_msgs/Pose pose
	Point position
		float64 x
		float64 y
		float64 z
	Quaternion orientation
		float64 x 0
		float64 y 0
		float64 z 0
		float64 w 1
""",
    "rai_interfaces/action/Task": """
# Goal
string task
string description
string priority

---
# Result
bool success
string report

---
# Feedback
string current_status
""",
    "/load_map": """
string filename
---
bool success
""",
    "/query_planner_interface": """
---

# The planning instances that could be used in the benchmark
PlannerInterfaceDescription[] planner_interfaces
	string name
	string pipeline_id
	string[] planner_ids

""",
}


TOPICS_AND_TYPES: Dict[str, str] = {
    # sample topics
    "/attached_collision_object": "moveit_msgs/msg/AttachedCollisionObject",
    "/camera_image_color": "sensor_msgs/msg/Image",
    "/camera_image_depth": "sensor_msgs/msg/Image",
    "/clock": "rosgraph_msgs/msg/Clock",
    "/collision_object": "moveit_msgs/msg/CollisionObject",
    "/color_camera_info": "sensor_msgs/msg/CameraInfo",
    "/color_camera_info5": "sensor_msgs/msg/CameraInfo",
    "/depth_camera_info5": "sensor_msgs/msg/CameraInfo",
    "/depth_image5": "sensor_msgs/msg/Image",
    # custom topics
    "/to_human": "rai_interfaces/msg/HRIMessage",
    "/send_audio": "rai_interfaces/msg/AudioMessage",
    "/send_detections": "rai_interfaces/msg/RAIDetectionArray",
}

SERVICES_AND_TYPES = {
    # sample interfaces
    "/load_map": "moveit_msgs/srv/LoadMap",
    "/query_planner_interface": "moveit_msgs/srv/QueryPlannerInterfaces",
    # custom interfaces
    "/manipulator_move_to": "rai_interfaces/srv/ManipulatorMoveTo",
    "/grounded_sam_segment": "rai_interfaces/srv/RAIGroundedSam",
    "/grounding_dino_classify": "rai_interfaces/srv/RAIGroundingDino",
    "/get_log_digest": "rai_interfaces/srv/StringList",
    "/rai_whoami_documentation_service": "rai_interfaces/srv/VectorStoreRetrieval",
    "rai/whatisee/get": "rai_interfaces/srv/WhatISee",
}

ACTIONS_AND_TYPES = {
    # custom actions
    "/perform_task": "rai_interfaces/action/Task",
    # some sample actions
    # "/execute_trajectory": "moveit_msgs/action/ExecuteTrajectory",
    # "/move_action": "moveit_msgs/action/MoveGroup",
    # "/follow_joint_trajectory": "control_msgs/action/FollowJointTrajectory",
    # "/gripper_cmd": "control_msgs/action/GripperCommand",
}

# only custom interfaces will be tested, so there no need for defualts for all of interfaces
DEFAULT_MESSAGES: Dict[str, Dict[str, Any]] = {
    "rai_interfaces/msg/HRIMessage": {
        "header": {"stamp": {"sec": 0, "nanosec": 0}, "frame_id": ""},
        "text": "",
        "images": [
            {
                "header": {"stamp": {"sec": 0, "nanosec": 0}, "frame_id": ""},
                "height": 0,
                "width": 0,
                "encoding": "",
                "is_bigendian": 0,
                "step": 0,
                "data": [],
            }
        ],
        "audios": [{"audio": [], "sample_rate": 0, "channels": 0}],
    },
    "rai_interfaces/msg/AudioMessage": {
        "audio": [],
        "sample_rate": 0,
        "channels": 0,
    },
    "rai_interfaces/srv/ManipulatorMoveTo": {
        "request": {
            "initial_gripper_state": False,
            "final_gripper_state": False,
            "target_pose": {
                "header": {"stamp": {"sec": 0, "nanosec": 0}, "frame_id": ""},
                "pose": {
                    "position": {"x": 0.0, "y": 0.0, "z": 0.0},
                    "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 0.0},
                },
            },
        },
        "response": {"success": False},
    },
    # "sensor_msgs/msg/CameraInfo":{
    #   "header": {
    #     "stamp": {
    #       "sec": 0,
    #       "nanosec": 0
    #     },
    #     "frame_id": ""
    #   },
    #   "height": 0,
    #   "width": 0,
    #   "distortion_model": "",
    #   "d": [],
    #   "k": [],
    #   "r": [],
    #   "p": [],
    #   "binning_x": 0,
    #   "binning_y": 0,
    #   "roi": {
    #     "x_offset": 0,
    #     "y_offset": 0,
    #     "height": 0,
    #     "width": 0,
    #     "do_rectify": False
    #   }
    # }
}


class CustomInterfacesTask(ROS2ToolCallingAgentTask):
    topic_strings = [
        f"topic: {topic}\ntype: {msg_type}\n"
        for topic, msg_type in TOPICS_AND_TYPES.items()
    ]
    service_strings = [
        f"service: {service}\ntype: {msg_type}\n"
        for service, msg_type in SERVICES_AND_TYPES.items()
    ]
    action_strings = [
        f"action: {action}\ntype: {msg_type}\n"
        for action, msg_type in SERVICES_AND_TYPES.items()
    ]

    def __init__(self, logger: loggers_type | None = None) -> None:
        super().__init__(logger=logger)
        self.expected_message_type = TOPICS_AND_TYPES[self.expected_topic]
        self.expected_message = DEFAULT_MESSAGES[self.expected_message_type]

    def get_system_prompt(self) -> str:
        return PROACTIVE_ROS2_EXPERT_SYSTEM_PROMPT

    @property
    @abstractmethod
    def expected_topic(self) -> str:
        pass

    def verify_tool_calls(self, response: dict[str, Any]):
        """It is expected that the agent will request:
        1. The tool that retrieves the ROS2 topics names and types to recognize what type of message to_human topic has
        2. The tool that retrieves interfaces to check HRIMessage type
        3. The tool to publish message with proper topic, message type and content

        Parameters
        ----------
        response : dict[str, Any]
            The response from the agent
        """
        messages = response["messages"]
        ai_messages: Sequence[AIMessage] = [
            message for message in messages if isinstance(message, AIMessage)
        ]
        self.logger.debug(ai_messages)
        if len(ai_messages) != 4:
            self.log_error(
                msg=f"Expected exactly 4 AI messages, but got {len(ai_messages)}."
            )
        if ai_messages:
            if not self._is_ai_message_requesting_get_ros2_topics_and_types(
                ai_messages[0]
            ):
                self.log_error(
                    msg="First AI message did not request ROS2 topics and types correctly."
                )
        if len(ai_messages) > 1:
            if self._check_tool_calls_num_in_ai_message(ai_messages[1], expected_num=1):
                self._check_tool_call(
                    tool_call=ai_messages[1].tool_calls[0],
                    expected_name="get_ros2_message_interface",
                    expected_args={"msg_type": self.expected_message_type},
                )

        if len(ai_messages) > 2:
            if self._check_tool_calls_num_in_ai_message(ai_messages[2], expected_num=1):
                self._check_tool_call(
                    tool_call=ai_messages[2].tool_calls[0],
                    expected_name="publish_ros2_message",
                    expected_args={
                        "topic": self.expected_topic,
                        "message": self.expected_message,
                        "message_type": self.expected_message_type,
                    },
                )
        if not self.result.errors:
            self.result.success = True


class PublishROS2HRIMessageTask(CustomInterfacesTask):
    complexity = "easy"

    def __init__(self, logger: loggers_type | None = None) -> None:
        super().__init__(logger=logger)
        self.expected_tools: List[BaseTool] = [
            MockGetROS2TopicsNamesAndTypesTool(
                mock_topics_names_and_types=self.topic_strings
            ),
            MockGetROS2MessageInterfaceTool(mock_interfaces=INTERFACES),
            MockPublishROS2MessageTool(
                available_topics=list(TOPICS_AND_TYPES.keys()),
                available_message_types=list(TOPICS_AND_TYPES.values()),
            ),
        ]

    @property
    def expected_topic(self) -> str:
        return "/to_human"

    def get_prompt(self) -> str:
        return "Publish message with text value: 'Hello!' to the /to_human topic. Before publishing check the message type of this topic and it's interface."


class PublishROS2AudioMessageTask(CustomInterfacesTask):
    complexity = "easy"

    expected_audio = [123, 456, 789]
    expected_sample_rate = 44100
    expected_channels = 2

    @property
    def expected_topic(self) -> str:
        return "/send_audio"

    def __init__(self, logger: loggers_type | None = None) -> None:
        super().__init__(logger=logger)
        self.expected_tools: List[BaseTool] = [
            MockGetROS2TopicsNamesAndTypesTool(
                mock_topics_names_and_types=self.topic_strings
            ),
            MockGetROS2MessageInterfaceTool(mock_interfaces=INTERFACES),
            MockPublishROS2MessageTool(
                available_topics=list(TOPICS_AND_TYPES.keys()),
                available_message_types=list(TOPICS_AND_TYPES.values()),
            ),
        ]

    def get_prompt(self) -> str:
        return (
            "Publish message to the /send_audio topic with audio samples [123, 456, 789], "
            "sample rate 44100, and 2 channels. Before publishing, check the message type "
            "of this topic and its interface."
        )


class PublishROS2DetectionArrayTask(CustomInterfacesTask):
    complexity = "easy"

    expected_detection_classes: List[str] = ["person", "car"]
    expected_detections: List[Any] = [
        {
            "header": {"stamp": {"sec": 0, "nanosec": 0}, "frame_id": "camera"},
            "results": [],
            "bbox": {
                "center": {"x": 320.0, "y": 240.0},
                "size": {"x": 50.0, "y": 50.0},
            },
        }
    ]
    expected_header: Dict[str, Any] = {
        "stamp": {"sec": 0, "nanosec": 0},
        "frame_id": "camera",
    }

    @property
    def expected_topic(self) -> str:
        return "/send_detections"

    def __init__(self, logger: loggers_type | None = None) -> None:
        super().__init__(logger=logger)
        self.expected_tools: List[BaseTool] = [
            MockGetROS2TopicsNamesAndTypesTool(
                mock_topics_names_and_types=self.topic_strings
            ),
            MockGetROS2MessageInterfaceTool(mock_interfaces=INTERFACES),
            MockPublishROS2MessageTool(
                available_topics=list(TOPICS_AND_TYPES.keys()),
                available_message_types=list(TOPICS_AND_TYPES.values()),
            ),
        ]

    def get_prompt(self) -> str:
        return (
            "Publish a detection message to the /send_detections topic. The message should have a header "
            f"with frame_id 'camera', one detection: {self.expected_detections}, and detection classes "
            f"{self.expected_detection_classes}. Before publishing, check the message type of this topic "
            "and its interface."
        )


class CallROS2ManipulatorMoveToServiceTask(CustomInterfacesTask):
    complexity = "easy"

    expected_initial_gripper_state = True
    expected_final_gripper_state = False
    expected_target_pose: Dict[str, Dict[str, Any]] = {
        "header": {"stamp": {"sec": 0, "nanosec": 0}, "frame_id": "world"},
        "pose": {
            "position": {"x": 1.0, "y": 2.0, "z": 3.0},
            "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
        },
    }

    @property
    def expected_service(self) -> str:
        return "/manipulator_move_to"

    def __init__(self, logger: loggers_type | None = None) -> None:
        super().__init__(logger=logger)
        self.expected_tools: List[BaseTool] = [
            MockGetROS2ServicesNamesAndTypesTool(
                mock_service_names_and_types=self.service_strings
            ),
            MockGetROS2MessageInterfaceTool(mock_interfaces=INTERFACES),
            MockCallROS2ServiceTool(
                available_services=list(SERVICES_AND_TYPES.keys()),
                available_service_types=list(SERVICES_AND_TYPES.values()),
            ),
        ]

    def get_prompt(self) -> str:
        return (
            f"Call service {self.expected_service} with a target_pose: {self.expected_target_pose}. "
            "Before calling the service, check the service type and its interface."
        )


class CallGroundedSAMSegmentTask(CustomInterfacesTask):
    complexity = "easy"

    expected_detections: Dict[str, Any] = {
        "header": {"stamp": {"sec": 0, "nanosec": 0}, "frame_id": "camera_frame"},
        "detections": [],
    }
    expected_source_img: Dict[str, Any] = {
        "header": {"stamp": {"sec": 0, "nanosec": 0}, "frame_id": "camera_frame"},
        "height": 480,
        "width": 640,
        "encoding": "rgb8",
        "is_bigendian": 0,
        "step": 1920,
        "data": [],
    }

    @property
    def expected_service(self) -> str:
        return "/grounded_sam_segment"

    def __init__(self, logger: loggers_type | None = None) -> None:
        super().__init__(logger=logger)
        self.expected_tools: List[BaseTool] = [
            MockGetROS2ServicesNamesAndTypesTool(
                mock_service_names_and_types=self.service_strings
            ),
            MockGetROS2MessageInterfaceTool(mock_interfaces=INTERFACES),
            MockCallROS2ServiceTool(
                available_services=list(SERVICES_AND_TYPES.keys()),
                available_service_types=list(SERVICES_AND_TYPES.values()),
            ),
        ]

    def get_prompt(self) -> str:
        return (
            "Call service /grounded_sam_segment with detections from frame 'camera_frame' and "
            "an RGB image of size 640x480. Before calling, look up the service type and its message structure."
        )


class CallGroundingDinoClassifyTask(CustomInterfacesTask):
    complexity = "easy"

    expected_classes = "bottle, book, chair"
    expected_box_threshold = 0.4
    expected_text_threshold = 0.25
    expected_source_img: Dict[str, Any] = {
        "header": {"stamp": {"sec": 0, "nanosec": 0}, "frame_id": "camera_frame"},
        "height": 480,
        "width": 640,
        "encoding": "rgb8",
        "is_bigendian": 0,
        "step": 1920,
        "data": [],
    }

    @property
    def expected_service(self) -> str:
        return "/grounding_dino_classify"

    def __init__(self, logger: loggers_type | None = None) -> None:
        super().__init__(logger=logger)
        self.expected_tools: List[BaseTool] = [
            MockGetROS2ServicesNamesAndTypesTool(
                mock_service_names_and_types=self.service_strings
            ),
            MockGetROS2MessageInterfaceTool(mock_interfaces=INTERFACES),
            MockCallROS2ServiceTool(
                available_services=list(SERVICES_AND_TYPES.keys()),
                available_service_types=list(SERVICES_AND_TYPES.values()),
            ),
        ]

    def get_prompt(self) -> str:
        return (
            f"Call the service /grounding_dino_classify with the following arguments: "
            f"classes='{self.expected_classes}', box_threshold={self.expected_box_threshold}, "
            f"text_threshold={self.expected_text_threshold}, and a 640x480 RGB image from frame 'camera_frame'. "
            "Before calling, look up the service type and its message structure."
        )


class CallGetLogDigestTask(CustomInterfacesTask):
    complexity = "easy"

    @property
    def expected_service(self) -> str:
        return "/get_log_digest"

    def __init__(self, logger: loggers_type | None = None) -> None:
        super().__init__(logger=logger)
        self.expected_tools: List[BaseTool] = [
            MockGetROS2ServicesNamesAndTypesTool(
                mock_service_names_and_types=self.service_strings
            ),
            MockGetROS2MessageInterfaceTool(mock_interfaces=INTERFACES),
            MockCallROS2ServiceTool(
                available_services=list(SERVICES_AND_TYPES.keys()),
                available_service_types=list(SERVICES_AND_TYPES.values()),
            ),
        ]

    def get_prompt(self) -> str:
        return (
            "Call the service /get_log_digest to retrieve a list of log strings. "
            "Before calling, look up the service type and its message structure. "
            "No request arguments are needed."
        )


class CallVectorStoreRetrievalTask(CustomInterfacesTask):
    complexity = "easy"

    expected_query = "What is the purpose of this robot?"

    @property
    def expected_service(self) -> str:
        return "/rai_whoami_documentation_service"

    def __init__(self, logger: loggers_type | None = None) -> None:
        super().__init__(logger=logger)
        self.expected_tools: List[BaseTool] = [
            MockGetROS2ServicesNamesAndTypesTool(
                mock_service_names_and_types=self.service_strings
            ),
            MockGetROS2MessageInterfaceTool(mock_interfaces=INTERFACES),
            MockCallROS2ServiceTool(
                available_services=list(SERVICES_AND_TYPES.keys()),
                available_service_types=list(SERVICES_AND_TYPES.values()),
            ),
        ]

    def get_prompt(self) -> str:
        return (
            f"Call the service rai_whoami_documentation_service with the query: '{self.expected_query}'. "
            "Before calling, look up the service type and its message structure."
        )


class CallWhatISeeTask(CustomInterfacesTask):
    complexity = "easy"

    expected_observations = ["table", "cup", "notebook"]
    expected_perception_source = "front_camera"
    expected_image: Dict[str, Any] = {
        "header": {"stamp": {"sec": 0, "nanosec": 0}, "frame_id": "camera_frame"},
        "height": 480,
        "width": 640,
        "encoding": "rgb8",
        "is_bigendian": 0,
        "step": 1920,
        "data": [],
    }
    expected_pose = {
        "position": {"x": 1.0, "y": 2.0, "z": 0.5},
        "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
    }

    @property
    def expected_service(self) -> str:
        return "rai/whatisee/get"

    def __init__(self, logger: loggers_type | None = None) -> None:
        super().__init__(logger=logger)
        self.expected_tools: List[BaseTool] = [
            MockGetROS2ServicesNamesAndTypesTool(
                mock_service_names_and_types=self.service_strings
            ),
            MockGetROS2MessageInterfaceTool(mock_interfaces=INTERFACES),
            MockCallROS2ServiceTool(
                available_services=list(SERVICES_AND_TYPES.keys()),
                available_service_types=list(SERVICES_AND_TYPES.values()),
            ),
        ]

    def get_prompt(self) -> str:
        return (
            f"Call the service rai/whatisee/get using the WhatISee interface. "
            f"Pass in observations {self.expected_observations}, source '{self.expected_perception_source}', "
            f"a 640x480 RGB image from 'camera_frame', and a pose at position {self.expected_pose['position']}."
            "Before calling, look up the service type and its message structure."
        )


class CallROS2CustomActionTask(CustomInterfacesTask):
    complexity = "easy"

    expected_task = "Where are you?"
    expected_description = ""
    expected_priority = "10"

    @property
    def expected_action(self) -> str:
        return "/perform_task"

    def __init__(self, logger: loggers_type | None = None) -> None:
        super().__init__(logger=logger)
        self.expected_tools: List[BaseTool] = [
            MockGetROS2ActionsNamesAndTypesTool(
                mock_actions_names_and_types=self.action_strings
            ),
            MockGetROS2MessageInterfaceTool(mock_interfaces=INTERFACES),
            MockStartROS2ActionTool(
                available_actions=list(ACTIONS_AND_TYPES.keys()),
                available_action_types=list(ACTIONS_AND_TYPES.values()),
            ),
        ]

    def get_prompt(self) -> str:
        return (
            "Call action /perform_task with the provided goal values: "
            "{priority: 10, description: '', task: 'Where are you?'}"
        )

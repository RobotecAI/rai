# Copyright (C) 2025 Robotec.AI
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

import logging
from abc import ABC
from typing import Any, Dict, List, Type

from langchain_core.tools import BaseTool
from pydantic import BaseModel
from rai.types import (
    BoundingBox2D,
    CameraInfo,
    Detection2D,
    Header,
    Image,
    Point,
    Pose,
    Pose2D,
    PoseStamped,
    Quaternion,
    Time,
)
from rai.types.rai_interfaces import (
    ManipulatorMoveToRequest,
    RAIDetectionArray,
    RAIGroundedSamRequest,
    RAIGroundingDinoRequest,
)

from rai_bench.tool_calling_agent.interfaces import Task, Validator
from rai_bench.tool_calling_agent.messages.base import Clock
from rai_bench.tool_calling_agent.messages.services import (
    StringListRequest,
    VectorStoreRetrievalRequest,
    WhatISeeRequest,
)
from rai_bench.tool_calling_agent.messages.topics import AudioMessage, HRIMessage
from rai_bench.tool_calling_agent.mocked_tools import (
    MockCallROS2ServiceTool,
    MockGetROS2MessageInterfaceTool,
    MockGetROS2ServicesNamesAndTypesTool,
    MockGetROS2TopicsNamesAndTypesTool,
    MockPublishROS2MessageTool,
)

loggers_type = logging.Logger

# dict of interfaces where keys are interfaces types and values are output
# of GetROS2MessageInterfaceTool which are same as ros2 interface show outputs
# the dict contains custom as well as couple other common interfaces
MOCK_INTERFACES: Dict[str, str] = {
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
string communication_id
int64 seq_no
bool seq_end
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
    "rai_interfaces/srv/WhatISee": """z
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


SERVICES_AND_TYPES = {
    # sample interfaces
    # "/load_map": "moveit_msgs/srv/LoadMap",
    # "/query_planner_interface": "moveit_msgs/srv/QueryPlannerInterfaces",
    # custom interfaces
    "/manipulator_move_to": "rai_interfaces/srv/ManipulatorMoveTo",
    "/grounded_sam_segment": "rai_interfaces/srv/RAIGroundedSam",
    "/grounding_dino_classify": "rai_interfaces/srv/RAIGroundingDino",
    "/get_log_digest": "rai_interfaces/srv/StringList",
    "/rai_whoami_documentation_service": "rai_interfaces/srv/VectorStoreRetrieval",
    "/rai/whatisee/get": "rai_interfaces/srv/WhatISee",
}

SERVICE_MODELS: Dict[str, Type[BaseModel]] = {
    "rai_interfaces/srv/ManipulatorMoveTo": ManipulatorMoveToRequest,
    "rai_interfaces/srv/RAIGroundedSam": RAIGroundedSamRequest,
    "rai_interfaces/srv/RAIGroundingDino": RAIGroundingDinoRequest,
    "rai_interfaces/srv/StringList": StringListRequest,
    "rai_interfaces/srv/VectorStoreRetrieval": VectorStoreRetrievalRequest,
    "rai_interfaces/srv/WhatISee": WhatISeeRequest,
}

TOPICS_AND_TYPES: Dict[str, str] = {
    # sample topics
    "/camera_image_color": "sensor_msgs/msg/Image",
    "/camera_image_depth": "sensor_msgs/msg/Image",
    "/clock": "rosgraph_msgs/msg/Clock",
    "/color_camera_info": "sensor_msgs/msg/CameraInfo",
    "/color_camera_info5": "sensor_msgs/msg/CameraInfo",
    "/depth_camera_info5": "sensor_msgs/msg/CameraInfo",
    "/depth_image5": "sensor_msgs/msg/Image",
    # custom topics
    "/to_human": "rai_interfaces/msg/HRIMessage",
    "/send_audio": "rai_interfaces/msg/AudioMessage",
    "/send_detections": "rai_interfaces/msg/RAIDetectionArray",
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
TOPIC_STRINGS = [
    f"topic: {topic}\ntype: {msg_type}\n"
    for topic, msg_type in TOPICS_AND_TYPES.items()
]
TOPIC_MODELS: Dict[str, Type[BaseModel]] = {
    "sensor_msgs/msg/CameraInfo": CameraInfo,
    "sensor_msgs/msg/Image": Image,
    "rosgraph_msgs/msg/Clock": Clock,
    "rai_interfaces/msg/HRIMessage": HRIMessage,
    "rai_interfaces/msg/AudioMessage": AudioMessage,
    "rai_interfaces/msg/RAIDetectionArray": RAIDetectionArray,
}

IMAGE_TOPICS: Dict[str, str] = {
    "/attached_collision_object": "moveit_msgs/msg/AttachedCollisionObject",
    "/camera_image_color": "sensor_msgs/msg/Image",
    "/camera_image_depth": "sensor_msgs/msg/Image",
    "/clock": "rosgraph_msgs/msg/Clock",
    "/collision_object": "moveit_msgs/msg/CollisionObject",
    "/color_camera_info": "sensor_msgs/msg/CameraInfo",
    "/color_camera_info5": "sensor_msgs/msg/CameraInfo",
    "/depth_camera_info5": "sensor_msgs/msg/CameraInfo",
}

SERVICE_STRINGS = [
    f"service: {service}\ntype: {msg_type}\n"
    for service, msg_type in SERVICES_AND_TYPES.items()
]


PROACTIVE_ROS2_EXPERT_SYSTEM_PROMPT = """You are a ROS 2 expert that want to solve tasks. You have access to various tools that allow you to query the ROS 2 system.
Be proactive and use the tools to answer questions.
Example of tool calls:
- get_ros2_message_interface, args: {'msg_type': 'geometry_msgs/msg/Twist'}
- publish_ros2_message, args: {'topic': '/cmd_vel', 'message_type': 'geometry_msgs/msg/Twist', 'message': {linear: {x: 0.5, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 1.0}}}
- get_ros2_message_interface, args: {'msg_type': 'turtlesim/srv/TeleportAbsolute'}
- publish_ros2_message, args: {'topic': '/turtle1/teleport_absolute', 'message_type': 'turtlesim/srv/TeleportAbsolute', 'message': {x: 5.0, y: 2.0, theta: 1.57}}"""


class CustomInterfaceTask(Task, ABC):
    @property
    def type(self) -> str:
        return "custom_interface"


class CustomInterfacesTopicTask(CustomInterfaceTask, ABC):
    def __init__(
        self,
        topic: str,
        validators: List[Validator],
        extra_tool_calls: int = 0,
        logger: loggers_type | None = None,
    ) -> None:
        super().__init__(
            validators=validators,
            extra_tool_calls=extra_tool_calls,
            logger=logger,
        )
        self.topic = topic

    @property
    def available_tools(self) -> List[BaseTool]:
        return [
            MockGetROS2TopicsNamesAndTypesTool(
                mock_topics_names_and_types=TOPIC_STRINGS
            ),
            MockGetROS2MessageInterfaceTool(mock_interfaces=MOCK_INTERFACES),
            MockPublishROS2MessageTool(
                available_topics=list(TOPICS_AND_TYPES.keys()),
                available_message_types=list(TOPICS_AND_TYPES.values()),
                available_topic_models=TOPIC_MODELS,
            ),
        ]

    def get_system_prompt(self) -> str:
        return PROACTIVE_ROS2_EXPERT_SYSTEM_PROMPT


class CustomInterfacesServiceTask(CustomInterfaceTask, ABC):
    def __init__(
        self,
        service: str,
        service_args: dict[str, Any],
        validators: List[Validator],
        extra_tool_calls: int = 0,
        logger: loggers_type | None = None,
    ) -> None:
        super().__init__(
            validators=validators,
            extra_tool_calls=extra_tool_calls,
            logger=logger,
        )
        self.service = service
        self.service_args = service_args

    @property
    def available_tools(self) -> List[BaseTool]:
        return [
            MockGetROS2ServicesNamesAndTypesTool(
                mock_service_names_and_types=SERVICE_STRINGS
            ),
            MockGetROS2MessageInterfaceTool(mock_interfaces=MOCK_INTERFACES),
            MockCallROS2ServiceTool(
                available_services=list(SERVICES_AND_TYPES.keys()),
                available_service_types=list(SERVICES_AND_TYPES.values()),
                available_service_models=SERVICE_MODELS,
            ),
        ]


# TODO (jm) add actions Tasks


# TODO (jm) should we and how to parametrize these classes?
class PublishROS2HRIMessageTextTask(CustomInterfacesTopicTask):
    complexity = "easy"

    def __init__(
        self,
        topic: str,
        text: str,
        validators: List[Validator],
        extra_tool_calls: int = 0,
        logger: logging.Logger | None = None,
    ) -> None:
        super().__init__(topic, validators, extra_tool_calls, logger)
        self.text = text

    def get_prompt(self) -> str:
        return (
            f"You need to publish a message to the topic '{self.topic}' with the text value: '{self.text}'.\n"
            "Before publishing, follow these steps:\n"
            "1. Use the tool to retrieve the available ROS2 topics and their message types.\n"
            f"2. Find the message type for the topic '{self.topic}'.\n"
            "3. Retrieve the full message interface definition for that type.\n"
            "4. Construct the message filling only the fields you are instructed to. Rest of the fields will have default values.\n"
            f"5. Publish the message to '{self.topic}' using the correct message type and interface.\n"
        )


class PublishROS2AudioMessageTask(CustomInterfacesTopicTask):
    complexity = "easy"
    expected_audio: List[int] = [123, 456, 789]
    expected_sample_rate: int = 44100
    expected_channels: int = 2

    def get_prompt(self) -> str:
        return (
            f"You need to publish a message to the topic '{self.topic}' with audio samples {self.expected_audio}, "
            f"sample rate {self.expected_sample_rate}, and {self.expected_channels} channels.\n"
            "Before publishing, follow these steps:\n"
            "1. Use the tool to retrieve the available ROS2 topics and their message types.\n"
            f"2. Find the message type for the topic '{self.topic}'.\n"
            "3. Retrieve the full message interface definition for that type.\n"
            "4. Construct the message filling only the fields you are instructed to. Rest of the fields will have default values.\n"
            f"5. Publish the message to '{self.topic}' using the correct message type and interface.\n"
        )


class PublishROS2DetectionArrayTask(CustomInterfacesTopicTask):
    complexity = "easy"

    expected_detection_classes: List[str] = ["person", "car"]
    expected_detections: List[Detection2D] = [
        Detection2D(
            bbox=BoundingBox2D(
                center=Pose2D(x=320.0, y=240.0, theta=0.0),
                size_x=50.0,
                size_y=50.0,
            )
        )
    ]

    def get_prompt(self) -> str:
        return (
            f"You need to publish a detection message to the topic '{self.topic}' with one detection:\n"
            f"{self.expected_detections[0].model_dump()} and detection classes {self.expected_detection_classes}.\n"
            "Before publishing, follow these steps:\n"
            "1. Use the tool to retrieve the available ROS2 topics and their message types.\n"
            f"2. Find the message type for the topic '{self.topic}'.\n"
            "3. Retrieve the full message interface definition for that type.\n"
            "4. Construct the message filling only the fields you are instructed to. Rest of the fields will have default values.\n"
            f"5. Publish the message to '{self.topic}' using the correct message type and interface.\n"
        )


class CallROS2ManipulatorMoveToServiceTask(CustomInterfacesServiceTask):
    complexity = "easy"

    expected_initial_gripper_state = True
    expected_final_gripper_state = False
    expected_target_pose: PoseStamped = PoseStamped(
        pose=Pose(
            position=Point(x=1.0, y=2.0, z=3.0),
            orientation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0),
        )
    )

    def get_prompt(self) -> str:
        return (
            f"You need to call the service '{self.service}' with a target_pose: "
            f"{self.expected_target_pose.model_dump()} and gripper states (initial: {self.expected_initial_gripper_state}, final: {self.expected_final_gripper_state}).\n"
            "Before calling, follow these steps:\n"
            "1. Use the tool to retrieve the available ROS2 services and their types.\n"
            f"2. Find the service type for '{self.service}'.\n"
            "3. Retrieve the full message interface definition for that service.\n"
            "4. Construct the request message filling only the fields you are instructed to. Rest of the fields will have default values.\n"
            f"5. Call the service '{self.service}' using the correct message type and interface.\n"
        )


class CallGroundedSAMSegmentTask(CustomInterfacesServiceTask):
    complexity = "easy"

    expected_detections: RAIDetectionArray = RAIDetectionArray(
        header=Header(stamp=Time(sec=0, nanosec=0), frame_id="camera_frame"),
        detections=[],
    )

    def get_prompt(self) -> str:
        return (
            f"You need to call the service '{self.service}' with detections: {self.expected_detections.model_dump()}\n"
            "Before calling, follow these steps:\n"
            "1. Use the tool to retrieve the available ROS2 services and their types.\n"
            f"2. Find the service type for '{self.service}'.\n"
            "3. Retrieve the full message interface definition for that service.\n"
            "4. Construct the request message filling only the fields you are instructed to. Rest of the fields will have default values.\n"
            f"5. Call the service '{self.service}' using the correct message type and interface.\n"
        )


class CallGroundingDinoClassify(CustomInterfacesServiceTask):
    complexity = "easy"

    expected_classes: str = "bottle, book, chair"
    expected_box_threshold: float = 0.4
    expected_text_threshold: float = 0.25

    def get_prompt(self) -> str:
        return (
            f"You need to call the service '{self.service}' with classes: '{self.expected_classes}', "
            f"box_threshold: {self.expected_box_threshold}, text_threshold: {self.expected_text_threshold}, "
            "Before calling, follow these steps:\n"
            "1. Use the tool to retrieve the available ROS2 services and their types.\n"
            f"2. Find the service type for '{self.service}'.\n"
            "3. Retrieve the full message interface definition for that service.\n"
            "4. Construct the request message filling only the fields you are instructed to. Rest of the fields will have default values.\n"
            f"5. Call the service '{self.service}' using the correct message type and interface.\n"
        )


class CallGetLogDigestTask(CustomInterfacesServiceTask):
    complexity = "easy"

    def get_prompt(self) -> str:
        return (
            f"You need to call the service '{self.service}' with an empty request.\n"
            "Before calling, follow these steps:\n"
            "1. Use the tool to retrieve the available ROS2 services and their types.\n"
            f"2. Find the service type for '{self.service}'.\n"
            "3. Retrieve the full message interface definition for that service.\n"
            "4. Construct the message filling only the fields you are instructed to. Rest of the fields will have default values.\n"
            f"5. Call the service '{self.service}' using the correct message type and interface.\n"
        )


class CallVectorStoreRetrievalTask(CustomInterfacesServiceTask):
    complexity = "easy"
    expected_query: str = "What is the purpose of this robot?"

    def get_prompt(self) -> str:
        return (
            f"You need to call the service '{self.service}' with the query: '{self.expected_query}'.\n"
            "Before calling, follow these steps:\n"
            "1. Use the tool to retrieve the available ROS2 services and their types.\n"
            f"2. Find the service type for '{self.service}'.\n"
            "3. Retrieve the full message interface definition for that service.\n"
            "4. Construct the request message filling only the fields you are instructed to. Rest of the fields will have default values.\n"
            f"5. Call the service '{self.service}' using the correct message type and interface.\n"
        )


class CallWhatISeeTask(CustomInterfacesServiceTask):
    complexity = "easy"

    expected_observations: List[str] = ["table", "cup", "notebook"]
    expected_perception_source: str = "front_camera"

    expected_image: Image = Image(
        header=Header(frame_id="camera_frame"),
        height=480,
        width=640,
    )

    expected_pose: Pose = Pose(
        position=Point(x=1.0, y=2.0, z=0.5),
        orientation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0),
    )

    def get_prompt(self) -> str:
        return (
            f"You need to call the service '{self.service}' with an empty request.\n"
            "Before calling, follow these steps:\n"
            "1. Use the tool to retrieve the available ROS2 services and their types.\n"
            f"2. Find the service type for '{self.service}'.\n"
            "3. Retrieve the full message interface definition for that service.\n"
            "4. Construct the request message filling only the fields you are instructed to. Rest of the fields will have default values.\n"
            f"5. Call the service '{self.service}' using the correct message type and interface.\n"
        )

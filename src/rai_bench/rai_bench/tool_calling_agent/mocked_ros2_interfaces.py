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

from typing import Dict, Type

from pydantic import BaseModel
from rai.types import (
    CameraInfo,
    Image,
)
from rai.types.rai_interfaces import (
    ManipulatorMoveToRequest,
    RAIDetectionArray,
    RAIGroundedSamRequest,
    RAIGroundingDinoRequest,
)

from rai_bench.tool_calling_agent.messages.actions import (
    AssistedTeleopGoal,
    BackUpGoal,
    ComputePathThroughPosesGoal,
    ComputePathToPoseGoal,
    DriveOnHeadingGoal,
    FollowPathGoal,
    FollowWaypointsGoal,
    NavigateThroughPosesGoal,
    NavigateToPoseGoal,
    SmoothPathGoal,
    SpinGoal,
    WaitGoal,
)
from rai_bench.tool_calling_agent.messages.base import Clock
from rai_bench.tool_calling_agent.messages.services import (
    StringListRequest,
    VectorStoreRetrievalRequest,
    WhatISeeRequest,
)
from rai_bench.tool_calling_agent.messages.topics import AudioMessage, HRIMessage

# dict of interfaces where keys are interfaces types and values are output
# of GetROS2MessageInterfaceTool which are same as ros2 interface show outputs
# the dict contains custom as well as couple other common interfaces

COMMON_INTERFACES: Dict[str, str] = {
    "std_srvs/srv/Empty": """# Empty service - no request or response
---
""",
    "std_srvs/srv/Trigger": """# Simple service to trigger an action
---
bool success   # indicate successful run of triggered service
string message # informational, e.g. for error messages
""",
    "std_srvs/srv/SetBool": """bool data # e.g. for hardware enabling / disabling
---
bool success   # indicate successful run of triggered service
string message # informational, e.g. for error messages
""",
    "std_srvs/srv/SetString": """string data
---
bool success
string message
""",
    "lifecycle_msgs/srv/ChangeState": """Transition transition
	uint8 id
	string label
---
bool success
""",
    "lifecycle_msgs/srv/GetState": """---
State current_state
	uint8 id
	string label
""",
    "lifecycle_msgs/srv/GetAvailableStates": """---
State[] available_states
	uint8 id
	string label
""",
    "lifecycle_msgs/srv/GetAvailableTransitions": """---
TransitionDescription[] available_transitions
	Transition transition
		uint8 id
		string label
	State start_state
		uint8 id
		string label
	State goal_state
		uint8 id
		string label
""",
    "rcl_interfaces/msg/ParameterEvent": """# This message is published when parameters change for a node
Parameter[] changed_parameters
	string name
	ParameterValue value
		uint8 type
		bool bool_value
		int64 integer_value
		float64 double_value
		string string_value
		byte[] byte_array_value
		bool[] bool_array_value
		int64[] integer_array_value
		float64[] double_array_value
		string[] string_array_value

Parameter[] deleted_parameters
	string name
	ParameterValue value
		uint8 type
		bool bool_value
		int64 integer_value
		float64 double_value
		string string_value
		byte[] byte_array_value
		bool[] bool_array_value
		int64[] integer_array_value
		float64[] double_array_value
		string[] string_array_value

string node
""",
    "rcl_interfaces/msg/Log": """# This message represents a log message published on the /rosout topic
# severity level constants
byte DEBUG=10
byte INFO=20
byte WARN=30
byte ERROR=40
byte FATAL=50

# message fields
builtin_interfaces/Time stamp
	int32 sec
	uint32 nanosec
byte level
string name      # name of the node
string msg       # message text
string file      # file the message came from
string function  # function the message came from
uint32 line      # line the message came from
""",
    "tf2_msgs/msg/TFMessage": """# An array of transforms with a header for the coordinate frame
geometry_msgs/TransformStamped[] transforms
	std_msgs/Header header
		builtin_interfaces/Time stamp
			int32 sec
			uint32 nanosec
		string frame_id
	string child_frame_id
	Transform transform
		Vector3 translation
			float64 x
			float64 y
			float64 z
		Quaternion rotation
			float64 x 0
			float64 y 0
			float64 z 0
			float64 w 1
""",
    "sensor_msgs/msg/JointState": """# This is a message that holds data to describe the state of a set of torque controlled joints.
#
# The state of each joint (revolute or prismatic) is defined by:
#  * the position of the joint (rad or m),
#  * the velocity of the joint (rad/s or m/s) and
#  * the effort that is applied in the joint (Nm or N).
#
# Each joint is uniquely identified by its name
# The header specifies the time at which the joint states were recorded. All the joint states
# in one message have to be recorded at the same time.
#
# This message consists of a multiple arrays, one for each part of the joint state.
# The goal is to make each of the fields optional. When e.g. your joints have no
# velocity or effort sensors, you can leave the velocity and effort arrays empty.
#
# All arrays in this message should have the same size.

std_msgs/Header header
	builtin_interfaces/Time stamp
		int32 sec
		uint32 nanosec
	string frame_id
string[] name
float64[] position
float64[] velocity
float64[] effort
""",
    "std_msgs/msg/String": """# Please look at the Standard ROS Messages documentation before using this.
# http://wiki.ros.org/std_msgs
string data
""",
    "bond/msg/Status": """# An array of bond ids that this node is maintaining
std_msgs/Header header
	builtin_interfaces/Time stamp
		int32 sec
		uint32 nanosec
	string frame_id
string id        # unique identifier for the bond
string instance_id # identifier for this instance of the node
bool active      # whether the bond is currently active
float32 heartbeat_timeout # timeout for heartbeat in seconds
float32 heartbeat_period  # period for heartbeat messages in seconds
""",
    "diagnostic_msgs/msg/DiagnosticArray": """# This message contains a list of diagnostic statuses
std_msgs/Header header
	builtin_interfaces/Time stamp
		int32 sec
		uint32 nanosec
	string frame_id

DiagnosticStatus[] status
	# Possible levels of operations
	byte OK=0
	byte WARN=1
	byte ERROR=2
	byte STALE=3

	byte level           # level of operation enumerated above
	string name          # a description of the test/component reporting
	string message       # a description of the status
	string hardware_id   # a hardware unique string
	KeyValue[] values    # an array of values associated with the status
		string key
		string value
""",
    "sensor_msgs/msg/PointCloud2": """# This message holds a collection of N-dimensional points, which may
# contain additional information such as normals, intensity, etc. The
# point data is stored as a binary blob, its format described by the
# contents of the \"fields\" array.

# The point cloud data may be organized 2d (image-like) or 1d
# (unordered). Point clouds organized as 2d images may be produced by
# camera depth sensors such as stereo or time-of-flight.

# Time of sensor data acquisition, and the coordinate frame ID (for 3d
# points).
std_msgs/Header header
	builtin_interfaces/Time stamp
		int32 sec
		uint32 nanosec
	string frame_id

# 2D structure of the point cloud. If the cloud is unordered, height is
# 1 and width is the length of the point cloud.
uint32 height
uint32 width

# Describes the channels and their layout in the binary data blob.
PointField[] fields
	uint8 INT8    = 1
	uint8 UINT8   = 2
	uint8 INT16   = 3
	uint8 UINT16  = 4
	uint8 INT32   = 5
	uint8 UINT32  = 6
	uint8 FLOAT32 = 7
	uint8 FLOAT64 = 8

	string name      # Name of field
	uint32 offset    # Offset from start of point struct
	uint8  datatype  # Datatype enumeration, see above
	uint32 count     # How many elements in the field

bool    is_bigendian # Is this data bigendian?
uint32  point_step   # Length of a point in bytes
uint32  row_step     # Length of a row in bytes
uint8[] data         # Actual point data, size is (row_step*height)

bool is_dense        # True if there are no invalid points
""",
    "sensor_msgs/msg/LaserScan": """# Single scan from a planar laser range-finder
#
# If you have another ranging device with different behavior (e.g. a sonar
# array), please find or create a different message, since applications
# will make fairly laser-specific assumptions about this data

std_msgs/Header header
	builtin_interfaces/Time stamp
		int32 sec
		uint32 nanosec
	string frame_id
float32 angle_min        # start angle of the scan [rad]
float32 angle_max        # end angle of the scan [rad]
float32 angle_increment  # angular distance between measurements [rad]

float32 time_increment   # time between measurements [seconds] - if your scanner
                         # is moving, this will be used in interpolating position
                         # of 3d points
float32 scan_time        # time between scans [seconds]

float32 range_min        # minimum range value [m]
float32 range_max        # maximum range value [m]

float32[] ranges         # range data [m] (Note: values < range_min or > range_max should be discarded)
float32[] intensities    # intensity data [device-specific units].  If your
                         # device does not provide intensities, please leave
                         # the array empty.
""",
    "nav_msgs/msg/Odometry": """# This represents an estimate of a position and velocity in free space.
# The pose in this message should be specified in the coordinate frame given by header.frame_id.
# The twist in this message should be specified in the coordinate frame given by the child_frame_id
std_msgs/Header header
	builtin_interfaces/Time stamp
		int32 sec
		uint32 nanosec
	string frame_id
string child_frame_id
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
	float64[36] covariance # Row-major representation of the 6x6 covariance matrix
geometry_msgs/TwistWithCovariance twist
	Twist twist
		Vector3  linear
			float64 x
			float64 y
			float64 z
		Vector3  angular
			float64 x
			float64 y
			float64 z
	float64[36] covariance # Row-major representation of the 6x6 covariance matrix
""",
    # Services from COMMON_SERVICES_AND_TYPES that are missing
    "tf2_msgs/srv/FrameGraph": """---
string frame_yaml
""",
    "composition_interfaces/srv/ListNodes": """---
# All unique node names within the container
string[] unique_names
# Full node names corresponding to each unique node name
string[] full_node_names
""",
    "composition_interfaces/srv/LoadNode": """LoadNodeRequest request
	string package_name
	string plugin_name
	string node_name
	string node_namespace
	string[] remap_rules
	Parameter[] parameters
		string name
		ParameterValue value
			uint8 type
			bool bool_value
			int64 integer_value
			float64 double_value
			string string_value
			byte[] byte_array_value
			bool[] bool_array_value
			int64[] integer_array_value
			float64[] double_array_value
			string[] string_array_value
	string[] extra_arguments
---
bool success
string error_message
string full_node_name
uint64 unique_id
""",
    "composition_interfaces/srv/UnloadNode": """uint64 unique_id
---
bool success
string error_message
""",
    "rcl_interfaces/srv/DescribeParameters": """string[] names
---
ParameterDescriptor[] descriptors
	string name
	uint8 type
	string description
	string additional_constraints
	bool read_only
	bool dynamic_typing
	ParameterValue floating_point_range
		uint8 type
		bool bool_value
		int64 integer_value
		float64 double_value
		string string_value
		byte[] byte_array_value
		bool[] bool_array_value
		int64[] integer_array_value
		float64[] double_array_value
		string[] string_array_value
	ParameterValue integer_range
		uint8 type
		bool bool_value
		int64 integer_value
		float64 double_value
		string string_value
		byte[] byte_array_value
		bool[] bool_array_value
		int64[] integer_array_value
		float64[] double_array_value
		string[] string_array_value
""",
    "rcl_interfaces/srv/GetParameterTypes": """string[] names
---
uint8[] types
# Possible parameter types:
uint8 PARAMETER_NOT_SET=0
uint8 PARAMETER_BOOL=1
uint8 PARAMETER_INTEGER=2
uint8 PARAMETER_DOUBLE=3
uint8 PARAMETER_STRING=4
uint8 PARAMETER_BYTE_ARRAY=5
uint8 PARAMETER_BOOL_ARRAY=6
uint8 PARAMETER_INTEGER_ARRAY=7
uint8 PARAMETER_DOUBLE_ARRAY=8
uint8 PARAMETER_STRING_ARRAY=9
""",
    "rcl_interfaces/srv/GetParameters": """string[] names
---
ParameterValue[] values
	uint8 type
	bool bool_value
	int64 integer_value
	float64 double_value
	string string_value
	byte[] byte_array_value
	bool[] bool_array_value
	int64[] integer_array_value
	float64[] double_array_value
	string[] string_array_value
""",
    "rcl_interfaces/srv/ListParameters": """ListParametersRequest request
	string[] prefixes
	uint64 depth
---
ListParametersResult result
	string[] names
	string[] prefixes
""",
    "rcl_interfaces/srv/SetParametersAtomically": """Parameter[] parameters
	string name
	ParameterValue value
		uint8 type
		bool bool_value
		int64 integer_value
		float64 double_value
		string string_value
		byte[] byte_array_value
		bool[] bool_array_value
		int64[] integer_array_value
		float64[] double_array_value
		string[] string_array_value
---
SetParametersResult result
	bool successful
	string reason
""",
    "gazebo_msgs/srv/GetWorldProperties": """# Service to get world properties
---
string[] model_names
string[] light_names
bool rendering_enabled
bool physics_enabled
bool physics_paused
float64 sim_time
""",
    "gazebo_msgs/srv/GetModelState": """string model_name
string relative_entity_name  # return pose relative to this entity
                             # an empty string will return world relative pose
---
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
geometry_msgs/Twist twist
	Vector3  linear
		float64 x
		float64 y
		float64 z
	Vector3  angular
		float64 x
		float64 y
		float64 z
bool success
string status_message
""",
    "gazebo_msgs/srv/DeleteEntity": """string name                       # Name of the Gazebo entity to be deleted. This can be either
                                  # a model or a light.
---
bool success                      # Return true if deletion is successful.
string status_message             # Comments if available.
""",
    "gazebo_msgs/srv/SpawnEntity": """string name                       # Name of the entity to be spawned (optional).
string xml                        # Entity XML description as a string, either URDF or SDF.
string robot_namespace            # Spawn robot and all ROS interfaces under this namespace
geometry_msgs/Pose initial_pose   # Initial entity pose.
	Point position
		float64 x
		float64 y
		float64 z
	Quaternion orientation
		float64 x 0
		float64 y 0
		float64 z 0
		float64 w 1
string reference_frame            # initial_pose is defined relative to the frame of this entity.
                                  # If left empty or "world" or "map", then gazebo world frame is
                                  # used.
                                  # If non-existent entity is specified, an error is returned
                                  # and the entity is not spawned.
---
bool success                      # Return true if spawned successfully.
string status_message             # Comments if available.""",
    "rcl_interfaces/srv/SetParameters": """# A list of parameters to set.
Parameter[] parameters
	string name
	ParameterValue value
		uint8 type
		bool bool_value
		int64 integer_value
		float64 double_value
		string string_value
		byte[] byte_array_value
		bool[] bool_array_value
		int64[] integer_array_value
		float64[] double_array_value
		string[] string_array_value

---
# Indicates whether setting each parameter succeeded or not and why.""",
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
}
MANIPULATION_INTERFACES: Dict[str, str] = {
    "moveit_msgs/action/ExecuteTrajectory": """# The trajectory to execute
RobotTrajectory trajectory
	trajectory_msgs/JointTrajectory joint_trajectory
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
	trajectory_msgs/MultiDOFJointTrajectory multi_dof_joint_trajectory
		std_msgs/Header header
			builtin_interfaces/Time stamp
				int32 sec
				uint32 nanosec
			string frame_id
		string[] joint_names
		MultiDOFJointTrajectoryPoint[] points
			geometry_msgs/Transform[] transforms
				Vector3 translation
					float64 x
					float64 y
					float64 z
				Quaternion rotation
					float64 x 0
					float64 y 0
					float64 z 0
					float64 w 1
			geometry_msgs/Twist[] velocities
				Vector3  linear
					float64 x
					float64 y
					float64 z
				Vector3  angular
					float64 x
					float64 y
					float64 z
			geometry_msgs/Twist[] accelerations
				Vector3  linear
					float64 x
					float64 y
					float64 z
				Vector3  angular
					float64 x
					float64 y
					float64 z
			builtin_interfaces/Duration time_from_start
				int32 sec
				uint32 nanosec

---

# Error code - encodes the overall reason for failure
MoveItErrorCodes error_code
	int32 val
	int32 SUCCESS=1
	int32 FAILURE=99999
	int32 PLANNING_FAILED=-1
	int32 INVALID_MOTION_PLAN=-2
	int32 MOTION_PLAN_INVALIDATED_BY_ENVIRONMENT_CHANGE=-3
	int32 CONTROL_FAILED=-4
	int32 UNABLE_TO_AQUIRE_SENSOR_DATA=-5
	int32 TIMED_OUT=-6
	int32 PREEMPTED=-7
	int32 START_STATE_IN_COLLISION=-10
	int32 START_STATE_VIOLATES_PATH_CONSTRAINTS=-11
	int32 START_STATE_INVALID=-26
	int32 GOAL_IN_COLLISION=-12
	int32 GOAL_VIOLATES_PATH_CONSTRAINTS=-13
	int32 GOAL_CONSTRAINTS_VIOLATED=-14
	int32 GOAL_STATE_INVALID=-27
	int32 UNRECOGNIZED_GOAL_TYPE=-28
	int32 INVALID_GROUP_NAME=-15
	int32 INVALID_GOAL_CONSTRAINTS=-16
	int32 INVALID_ROBOT_STATE=-17
	int32 INVALID_LINK_NAME=-18
	int32 INVALID_OBJECT_NAME=-19
	int32 FRAME_TRANSFORM_FAILURE=-21
	int32 COLLISION_CHECKING_UNAVAILABLE=-22
	int32 ROBOT_STATE_STALE=-23
	int32 SENSOR_INFO_STALE=-24
	int32 COMMUNICATION_FAILURE=-25
	int32 CRASH=-29
	int32 ABORT=-30
	int32 NO_IK_SOLUTION=-31

---

# The internal state that the move group action currently is in
string state}
""",
    "moveit_msgs/action/MoveGroup": """# Motion planning request to pass to planner
MotionPlanRequest request
	WorkspaceParameters workspace_parameters
		std_msgs/Header header
			builtin_interfaces/Time stamp
				int32 sec
				uint32 nanosec
			string frame_id
		geometry_msgs/Vector3 min_corner
			float64 x
			float64 y
			float64 z
		geometry_msgs/Vector3 max_corner
			float64 x
			float64 y
			float64 z
	RobotState start_state
		sensor_msgs/JointState joint_state
			#
			#
			#
			#
			std_msgs/Header header
				builtin_interfaces/Time stamp
					int32 sec
					uint32 nanosec
				string frame_id
			string[] name
			float64[] position
			float64[] velocity
			float64[] effort
		sensor_msgs/MultiDOFJointState multi_dof_joint_state
			#
			#
			#
			#
			std_msgs/Header header
				builtin_interfaces/Time stamp
					int32 sec
					uint32 nanosec
				string frame_id
			string[] joint_names
			geometry_msgs/Transform[] transforms
				Vector3 translation
					float64 x
					float64 y
					float64 z
				Quaternion rotation
					float64 x 0
					float64 y 0
					float64 z 0
					float64 w 1
			geometry_msgs/Twist[] twist
				Vector3  linear
					float64 x
					float64 y
					float64 z
				Vector3  angular
					float64 x
					float64 y
					float64 z
			geometry_msgs/Wrench[] wrench
				Vector3  force
					float64 x
					float64 y
					float64 z
				Vector3  torque
					float64 x
					float64 y
					float64 z
		AttachedCollisionObject[] attached_collision_objects
			string link_name
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
			string[] touch_links
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
			float64 weight
		bool is_diff
	Constraints[] goal_constraints
		string name
		JointConstraint[] joint_constraints
			string joint_name
			float64 position
			float64 tolerance_above
			float64 tolerance_below
			float64 weight
		PositionConstraint[] position_constraints
			std_msgs/Header header
				builtin_interfaces/Time stamp
					int32 sec
					uint32 nanosec
				string frame_id
			string link_name
			geometry_msgs/Vector3 target_point_offset
				float64 x
				float64 y
				float64 z
			BoundingVolume constraint_region
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
			float64 weight
		OrientationConstraint[] orientation_constraints
			std_msgs/Header header
				builtin_interfaces/Time stamp
					int32 sec
					uint32 nanosec
				string frame_id
			geometry_msgs/Quaternion orientation
				float64 x 0
				float64 y 0
				float64 z 0
				float64 w 1
			string link_name
			float64 absolute_x_axis_tolerance
			float64 absolute_y_axis_tolerance
			float64 absolute_z_axis_tolerance
			uint8 parameterization
			uint8 XYZ_EULER_ANGLES=0
			uint8 ROTATION_VECTOR=1
			float64 weight
		VisibilityConstraint[] visibility_constraints
			float64 target_radius
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
			int32 cone_sides
			geometry_msgs/PoseStamped sensor_pose
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
			float64 max_view_angle
			float64 max_range_angle
			uint8 SENSOR_Z=0
			uint8 SENSOR_Y=1
			uint8 SENSOR_X=2
			uint8 sensor_view_direction
			float64 weight
	Constraints path_constraints
		string name
		JointConstraint[] joint_constraints
			string joint_name
			float64 position
			float64 tolerance_above
			float64 tolerance_below
			float64 weight
		PositionConstraint[] position_constraints
			std_msgs/Header header
				builtin_interfaces/Time stamp
					int32 sec
					uint32 nanosec
				string frame_id
			string link_name
			geometry_msgs/Vector3 target_point_offset
				float64 x
				float64 y
				float64 z
			BoundingVolume constraint_region
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
			float64 weight
		OrientationConstraint[] orientation_constraints
			std_msgs/Header header
				builtin_interfaces/Time stamp
					int32 sec
					uint32 nanosec
				string frame_id
			geometry_msgs/Quaternion orientation
				float64 x 0
				float64 y 0
				float64 z 0
				float64 w 1
			string link_name
			float64 absolute_x_axis_tolerance
			float64 absolute_y_axis_tolerance
			float64 absolute_z_axis_tolerance
			uint8 parameterization
			uint8 XYZ_EULER_ANGLES=0
			uint8 ROTATION_VECTOR=1
			float64 weight
		VisibilityConstraint[] visibility_constraints
			float64 target_radius
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
			int32 cone_sides
			geometry_msgs/PoseStamped sensor_pose
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
			float64 max_view_angle
			float64 max_range_angle
			uint8 SENSOR_Z=0
			uint8 SENSOR_Y=1
			uint8 SENSOR_X=2
			uint8 sensor_view_direction
			float64 weight
	TrajectoryConstraints trajectory_constraints
		Constraints[] constraints
			string name
			JointConstraint[] joint_constraints
				string joint_name
				float64 position
				float64 tolerance_above
				float64 tolerance_below
				float64 weight
			PositionConstraint[] position_constraints
				std_msgs/Header header
					builtin_interfaces/Time stamp
						int32 sec
						uint32 nanosec
					string frame_id
				string link_name
				geometry_msgs/Vector3 target_point_offset
					float64 x
					float64 y
					float64 z
				BoundingVolume constraint_region
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
				float64 weight
			OrientationConstraint[] orientation_constraints
				std_msgs/Header header
					builtin_interfaces/Time stamp
						int32 sec
						uint32 nanosec
					string frame_id
				geometry_msgs/Quaternion orientation
					float64 x 0
					float64 y 0
					float64 z 0
					float64 w 1
				string link_name
				float64 absolute_x_axis_tolerance
				float64 absolute_y_axis_tolerance
				float64 absolute_z_axis_tolerance
				uint8 parameterization
				uint8 XYZ_EULER_ANGLES=0
				uint8 ROTATION_VECTOR=1
				float64 weight
			VisibilityConstraint[] visibility_constraints
				float64 target_radius
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
				int32 cone_sides
				geometry_msgs/PoseStamped sensor_pose
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
				float64 max_view_angle
				float64 max_range_angle
				uint8 SENSOR_Z=0
				uint8 SENSOR_Y=1
				uint8 SENSOR_X=2
				uint8 sensor_view_direction
				float64 weight
	GenericTrajectory[] reference_trajectories
		std_msgs/Header header
			builtin_interfaces/Time stamp
				int32 sec
				uint32 nanosec
			string frame_id
		trajectory_msgs/JointTrajectory[] joint_trajectory
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
		moveit_msgs/CartesianTrajectory[] cartesian_trajectory
			std_msgs/Header header
				builtin_interfaces/Time stamp
					int32 sec
					uint32 nanosec
				string frame_id
			string tracked_frame
			CartesianTrajectoryPoint[] points
				CartesianPoint point
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
					geometry_msgs/Twist velocity
						Vector3  linear
							float64 x
							float64 y
							float64 z
						Vector3  angular
							float64 x
							float64 y
							float64 z
					geometry_msgs/Accel acceleration
						Vector3  linear
							float64 x
							float64 y
							float64 z
						Vector3  angular
							float64 x
							float64 y
							float64 z
				builtin_interfaces/Duration time_from_start
					int32 sec
					uint32 nanosec
	string pipeline_id
	string planner_id
	string group_name
	int32 num_planning_attempts
	float64 allowed_planning_time
	float64 max_velocity_scaling_factor
	float64 max_acceleration_scaling_factor
	string cartesian_speed_end_effector_link
	float64 max_cartesian_speed #

# Planning options
PlanningOptions planning_options
	PlanningScene planning_scene_diff
		string name
		RobotState robot_state
			sensor_msgs/JointState joint_state
				#
				#
				#
				#
				std_msgs/Header header
					builtin_interfaces/Time stamp
						int32 sec
						uint32 nanosec
					string frame_id
				string[] name
				float64[] position
				float64[] velocity
				float64[] effort
			sensor_msgs/MultiDOFJointState multi_dof_joint_state
				#
				#
				#
				#
				std_msgs/Header header
					builtin_interfaces/Time stamp
						int32 sec
						uint32 nanosec
					string frame_id
				string[] joint_names
				geometry_msgs/Transform[] transforms
					Vector3 translation
						float64 x
						float64 y
						float64 z
					Quaternion rotation
						float64 x 0
						float64 y 0
						float64 z 0
						float64 w 1
				geometry_msgs/Twist[] twist
					Vector3  linear
						float64 x
						float64 y
						float64 z
					Vector3  angular
						float64 x
						float64 y
						float64 z
				geometry_msgs/Wrench[] wrench
					Vector3  force
						float64 x
						float64 y
						float64 z
					Vector3  torque
						float64 x
						float64 y
						float64 z
			AttachedCollisionObject[] attached_collision_objects
				string link_name
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
				string[] touch_links
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
				float64 weight
			bool is_diff
		string robot_model_name
		geometry_msgs/TransformStamped[] fixed_frame_transforms
			#
			#
			std_msgs/Header header
				builtin_interfaces/Time stamp
					int32 sec
					uint32 nanosec
				string frame_id
			string child_frame_id
			Transform transform
				Vector3 translation
					float64 x
					float64 y
					float64 z
				Quaternion rotation
					float64 x 0
					float64 y 0
					float64 z 0
					float64 w 1
		AllowedCollisionMatrix allowed_collision_matrix
			string[] entry_names
			AllowedCollisionEntry[] entry_values
				bool[] enabled
			string[] default_entry_names
			bool[] default_entry_values
		LinkPadding[] link_padding
			string link_name
			float64 padding
		LinkScale[] link_scale
			string link_name
			float64 scale
		ObjectColor[] object_colors
			string id
			std_msgs/ColorRGBA color
				float32 r
				float32 g
				float32 b
				float32 a
		PlanningSceneWorld world
			CollisionObject[] collision_objects
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
			octomap_msgs/OctomapWithPose octomap
				std_msgs/Header header
					builtin_interfaces/Time stamp
						int32 sec
						uint32 nanosec
					string frame_id
				geometry_msgs/Pose origin
					Point position
						float64 x
						float64 y
						float64 z
					Quaternion orientation
						float64 x 0
						float64 y 0
						float64 z 0
						float64 w 1
				octomap_msgs/Octomap octomap
					std_msgs/Header header
						builtin_interfaces/Time stamp
							int32 sec
							uint32 nanosec
						string frame_id
					bool binary
					string id
					float64 resolution
					int8[] data
		bool is_diff
	bool plan_only
	bool look_around
	int32 look_around_attempts
	float64 max_safe_execution_cost
	bool replan
	int32 replan_attempts
	float64 replan_delay

---

# An error code reflecting what went wrong
MoveItErrorCodes error_code
	int32 val
	int32 SUCCESS=1
	int32 FAILURE=99999
	int32 PLANNING_FAILED=-1
	int32 INVALID_MOTION_PLAN=-2
	int32 MOTION_PLAN_INVALIDATED_BY_ENVIRONMENT_CHANGE=-3
	int32 CONTROL_FAILED=-4
	int32 UNABLE_TO_AQUIRE_SENSOR_DATA=-5
	int32 TIMED_OUT=-6
	int32 PREEMPTED=-7
	int32 START_STATE_IN_COLLISION=-10
	int32 START_STATE_VIOLATES_PATH_CONSTRAINTS=-11
	int32 START_STATE_INVALID=-26
	int32 GOAL_IN_COLLISION=-12
	int32 GOAL_VIOLATES_PATH_CONSTRAINTS=-13
	int32 GOAL_CONSTRAINTS_VIOLATED=-14
	int32 GOAL_STATE_INVALID=-27
	int32 UNRECOGNIZED_GOAL_TYPE=-28
	int32 INVALID_GROUP_NAME=-15
	int32 INVALID_GOAL_CONSTRAINTS=-16
	int32 INVALID_ROBOT_STATE=-17
	int32 INVALID_LINK_NAME=-18
	int32 INVALID_OBJECT_NAME=-19
	int32 FRAME_TRANSFORM_FAILURE=-21
	int32 COLLISION_CHECKING_UNAVAILABLE=-22
	int32 ROBOT_STATE_STALE=-23
	int32 SENSOR_INFO_STALE=-24
	int32 COMMUNICATION_FAILURE=-25
	int32 CRASH=-29
	int32 ABORT=-30
	int32 NO_IK_SOLUTION=-31

# The full starting state of the robot at the start of the trajectory
moveit_msgs/RobotState trajectory_start
	sensor_msgs/JointState joint_state
		#
		#
		#
		#
		std_msgs/Header header
			builtin_interfaces/Time stamp
				int32 sec
				uint32 nanosec
			string frame_id
		string[] name
		float64[] position
		float64[] velocity
		float64[] effort
	sensor_msgs/MultiDOFJointState multi_dof_joint_state
		#
		#
		#
		#
		std_msgs/Header header
			builtin_interfaces/Time stamp
				int32 sec
				uint32 nanosec
			string frame_id
		string[] joint_names
		geometry_msgs/Transform[] transforms
			Vector3 translation
				float64 x
				float64 y
				float64 z
			Quaternion rotation
				float64 x 0
				float64 y 0
				float64 z 0
				float64 w 1
		geometry_msgs/Twist[] twist
			Vector3  linear
				float64 x
				float64 y
				float64 z
			Vector3  angular
				float64 x
				float64 y
				float64 z
		geometry_msgs/Wrench[] wrench
			Vector3  force
				float64 x
				float64 y
				float64 z
			Vector3  torque
				float64 x
				float64 y
				float64 z
	AttachedCollisionObject[] attached_collision_objects
		string link_name
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
		string[] touch_links
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
		float64 weight
	bool is_diff

# The trajectory that moved group produced for execution
moveit_msgs/RobotTrajectory planned_trajectory
	trajectory_msgs/JointTrajectory joint_trajectory
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
	trajectory_msgs/MultiDOFJointTrajectory multi_dof_joint_trajectory
		std_msgs/Header header
			builtin_interfaces/Time stamp
				int32 sec
				uint32 nanosec
			string frame_id
		string[] joint_names
		MultiDOFJointTrajectoryPoint[] points
			geometry_msgs/Transform[] transforms
				Vector3 translation
					float64 x
					float64 y
					float64 z
				Quaternion rotation
					float64 x 0
					float64 y 0
					float64 z 0
					float64 w 1
			geometry_msgs/Twist[] velocities
				Vector3  linear
					float64 x
					float64 y
					float64 z
				Vector3  angular
					float64 x
					float64 y
					float64 z
			geometry_msgs/Twist[] accelerations
				Vector3  linear
					float64 x
					float64 y
					float64 z
				Vector3  angular
					float64 x
					float64 y
					float64 z
			builtin_interfaces/Duration time_from_start
				int32 sec
				uint32 nanosec

# The trace of the trajectory recorded during execution
moveit_msgs/RobotTrajectory executed_trajectory
	trajectory_msgs/JointTrajectory joint_trajectory
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
	trajectory_msgs/MultiDOFJointTrajectory multi_dof_joint_trajectory
		std_msgs/Header header
			builtin_interfaces/Time stamp
				int32 sec
				uint32 nanosec
			string frame_id
		string[] joint_names
		MultiDOFJointTrajectoryPoint[] points
			geometry_msgs/Transform[] transforms
				Vector3 translation
					float64 x
					float64 y
					float64 z
				Quaternion rotation
					float64 x 0
					float64 y 0
					float64 z 0
					float64 w 1
			geometry_msgs/Twist[] velocities
				Vector3  linear
					float64 x
					float64 y
					float64 z
				Vector3  angular
					float64 x
					float64 y
					float64 z
			geometry_msgs/Twist[] accelerations
				Vector3  linear
					float64 x
					float64 y
					float64 z
				Vector3  angular
					float64 x
					float64 y
					float64 z
			builtin_interfaces/Duration time_from_start
				int32 sec
				uint32 nanosec

# The amount of time it took to complete the motion plan
float64 planning_time

---

# The internal state that the move group action currently is in
string state
""",
    "control_msgs/action/FollowJointTrajectory": """# The trajectory for all revolute, continuous or prismatic joints
trajectory_msgs/JointTrajectory trajectory
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
# The trajectory for all planar or floating joints (i.e. individual joints with more than one DOF)
trajectory_msgs/MultiDOFJointTrajectory multi_dof_trajectory
	std_msgs/Header header
		builtin_interfaces/Time stamp
			int32 sec
			uint32 nanosec
		string frame_id
	string[] joint_names
	MultiDOFJointTrajectoryPoint[] points
		geometry_msgs/Transform[] transforms
			Vector3 translation
				float64 x
				float64 y
				float64 z
			Quaternion rotation
				float64 x 0
				float64 y 0
				float64 z 0
				float64 w 1
		geometry_msgs/Twist[] velocities
			Vector3  linear
				float64 x
				float64 y
				float64 z
			Vector3  angular
				float64 x
				float64 y
				float64 z
		geometry_msgs/Twist[] accelerations
			Vector3  linear
				float64 x
				float64 y
				float64 z
			Vector3  angular
				float64 x
				float64 y
				float64 z
		builtin_interfaces/Duration time_from_start
			int32 sec
			uint32 nanosec

# Tolerances for the trajectory.  If the measured joint values fall
# outside the tolerances the trajectory goal is aborted.  Any
# tolerances that are not specified (by being omitted or set to 0) are
# set to the defaults for the action server (often taken from the
# parameter server).

# Tolerances applied to the joints as the trajectory is executed.  If
# violated, the goal aborts with error_code set to
# PATH_TOLERANCE_VIOLATED.
JointTolerance[] path_tolerance
	#
	string name
	float64 position  #
	float64 velocity  #
	float64 acceleration  #
JointComponentTolerance[] component_path_tolerance
	uint16 X_AXIS=1
	uint16 Y_AXIS=2
	uint16 Z_AXIS=3
	uint16 TRANSLATION=4
	uint16 ROTATION=5
	string joint_name
	uint16 component
	float64 position
	float64 velocity
	float64 acceleration

# To report success, the joints must be within goal_tolerance of the
# final trajectory value.  The goal must be achieved by time the
# trajectory ends plus goal_time_tolerance.  (goal_time_tolerance
# allows some leeway in time, so that the trajectory goal can still
# succeed even if the joints reach the goal some time after the
# precise end time of the trajectory).
#
# If the joints are not within goal_tolerance after "trajectory finish
# time" + goal_time_tolerance, the goal aborts with error_code set to
# GOAL_TOLERANCE_VIOLATED
JointTolerance[] goal_tolerance
	#
	string name
	float64 position  #
	float64 velocity  #
	float64 acceleration  #
JointComponentTolerance[] component_goal_tolerance
	uint16 X_AXIS=1
	uint16 Y_AXIS=2
	uint16 Z_AXIS=3
	uint16 TRANSLATION=4
	uint16 ROTATION=5
	string joint_name
	uint16 component
	float64 position
	float64 velocity
	float64 acceleration
builtin_interfaces/Duration goal_time_tolerance
	int32 sec
	uint32 nanosec

---
int32 error_code
int32 SUCCESSFUL = 0
int32 INVALID_GOAL = -1
int32 INVALID_JOINTS = -2
int32 OLD_HEADER_TIMESTAMP = -3
int32 PATH_TOLERANCE_VIOLATED = -4
int32 GOAL_TOLERANCE_VIOLATED = -5

# Human readable description of the error code. Contains complementary
# information that is especially useful when execution fails, for instance:
# - INVALID_GOAL: The reason for the invalid goal (e.g., the requested
#   trajectory is in the past).
# - INVALID_JOINTS: The mismatch between the expected controller joints
#   and those provided in the goal.
# - PATH_TOLERANCE_VIOLATED and GOAL_TOLERANCE_VIOLATED: Which joint
#   violated which tolerance, and by how much.
string error_string

---
std_msgs/Header header
	builtin_interfaces/Time stamp
		int32 sec
		uint32 nanosec
	string frame_id
string[] joint_names
trajectory_msgs/JointTrajectoryPoint desired
	float64[] positions
	float64[] velocities
	float64[] accelerations
	float64[] effort
	builtin_interfaces/Duration time_from_start
		int32 sec
		uint32 nanosec
trajectory_msgs/JointTrajectoryPoint actual
	float64[] positions
	float64[] velocities
	float64[] accelerations
	float64[] effort
	builtin_interfaces/Duration time_from_start
		int32 sec
		uint32 nanosec
trajectory_msgs/JointTrajectoryPoint error
	float64[] positions
	float64[] velocities
	float64[] accelerations
	float64[] effort
	builtin_interfaces/Duration time_from_start
		int32 sec
		uint32 nanosec

string[] multi_dof_joint_names
trajectory_msgs/MultiDOFJointTrajectoryPoint multi_dof_desired
	geometry_msgs/Transform[] transforms
		Vector3 translation
			float64 x
			float64 y
			float64 z
		Quaternion rotation
			float64 x 0
			float64 y 0
			float64 z 0
			float64 w 1
	geometry_msgs/Twist[] velocities
		Vector3  linear
			float64 x
			float64 y
			float64 z
		Vector3  angular
			float64 x
			float64 y
			float64 z
	geometry_msgs/Twist[] accelerations
		Vector3  linear
			float64 x
			float64 y
			float64 z
		Vector3  angular
			float64 x
			float64 y
			float64 z
	builtin_interfaces/Duration time_from_start
		int32 sec
		uint32 nanosec
trajectory_msgs/MultiDOFJointTrajectoryPoint multi_dof_actual
	geometry_msgs/Transform[] transforms
		Vector3 translation
			float64 x
			float64 y
			float64 z
		Quaternion rotation
			float64 x 0
			float64 y 0
			float64 z 0
			float64 w 1
	geometry_msgs/Twist[] velocities
		Vector3  linear
			float64 x
			float64 y
			float64 z
		Vector3  angular
			float64 x
			float64 y
			float64 z
	geometry_msgs/Twist[] accelerations
		Vector3  linear
			float64 x
			float64 y
			float64 z
		Vector3  angular
			float64 x
			float64 y
			float64 z
	builtin_interfaces/Duration time_from_start
		int32 sec
		uint32 nanosec
trajectory_msgs/MultiDOFJointTrajectoryPoint multi_dof_error
	geometry_msgs/Transform[] transforms
		Vector3 translation
			float64 x
			float64 y
			float64 z
		Quaternion rotation
			float64 x 0
			float64 y 0
			float64 z 0
			float64 w 1
	geometry_msgs/Twist[] velocities
		Vector3  linear
			float64 x
			float64 y
			float64 z
		Vector3  angular
			float64 x
			float64 y
			float64 z
	geometry_msgs/Twist[] accelerations
		Vector3  linear
			float64 x
			float64 y
			float64 z
		Vector3  angular
			float64 x
			float64 y
			float64 z
	builtin_interfaces/Duration time_from_start
		int32 sec
		uint32 nanosec
""",
    "/panda_hand_controller/gripper_cmd": """GripperCommand command
float64 position
float64 max_effort
---
float64 position  # The current gripper gap size (in meters)
float64 effort    # The current effort exerted (in Newtons)
bool stalled      # True iff the gripper is exerting max effort and not moving
bool reached_goal # True iff the gripper position has reached the commanded setpoint
---
float64 position  # The current gripper gap size (in meters)
float64 effort    # The current effort exerted (in Newtons)
bool stalled      # True iff the gripper is exerting max effort and not moving
bool reached_goal # True iff the gripper position has reached the commanded setpoint
""",
}


NAVIGATION_INTERFACES: Dict[str, str] = {
    "nav2_msgs/action/NavigateToPose": """#goal definition
geometry_msgs/PoseStamped pose
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
string behavior_tree
---
#result definition
std_msgs/Empty result
---
#feedback definition
geometry_msgs/PoseStamped current_pose
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
builtin_interfaces/Duration navigation_time
	int32 sec
	uint32 nanosec
builtin_interfaces/Duration estimated_time_remaining
	int32 sec
	uint32 nanosec
int16 number_of_recoveries
float32 distance_remaining
""",
    "nav2_msgs/action/AssistedTeleop": """#goal definition
builtin_interfaces/Duration time_allowance
	int32 sec
	uint32 nanosec
---
#result definition
builtin_interfaces/Duration total_elapsed_time
	int32 sec
	uint32 nanosec
---
#feedback
builtin_interfaces/Duration current_teleop_duration
	int32 sec
	uint32 nanosec""",
    "nav2_msgs/action/BackUp": """#goal definition
geometry_msgs/Point target
	float64 x
	float64 y
	float64 z
float32 speed
builtin_interfaces/Duration time_allowance
	int32 sec
	uint32 nanosec
---
#result definition
builtin_interfaces/Duration total_elapsed_time
	int32 sec
	uint32 nanosec
---
#feedback definition
float32 distance_traveled""",
    "nav2_msgs/action/ComputePathThroughPoses": """#goal definition
geometry_msgs/PoseStamped[] goals
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
geometry_msgs/PoseStamped start
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
string planner_id
bool use_start # If false, use current robot pose as path start, if true, use start above instead
---
#result definition
nav_msgs/Path path
	std_msgs/Header header
		builtin_interfaces/Time stamp
			int32 sec
			uint32 nanosec
		string frame_id
	geometry_msgs/PoseStamped[] poses
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
builtin_interfaces/Duration planning_time
	int32 sec
	uint32 nanosec
---
#feedback definition""",
    "nav2_msgs/action/ComputePathToPose": """#goal definition
geometry_msgs/PoseStamped goal
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
geometry_msgs/PoseStamped start
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
string planner_id
bool use_start # If false, use current robot pose as path start, if true, use start above instead
---
#result definition
nav_msgs/Path path
	std_msgs/Header header
		builtin_interfaces/Time stamp
			int32 sec
			uint32 nanosec
		string frame_id
	geometry_msgs/PoseStamped[] poses
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
builtin_interfaces/Duration planning_time
	int32 sec
	uint32 nanosec
---
#feedback definition""",
    "nav2_msgs/action/DriveOnHeading": """#goal definition
geometry_msgs/Point target
	float64 x
	float64 y
	float64 z
float32 speed
builtin_interfaces/Duration time_allowance
	int32 sec
	uint32 nanosec
---
#result definition
builtin_interfaces/Duration total_elapsed_time
	int32 sec
	uint32 nanosec
---
#feedback definition
float32 distance_traveled""",
    "nav2_msgs/action/FollowPath": """#goal definition
nav_msgs/Path path
	std_msgs/Header header
		builtin_interfaces/Time stamp
			int32 sec
			uint32 nanosec
		string frame_id
	geometry_msgs/PoseStamped[] poses
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
string controller_id
string goal_checker_id
---
#result definition
std_msgs/Empty result
---
#feedback definition
float32 distance_to_goal
float32 speed""",
    "nav2_msgs/action/FollowWaypoints": """#goal definition
geometry_msgs/PoseStamped[] poses
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
#result definition
int32[] missed_waypoints
---
#feedback definition
uint32 current_waypoint""",
    "nav2_msgs/action/NavigateThroughPoses": """#goal definition
geometry_msgs/PoseStamped[] poses
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
string behavior_tree
---
#result definition
std_msgs/Empty result
---
#feedback definition
geometry_msgs/PoseStamped current_pose
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
builtin_interfaces/Duration navigation_time
	int32 sec
	uint32 nanosec
builtin_interfaces/Duration estimated_time_remaining
	int32 sec
	uint32 nanosec
int16 number_of_recoveries
float32 distance_remaining
int16 number_of_poses_remaining
""",
    "nav2_msgs/action/SmoothPath": """#goal definition
nav_msgs/Path path
	std_msgs/Header header
		builtin_interfaces/Time stamp
			int32 sec
			uint32 nanosec
		string frame_id
	geometry_msgs/PoseStamped[] poses
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
string smoother_id
builtin_interfaces/Duration max_smoothing_duration
	int32 sec
	uint32 nanosec
bool check_for_collisions
---
#result definition
nav_msgs/Path path
	std_msgs/Header header
		builtin_interfaces/Time stamp
			int32 sec
			uint32 nanosec
		string frame_id
	geometry_msgs/PoseStamped[] poses
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
builtin_interfaces/Duration smoothing_duration
	int32 sec
	uint32 nanosec
bool was_completed
---
#feedback definition
""",
    "nav2_msgs/action/Wait": """#goal definition
builtin_interfaces/Duration time
	int32 sec
	uint32 nanosec
---
#result definition
builtin_interfaces/Duration total_elapsed_time
	int32 sec
	uint32 nanosec
---
#feedback definition
builtin_interfaces/Duration time_left
	int32 sec
	uint32 nanosec""",
}

CUSTOM_INTERFACES: Dict[str, str] = {
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

COMMON_TOPICS_AND_TYPES: Dict[str, str] = {
    "/clock": "rosgraph_msgs/msg/Clock",
    "/parameter_events": "rcl_interfaces/msg/ParameterEvent",
    "/rosout": "rcl_interfaces/msg/Log",
    "/tf": "tf2_msgs/msg/TFMessage",
    "/tf_static": "tf2_msgs/msg/TFMessage",
    "/joint_states": "sensor_msgs/msg/JointState",
    "/robot_description": "std_msgs/msg/String",
    "/robot_description_semantic": "std_msgs/msg/String",
    "/bond": "bond/msg/Status",
    "/diagnostics": "diagnostic_msgs/msg/DiagnosticArray",
    # Perception topics
    "/color_camera_info5": "sensor_msgs/msg/CameraInfo",
    "/color_image5": "sensor_msgs/msg/Image",
    "/depth_camera_info5": "sensor_msgs/msg/CameraInfo",
    "/depth_image5": "sensor_msgs/msg/Image",
    "/pointcloud": "sensor_msgs/msg/PointCloud2",
    "/scan": "sensor_msgs/msg/LaserScan",
}

MANIPULATION_TOPICS_AND_TYPES: Dict[str, str] = {
    # MoveIt2 planning and execution
    "/move_action/_action/feedback": "moveit_msgs/action/MoveGroup_FeedbackMessage",
    "/move_action/_action/status": "action_msgs/msg/GoalStatusArray",
    "/execute_trajectory/_action/feedback": "moveit_msgs/action/ExecuteTrajectory_FeedbackMessage",
    "/execute_trajectory/_action/status": "action_msgs/msg/GoalStatusArray",
    "/motion_plan_request": "moveit_msgs/msg/MotionPlanRequest",
    "/display_planned_path": "moveit_msgs/msg/DisplayTrajectory",
    "/trajectory_execution_event": "std_msgs/msg/String",
    # Planning scene management
    "/planning_scene": "moveit_msgs/msg/PlanningScene",
    "/planning_scene_world": "moveit_msgs/msg/PlanningSceneWorld",
    "/monitored_planning_scene": "moveit_msgs/msg/PlanningScene",
    "/collision_object": "moveit_msgs/msg/CollisionObject",
    "/attached_collision_object": "moveit_msgs/msg/AttachedCollisionObject",
    "/display_contacts": "visualization_msgs/msg/MarkerArray",
    # Arm and gripper controllers
    "/panda_arm_controller/follow_joint_trajectory/_action/feedback": "control_msgs/action/FollowJointTrajectory_FeedbackMessage",
    "/panda_arm_controller/follow_joint_trajectory/_action/status": "action_msgs/msg/GoalStatusArray",
    "/panda_hand_controller/gripper_cmd/_action/feedback": "control_msgs/action/GripperCommand_FeedbackMessage",
    "/panda_hand_controller/gripper_cmd/_action/status": "action_msgs/msg/GoalStatusArray",
}


NAVIGATION_TOPICS_AND_TYPES: Dict[str, str] = {
    # Main navigation actions
    "/navigate_to_pose/_action/feedback": "nav2_msgs/action/NavigateToPose_FeedbackMessage",
    "/navigate_to_pose/_action/status": "action_msgs/msg/GoalStatusArray",
    "/navigate_through_poses/_action/feedback": "nav2_msgs/action/NavigateThroughPoses_FeedbackMessage",
    "/navigate_through_poses/_action/status": "action_msgs/msg/GoalStatusArray",
    "/follow_path/_action/feedback": "nav2_msgs/action/FollowPath_FeedbackMessage",
    "/follow_path/_action/status": "action_msgs/msg/GoalStatusArray",
    "/follow_waypoints/_action/feedback": "nav2_msgs/action/FollowWaypoints_FeedbackMessage",
    "/follow_waypoints/_action/status": "action_msgs/msg/GoalStatusArray",
    # Path planning actions
    "/compute_path_to_pose/_action/feedback": "nav2_msgs/action/ComputePathToPose_FeedbackMessage",
    "/compute_path_to_pose/_action/status": "action_msgs/msg/GoalStatusArray",
    "/compute_path_through_poses/_action/feedback": "nav2_msgs/action/ComputePathThroughPoses_FeedbackMessage",
    "/compute_path_through_poses/_action/status": "action_msgs/msg/GoalStatusArray",
    "/smooth_path/_action/feedback": "nav2_msgs/action/SmoothPath_FeedbackMessage",
    "/smooth_path/_action/status": "action_msgs/msg/GoalStatusArray",
    # Behavior actions
    "/assisted_teleop/_action/feedback": "nav2_msgs/action/AssistedTeleop_FeedbackMessage",
    "/assisted_teleop/_action/status": "action_msgs/msg/GoalStatusArray",
    "/backup/_action/feedback": "nav2_msgs/action/BackUp_FeedbackMessage",
    "/backup/_action/status": "action_msgs/msg/GoalStatusArray",
    "/drive_on_heading/_action/feedback": "nav2_msgs/action/DriveOnHeading_FeedbackMessage",
    "/drive_on_heading/_action/status": "action_msgs/msg/GoalStatusArray",
    "/spin/_action/feedback": "nav2_msgs/action/Spin_FeedbackMessage",
    "/spin/_action/status": "action_msgs/msg/GoalStatusArray",
    "/wait/_action/feedback": "nav2_msgs/action/Wait_FeedbackMessage",
    "/wait/_action/status": "action_msgs/msg/GoalStatusArray",
    # Costmaps and mapping
    "/global_costmap/costmap": "nav_msgs/msg/OccupancyGrid",
    "/global_costmap/costmap_raw": "nav2_msgs/msg/Costmap",
    "/global_costmap/costmap_updates": "map_msgs/msg/OccupancyGridUpdate",
    "/global_costmap/footprint": "geometry_msgs/msg/Polygon",
    "/global_costmap/published_footprint": "geometry_msgs/msg/PolygonStamped",
    "/global_costmap/scan": "sensor_msgs/msg/LaserScan",
    "/local_costmap/costmap": "nav_msgs/msg/OccupancyGrid",
    "/local_costmap/costmap_raw": "nav2_msgs/msg/Costmap",
    "/local_costmap/costmap_updates": "map_msgs/msg/OccupancyGridUpdate",
    "/local_costmap/footprint": "geometry_msgs/msg/Polygon",
    "/local_costmap/published_footprint": "geometry_msgs/msg/PolygonStamped",
    "/local_costmap/scan": "sensor_msgs/msg/LaserScan",
    "/map": "nav_msgs/msg/OccupancyGrid",
    "/map_metadata": "nav_msgs/msg/MapMetaData",
    # SLAM
    "/slam_toolbox/feedback": "visualization_msgs/msg/InteractiveMarkerFeedback",
    "/slam_toolbox/graph_visualization": "visualization_msgs/msg/MarkerArray",
    "/slam_toolbox/scan_visualization": "sensor_msgs/msg/LaserScan",
    "/slam_toolbox/update": "visualization_msgs/msg/InteractiveMarkerUpdate",
    # Path planning and visualization
    "/plan": "nav_msgs/msg/Path",
    "/plan_smoothed": "nav_msgs/msg/Path",
    "/unsmoothed_plan": "nav_msgs/msg/Path",
    "/transformed_global_plan": "nav_msgs/msg/Path",
    "/trajectories": "visualization_msgs/msg/MarkerArray",
    # Control and goals
    "/cmd_vel_nav": "geometry_msgs/msg/Twist",
    "/cmd_vel_teleop": "geometry_msgs/msg/Twist",
    "/goal_pose": "geometry_msgs/msg/PoseStamped",
    "/pose": "geometry_msgs/msg/PoseWithCovarianceStamped",
    "/preempt_teleop": "std_msgs/msg/Empty",
    "/speed_limit": "nav2_msgs/msg/SpeedLimit",
    # Behavior tree
    "/behavior_tree_log": "nav2_msgs/msg/BehaviorTreeLog",
    # Other
    "/led_strip": "sensor_msgs/msg/Image",
    # Lifecycle management
    "/behavior_server/transition_event": "lifecycle_msgs/msg/TransitionEvent",
    "/bt_navigator/transition_event": "lifecycle_msgs/msg/TransitionEvent",
    "/controller_server/transition_event": "lifecycle_msgs/msg/TransitionEvent",
    "/global_costmap/global_costmap/transition_event": "lifecycle_msgs/msg/TransitionEvent",
    "/local_costmap/local_costmap/transition_event": "lifecycle_msgs/msg/TransitionEvent",
    "/map_saver/transition_event": "lifecycle_msgs/msg/TransitionEvent",
    "/planner_server/transition_event": "lifecycle_msgs/msg/TransitionEvent",
    "/smoother_server/transition_event": "lifecycle_msgs/msg/TransitionEvent",
    "/velocity_smoother/transition_event": "lifecycle_msgs/msg/TransitionEvent",
    "/waypoint_follower/transition_event": "lifecycle_msgs/msg/TransitionEvent",
}

CUSTOM_TOPICS_AND_TYPES: Dict[str, str] = {
    "/to_human": "rai_interfaces/msg/HRIMessage",
    "/audio_message": "rai_interfaces/msg/AudioMessage",
    "/detection_array": "rai_interfaces/msg/RAIDetectionArray",
}


COMMON_SERVICES_AND_TYPES: Dict[str, str] = {
    # Core infrastructure
    "/tf2_frames": "tf2_msgs/srv/FrameGraph",
    # Container management
    "/nav2_container/_container/list_nodes": "composition_interfaces/srv/ListNodes",
    "/nav2_container/_container/load_node": "composition_interfaces/srv/LoadNode",
    "/nav2_container/_container/unload_node": "composition_interfaces/srv/UnloadNode",
    # Robot state and transforms
    "/robot_state_publisher/describe_parameters": "rcl_interfaces/srv/DescribeParameters",
    "/robot_state_publisher/get_parameter_types": "rcl_interfaces/srv/GetParameterTypes",
    "/robot_state_publisher/get_parameters": "rcl_interfaces/srv/GetParameters",
    "/robot_state_publisher/list_parameters": "rcl_interfaces/srv/ListParameters",
    "/robot_state_publisher/set_parameters": "rcl_interfaces/srv/SetParameters",
    "/robot_state_publisher/set_parameters_atomically": "rcl_interfaces/srv/SetParametersAtomically",
    "/static_transform_publisher/describe_parameters": "rcl_interfaces/srv/DescribeParameters",
    "/static_transform_publisher/get_parameter_types": "rcl_interfaces/srv/GetParameterTypes",
    "/static_transform_publisher/get_parameters": "rcl_interfaces/srv/GetParameters",
    "/static_transform_publisher/list_parameters": "rcl_interfaces/srv/ListParameters",
    "/static_transform_publisher/set_parameters": "rcl_interfaces/srv/SetParameters",
    "/static_transform_publisher/set_parameters_atomically": "rcl_interfaces/srv/SetParametersAtomically",
    # Simulation/Gazebo services
    "/delete_entity": "gazebo_msgs/srv/DeleteEntity",
    "/get_available_spawnable_names": "gazebo_msgs/srv/GetWorldProperties",
    "/get_spawn_point_info": "gazebo_msgs/srv/GetModelState",
    "/get_spawn_points_names": "gazebo_msgs/srv/GetWorldProperties",
    "/spawn_entity": "gazebo_msgs/srv/SpawnEntity",
    # Parameter services (common pattern for all nodes)
    "/launch_ros_138640/describe_parameters": "rcl_interfaces/srv/DescribeParameters",
    "/launch_ros_138640/get_parameter_types": "rcl_interfaces/srv/GetParameterTypes",
    "/launch_ros_138640/get_parameters": "rcl_interfaces/srv/GetParameters",
    "/launch_ros_138640/list_parameters": "rcl_interfaces/srv/ListParameters",
    "/launch_ros_138640/set_parameters": "rcl_interfaces/srv/SetParameters",
    "/launch_ros_138640/set_parameters_atomically": "rcl_interfaces/srv/SetParametersAtomically",
    "/launch_ros_2375507/describe_parameters": "rcl_interfaces/srv/DescribeParameters",
    "/launch_ros_2375507/get_parameter_types": "rcl_interfaces/srv/GetParameterTypes",
    "/launch_ros_2375507/get_parameters": "rcl_interfaces/srv/GetParameters",
    "/launch_ros_2375507/list_parameters": "rcl_interfaces/srv/ListParameters",
    "/launch_ros_2375507/set_parameters": "rcl_interfaces/srv/SetParameters",
    "/launch_ros_2375507/set_parameters_atomically": "rcl_interfaces/srv/SetParametersAtomically",
    "/o3de_ros2_node/describe_parameters": "rcl_interfaces/srv/DescribeParameters",
    "/o3de_ros2_node/get_parameter_types": "rcl_interfaces/srv/GetParameterTypes",
    "/o3de_ros2_node/get_parameters": "rcl_interfaces/srv/GetParameters",
    "/o3de_ros2_node/list_parameters": "rcl_interfaces/srv/ListParameters",
    "/o3de_ros2_node/set_parameters": "rcl_interfaces/srv/SetParameters",
    "/o3de_ros2_node/set_parameters_atomically": "rcl_interfaces/srv/SetParametersAtomically",
    # AI/ML services (custom interfaces for perception and documentation)
    "/grounded_sam/describe_parameters": "rcl_interfaces/srv/DescribeParameters",
    "/grounded_sam/get_parameter_types": "rcl_interfaces/srv/GetParameterTypes",
    "/grounded_sam/get_parameters": "rcl_interfaces/srv/GetParameters",
    "/grounded_sam/list_parameters": "rcl_interfaces/srv/ListParameters",
    "/grounded_sam/set_parameters": "rcl_interfaces/srv/SetParameters",
    "/grounded_sam/set_parameters_atomically": "rcl_interfaces/srv/SetParametersAtomically",
    "/grounding_dino/describe_parameters": "rcl_interfaces/srv/DescribeParameters",
    "/grounding_dino/get_parameter_types": "rcl_interfaces/srv/GetParameterTypes",
    "/grounding_dino/get_parameters": "rcl_interfaces/srv/GetParameters",
    "/grounding_dino/list_parameters": "rcl_interfaces/srv/ListParameters",
    "/grounding_dino/set_parameters": "rcl_interfaces/srv/SetParameters",
    "/grounding_dino/set_parameters_atomically": "rcl_interfaces/srv/SetParametersAtomically",
    "/rai_ros2_ari_connector_b6ed00ab6356/describe_parameters": "rcl_interfaces/srv/DescribeParameters",
    "/rai_ros2_ari_connector_b6ed00ab6356/get_parameter_types": "rcl_interfaces/srv/GetParameterTypes",
    "/rai_ros2_ari_connector_b6ed00ab6356/get_parameters": "rcl_interfaces/srv/GetParameters",
    "/rai_ros2_ari_connector_b6ed00ab6356/list_parameters": "rcl_interfaces/srv/ListParameters",
    "/rai_ros2_ari_connector_b6ed00ab6356/set_parameters": "rcl_interfaces/srv/SetParameters",
    "/rai_ros2_ari_connector_b6ed00ab6356/set_parameters_atomically": "rcl_interfaces/srv/SetParametersAtomically",
}

MANIPULATION_SERVICES_AND_TYPES: Dict[str, str] = {
    # MoveIt2 planning services
    "/apply_planning_scene": "moveit_msgs/srv/ApplyPlanningScene",
    "/check_state_validity": "moveit_msgs/srv/GetStateValidity",
    "/clear_octomap": "std_srvs/srv/Empty",
    "/compute_cartesian_path": "moveit_msgs/srv/GetCartesianPath",
    "/compute_fk": "moveit_msgs/srv/GetPositionFK",
    "/compute_ik": "moveit_msgs/srv/GetPositionIK",
    "/get_planner_params": "moveit_msgs/srv/GetPlannerParams",
    "/get_planning_scene": "moveit_msgs/srv/GetPlanningScene",
    "/load_map": "moveit_msgs/srv/LoadMap",
    "/plan_kinematic_path": "moveit_msgs/srv/GetMotionPlan",
    "/query_planner_interface": "moveit_msgs/srv/QueryPlannerInterfaces",
    "/save_map": "moveit_msgs/srv/SaveMap",
    "/set_planner_params": "moveit_msgs/srv/SetPlannerParams",
    # Custom manipulation interfaces
    "/reset_manipulator": "std_srvs/srv/Trigger",
    # MoveIt2 component parameter services
    "/move_group/describe_parameters": "rcl_interfaces/srv/DescribeParameters",
    "/move_group/get_parameter_types": "rcl_interfaces/srv/GetParameterTypes",
    "/move_group/get_parameters": "rcl_interfaces/srv/GetParameters",
    "/move_group/list_parameters": "rcl_interfaces/srv/ListParameters",
    "/move_group/set_parameters": "rcl_interfaces/srv/SetParameters",
    "/move_group/set_parameters_atomically": "rcl_interfaces/srv/SetParametersAtomically",
    "/move_group_private_96220314512624/describe_parameters": "rcl_interfaces/srv/DescribeParameters",
    "/move_group_private_96220314512624/get_parameter_types": "rcl_interfaces/srv/GetParameterTypes",
    "/move_group_private_96220314512624/get_parameters": "rcl_interfaces/srv/GetParameters",
    "/move_group_private_96220314512624/list_parameters": "rcl_interfaces/srv/ListParameters",
    "/move_group_private_96220314512624/set_parameters": "rcl_interfaces/srv/SetParameters",
    "/move_group_private_96220314512624/set_parameters_atomically": "rcl_interfaces/srv/SetParametersAtomically",
    "/moveit_simple_controller_manager/describe_parameters": "rcl_interfaces/srv/DescribeParameters",
    "/moveit_simple_controller_manager/get_parameter_types": "rcl_interfaces/srv/GetParameterTypes",
    "/moveit_simple_controller_manager/get_parameters": "rcl_interfaces/srv/GetParameters",
    "/moveit_simple_controller_manager/list_parameters": "rcl_interfaces/srv/ListParameters",
    "/moveit_simple_controller_manager/set_parameters": "rcl_interfaces/srv/SetParameters",
    "/moveit_simple_controller_manager/set_parameters_atomically": "rcl_interfaces/srv/SetParametersAtomically",
    # Arm controller services
    "/arm_controller/describe_parameters": "rcl_interfaces/srv/DescribeParameters",
    "/arm_controller/get_parameter_types": "rcl_interfaces/srv/GetParameterTypes",
    "/arm_controller/get_parameters": "rcl_interfaces/srv/GetParameters",
    "/arm_controller/list_parameters": "rcl_interfaces/srv/ListParameters",
    "/arm_controller/set_parameters": "rcl_interfaces/srv/SetParameters",
    "/arm_controller/set_parameters_atomically": "rcl_interfaces/srv/SetParametersAtomically",
    "/state_controller/describe_parameters": "rcl_interfaces/srv/DescribeParameters",
    "/state_controller/get_parameter_types": "rcl_interfaces/srv/GetParameterTypes",
    "/state_controller/get_parameters": "rcl_interfaces/srv/GetParameters",
    "/state_controller/list_parameters": "rcl_interfaces/srv/ListParameters",
    "/state_controller/set_parameters": "rcl_interfaces/srv/SetParameters",
    "/state_controller/set_parameters_atomically": "rcl_interfaces/srv/SetParametersAtomically",
}

NAVIGATION_SERVICES_AND_TYPES: Dict[str, str] = {
    # Action services for navigation behaviors
    "/assisted_teleop/_action/cancel_goal": "action_msgs/srv/CancelGoal",
    "/assisted_teleop/_action/get_result": "nav2_msgs/action/AssistedTeleop_GetResult",
    "/assisted_teleop/_action/send_goal": "nav2_msgs/action/AssistedTeleop_SendGoal",
    "/backup/_action/cancel_goal": "action_msgs/srv/CancelGoal",
    "/backup/_action/get_result": "nav2_msgs/action/BackUp_GetResult",
    "/backup/_action/send_goal": "nav2_msgs/action/BackUp_SendGoal",
    "/drive_on_heading/_action/cancel_goal": "action_msgs/srv/CancelGoal",
    "/drive_on_heading/_action/get_result": "nav2_msgs/action/DriveOnHeading_GetResult",
    "/drive_on_heading/_action/send_goal": "nav2_msgs/action/DriveOnHeading_SendGoal",
    "/follow_path/_action/cancel_goal": "action_msgs/srv/CancelGoal",
    "/follow_path/_action/get_result": "nav2_msgs/action/FollowPath_GetResult",
    "/follow_path/_action/send_goal": "nav2_msgs/action/FollowPath_SendGoal",
    "/follow_waypoints/_action/cancel_goal": "action_msgs/srv/CancelGoal",
    "/follow_waypoints/_action/get_result": "nav2_msgs/action/FollowWaypoints_GetResult",
    "/follow_waypoints/_action/send_goal": "nav2_msgs/action/FollowWaypoints_SendGoal",
    "/spin/_action/cancel_goal": "action_msgs/srv/CancelGoal",
    "/spin/_action/get_result": "nav2_msgs/action/Spin_GetResult",
    "/spin/_action/send_goal": "nav2_msgs/action/Spin_SendGoal",
    "/wait/_action/cancel_goal": "action_msgs/srv/CancelGoal",
    "/wait/_action/get_result": "nav2_msgs/action/Wait_GetResult",
    "/wait/_action/send_goal": "nav2_msgs/action/Wait_SendGoal",
    # Path planning action services
    "/compute_path_through_poses/_action/cancel_goal": "action_msgs/srv/CancelGoal",
    "/compute_path_through_poses/_action/get_result": "nav2_msgs/action/ComputePathThroughPoses_GetResult",
    "/compute_path_through_poses/_action/send_goal": "nav2_msgs/action/ComputePathThroughPoses_SendGoal",
    "/compute_path_to_pose/_action/cancel_goal": "action_msgs/srv/CancelGoal",
    "/compute_path_to_pose/_action/get_result": "nav2_msgs/action/ComputePathToPose_GetResult",
    "/compute_path_to_pose/_action/send_goal": "nav2_msgs/action/ComputePathToPose_SendGoal",
    "/smooth_path/_action/cancel_goal": "action_msgs/srv/CancelGoal",
    "/smooth_path/_action/get_result": "nav2_msgs/action/SmoothPath_GetResult",
    "/smooth_path/_action/send_goal": "nav2_msgs/action/SmoothPath_SendGoal",
    # Main navigation action services
    "/navigate_through_poses/_action/cancel_goal": "action_msgs/srv/CancelGoal",
    "/navigate_through_poses/_action/get_result": "nav2_msgs/action/NavigateThroughPoses_GetResult",
    "/navigate_through_poses/_action/send_goal": "nav2_msgs/action/NavigateThroughPoses_SendGoal",
    "/navigate_to_pose/_action/cancel_goal": "action_msgs/srv/CancelGoal",
    "/navigate_to_pose/_action/get_result": "nav2_msgs/action/NavigateToPose_GetResult",
    "/navigate_to_pose/_action/send_goal": "nav2_msgs/action/NavigateToPose_SendGoal",
    # Costmap management services
    "/global_costmap/clear_around_global_costmap": "nav2_msgs/srv/ClearCostmapAroundRobot",
    "/global_costmap/clear_entirely_global_costmap": "nav2_msgs/srv/ClearEntireCostmap",
    "/global_costmap/clear_except_global_costmap": "nav2_msgs/srv/ClearCostmapExceptRegion",
    "/global_costmap/get_costmap": "nav2_msgs/srv/GetCostmap",
    "/local_costmap/clear_around_local_costmap": "nav2_msgs/srv/ClearCostmapAroundRobot",
    "/local_costmap/clear_entirely_local_costmap": "nav2_msgs/srv/ClearEntireCostmap",
    "/local_costmap/clear_except_local_costmap": "nav2_msgs/srv/ClearCostmapExceptRegion",
    "/local_costmap/get_costmap": "nav2_msgs/srv/GetCostmap",
    # Path validation
    "/is_path_valid": "nav2_msgs/srv/IsPathValid",
    # SLAM services
    "/slam_toolbox/clear_changes": "slam_toolbox/srv/Clear",
    "/slam_toolbox/clear_queue": "slam_toolbox/srv/ClearQueue",
    "/slam_toolbox/describe_parameters": "rcl_interfaces/srv/DescribeParameters",
    "/slam_toolbox/deserialize_map": "slam_toolbox/srv/DeserializePoseGraph",
    "/slam_toolbox/dynamic_map": "nav_msgs/srv/GetMap",
    "/slam_toolbox/get_interactive_markers": "visualization_msgs/srv/GetInteractiveMarkers",
    "/slam_toolbox/get_parameter_types": "rcl_interfaces/srv/GetParameterTypes",
    "/slam_toolbox/get_parameters": "rcl_interfaces/srv/GetParameters",
    "/slam_toolbox/list_parameters": "rcl_interfaces/srv/ListParameters",
    "/slam_toolbox/manual_loop_closure": "slam_toolbox/srv/LoopClosure",
    "/slam_toolbox/pause_new_measurements": "slam_toolbox/srv/Pause",
    "/slam_toolbox/save_map": "slam_toolbox/srv/SaveMap",
    "/slam_toolbox/serialize_map": "slam_toolbox/srv/SerializePoseGraph",
    "/slam_toolbox/set_parameters": "rcl_interfaces/srv/SetParameters",
    "/slam_toolbox/set_parameters_atomically": "rcl_interfaces/srv/SetParametersAtomically",
    "/slam_toolbox/toggle_interactive_mode": "slam_toolbox/srv/ToggleInteractive",
    # Map saving
    "/map_saver/change_state": "lifecycle_msgs/srv/ChangeState",
    "/map_saver/describe_parameters": "rcl_interfaces/srv/DescribeParameters",
    "/map_saver/get_available_states": "lifecycle_msgs/srv/GetAvailableStates",
    "/map_saver/get_available_transitions": "lifecycle_msgs/srv/GetAvailableTransitions",
    "/map_saver/get_parameter_types": "rcl_interfaces/srv/GetParameterTypes",
    "/map_saver/get_parameters": "rcl_interfaces/srv/GetParameters",
    "/map_saver/get_state": "lifecycle_msgs/srv/GetState",
    "/map_saver/get_transition_graph": "lifecycle_msgs/srv/GetAvailableTransitions",
    "/map_saver/list_parameters": "rcl_interfaces/srv/ListParameters",
    "/map_saver/save_map": "nav2_msgs/srv/SaveMap",
    "/map_saver/set_parameters": "rcl_interfaces/srv/SetParameters",
    "/map_saver/set_parameters_atomically": "rcl_interfaces/srv/SetParametersAtomically",
    # Navigation server lifecycle and parameter services
    "/behavior_server/change_state": "lifecycle_msgs/srv/ChangeState",
    "/behavior_server/describe_parameters": "rcl_interfaces/srv/DescribeParameters",
    "/behavior_server/get_available_states": "lifecycle_msgs/srv/GetAvailableStates",
    "/behavior_server/get_available_transitions": "lifecycle_msgs/srv/GetAvailableTransitions",
    "/behavior_server/get_parameter_types": "rcl_interfaces/srv/GetParameterTypes",
    "/behavior_server/get_parameters": "rcl_interfaces/srv/GetParameters",
    "/behavior_server/get_state": "lifecycle_msgs/srv/GetState",
    "/behavior_server/get_transition_graph": "lifecycle_msgs/srv/GetAvailableTransitions",
    "/behavior_server/list_parameters": "rcl_interfaces/srv/ListParameters",
    "/behavior_server/set_parameters": "rcl_interfaces/srv/SetParameters",
    "/behavior_server/set_parameters_atomically": "rcl_interfaces/srv/SetParametersAtomically",
    "/bt_navigator/change_state": "lifecycle_msgs/srv/ChangeState",
    "/bt_navigator/describe_parameters": "rcl_interfaces/srv/DescribeParameters",
    "/bt_navigator/get_available_states": "lifecycle_msgs/srv/GetAvailableStates",
    "/bt_navigator/get_available_transitions": "lifecycle_msgs/srv/GetAvailableTransitions",
    "/bt_navigator/get_parameter_types": "rcl_interfaces/srv/GetParameterTypes",
    "/bt_navigator/get_parameters": "rcl_interfaces/srv/GetParameters",
    "/bt_navigator/get_state": "lifecycle_msgs/srv/GetState",
    "/bt_navigator/get_transition_graph": "lifecycle_msgs/srv/GetAvailableTransitions",
    "/bt_navigator/list_parameters": "rcl_interfaces/srv/ListParameters",
    "/bt_navigator/set_parameters": "rcl_interfaces/srv/SetParameters",
    "/bt_navigator/set_parameters_atomically": "rcl_interfaces/srv/SetParametersAtomically",
    "/bt_navigator_navigate_through_poses_rclcpp_node/describe_parameters": "rcl_interfaces/srv/DescribeParameters",
    "/bt_navigator_navigate_through_poses_rclcpp_node/get_parameter_types": "rcl_interfaces/srv/GetParameterTypes",
    "/bt_navigator_navigate_through_poses_rclcpp_node/get_parameters": "rcl_interfaces/srv/GetParameters",
    "/bt_navigator_navigate_through_poses_rclcpp_node/list_parameters": "rcl_interfaces/srv/ListParameters",
    "/bt_navigator_navigate_through_poses_rclcpp_node/set_parameters": "rcl_interfaces/srv/SetParameters",
    "/bt_navigator_navigate_through_poses_rclcpp_node/set_parameters_atomically": "rcl_interfaces/srv/SetParametersAtomically",
    "/bt_navigator_navigate_to_pose_rclcpp_node/describe_parameters": "rcl_interfaces/srv/DescribeParameters",
    "/bt_navigator_navigate_to_pose_rclcpp_node/get_parameter_types": "rcl_interfaces/srv/GetParameterTypes",
    "/bt_navigator_navigate_to_pose_rclcpp_node/get_parameters": "rcl_interfaces/srv/GetParameters",
    "/bt_navigator_navigate_to_pose_rclcpp_node/list_parameters": "rcl_interfaces/srv/ListParameters",
    "/bt_navigator_navigate_to_pose_rclcpp_node/set_parameters": "rcl_interfaces/srv/SetParameters",
    "/bt_navigator_navigate_to_pose_rclcpp_node/set_parameters_atomically": "rcl_interfaces/srv/SetParametersAtomically",
    "/controller_server/change_state": "lifecycle_msgs/srv/ChangeState",
    "/controller_server/describe_parameters": "rcl_interfaces/srv/DescribeParameters",
    "/controller_server/get_available_states": "lifecycle_msgs/srv/GetAvailableStates",
    "/controller_server/get_available_transitions": "lifecycle_msgs/srv/GetAvailableTransitions",
    "/controller_server/get_parameter_types": "rcl_interfaces/srv/GetParameterTypes",
    "/controller_server/get_parameters": "rcl_interfaces/srv/GetParameters",
    "/controller_server/get_state": "lifecycle_msgs/srv/GetState",
    "/controller_server/get_transition_graph": "lifecycle_msgs/srv/GetAvailableTransitions",
    "/controller_server/list_parameters": "rcl_interfaces/srv/ListParameters",
    "/controller_server/set_parameters": "rcl_interfaces/srv/SetParameters",
    "/controller_server/set_parameters_atomically": "rcl_interfaces/srv/SetParametersAtomically",
    "/global_costmap/global_costmap/change_state": "lifecycle_msgs/srv/ChangeState",
    "/global_costmap/global_costmap/describe_parameters": "rcl_interfaces/srv/DescribeParameters",
    "/global_costmap/global_costmap/get_available_states": "lifecycle_msgs/srv/GetAvailableStates",
    "/global_costmap/global_costmap/get_available_transitions": "lifecycle_msgs/srv/GetAvailableTransitions",
    "/global_costmap/global_costmap/get_parameter_types": "rcl_interfaces/srv/GetParameterTypes",
    "/global_costmap/global_costmap/get_parameters": "rcl_interfaces/srv/GetParameters",
    "/global_costmap/global_costmap/get_state": "lifecycle_msgs/srv/GetState",
    "/global_costmap/global_costmap/get_transition_graph": "lifecycle_msgs/srv/GetAvailableTransitions",
    "/global_costmap/global_costmap/list_parameters": "rcl_interfaces/srv/ListParameters",
    "/global_costmap/global_costmap/set_parameters": "rcl_interfaces/srv/SetParameters",
    "/global_costmap/global_costmap/set_parameters_atomically": "rcl_interfaces/srv/SetParametersAtomically",
    "/local_costmap/local_costmap/change_state": "lifecycle_msgs/srv/ChangeState",
    "/local_costmap/local_costmap/describe_parameters": "rcl_interfaces/srv/DescribeParameters",
    "/local_costmap/local_costmap/get_available_states": "lifecycle_msgs/srv/GetAvailableStates",
    "/local_costmap/local_costmap/get_available_transitions": "lifecycle_msgs/srv/GetAvailableTransitions",
    "/local_costmap/local_costmap/get_parameter_types": "rcl_interfaces/srv/GetParameterTypes",
    "/local_costmap/local_costmap/get_parameters": "rcl_interfaces/srv/GetParameters",
    "/local_costmap/local_costmap/get_state": "lifecycle_msgs/srv/GetState",
    "/local_costmap/local_costmap/get_transition_graph": "lifecycle_msgs/srv/GetAvailableTransitions",
    "/local_costmap/local_costmap/list_parameters": "rcl_interfaces/srv/ListParameters",
    "/local_costmap/local_costmap/set_parameters": "rcl_interfaces/srv/SetParameters",
    "/local_costmap/local_costmap/set_parameters_atomically": "rcl_interfaces/srv/SetParametersAtomically",
    "/planner_server/change_state": "lifecycle_msgs/srv/ChangeState",
    "/planner_server/describe_parameters": "rcl_interfaces/srv/DescribeParameters",
    "/planner_server/get_available_states": "lifecycle_msgs/srv/GetAvailableStates",
    "/planner_server/get_available_transitions": "lifecycle_msgs/srv/GetAvailableTransitions",
    "/planner_server/get_parameter_types": "rcl_interfaces/srv/GetParameterTypes",
    "/planner_server/get_parameters": "rcl_interfaces/srv/GetParameters",
    "/planner_server/get_state": "lifecycle_msgs/srv/GetState",
    "/planner_server/get_transition_graph": "lifecycle_msgs/srv/GetAvailableTransitions",
    "/planner_server/list_parameters": "rcl_interfaces/srv/ListParameters",
    "/planner_server/set_parameters": "rcl_interfaces/srv/SetParameters",
    "/planner_server/set_parameters_atomically": "rcl_interfaces/srv/SetParametersAtomically",
    "/smoother_server/change_state": "lifecycle_msgs/srv/ChangeState",
    "/smoother_server/describe_parameters": "rcl_interfaces/srv/DescribeParameters",
    "/smoother_server/get_available_states": "lifecycle_msgs/srv/GetAvailableStates",
    "/smoother_server/get_available_transitions": "lifecycle_msgs/srv/GetAvailableTransitions",
    "/smoother_server/get_parameter_types": "rcl_interfaces/srv/GetParameterTypes",
    "/smoother_server/get_parameters": "rcl_interfaces/srv/GetParameters",
    "/smoother_server/get_state": "lifecycle_msgs/srv/GetState",
    "/smoother_server/get_transition_graph": "lifecycle_msgs/srv/GetAvailableTransitions",
    "/smoother_server/list_parameters": "rcl_interfaces/srv/ListParameters",
    "/smoother_server/set_parameters": "rcl_interfaces/srv/SetParameters",
    "/smoother_server/set_parameters_atomically": "rcl_interfaces/srv/SetParametersAtomically",
    "/velocity_smoother/change_state": "lifecycle_msgs/srv/ChangeState",
    "/velocity_smoother/describe_parameters": "rcl_interfaces/srv/DescribeParameters",
    "/velocity_smoother/get_available_states": "lifecycle_msgs/srv/GetAvailableStates",
    "/velocity_smoother/get_available_transitions": "lifecycle_msgs/srv/GetAvailableTransitions",
    "/velocity_smoother/get_parameter_types": "rcl_interfaces/srv/GetParameterTypes",
    "/velocity_smoother/get_parameters": "rcl_interfaces/srv/GetParameters",
    "/velocity_smoother/get_state": "lifecycle_msgs/srv/GetState",
    "/velocity_smoother/get_transition_graph": "lifecycle_msgs/srv/GetAvailableTransitions",
    "/velocity_smoother/list_parameters": "rcl_interfaces/srv/ListParameters",
    "/velocity_smoother/set_parameters": "rcl_interfaces/srv/SetParameters",
    "/velocity_smoother/set_parameters_atomically": "rcl_interfaces/srv/SetParametersAtomically",
    "/waypoint_follower/change_state": "lifecycle_msgs/srv/ChangeState",
    "/waypoint_follower/describe_parameters": "rcl_interfaces/srv/DescribeParameters",
    "/waypoint_follower/get_available_states": "lifecycle_msgs/srv/GetAvailableStates",
    "/waypoint_follower/get_available_transitions": "lifecycle_msgs/srv/GetAvailableTransitions",
    "/waypoint_follower/get_parameter_types": "rcl_interfaces/srv/GetParameterTypes",
    "/waypoint_follower/get_parameters": "rcl_interfaces/srv/GetParameters",
    "/waypoint_follower/get_state": "lifecycle_msgs/srv/GetState",
    "/waypoint_follower/get_transition_graph": "lifecycle_msgs/srv/GetAvailableTransitions",
    "/waypoint_follower/list_parameters": "rcl_interfaces/srv/ListParameters",
    "/waypoint_follower/set_parameters": "rcl_interfaces/srv/SetParameters",
    "/waypoint_follower/set_parameters_atomically": "rcl_interfaces/srv/SetParametersAtomically",
    # Lifecycle management services
    "/lifecycle_manager_navigation/describe_parameters": "rcl_interfaces/srv/DescribeParameters",
    "/lifecycle_manager_navigation/get_parameter_types": "rcl_interfaces/srv/GetParameterTypes",
    "/lifecycle_manager_navigation/get_parameters": "rcl_interfaces/srv/GetParameters",
    "/lifecycle_manager_navigation/is_active": "std_srvs/srv/Trigger",
    "/lifecycle_manager_navigation/list_parameters": "rcl_interfaces/srv/ListParameters",
    "/lifecycle_manager_navigation/manage_nodes": "nav2_msgs/srv/ManageLifecycleNodes",
    "/lifecycle_manager_navigation/set_parameters": "rcl_interfaces/srv/SetParameters",
    "/lifecycle_manager_navigation/set_parameters_atomically": "rcl_interfaces/srv/SetParametersAtomically",
    "/lifecycle_manager_slam/describe_parameters": "rcl_interfaces/srv/DescribeParameters",
    "/lifecycle_manager_slam/get_parameter_types": "rcl_interfaces/srv/GetParameterTypes",
    "/lifecycle_manager_slam/get_parameters": "rcl_interfaces/srv/GetParameters",
    "/lifecycle_manager_slam/is_active": "std_srvs/srv/Trigger",
    "/lifecycle_manager_slam/list_parameters": "rcl_interfaces/srv/ListParameters",
    "/lifecycle_manager_slam/manage_nodes": "nav2_msgs/srv/ManageLifecycleNodes",
    "/lifecycle_manager_slam/set_parameters": "rcl_interfaces/srv/SetParameters",
    "/lifecycle_manager_slam/set_parameters_atomically": "rcl_interfaces/srv/SetParametersAtomically",
}
CUSTOM_SERVICES_AND_TYPES: Dict[str, str] = {
    "/grounded_sam_segment": "rai_interfaces/srv/RAIGroundedSam",
    "/grounding_dino_classify": "rai_interfaces/srv/RAIGroundingDino",
    "/manipulator_move_to": "rai_interfaces/srv/ManipulatorMoveTo",
    "/get_log_digest": "rai_interfaces/srv/StringList",
    "/rai_whoami_documentation_service": "rai_interfaces/srv/VectorStoreRetrieval",
    "/rai_whatisee_get": "rai_interfaces/srv/WhatISee",
}

MANIPULATION_ACTIONS_AND_TYPES: Dict[str, str] = {
    "/move_action": "moveit_msgs/action/MoveGroup",
    "/execute_trajectory": "moveit_msgs/action/ExecuteTrajectory",
    "/panda_arm_controller/follow_joint_trajectory": "control_msgs/action/FollowJointTrajectory",
    "/arm_controller/follow_joint_trajectory": "control_msgs/action/FollowJointTrajectory",
    "/panda_hand_controller/gripper_cmd": "control_msgs/action/GripperCommand",
    "/gripper_controller/gripper_cmd": "control_msgs/action/GripperCommand",
    "/pickup": "moveit_msgs/action/Pickup",
    "/place": "moveit_msgs/action/Place",
}
NAVIGATION_ACTIONS_AND_TYPES: Dict[str, str] = {
    "/navigate_to_pose": "nav2_msgs/action/NavigateToPose",
    "/navigate_through_poses": "nav2_msgs/action/Nmoveit_msgs/action/MoveGroupmoveit_msgs/action/MoveGroupavigateThroughPoses",
    "/follow_path": "nav2_msgs/action/FollowPath",
    "/follow_waypoints": "nav2_msgs/action/FollowWaypoints",
    "/compute_path_to_pose": "nav2_msgs/action/ComputePathToPose",
    "/compute_path_through_poses": "nav2_msgs/action/ComputePathThroughPoses",
    "/smooth_path": "nav2_msgs/action/SmoothPath",
    "/spin": "nav2_msgs/action/Spin",
    "/backup": "nav2_msgs/action/BackUp",
    "/drive_on_heading": "nav2_msgs/action/DriveOnHeading",
    "/wait": "nav2_msgs/action/Wait",
    "/assisted_teleop": "nav2_msgs/action/AssistedTeleop",
    "/clear_costmap": "nav2_msgs/action/ClearEntireCostmap",
}
COMMON_TOPIC_MODELS: Dict[str, Type[BaseModel]] = {
    "sensor_msgs/msg/CameraInfo": CameraInfo,
    "sensor_msgs/msg/Image": Image,
    "rosgraph_msgs/msg/Clock": Clock,
}

CUSTOM_TOPIC_MODELS: Dict[str, Type[BaseModel]] = {
    "rai_interfaces/msg/HRIMessage": HRIMessage,
    "rai_interfaces/msg/AudioMessage": AudioMessage,
    "rai_interfaces/msg/RAIDetectionArray": RAIDetectionArray,
}

CUSTOM_SERVICE_MODELS: Dict[str, Type[BaseModel]] = {
    "rai_interfaces/srv/ManipulatorMoveTo": ManipulatorMoveToRequest,
    "rai_interfaces/srv/RAIGroundedSam": RAIGroundedSamRequest,
    "rai_interfaces/srv/RAIGroundingDino": RAIGroundingDinoRequest,
    "rai_interfaces/srv/StringList": StringListRequest,
    "rai_interfaces/srv/VectorStoreRetrieval": VectorStoreRetrievalRequest,
    "rai_interfaces/srv/WhatISee": WhatISeeRequest,
}
MANIPULATION_ACTION_MODELS: Dict[str, Type[BaseModel]] = {}
NAVIGATION_ACTION_MODELS: Dict[str, Type[BaseModel]] = {
    "nav2_msgs/action/NavigateToPose": NavigateToPoseGoal,
    "nav2_msgs/action/Spin": SpinGoal,
    "nav2_msgs/action/AssistedTeleop": AssistedTeleopGoal,
    "nav2_msgs/action/BackUp": BackUpGoal,
    "nav2_msgs/action/ComputePathThroughPoses": ComputePathThroughPosesGoal,
    "nav2_msgs/action/ComputePathToPose": ComputePathToPoseGoal,
    "nav2_msgs/action/DriveOnHeading": DriveOnHeadingGoal,
    "nav2_msgs/action/FollowPath": FollowPathGoal,
    "nav2_msgs/action/FollowWaypoints": FollowWaypointsGoal,
    "nav2_msgs/action/NavigateThroughPoses": NavigateThroughPosesGoal,
    "nav2_msgs/action/SmoothPath": SmoothPathGoal,
    "nav2_msgs/action/Wait": WaitGoal,
}

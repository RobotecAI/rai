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
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Literal, Sequence, Type

from langchain_core.messages import AIMessage, ToolCall
from langchain_core.runnables.config import DEFAULT_RECURSION_LIMIT
from langchain_core.tools import BaseTool
from pydantic import BaseModel

from rai_bench.tool_calling_agent_bench.messages.base import Clock
from rai_bench.tool_calling_agent_bench.messages.services import (
    ManipulatorMoveToRequest,
    RAIGroundedSamRequest,
    RAIGroundingDinoRequest,
    StringListRequest,
    VectorStoreRetrievalRequest,
    WhatISeeRequest,
)
from rai_bench.tool_calling_agent_bench.messages.topics import (
    AudioMessage,
    CameraInfo,
    HRIMessage,
    Image,
    RAIDetectionArray,
)
from rai_bench.tool_calling_agent_bench.mocked_tools import (
    MockCallROS2ServiceTool,
    MockCancelROS2ActionTool,
    MockGetROS2ActionIDsTool,
    MockGetROS2ActionsNamesAndTypesTool,
    MockGetROS2ImageTool,
    MockGetROS2MessageInterfaceTool,
    MockGetROS2ServicesNamesAndTypesTool,
    MockGetROS2TopicsNamesAndTypesTool,
    MockMoveToPointTool,
    MockPublishROS2MessageTool,
    MockStartROS2ActionTool,
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


class Result(BaseModel):
    success: bool = False
    errors: list[str] = []


class ToolCallingAgentTask(ABC):
    """Abstract class for tool calling agent tasks. Contains methods for requested tool calls verification.

    Parameters
    ----------
    logger : loggers_type | None, optional
        Logger, by default None
    """

    complexity: Literal["easy", "medium", "hard"]
    recursion_limit: int = DEFAULT_RECURSION_LIMIT

    def __init__(
        self,
        logger: loggers_type | None = None,
    ) -> None:
        if logger:
            self.logger = logger
        else:
            self.logger = logging.getLogger(__name__)
        self.expected_tools: List[BaseTool] = []
        self.result = Result()

    @abstractmethod
    def get_system_prompt(self) -> str:
        """Get the system prompt that will be passed to agent

        Returns
        -------
        str
            System prompt
        """
        pass

    @abstractmethod
    def get_prompt(self) -> str:
        """Get the task instruction - the prompt that will be passed to agent.

        Returns
        -------
        str
            Prompt
        """
        pass

    @abstractmethod
    def verify_tool_calls(self, response: dict[str, Any]):
        """Verify correctness of the tool calls from the agent's response.

        Note
        ----
        This method should set self.result.success to True if the verification is successful and append occuring errors related to verification to self.result.errors.

        Parameters
        ----------
        response : dict[str, Any]
            Agent's response
        """
        pass

    def _check_topic_tool_call_field(
        self,
        tool_call: ToolCall,
        expected_name: str,
        expected_topic: str,
        expected_message_type: str,
        field_path: str,
        expected_value: Any,
    ) -> bool:
        """
        Verifies a tool call for a topic publishing operation.

        Parameters
        ----------
        tool_call : ToolCall
            The tool call dictionary containing keys such as "name" and "args".
        expected_name : str
            The expected tool call name (e.g., "publish_ros2_message").
        expected_topic : str
            The expected topic name in the tool call's arguments.
        expected_message_type : str
            The expected message type (e.g., "rai_interfaces/msg/HRIMessage").
        field_path : str
            Dot-separated path to the field inside the message (e.g., "header.frame_id").
        expected_value : Any
            The expected value at the given field path.

        Returns
        -------
        bool
            True if all conditions are met; False otherwise.
        """
        # Check tool call name.
        if tool_call.get("name") != expected_name:
            self.log_error(
                f"Expected tool call name '{expected_name}', but got '{tool_call.get('name')}'."
            )
            return False

        args = tool_call.get("args", {})

        # Check topic.
        if args.get("topic") != expected_topic:
            self.log_error(
                f"Expected topic '{expected_topic}', but got '{args.get('topic')}'."
            )
            return False

        # Check message type.
        if args.get("message_type") != expected_message_type:
            self.log_error(
                f"Expected message type '{expected_message_type}', but got '{args.get('message_type')}'."
            )
            return False

        # Traverse the message field.
        message = args.get("message")
        if message is None:
            self.log_error("Tool call does not contain a 'message' argument.")
            return False

        keys = field_path.split(".")
        value: Any = message
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                self.log_error(f"Field path '{field_path}' not found in the message.")
                return False

        if value != expected_value:
            self.log_error(
                f"Expected value for field '{field_path}' is '{expected_value}', but got '{value}'."
            )
            return False

        return True

    def _check_service_tool_call_field(
        self,
        tool_call: ToolCall,
        expected_name: str,
        expected_service: str,
        expected_service_type: str,
        field_path: str,
        expected_value: Any,
    ) -> bool:
        """
        Verifies a tool call for a service call.

        Parameters
        ----------
        tool_call : ToolCall
            The tool call dictionary containing keys such as "name" and "args".
        expected_name : str
            The expected tool call name (e.g., "call_ros2_service").
        expected_service : str
            The expected service name in the tool call's arguments.
        expected_message_type : str
            The expected message type.
        field_path : str
            Dot-separated path to the field inside the message.
        expected_value : Any
            The expected value at the given field path.

        Returns
        -------
        bool
            True if all conditions are met; False otherwise.
        """
        if tool_call.get("name") != expected_name:
            self.log_error(
                f"Expected tool call name '{expected_name}', but got '{tool_call.get('name')}'."
            )
            return False

        args = tool_call.get("args", {})

        # Check service.
        if args.get("service_name") != expected_service:
            self.log_error(
                f"Expected service '{expected_service}', but got '{args.get('service')}'."
            )
            return False

        # Check message type.
        if args.get("service_type") != expected_service_type:
            self.log_error(
                f"Expected message type '{expected_service_type}', but got '{args.get('service_type')}'."
            )
            return False

        service_args = args.get("service_args")
        if service_args is None:
            self.log_error("Tool call does not contain a 'service_args' argument.")
            return False

        if field_path == "":
            if service_args == {}:
                return True
            else:
                self.log_error(f"Expected empty service_args, but got: {service_args}")
                return False

        keys = field_path.split(".")
        value: Any = service_args
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                self.log_error(f"Field path '{field_path}' not found in the message.")
                return False

        if value != expected_value:
            self.log_error(
                f"Expected value for field '{field_path}' is '{expected_value}', but got '{value}'."
            )
            return False

        return True

    def _check_tool_call(
        self,
        tool_call: ToolCall,
        expected_name: str,
        expected_args: dict[str, Any],
        expected_optional_args: dict[str, Any] = {},
    ) -> bool:
        """Helper method to check if a tool call has the expected name and arguments.

        Parameters
        ----------
        tool_call : ToolCall
            The tool call to check
        expected_name : str
            The expected name of the tool
        expected_args : dict[str, Any]
            The expected arguments dictionary that must be present
        expected_optional_args : dict[str, Any], optional
            Optional arguments dictionary that can be present but don't need to be (e.g. timeout). If value of an optional argument does not matter, set it to {}

        Returns
        -------
        bool
            True if the tool call matches the expected name and args, False otherwise
        """
        if tool_call["name"] != expected_name:
            self.log_error(
                msg=f"Expected tool call name should be '{expected_name}', but got {tool_call['name']}"
            )
            return False

        # Check that all required arguments are present and have the expected values
        for arg_name, arg_value in expected_args.items():
            if arg_name in tool_call["args"]:
                if tool_call["args"][arg_name] != arg_value:
                    self.log_error(
                        msg=f"Expected argument '{arg_name}' should have value '{arg_value}', but got '{tool_call['args'][arg_name]}'"
                    )
                    return False
            else:
                self.log_error(
                    msg=f"Required argument '{arg_name}' missing in tool call {expected_name}."
                )
                return False

        # Check that no unexpected arguments are present (except for optional ones)
        for arg_name, arg_value in tool_call["args"].items():
            if arg_name not in expected_args:
                # If this argument is not required, check if it's an allowed optional argument
                if not expected_optional_args or arg_name not in expected_optional_args:
                    self.log_error(
                        msg=f"Unexpected argument '{arg_name}' found in tool call {expected_name}."
                    )
                    return False
                # If optional argument has expected value, check if the value is correct
                elif expected_optional_args[arg_name]:
                    if expected_optional_args[arg_name] != arg_value:
                        self.log_error(
                            msg=f"Optional argument '{arg_name}' has incorrect value '{arg_value}' in tool call {expected_name}."
                        )
                        return False

        return True

    def _check_multiple_tool_calls(
        self, message: AIMessage, expected_tool_calls: list[dict[str, Any]]
    ) -> bool:
        """Helper method to check multiple tool calls in a single AIMessage.

        Parameters
        ----------
        message : AIMessage
            The AIMessage to check
        expected_tool_calls : list[dict[str, Any]]
            A list of dictionaries, each containing expected 'name', 'args', and optional 'optional_args' for a tool call

        Returns
        -------
        bool
            True if all tool calls match expected patterns, False otherwise
        """
        if not self._check_tool_calls_num_in_ai_message(
            message, len(expected_tool_calls)
        ):
            return False

        matched_calls = [False] * len(expected_tool_calls)
        error_occurs = False

        for tool_call in message.tool_calls:
            found_match = False

            for i, expected in enumerate(expected_tool_calls):
                if matched_calls[i]:
                    continue

                expected_name = expected["name"]
                expected_args = expected["args"]
                expected_optional_args = expected.get("optional_args", {})

                if self._check_tool_call(
                    tool_call=tool_call,
                    expected_name=expected_name,
                    expected_args=expected_args,
                    expected_optional_args=expected_optional_args,
                ):
                    matched_calls[i] = True
                    found_match = True
                    break

            if not found_match:
                self.log_error(
                    msg=f"Tool call {tool_call['name']} with args {tool_call['args']} does not match any expected call"
                )
                error_occurs = True

        return not error_occurs

    def _check_tool_calls_num_in_ai_message(
        self, message: AIMessage, expected_num: int
    ) -> bool:
        """Helper method to check number of tool calls in a single AIMessage.

        Parameters
        ----------
        message : AIMessage
            The AIMessage to check
        expected_num : int
            The expected number of tool calls

        Returns
        -------
        bool
            True if the number of tool calls in the message matches the expected number, False otherwise
        """
        if len(message.tool_calls) != expected_num:
            self.log_error(
                msg=f"Expected number of tool calls should be {expected_num}, but got {len(message.tool_calls)}"
            )
            return False
        return True

    def log_error(self, msg: str):
        self.logger.error(msg)
        self.result.errors.append(msg)


class ROS2ToolCallingAgentTask(ToolCallingAgentTask, ABC):
    """Abstract class for ROS2 related tasks for tool calling agent.

    Parameters
    ----------
    logger : loggers_type | None
        Logger for the task.
    """

    def __init__(self, logger: loggers_type | None = None) -> None:
        super().__init__(logger)

    def _is_ai_message_requesting_get_ros2_topics_and_types(
        self, ai_message: AIMessage
    ) -> bool:
        """Helper method to check if the given AIMessage is calling the exactly one tool that gets ROS2 topics names and types correctly.

        Parameters
        ----------
        ai_message : AIMessage
            The AIMessage to check

        Returns
        -------
        bool
            True if the ai_message is requesting get_ros2_topics_names_and_types correctly, False otherwise
        """
        if not self._check_tool_calls_num_in_ai_message(ai_message, expected_num=1):
            return False

        tool_call: ToolCall = ai_message.tool_calls[0]
        if not self._check_tool_call(
            tool_call=tool_call,
            expected_name="get_ros2_topics_names_and_types",
            expected_args={},
        ):
            return False
        return True

    def _is_ai_message_requesting_get_ros2_services_and_types(
        self, ai_message: AIMessage
    ) -> bool:
        """Helper method to check if the given AIMessage is calling the exactly one tool that gets ROS2 service names and types correctly.

        Parameters
        ----------
        ai_message : AIMessage
            The AIMessage to check

        Returns
        -------
        bool
            True if the ai_message is requesting get_ros2_service_names_and_types correctly, False otherwise
        """
        if not self._check_tool_calls_num_in_ai_message(ai_message, expected_num=1):
            return False

        tool_call: ToolCall = ai_message.tool_calls[0]
        if not self._check_tool_call(
            tool_call=tool_call,
            expected_name="get_ros2_services_names_and_types",
            expected_args={},
        ):
            return False
        return True

    def _is_ai_message_requesting_get_ros2_actions_and_types(
        self, ai_message: AIMessage
    ) -> bool:
        """Helper method to check if the given AIMessage is calling the exactly one tool that gets ROS2 actions names and types correctly.

        Parameters
        ----------
        ai_message : AIMessage
            The AIMessage to check

        Returns
        -------
        bool
            True if the ai_message is requesting get_ros2_actions_names_and_types correctly, False otherwise
        """
        if not self._check_tool_calls_num_in_ai_message(ai_message, expected_num=1):
            return False

        tool_call: ToolCall = ai_message.tool_calls[0]
        if not self._check_tool_call(
            tool_call=tool_call,
            expected_name="get_ros2_actions_names_and_types",
            expected_args={},
        ):
            return False
        return True

    def get_tool_calls(self, response: dict[str, Any]) -> list[ToolCall]:
        """Extracts all tool calls from the response, flattened across all AI messages."""
        tool_calls: List[ToolCall] = []
        for msg in response["messages"]:
            if isinstance(msg, AIMessage):
                tool_calls.extend(msg.tool_calls)
        return tool_calls


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


class CustomInterfacesTopicTask(ROS2ToolCallingAgentTask, ABC):
    def __init__(self, logger: loggers_type | None = None) -> None:
        super().__init__(logger=logger)

        self.expected_tools: List[BaseTool] = [
            MockGetROS2TopicsNamesAndTypesTool(
                mock_topics_names_and_types=TOPIC_STRINGS
            ),
            MockGetROS2MessageInterfaceTool(mock_interfaces=MOCK_INTERFACES),
            MockPublishROS2MessageTool(
                available_topics=list(TOPICS_AND_TYPES.keys()),
                available_message_types=list(TOPICS_AND_TYPES.values()),
                available_topic_models=TOPIC_MODELS,
            ),
            MockCancelROS2ActionTool(),
            MockGetROS2ActionIDsTool(),
            MockMoveToPointTool(manipulator_frame="base_link"),
            MockGetROS2ImageTool(available_topics=list(IMAGE_TOPICS.keys())),
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

    @property
    @abstractmethod
    def expected_topic(self) -> str:
        pass

    @property
    def expected_message_type(self) -> str:
        return TOPICS_AND_TYPES[self.expected_topic]

    @property
    def extra_calls(self) -> int:
        return 0

    def verify_list_and_get_interface_tool_calls(
        self, tool_calls: List[ToolCall]
    ) -> tuple[bool, list[ToolCall]]:
        """
        Verifies tool calls in this required order:
        1. get_ros2_topics_and_types
        2. get_ros2_message_interface (with correct msg_type)

        Returns
        -------
        Tuple[bool, List[AIMessage]]
            Success flag and remaining messages (to be used in `verify_message_tool_call`)
        """

        expected_core_calls = 3
        max_allowed = expected_core_calls + self.extra_calls
        if len(tool_calls) > max_allowed:
            self.log_error(
                f"Too many tool calls. Expected at most {max_allowed}, got {len(tool_calls)}."
            )
            return False, []

        stage = 0  # 0: expect topics, 1: expect interface
        for idx, call in enumerate(tool_calls):
            if stage == 0 and call["name"] == "get_ros2_topics_names_and_types":
                stage = 1
                continue

            if stage == 1 and call["name"] == "get_ros2_message_interface":
                if call["args"].get("msg_type") == self.expected_message_type:
                    stage = 2
                    return True, tool_calls[idx + 1 :]

        self.log_error("Required tool calls not found in order: topics → interface")
        return False, []

    @abstractmethod
    def verify_message_tool_call(self, tool_calls: List[ToolCall]) -> bool:
        """
        Search the remaining AI messages for the expected publish/service tool call.
        """
        pass

    def verify_tool_calls(self, response: dict[str, Any]):
        """
        Validates the full sequence of AI tool calls with support for extras and ordering.

        Steps:
        1. Get topics
        2. Get message interface
        3. Call publish/service with expected content
        """
        messages = response["messages"]
        ai_messages: Sequence[AIMessage] = [
            msg for msg in messages if isinstance(msg, AIMessage)
        ]
        self.logger.debug(f"AI messages: {ai_messages}")
        tool_calls = self.get_tool_calls(response)

        # success, remaining_tool_calls = self.verify_list_and_get_interface_tool_calls(
        #     tool_calls
        # )
        # if success and self.verify_message_tool_call(remaining_tool_calls):
        #     self.result.success = True
        if self.verify_message_tool_call(tool_calls):
            self.result.success = True


class CustomInterfacesServiceTask(ROS2ToolCallingAgentTask, ABC):
    def __init__(self, logger: loggers_type | None = None) -> None:
        super().__init__(logger=logger)
        self.expected_tools: List[BaseTool] = [
            MockGetROS2TopicsNamesAndTypesTool(
                mock_topics_names_and_types=TOPIC_STRINGS
            ),
            MockGetROS2MessageInterfaceTool(mock_interfaces=MOCK_INTERFACES),
            MockPublishROS2MessageTool(
                available_topics=list(TOPICS_AND_TYPES.keys()),
                available_message_types=list(TOPICS_AND_TYPES.values()),
                available_topic_models=TOPIC_MODELS,
            ),
            MockCancelROS2ActionTool(),
            MockGetROS2ActionIDsTool(),
            MockMoveToPointTool(manipulator_frame="base_link"),
            MockGetROS2ImageTool(available_topics=list(IMAGE_TOPICS.keys())),
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

    @property
    @abstractmethod
    def expected_service(self) -> str:
        pass

    @property
    def expected_service_type(self) -> str:
        return SERVICES_AND_TYPES[self.expected_service]

    @property
    def extra_calls(self) -> int:
        return 0

    def verify_list_and_get_interface_tool_calls(
        self, tool_calls: List[ToolCall]
    ) -> tuple[bool, list[ToolCall]]:
        """
        Verifies tool calls in this required order:
        1. get_ros2_services_names_and_types
        2. get_ros2_message_interface (with correct msg_type)

        Returns
        -------
        Tuple[bool, List[ToolCall]]
            Success flag and remaining tool calls for message verification
        """
        expected_core_calls = 3
        max_allowed = expected_core_calls + self.extra_calls
        if len(tool_calls) > max_allowed:
            self.log_error(
                f"Too many tool calls. Expected at most {max_allowed}, got {len(tool_calls)}."
            )
            return False, []

        stage = 0  # 0: expect service list, 1: expect interface
        for idx, call in enumerate(tool_calls):
            if stage == 0 and call["name"] == "get_ros2_services_names_and_types":
                stage = 1
                continue

            if stage == 1 and call["name"] == "get_ros2_message_interface":
                if call["args"].get("msg_type") == self.expected_service_type:
                    stage = 2
                    return True, tool_calls[idx + 1 :]

        self.log_error("Required tool calls not found in order: services → interface")
        return False, []

    @abstractmethod
    def verify_message_tool_call(self, tool_calls: List[ToolCall]) -> bool:
        """Search the remaining tool calls for the expected service call."""
        pass

    def verify_tool_calls(self, response: dict[str, Any]):
        """
        Full tool call sequence verification:
        1. Get services
        2. Get message interface
        3. Call service with expected values
        """
        messages = response["messages"]
        ai_messages: Sequence[AIMessage] = [
            msg for msg in messages if isinstance(msg, AIMessage)
        ]
        self.logger.debug(f"AI messages: {ai_messages}")
        tool_calls = self.get_tool_calls(response)

        # success, remaining_tool_calls = self.verify_list_and_get_interface_tool_calls(
        #     tool_calls
        # )
        # if success and

        if self.verify_message_tool_call(tool_calls):
            self.result.success = True


class CustomInterfacesActionTask(ROS2ToolCallingAgentTask, ABC):
    ACTIONS_AND_TYPES = {
        # custom actions
        "/perform_task": "rai_interfaces/action/Task",
        # some sample actions
        # "/execute_trajectory": "moveit_msgs/action/ExecuteTrajectory",
        # "/move_action": "moveit_msgs/action/MoveGroup",
        # "/follow_joint_trajectory": "control_msgs/action/FollowJointTrajectory",
        # "/gripper_cmd": "control_msgs/action/GripperCommand",
    }

    action_strings = [
        f"action: {action}\ntype: {msg_type}\n"
        for action, msg_type in ACTIONS_AND_TYPES.items()
    ]

    def __init__(self, logger: loggers_type | None = None) -> None:
        super().__init__(logger=logger)
        self.expected_tools: List[BaseTool] = [
            MockGetROS2ActionsNamesAndTypesTool(
                mock_actions_names_and_types=self.action_strings
            ),
            MockGetROS2MessageInterfaceTool(mock_interfaces=MOCK_INTERFACES),
            MockStartROS2ActionTool(
                available_actions=list(self.ACTIONS_AND_TYPES.keys()),
                available_action_types=list(self.ACTIONS_AND_TYPES.values()),
            ),
        ]

    @property
    @abstractmethod
    def expected_action(self) -> str:
        pass

    @property
    @abstractmethod
    def expected_message(self) -> BaseModel:
        pass

    @property
    def expected_action_type(self) -> str:
        return self.ACTIONS_AND_TYPES[self.expected_action]

    def verify_tool_calls(self, response: dict[str, Any]):
        """It is expected that the agent will request:
        1. The tool that retrieves the topics names and types to recognize what type of message to_human topic has
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
            if not self._is_ai_message_requesting_get_ros2_actions_and_types(
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
                    expected_args={"msg_type": self.expected_action_type},
                )

        if len(ai_messages) > 2:
            if self._check_tool_calls_num_in_ai_message(ai_messages[2], expected_num=1):
                self._check_tool_call(
                    tool_call=ai_messages[2].tool_calls[0],
                    expected_name="start_ros2_action",
                    expected_args={
                        "action_name": self.expected_action,
                        "action_args": self.expected_message.model_dump(),
                        "action_type": self.expected_action_type,
                    },
                )
        if not self.result.errors:
            self.result.success = True

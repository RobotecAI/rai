# Copyright (C) 2026 Robotec.AI
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

"""Tool argument constraints are declared on the args_schema, so a model that
sends an empty topic name, a non-positive timeout or a NaN coordinate gets a
ValidationError that ToolRunner reports back to it instead of a live call."""

import math

import pytest
from pydantic import ValidationError
from rai.tools.ros2.generic.actions import StartROS2ActionToolInput
from rai.tools.ros2.generic.services import CallROS2ServiceToolInput
from rai.tools.ros2.generic.topics import (
    GetROS2ImageToolInput,
    GetROS2TransformToolInput,
    PublishROS2MessageToolInput,
    ReceiveROS2MessageToolInput,
)
from rai.tools.ros2.navigation.nav2 import NavigateToPoseToolInput

VALID = {
    PublishROS2MessageToolInput: dict(
        topic="/chatter", message={}, message_type="std_msgs/msg/String"
    ),
    ReceiveROS2MessageToolInput: dict(topic="/chatter", timeout_sec=1.0),
    GetROS2ImageToolInput: dict(topic="/camera/image_raw", timeout_sec=1.0),
    GetROS2TransformToolInput: dict(
        target_frame="map", source_frame="base_link", timeout_sec=1.0
    ),
    CallROS2ServiceToolInput: dict(
        service_name="/set_bool", service_type="std_srvs/srv/SetBool", timeout_sec=1.0
    ),
    StartROS2ActionToolInput: dict(
        action_name="/navigate",
        action_type="nav2_msgs/action/NavigateToPose",
        action_args={},
    ),
    NavigateToPoseToolInput: dict(x=1.0, y=2.0, z=0.0, yaw=0.0),
}

EMPTY_NAME_FIELDS = [
    (PublishROS2MessageToolInput, "topic"),
    (PublishROS2MessageToolInput, "message_type"),
    (ReceiveROS2MessageToolInput, "topic"),
    (GetROS2ImageToolInput, "topic"),
    (GetROS2TransformToolInput, "target_frame"),
    (GetROS2TransformToolInput, "source_frame"),
    (CallROS2ServiceToolInput, "service_name"),
    (CallROS2ServiceToolInput, "service_type"),
    (StartROS2ActionToolInput, "action_name"),
    (StartROS2ActionToolInput, "action_type"),
]

TIMEOUT_MODELS = [
    ReceiveROS2MessageToolInput,
    GetROS2ImageToolInput,
    GetROS2TransformToolInput,
    CallROS2ServiceToolInput,
]


@pytest.mark.parametrize("model", VALID)
def test_valid_args_accepted(model):
    model(**VALID[model])


@pytest.mark.parametrize("model,field", EMPTY_NAME_FIELDS)
def test_empty_names_rejected(model, field):
    with pytest.raises(ValidationError):
        model(**{**VALID[model], field: ""})


@pytest.mark.parametrize("model", TIMEOUT_MODELS)
@pytest.mark.parametrize("timeout_sec", [0, -1.0])
def test_non_positive_timeout_rejected(model, timeout_sec):
    with pytest.raises(ValidationError):
        model(**{**VALID[model], "timeout_sec": timeout_sec})


@pytest.mark.parametrize("field", ["x", "y", "z", "yaw"])
@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_non_finite_pose_rejected(field, value):
    with pytest.raises(ValidationError):
        NavigateToPoseToolInput(**{**VALID[NavigateToPoseToolInput], field: value})

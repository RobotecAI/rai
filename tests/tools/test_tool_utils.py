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


import pytest
from geometry_msgs.msg import Point, TransformStamped
from nav2_msgs.action import NavigateToPose
from sensor_msgs.msg import Image
from tf2_msgs.msg import TFMessage

from rai.tools.ros2.utils import ros2_message_to_dict


# TODO(`maciejmajek`): Add custom RAI messages?
@pytest.mark.parametrize(
    "message",
    [
        Point(),
        Image(),
        TFMessage(),
        TransformStamped(),
        NavigateToPose.Goal(),
        NavigateToPose.Result(),
        NavigateToPose.Feedback(),
    ],
    ids=lambda x: x.__class__.__name__,
)
def test_ros2_message_to_dict(message):
    assert ros2_message_to_dict(message)

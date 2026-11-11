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

import sys

import pytest


def pytest_configure(config):
    if sys.prefix != sys.base_prefix and not any(
        p.endswith("dist-packages") for p in sys.path
    ):
        raise pytest.UsageError(
            "The venv cannot see the system site-packages that ROS 2 needs. "
            "Recreate it with scripts/dev_env.sh (uv venv --system-site-packages)."
        )


# 3D gripping point detection strategy
def pytest_addoption(parser):
    parser.addoption(
        "--strategy", action="store", default="centroid", help="Gripping point strategy"
    )


@pytest.fixture
def strategy(request):
    return request.config.getoption("--strategy")


@pytest.fixture
def ros2_image():
    import numpy as np
    from sensor_msgs.msg import Image

    msg = Image()
    msg.header.frame_id = "test_frame"
    msg.height = 100
    msg.width = 100
    msg.encoding = "rgb8"
    msg.is_bigendian = False
    msg.step = 300
    msg.data = np.zeros((100, 100, 3), dtype=np.uint8).tobytes()
    return msg

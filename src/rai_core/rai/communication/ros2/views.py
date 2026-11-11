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

"""numpy views of sensor messages, without copying the message data.

rclpy deserializes uint8[] fields into array.array, so np.frombuffer gives a
view for free. A view aliases the message: copy it to keep it past the message
or to change it without changing the message.
"""

import numpy as np
from numpy.typing import NDArray
from sensor_msgs.msg import CompressedImage, Image, PointCloud2, PointField

from rai.communication.ros2.images import ENCODINGS, compressed_imgmsg_to_numpy

_POINT_TYPES = {
    PointField.INT8: "i1",
    PointField.UINT8: "u1",
    PointField.INT16: "i2",
    PointField.UINT16: "u2",
    PointField.INT32: "i4",
    PointField.UINT32: "u4",
    PointField.FLOAT32: "f4",
    PointField.FLOAT64: "f8",
}


def image(msg: Image | CompressedImage) -> NDArray:
    """(H, W) or (H, W, C) view of an Image in its own encoding and channel order.
    A CompressedImage is decoded (bgr8 for color), which copies."""
    if isinstance(msg, CompressedImage):
        return compressed_imgmsg_to_numpy(msg)
    if msg.encoding not in ENCODINGS:
        raise ValueError(f"Unsupported image encoding: {msg.encoding}")
    dtype, channels = ENCODINGS[msg.encoding]
    dt = np.dtype(dtype).newbyteorder(">" if msg.is_bigendian else "<")
    rows = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.step)
    pixels = rows[:, : msg.width * channels * dt.itemsize].view(dt)
    if channels == 1:
        return pixels.reshape(msg.height, msg.width)
    return pixels.reshape(msg.height, msg.width, channels)


def points(msg: PointCloud2) -> NDArray:
    """Structured view of a PointCloud2, one record per point with the message's
    fields (e.g. view["x"]); shape (width,) or (height, width)."""
    order = ">" if msg.is_bigendian else "<"
    dtype = np.dtype(
        {
            "names": [f.name for f in msg.fields],
            "formats": [
                (order + _POINT_TYPES[f.datatype], (f.count,))
                if f.count > 1
                else order + _POINT_TYPES[f.datatype]
                for f in msg.fields
            ],
            "offsets": [f.offset for f in msg.fields],
            "itemsize": msg.point_step,
        }
    )
    rows = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.row_step)
    cloud = rows[:, : msg.width * msg.point_step].view(dtype).reshape(-1, msg.width)
    return cloud[0] if msg.height == 1 else cloud

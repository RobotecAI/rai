import base64
import datetime
import io
import time
from dataclasses import dataclass

import rclpy
from PIL import Image
from PIL.Image import Image as PILImage
from pydantic import BaseModel
from scipy.spatial.transform import Rotation

from rai.node import RaiBaseNode
from rai.tools.ros.native import GetCameraImage
from rai.tools.ros.utils import get_transform


@dataclass
class Position:
    x: float
    y: float
    z: float
    yaw: float


class Observation(BaseModel):
    image: PILImage
    position: Position
    timestamp: float


rclpy.init()
node = RaiBaseNode(node_name="test_node")

get_camera_image = GetCameraImage(node=node)
# get current cet timestamp
cet_timestamp = datetime.datetime.now(datetime.timezone.utc).timestamp()

# get_msg_from_topic = GetMsgFromTopic(node=node)

while True:
    _, image = get_camera_image._run(topic_name="/camera/image_raw")
    position = get_transform(node=node, target_frame="base_link", source_frame="map")
    cet_timestamp = datetime.datetime.now(datetime.timezone.utc).timestamp()
    # get Pil image from base_64 string
    pose = position.transform.translation  # type: ignore
    # calculate yaw from quaternion
    quat = Rotation.from_quat(
        [
            position.transform.rotation.x,  # type: ignore
            position.transform.rotation.y,  # type: ignore
            position.transform.rotation.z,  # type: ignore
            position.transform.rotation.w,  # type: ignore
        ]
    )
    yaw = quat.as_euler("xyz", degrees=False)[2]  # type: ignore
    pil_image = Image.open(io.BytesIO(base64.b64decode(image["images"][0])))
    observation = Observation(
        image=pil_image,
        position=Position(x=pose.x, y=pose.y, z=pose.z, yaw=yaw),  # type: ignore
        timestamp=cet_timestamp,
    )
    print(observation)

    time.sleep(10.0)

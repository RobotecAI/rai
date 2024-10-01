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

from typing import Optional

import rclpy
from pydantic import Field
from rclpy import Future

from rai.node import RaiBaseNode
from rai.tools.ros import Ros2BaseTool
from rai_interfaces.srv import RAIGroundingDino


class CompositeBaseTool(Ros2BaseTool):
    node: RaiBaseNode = Field(..., exclude=True, required=True)

    def _spin(self, future: Future) -> Optional[RAIGroundingDino.Response]:
        rclpy.spin_once(self.node)
        if future.done():
            try:
                response = future.result()
            except Exception as e:
                self.node.get_logger().info("Service call failed %r" % (e,))
                raise Exception("Service call failed %r" % (e,))
            else:
                assert response is not None
                self.node.get_logger().info(f"{response.detections}")
                return response
        return None

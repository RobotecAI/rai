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

from typing import Any, Dict, Literal, Optional

from rai.communication import ARIMessage, HRIMessage, HRIPayload


class ROS2ARIMessage(ARIMessage):
    def __init__(self, payload: Any, metadata: Optional[Dict[str, Any]] = None):
        super().__init__(payload, metadata)


class ROS2HRIMessage(HRIMessage):
    def __init__(self, payload: HRIPayload, message_author: Literal["ai", "human"]):
        super().__init__(payload, message_author)

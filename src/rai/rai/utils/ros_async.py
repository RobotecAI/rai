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

import time
from typing import Any, Optional

import rclpy.task


def get_future_result(
    future: rclpy.task.Future, timeout_sec: float = 5.0
) -> Optional[Any]:
    """Replaces rclpy.spin_until_future_complete"""
    result = None

    def callback(future: rclpy.task.Future) -> None:
        nonlocal result
        result = future.result()

    future.add_done_callback(callback)

    ts = time.perf_counter()
    while result is None and time.perf_counter() - ts < timeout_sec:
        time.sleep(0.1)

    return result

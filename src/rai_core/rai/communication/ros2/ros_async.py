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

import logging
import threading
from typing import Any

import rclpy
import rclpy.task

logger = logging.getLogger(__name__)


def get_future_result(
    future: rclpy.task.Future, timeout_sec: float = 5.0
) -> Any | None:
    """Replaces rclpy.spin_until_future_complete"""
    result = None
    event = threading.Event()

    def callback(future: rclpy.task.Future) -> None:
        nonlocal result
        result = future.result()
        event.set()

    future.add_done_callback(callback)

    timed_out = not event.wait(timeout=timeout_sec)
    if timed_out:
        logger.warning(f"Future timed out after {timeout_sec}s")

    return result

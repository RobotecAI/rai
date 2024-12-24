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

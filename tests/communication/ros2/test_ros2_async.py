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

import concurrent.futures
import threading
import time

import pytest
from rai.communication.ros2.ros_async import get_future_result
from rclpy.task import Future


# Basic functionality tests
def test_get_future_result():
    """Test with a future that's already completed."""
    future = Future()
    future.set_result(1)
    result = get_future_result(future)
    assert result == 1


def test_get_future_result_timeout():
    """Test that a future that never completes returns None after timeout."""
    future = Future()
    result = get_future_result(future, timeout_sec=0.1)
    assert result is None


def test_get_future_result_timeout_logs_warning(caplog):
    """Test that timeout logs a warning message."""
    import logging

    logger = logging.getLogger("rai.communication.ros2.ros_async")
    logger.propagate = True

    future = Future()
    with caplog.at_level(logging.WARNING, logger=logger.name):
        result = get_future_result(future, timeout_sec=0.1)

    assert result is None
    assert any(
        "Future timed out after 0.1s" in record.message for record in caplog.records
    )


# Test future completing during wait
def test_get_future_result_completes_during_wait():
    """Test when the future completes while waiting."""
    future = Future()

    def complete_future():
        time.sleep(0.2)
        future.set_result(42)

    thread = threading.Thread(target=complete_future, daemon=True)
    thread.start()

    result = get_future_result(future, timeout_sec=1.0)
    assert result == 42
    thread.join(timeout=1.0)


# Exception handling tests
def test_get_future_result_with_exception():
    """Test how the function handles exceptions raised in the future."""
    future = Future()
    future.set_exception(RuntimeError("Test error"))

    with pytest.raises(RuntimeError, match="Test error"):
        _ = get_future_result(future)


def test_get_future_result_with_custom_exception():
    """Test with a custom exception type."""
    future = Future()

    class CustomError(Exception):
        pass

    future.set_exception(CustomError("Custom error message"))

    with pytest.raises(CustomError, match="Custom error message"):
        _ = get_future_result(future)


# Cancelled future tests
@pytest.mark.xfail(reason="TODO: define behavior for cancelled future")
def test_get_future_result_cancelled():
    """Test behavior when future is cancelled before calling get_future_result."""
    future = Future()
    future.cancel()

    with pytest.raises(Exception):
        _ = get_future_result(future)


@pytest.mark.xfail(reason="TODO: define behavior for cancelled future")
def test_get_future_result_cancelled_during_wait():
    """Test behavior when future is cancelled while waiting."""
    future = Future()

    def cancel_future():
        time.sleep(0.05)
        future.cancel()

    thread = threading.Thread(target=cancel_future, daemon=True)
    thread.start()

    with pytest.raises(Exception):  # Should raise when cancelled
        _ = get_future_result(future, timeout_sec=1.0)

    thread.join(timeout=1.0)


# Edge case timeout tests
def test_get_future_result_zero_timeout():
    """Test with zero timeout."""
    future = Future()
    result = get_future_result(future, timeout_sec=0.0)
    assert result is None


def test_get_future_result_very_short_timeout():
    """Test with very short timeout."""
    future = Future()
    result = get_future_result(future, timeout_sec=0.001)
    assert result is None


def test_get_future_result_long_timeout():
    """Test with long timeout but future completes quickly."""
    future = Future()
    future.set_result("quick")

    start = time.time()
    result = get_future_result(future, timeout_sec=10.0)
    elapsed = time.time() - start

    assert result == "quick"
    assert elapsed < 1.0  # Should return immediately, not wait for timeout


# Thread safety / Concurrent calls tests
def test_get_future_result_concurrent_calls():
    """Test multiple concurrent calls to get_future_result."""
    futures = [Future() for _ in range(5)]
    for i, f in enumerate(futures):
        f.set_result(i)

    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        results = list(executor.map(get_future_result, futures))

    assert results == [0, 1, 2, 3, 4]


def test_get_future_result_multiple_callbacks():
    """Test that adding multiple callbacks doesn't interfere."""
    future = Future()
    callback_results: list[int] = []

    def other_callback(f: Future) -> None:
        res = f.result()
        if isinstance(res, int):
            callback_results.append(res)

    future.add_done_callback(other_callback)
    future.add_done_callback(other_callback)

    future.set_result(100)
    result = get_future_result(future)

    assert result == 100
    assert callback_results == [100, 100]


# Stress test
def test_get_future_result_many_sequential_calls():
    """Test many sequential calls to ensure no resource leaks."""
    for i in range(100):
        future = Future()
        future.set_result(i)
        result = get_future_result(future, timeout_sec=0.1)
        assert result == i


def test_get_future_result_ambiguous_none():
    """
    Test demonstrating the ambiguity issue:
    Function returns None for both timeout and when actual result is None.
    This is a known limitation of the current implementation.
    """
    future1 = Future()
    future1.set_result(None)
    result1 = get_future_result(future1, timeout_sec=0.1)

    future2 = Future()
    result2 = get_future_result(future2, timeout_sec=0.1)

    assert result1 is None
    assert result2 is None

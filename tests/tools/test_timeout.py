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

import time

import pytest
from rai.tools.timeout import RaiTimeoutError, timeout_method


class TestClass:
    @timeout_method(1.0)
    def fast_method(self):
        return "success"

    @timeout_method(0.5)
    def slow_method(self):
        time.sleep(1.0)
        return "should not reach here"

    @timeout_method(0.5, "Custom timeout message")
    def slow_method_custom(self):
        time.sleep(1.0)
        return "should not reach here"

    @timeout_method(1.0)
    def method_with_args(self, arg1, arg2, kwarg=None):
        return f"{arg1}_{arg2}_{kwarg}"


@pytest.fixture
def obj():
    return TestClass()


def test_timeout_method_successful_execution(obj):
    assert obj.fast_method() == "success"
    assert obj.method_with_args("a", "b") == "a_b_None"
    assert obj.method_with_args("x", "y", kwarg="z") == "x_y_z"


def test_timeout_method_raises_timeout_error(obj):
    with pytest.raises(RaiTimeoutError) as exc_info:
        obj.slow_method()
    assert "slow_method" in str(exc_info.value)
    assert "TestClass" in str(exc_info.value)
    assert "0.5 seconds" in str(exc_info.value)

    with pytest.raises(RaiTimeoutError) as exc_info:
        obj.slow_method_custom()
    assert str(exc_info.value) == "Custom timeout message"

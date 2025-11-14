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

import rclpy
from rai.communication.ros2.context import ROS2Context
from rclpy.utilities import get_default_context


def test_ros2_context_wrapper():
    context = ROS2Context()

    def test_function():
        assert rclpy.ok()

    test_function = context(test_function)
    assert not rclpy.ok()
    test_function()
    assert not rclpy.ok()


def test_ros2_context_decorator():
    @ROS2Context()
    def test_function():
        assert rclpy.ok()

    assert not rclpy.ok()
    test_function()
    assert not rclpy.ok()


def test_ros2_context_decorator_domain_id():
    @ROS2Context(domain_id=13)
    def test_function():
        assert rclpy.ok()
        assert get_default_context().get_domain_id() == 13

    assert not rclpy.ok()
    test_function()
    assert not rclpy.ok()


def test_ros2_context_enter():
    assert not rclpy.ok()
    with ROS2Context() as ctx:
        assert ctx.is_initialized
        assert rclpy.ok()
    assert not rclpy.ok()


def test_ros2_context_enter_domain_id():
    assert not rclpy.ok()
    with ROS2Context(domain_id=13) as ctx:
        assert ctx.is_initialized
        assert rclpy.ok()
        assert get_default_context().get_domain_id() == 13
    assert not rclpy.ok()

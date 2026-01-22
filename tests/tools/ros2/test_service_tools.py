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

import pytest

try:
    import rclpy  # noqa: F401
except ImportError:
    pytest.skip("ROS2 is not installed", allow_module_level=True)


import time

from rai.communication.ros2.connectors import ROS2Connector
from rai.tools.ros2 import (
    CallROS2ServiceTool,
    GetROS2ServicesNamesAndTypesTool,
)

from tests.communication.ros2.helpers import (
    ServiceServer,
    multi_threaded_spinner,
    ros_setup,
    shutdown_executors_and_threads,
)

_ = ros_setup  # Explicitly use the fixture to prevent pytest warnings


def test_service_call_tool(ros_setup: None, request: pytest.FixtureRequest) -> None:
    service_name = f"{request.node.originalname}_service"  # type: ignore
    connector = ROS2Connector()
    server = ServiceServer(service_name=service_name)
    executors, threads = multi_threaded_spinner([server])
    tool = CallROS2ServiceTool(connector=connector)
    try:
        response = tool._run(  # type: ignore
            service_name=service_name,
            service_type="std_srvs/srv/SetBool",
            service_args={},
        )
        assert "Test service called" in response
        assert "success=True" in response
    finally:
        shutdown_executors_and_threads(executors, threads)


def test_service_call_tool_with_forbidden_service(
    ros_setup: None, request: pytest.FixtureRequest
) -> None:
    service_name = f"{request.node.originalname}_service"  # type: ignore
    connector = ROS2Connector()
    tool = CallROS2ServiceTool(connector=connector, forbidden=[service_name])
    with pytest.raises(ValueError):
        tool._run(
            service_name=service_name,
            service_type="std_srvs/srv/SetBool",
            service_args={},
        )


def test_service_call_tool_with_writable_service(
    ros_setup: None, request: pytest.FixtureRequest
) -> None:
    service_name = f"{request.node.originalname}_service"  # type: ignore
    connector = ROS2Connector()
    server = ServiceServer(service_name=service_name)
    executors, threads = multi_threaded_spinner([server])
    tool = CallROS2ServiceTool(connector=connector, writable=[service_name])
    try:
        response = tool._run(  # type: ignore
            service_name=service_name,
            service_type="std_srvs/srv/SetBool",
            service_args={},
        )
        assert "Test service called" in response
        assert "success=True" in response
    finally:
        shutdown_executors_and_threads(executors, threads)


def test_get_services_names_and_types_tool_no_restrictions(
    ros_setup: None, request: pytest.FixtureRequest
) -> None:
    service_name = f"/{request.node.originalname}_service"  # type: ignore
    connector = ROS2Connector()
    server = ServiceServer(service_name=service_name)
    executors, threads = multi_threaded_spinner([server])
    try:
        time.sleep(0.2)
        tool = GetROS2ServicesNamesAndTypesTool(connector=connector)
        response = tool._run()  # type: ignore
        assert response != ""
        assert service_name in response
        assert "service:" in response
        assert "type:" in response
    finally:
        shutdown_executors_and_threads(executors, threads)


def test_get_services_names_and_types_tool_with_writable(
    ros_setup: None, request: pytest.FixtureRequest
) -> None:
    writable_service = f"/{request.node.originalname}_writable_service"  # type: ignore
    non_writable_service = f"/{request.node.originalname}_non_writable_service"  # type: ignore
    connector = ROS2Connector()
    server1 = ServiceServer(service_name=writable_service)
    server2 = ServiceServer(service_name=non_writable_service)
    executors, threads = multi_threaded_spinner([server1, server2])
    try:
        time.sleep(0.2)
        tool = GetROS2ServicesNamesAndTypesTool(
            connector=connector, writable=[writable_service]
        )
        response = tool._run()  # type: ignore
        assert response != ""
        assert writable_service in response
        assert non_writable_service not in response
    finally:
        shutdown_executors_and_threads(executors, threads)


def test_get_services_names_and_types_tool_with_forbidden(
    ros_setup: None, request: pytest.FixtureRequest
) -> None:
    service_name = f"/{request.node.originalname}_service"  # type: ignore
    forbidden_service = f"/{request.node.originalname}_forbidden_service"  # type: ignore
    connector = ROS2Connector()
    server1 = ServiceServer(service_name=service_name)
    server2 = ServiceServer(service_name=forbidden_service)
    executors, threads = multi_threaded_spinner([server1, server2])
    try:
        time.sleep(0.2)
        tool = GetROS2ServicesNamesAndTypesTool(
            connector=connector, forbidden=[forbidden_service]
        )
        response = tool._run()  # type: ignore
        assert response != ""
        assert service_name in response
        assert forbidden_service not in response
    finally:
        shutdown_executors_and_threads(executors, threads)


def test_get_services_names_and_types_tool_with_writable_and_forbidden(
    ros_setup: None, request: pytest.FixtureRequest
) -> None:
    writable_service = f"/{request.node.originalname}_writable_service"  # type: ignore
    forbidden_service = f"/{request.node.originalname}_forbidden_service"  # type: ignore
    connector = ROS2Connector()
    server1 = ServiceServer(service_name=writable_service)
    server2 = ServiceServer(service_name=forbidden_service)
    executors, threads = multi_threaded_spinner([server1, server2])
    try:
        time.sleep(0.2)
        tool = GetROS2ServicesNamesAndTypesTool(
            connector=connector,
            writable=[writable_service],
            forbidden=[forbidden_service],
        )
        response = tool._run()  # type: ignore
        assert response != ""
        assert writable_service in response
        assert forbidden_service not in response
    finally:
        shutdown_executors_and_threads(executors, threads)

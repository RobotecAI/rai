# Copyright (C) 2025 Julia Jia
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

from typing import Any, Optional


class ROS2ServiceError(Exception):
    """Exception raised for ROS2 service-related errors."""

    def __init__(
        self,
        service_name: str,
        timeout_sec: float,
        service_state: Optional[str] = None,
        suggestion: Optional[str] = None,
        underlying_error: Optional[Exception] = None,
    ):
        self.service_name = service_name
        self.timeout_sec = timeout_sec
        self.service_state = service_state  # "exists", "ready", "unavailable"
        self.suggestion = (
            suggestion  # "Check if service is running", "Try increasing timeout"
        )
        self.underlying_error = underlying_error
        message = f"Service {service_name} error: {service_state or 'unavailable'}"
        if suggestion:
            message += f". {suggestion}"
        super().__init__(message)


class ROS2ParameterError(Exception):
    """Exception raised for ROS2 parameter-related errors."""

    def __init__(
        self,
        param_name: str,
        expected_type: Optional[str] = None,
        expected_value: Optional[str] = None,
        suggestion: Optional[str] = None,
        default_value: Any = None,
    ):
        self.param_name = param_name
        self.expected_type = expected_type
        self.expected_value = expected_value
        self.suggestion = suggestion  # "Set in launch file", "Check config YAML"
        self.default_value = default_value
        message = f"Parameter {param_name} error: {suggestion or 'missing or invalid'}"
        if expected_type:
            message += f" (expected type: {expected_type})"
        super().__init__(message)

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


from typing import Any, Dict

import pytest
from rai.types.rai_interfaces import ManipulatorMoveToRequest
from rcl_interfaces.srv import SetParameters

from rai_bench.tool_calling_agent.mocked_tools import ServiceValidator


class TestServiceValidator:
    """Test suite for ServiceValidator using real ROS 2 interfaces and custom models"""

    @pytest.fixture
    def custom_models(self) -> Dict[str, Any]:
        """Fixture providing actual custom Pydantic models"""
        return {
            "rai_interfaces/srv/ManipulatorMoveTo": ManipulatorMoveToRequest,
        }

    @pytest.fixture
    def validator(self, custom_models: Dict[str, Any]) -> ServiceValidator:
        """Fixture providing ServiceValidator instance with real models"""
        return ServiceValidator(custom_models)

    def test_get_ros2_service_class_caching(self, validator: ServiceValidator):
        """Test that service classes are cached properly"""
        # First call
        result1 = validator.get_ros2_service_class("rcl_interfaces/srv/SetParameters")
        # Second call (should use cache)
        result2 = validator.get_ros2_service_class("rcl_interfaces/srv/SetParameters")

        assert result1 == result2 == SetParameters
        assert "rcl_interfaces/srv/SetParameters" in validator.ros2_services_cache

    def test_get_ros2_service_class_invalid_format_too_few_parts(
        self, validator: ServiceValidator
    ):
        """Test error handling for service type with too few parts"""
        with pytest.raises(ValueError, match="is invalid"):
            validator.get_ros2_service_class("invalid_format")

    def test_get_ros2_service_class_invalid_format_wrong_middle_part(
        self, validator: ServiceValidator
    ):
        """Test error handling for service type with wrong middle part"""
        with pytest.raises(ValueError, match="is invalid"):
            validator.get_ros2_service_class("rcl_interfaces/msg/SetParameters")

    def test_get_ros2_service_class_nonexistent_package(
        self, validator: ServiceValidator
    ):
        """Test error handling when package doesn't exist"""
        with pytest.raises(ImportError):
            validator.get_ros2_service_class("nonexistent_package/srv/TestService")

    def test_get_ros2_service_class_nonexistent_service(
        self, validator: ServiceValidator
    ):
        """Test error handling when service doesn't exist in package"""
        with pytest.raises(AttributeError):
            validator.get_ros2_service_class("rcl_interfaces/srv/NonexistentService")

    def test_validate_with_ros2_setparameters_valid_args(
        self, validator: ServiceValidator
    ):
        """Test successful validation with valid SetParameters arguments"""
        args: Dict[str, Any] = {
            "parameters": [
                {
                    "name": "test_param",
                    "value": {
                        "type": 2,  # PARAMETER_INTEGER
                        "bool_value": False,
                        "integer_value": 42,
                        "double_value": 0.0,
                        "string_value": "",
                    },
                }
            ]
        }

        validator.validate_with_ros2("rcl_interfaces/srv/SetParameters", args)

    def test_validate_with_ros2_setparameters_minimal_valid_args(
        self, validator: ServiceValidator
    ):
        """Test validation with minimal valid SetParameters arguments"""
        args: Dict[str, Any] = {
            "parameters": [
                {
                    "name": "test_param",
                    "value": {
                        "type": 4,  # PARAMETER_STRING
                        "string_value": "test_value",
                    },
                }
            ]
        }

        validator.validate_with_ros2("rcl_interfaces/srv/SetParameters", args)

    def test_validate_with_ros2_invalid_field_name(self, validator: ServiceValidator):
        """Test validation with invalid field in SetParameters"""
        args: Dict[str, Any] = {"parameters": [], "invalid_field": "should_not_exist"}

        with pytest.raises(AttributeError):  # set_message_fields will raise
            validator.validate_with_ros2("rcl_interfaces/srv/SetParameters", args)

    def test_validate_with_ros2_wrong_parameter_type(self, validator: ServiceValidator):
        """Test validation with wrong parameter structure"""
        args: Dict[str, Any] = {
            "parameters": [{"name": "test_param", "value": "should_be_dict_not_string"}]
        }

        with pytest.raises(TypeError):
            validator.validate_with_ros2("rcl_interfaces/srv/SetParameters", args)

    def test_validate_with_ros2_empty_args(self, validator: ServiceValidator):
        """Test validation with empty args (should use defaults)"""
        args: Dict[str, Any] = {}

        # Should work - ROS 2 messages have default values
        validator.validate_with_ros2("rcl_interfaces/srv/GetParameters", args)

    def test_validate_with_custom_model_valid_args(self, validator: ServiceValidator):
        """Test successful validation with valid ManipulatorMoveTo arguments"""
        args: Dict[str, Any] = {
            "initial_gripper_state": True,
            "final_gripper_state": False,
            "target_pose": {
                "header": {
                    "stamp": {"sec": 0, "nanosec": 0},
                    "frame_id": "base_link",
                },
                "pose": {
                    "position": {"x": 1.0, "y": 2.0, "z": 3.0},
                    "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
                },
            },
        }

        validator.validate_with_custom("rai_interfaces/srv/ManipulatorMoveTo", args)

    def test_validate_with_custom_manipulator_minimal_args(
        self, validator: ServiceValidator
    ):
        """Test validation with minimal ManipulatorMoveTo arguments (using defaults)"""
        args: Dict[str, Any] = {}  # All fields have defaults

        validator.validate_with_custom("rai_interfaces/srv/ManipulatorMoveTo", args)

    def test_validate_with_custom_manipulator_invalid_type(
        self, validator: ServiceValidator
    ):
        """Test validation with invalid type for ManipulatorMoveTo"""
        args: Dict[str, Any] = {
            "initial_gripper_state": "should_be_bool_not_string",
            "final_gripper_state": False,
        }

        with pytest.raises(ValueError, match="Pydantic validation failed"):
            validator.validate_with_custom("rai_interfaces/srv/ManipulatorMoveTo", args)

    def test_validate_with_custom_service_not_in_models(
        self, validator: ServiceValidator
    ):
        """Test custom validation when service type not in custom models"""
        args: Dict[str, Any] = {"some_field": "value"}

        with pytest.raises(ValueError, match="is invalid custom type"):
            validator.validate_with_custom("unknown/srv/Service", args)

    def test_validate_routes_to_custom_when_available(
        self, validator: ServiceValidator
    ):
        """Test that validate() uses custom models when service type is in custom_models"""
        args = {"initial_gripper_state": True}

        # Should route to custom validation (not ROS 2)
        validator.validate("rai_interfaces/srv/ManipulatorMoveTo", args)

    def test_validate_service_not_in_custom_or_ros2(self, validator: ServiceValidator):
        """Test validation when service exists in neither custom models nor ROS 2"""
        args = {"some_field": "value"}

        with pytest.raises(ImportError):
            validator.validate("nonexistent_package/srv/NonexistentService", args)

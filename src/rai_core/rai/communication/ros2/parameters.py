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

from typing import Any

import rclpy
from rclpy.parameter import Parameter, ParameterType

from .exceptions import ROS2ParameterError

# TODO(juliaj): Re-evaluate whether this is over-engineering. Consider if the
# convenience of automatic type extraction justifies a separate module, or if
# direct ROS2 parameter access is sufficient.


def get_param_value(node: rclpy.node.Node, name: str, default: Any = None) -> Any:
    """Get parameter value from node with automatic type extraction.

    Args:
        node: ROS2 node.
        name: Parameter name.
        default: Default value if parameter not found.

    Returns:
        Parameter value as Python type, or default if not found.

    Raises:
        ROS2ParameterError: If parameter not found and no default provided.

    Example:
        Instead of:
            value = node.get_parameter("my_param").get_parameter_value().string_value
        Use:
            value = get_param_value(node, "my_param", default="")
    """
    if not node.has_parameter(name):
        if default is not None:
            return default
        raise ROS2ParameterError(
            param_name=name,
            suggestion=f"Parameter '{name}' not found and no default provided",
        )

    param = node.get_parameter(name)
    return _extract_param_value(param)


def _extract_param_value(param: Parameter) -> Any:
    """Extract Python value from ROS2 Parameter object."""
    param_value = param.get_parameter_value()
    # param.type_ is an enum (rclpy.parameter.Parameter.Type), get its integer value
    param_type = param.type_.value if hasattr(param.type_, "value") else param.type_

    if param_type == ParameterType.PARAMETER_BOOL:
        return param_value.bool_value
    elif param_type == ParameterType.PARAMETER_INTEGER:
        return param_value.integer_value
    elif param_type == ParameterType.PARAMETER_DOUBLE:
        return param_value.double_value
    elif param_type == ParameterType.PARAMETER_STRING:
        return param_value.string_value
    elif param_type == ParameterType.PARAMETER_BYTE_ARRAY:
        return list(param_value.byte_array_value)
    elif param_type == ParameterType.PARAMETER_BOOL_ARRAY:
        return list(param_value.bool_array_value)
    elif param_type == ParameterType.PARAMETER_INTEGER_ARRAY:
        return list(param_value.integer_array_value)
    elif param_type == ParameterType.PARAMETER_DOUBLE_ARRAY:
        return list(param_value.double_array_value)
    elif param_type == ParameterType.PARAMETER_STRING_ARRAY:
        return list(param_value.string_array_value)
    else:
        return None

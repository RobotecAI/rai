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


from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, SetEnvironmentVariable
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    # Launch argument to control legacy service name registration
    # Default: True (backward compatible - existing apps continue working)
    # Set to False for new apps using only new service names
    enable_legacy_arg = DeclareLaunchArgument(
        "enable_legacy_service_names",
        default_value="true",
        description="Enable legacy service names for backward compatibility (default: true)",
    )

    return LaunchDescription(
        [
            enable_legacy_arg,
            SetEnvironmentVariable(
                "ENABLE_LEGACY_SERVICE_NAMES",
                LaunchConfiguration("enable_legacy_service_names"),
            ),
            ExecuteProcess(
                cmd=["python", "run_perception_services.py"],
                cwd="src/rai_extensions/rai_perception/rai_perception/scripts",
                output="screen",
            ),
        ]
    )

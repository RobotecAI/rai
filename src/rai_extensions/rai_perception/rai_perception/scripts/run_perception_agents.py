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


import warnings


def main():
    """Deprecated: Use run_perception_services.py instead.

    This script is deprecated and will be removed in a future version.
    Use 'python -m rai_perception.scripts.run_perception_services' instead.
    """
    warnings.warn(
        "run_perception_agents.py is deprecated. "
        "Use run_perception_services.py instead. "
        "This script will be removed in a future version.",
        DeprecationWarning,
        stacklevel=2,
    )
    from rai_perception.scripts.run_perception_services import main as services_main

    services_main()


if __name__ == "__main__":
    main()

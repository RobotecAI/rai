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

from .basic_tasks import get_basic_tasks
from .custom_interfaces_tasks import get_custom_interfaces_tasks
from .manipulation_tasks import get_manipulation_tasks
from .navigation_tasks import get_navigation_tasks
from .spatial_reasoning_tasks import get_spatial_tasks

__all__ = [
    "get_basic_tasks",
    "get_custom_interfaces_tasks",
    "get_manipulation_tasks",
    "get_navigation_tasks",
    "get_spatial_tasks",
]

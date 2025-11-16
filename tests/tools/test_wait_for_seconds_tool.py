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

from unittest.mock import patch

from rai.tools.time import WaitForSecondsTool


def test_wait_for_seconds_tool_caps_duration():
    tool = WaitForSecondsTool()

    with patch("rai.tools.time.time.sleep") as mock_sleep:
        result = tool._run(15)

    mock_sleep.assert_called_once_with(10)
    assert result == "Waited for 10 seconds."

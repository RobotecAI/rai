# Copyright (C) 2026 Robotec.AI
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

from rai.tools.names import require_non_empty_name


@pytest.mark.parametrize("value", ["", "   ", None, 1, True])
def test_require_non_empty_name_rejects(value):
    with pytest.raises(ValueError, match="non-empty"):
        require_non_empty_name(value, name="topic")


def test_require_non_empty_name_strips():
    assert require_non_empty_name("  /chatter ", name="topic") == "/chatter"

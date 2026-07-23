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

import math

import pytest

from rai.tools.timeout import require_positive_timeout


@pytest.mark.parametrize("value", [0, -1, -0.01, True, False, "1", None, math.nan, object()])
def test_require_positive_timeout_rejects_invalid(value):
    with pytest.raises(ValueError, match="timeout"):
        require_positive_timeout(value)


@pytest.mark.parametrize("value,expected", [(1, 1.0), (0.5, 0.5), (2, 2.0)])
def test_require_positive_timeout_accepts_positive(value, expected):
    assert require_positive_timeout(value) == expected

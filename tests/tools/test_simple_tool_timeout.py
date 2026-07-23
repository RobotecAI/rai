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

from rai.tools.positive_params import require_positive_number


@pytest.mark.parametrize("bad", [0, -1, 0.0])
def test_simple_timeout_rejects_non_positive(bad):
    with pytest.raises(ValueError, match="timeout_sec must be positive"):
        require_positive_number(bad, name="timeout_sec")


def test_simple_timeout_rejects_bool():
    with pytest.raises(TypeError, match="timeout_sec must be a positive number"):
        require_positive_number(False, name="timeout_sec")  # type: ignore[arg-type]


def test_simple_timeout_accepts_positive():
    assert require_positive_number(5.0, name="timeout_sec") == 5.0

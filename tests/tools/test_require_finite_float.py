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

from rai.tools.numbers import require_finite_float


@pytest.mark.parametrize("value", [True, False, "1", None, math.nan, math.inf, -math.inf])
def test_require_finite_float_rejects(value):
    with pytest.raises(ValueError, match="finite"):
        require_finite_float(value, name="x")


def test_require_finite_float_accepts():
    assert require_finite_float(1.5, name="x") == 1.5
    assert require_finite_float(0, name="x") == 0.0

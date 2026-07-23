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

from rai_sim.positive_params import require_positive_number


@pytest.mark.parametrize("bad", [0, -1, 0.0])
def test_o3de_timeout_params_reject_non_positive(bad):
    with pytest.raises(ValueError, match="must be positive"):
        require_positive_number(bad, name="timeout")
    with pytest.raises(ValueError, match="must be positive"):
        require_positive_number(bad, name="stale_timeout")
    with pytest.raises(ValueError, match="must be positive"):
        require_positive_number(bad, name="poll_interval")


@pytest.mark.parametrize("bad", [True, False, "5", None])
def test_o3de_timeout_params_reject_types(bad):
    with pytest.raises(TypeError, match="must be a positive number"):
        require_positive_number(bad, name="timeout")  # type: ignore[arg-type]


def test_o3de_timeout_params_accept_positive():
    assert require_positive_number(15, name="timeout") == 15.0
    assert require_positive_number(0.5, name="poll_interval") == 0.5

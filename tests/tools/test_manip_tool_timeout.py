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
def test_manip_timeouts_reject_non_positive(bad):
    with pytest.raises(ValueError, match="must be positive"):
        require_positive_number(bad, name="timeout_sec")
    with pytest.raises(ValueError, match="must be positive"):
        require_positive_number(bad, name="service_availability_timeout_sec")


def test_manip_timeouts_accept_positive():
    assert require_positive_number(20.0, name="timeout_sec") == 20.0

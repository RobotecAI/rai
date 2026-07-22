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

from rai.communication.timeout_validate import require_positive_timeout


@pytest.mark.parametrize("bad", [0, -1, -0.5, 0.0])
def test_require_positive_timeout_rejects_non_positive(bad):
    with pytest.raises(ValueError, match="timeout_sec must be positive"):
        require_positive_timeout(bad)


def test_require_positive_timeout_rejects_bool():
    with pytest.raises(TypeError, match="timeout_sec must be a positive number"):
        require_positive_timeout(False)  # type: ignore[arg-type]


def test_require_positive_timeout_rejects_str():
    with pytest.raises(TypeError, match="timeout_sec must be a positive number"):
        require_positive_timeout("5")  # type: ignore[arg-type]


def test_require_positive_timeout_accepts_positive():
    assert require_positive_timeout(5) == 5.0
    assert require_positive_timeout(0.1) == 0.1

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

"""VAD and wake word thresholds come from config.toml, where an out-of-range
value silently turns detection permanently on or permanently off."""

import pytest

from rai_s2s.asr.agents.initialization import VADConfig, WWConfig


@pytest.mark.parametrize("config_cls", [VADConfig, WWConfig])
@pytest.mark.parametrize("threshold", [0.0, -0.1, 1.5])
def test_threshold_out_of_range_rejected(config_cls, threshold):
    with pytest.raises(ValueError, match="threshold"):
        config_cls(threshold=threshold)


def test_negative_silence_grace_period_rejected():
    with pytest.raises(ValueError, match="silence_grace_period"):
        VADConfig(silence_grace_period=-1.0)


def test_defaults_are_valid():
    assert VADConfig().threshold == 0.5
    assert WWConfig().threshold == 0.01

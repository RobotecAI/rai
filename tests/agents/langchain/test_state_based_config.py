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

import pytest
from pydantic import ValidationError
from rai.agents.langchain.state_based_agent import StateBasedConfig


def test_time_interval_must_be_positive():
    for bad in (0, -1.0, 0.0):
        with pytest.raises(ValidationError):
            StateBasedConfig(aggregators={}, time_interval=bad)


def test_max_workers_must_be_positive():
    for bad in (0, -1):
        with pytest.raises(ValidationError):
            StateBasedConfig(aggregators={}, max_workers=bad)


def test_defaults_and_positive_ok():
    cfg = StateBasedConfig(aggregators={})
    assert cfg.time_interval == 5.0
    assert cfg.max_workers == 8
    cfg2 = StateBasedConfig(aggregators={}, time_interval=0.1, max_workers=1)
    assert cfg2.time_interval == 0.1
    assert cfg2.max_workers == 1

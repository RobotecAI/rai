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
from langchain_core.messages import BaseMessage
from rai.aggregators.base import BaseAggregator


class DummyAggregator(BaseAggregator[int]):
    def get(self) -> BaseMessage | None:
        return None


def test_base_aggregator_rejects_nonpositive_max_size():
    with pytest.raises(ValueError, match="max_size must be positive"):
        DummyAggregator(max_size=0)
    with pytest.raises(ValueError, match="max_size must be positive"):
        DummyAggregator(max_size=-1)


def test_base_aggregator_accepts_none_and_positive_max_size():
    unbounded = DummyAggregator(max_size=None)
    assert unbounded.max_size is None
    for i in range(5):
        unbounded(i)
    assert unbounded.get_buffer() == [0, 1, 2, 3, 4]

    bounded = DummyAggregator(max_size=2)
    for i in range(5):
        bounded(i)
    assert bounded.get_buffer() == [3, 4]

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
from rai.agents.langchain.callback import HRICallbackHandler


def test_hri_callback_handler_rejects_nonpositive_max_buffer_size():
    with pytest.raises(ValueError, match="max_buffer_size must be positive"):
        HRICallbackHandler(connectors={}, max_buffer_size=0)
    with pytest.raises(ValueError, match="max_buffer_size must be positive"):
        HRICallbackHandler(connectors={}, max_buffer_size=-1)


def test_hri_callback_handler_rejects_non_int_max_buffer_size():
    with pytest.raises(TypeError, match="max_buffer_size must be a positive int"):
        HRICallbackHandler(connectors={}, max_buffer_size=False)
    with pytest.raises(TypeError, match="max_buffer_size must be a positive int"):
        HRICallbackHandler(connectors={}, max_buffer_size=1.5)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="max_buffer_size must be a positive int"):
        HRICallbackHandler(connectors={}, max_buffer_size="200")  # type: ignore[arg-type]


def test_hri_callback_handler_accepts_positive_max_buffer_size():
    handler = HRICallbackHandler(connectors={}, max_buffer_size=1)
    assert handler.max_buffer_size == 1
    default_handler = HRICallbackHandler(connectors={})
    assert default_handler.max_buffer_size == 200

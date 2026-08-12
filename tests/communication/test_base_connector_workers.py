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
from rai.communication.base_connector import BaseConnector, BaseMessage


class _Msg(BaseMessage):
    pass


class _Conn(BaseConnector[_Msg]):
    def send_message(self, message, target, **kwargs):
        pass

    def receive_message(self, source, timeout_sec, **kwargs):
        pass

    def service_call(self, message, target, timeout_sec, **kwargs):
        pass


@pytest.mark.parametrize("bad", [0, -1, 1.5, True, "4", None])
def test_callback_max_workers_rejects_non_positive(bad):
    with pytest.raises(ValueError, match="callback_max_workers must be a positive int"):
        _Conn(callback_max_workers=bad)  # type: ignore[arg-type]


def test_callback_max_workers_accepts_positive():
    c = _Conn(callback_max_workers=2)
    assert c.callback_max_workers == 2
    c.callback_executor.shutdown(wait=False)

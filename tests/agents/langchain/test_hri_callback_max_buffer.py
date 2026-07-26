# Copyright (C) 2026 Robotec.AI
import pytest

from rai.agents.langchain.callback import HRICallbackHandler


def test_hri_callback_rejects_non_positive_max_buffer_size():
    with pytest.raises(ValueError, match="max_buffer_size"):
        HRICallbackHandler(connectors={}, max_buffer_size=0)
    with pytest.raises(ValueError, match="max_buffer_size"):
        HRICallbackHandler(connectors={}, max_buffer_size=-5)
    with pytest.raises(ValueError, match="max_buffer_size"):
        HRICallbackHandler(connectors={}, max_buffer_size=True)  # type: ignore[arg-type]


def test_hri_callback_accepts_positive_max_buffer_size():
    h = HRICallbackHandler(connectors={}, max_buffer_size=16)
    assert h.max_buffer_size == 16

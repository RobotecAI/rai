# Copyright (C) 2026 Robotec.AI
from unittest.mock import MagicMock

import pytest

from rai.communication.ros2.context import NodeDiscovery


def test_node_discovery_rejects_non_positive_period_sec():
    node = MagicMock()
    with pytest.raises(ValueError, match="period_sec"):
        NodeDiscovery(node, period_sec=0)
    with pytest.raises(ValueError, match="period_sec"):
        NodeDiscovery(node, period_sec=-1.0)
    with pytest.raises(ValueError, match="period_sec"):
        NodeDiscovery(node, period_sec=float("nan"))


def test_node_discovery_accepts_positive_period_sec():
    node = MagicMock()
    d = NodeDiscovery(node, period_sec=0.25)
    assert d.period_sec == 0.25
    node.create_timer.assert_called()

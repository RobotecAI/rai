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

"""Tests for the ``context`` parameter of ROS2BaseConnector / ROS2Connector.

The parameter accepts three forms:

- ``None``            – use the process-global default context (existing behaviour).
- ``int``             – domain-ID shorthand; connector creates and owns a fresh
                        ``rclpy.Context`` for that domain.
- ``rclpy.Context``   – caller-supplied, already-initialised context; connector
                        does not manage its lifecycle.

Each form is tested for:
  * correct internal state (``_rclpy_context``, ``_owns_context``)
  * correct wiring of the Node and executor to the resolved context
  * correct lifecycle behaviour on ``connector.shutdown()``
  * isolation / coexistence between multiple connectors
"""

from typing import Generator
from unittest.mock import MagicMock, patch

import pytest
import rclpy
import rclpy.context
from rai.communication.ros2 import ROS2Connector

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def ros_setup() -> Generator[None, None, None]:
    """Module-scoped global rclpy initialisation for context=None tests."""
    rclpy.init()
    yield
    rclpy.shutdown()


_ = ros_setup  # keep linters quiet about "unused import"


@pytest.fixture
def external_context() -> Generator[rclpy.Context, None, None]:
    """Provides a function-scoped, already-initialised rclpy.Context on domain 60.

    The connector under test must NOT shut this down — the fixture owns it.
    """
    ctx = rclpy.Context()
    rclpy.init(context=ctx, domain_id=60)
    yield ctx
    if ctx.ok():
        rclpy.shutdown(context=ctx)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Patches applied when testing the auto-init code path.  We replace all ROS2
# infrastructure with mocks so the test does not need a live ROS2 environment.
_INFRA_PATCHES = [
    "rai.communication.ros2.connectors.base.Node",
    "rai.communication.ros2.connectors.base.ROS2TopicAPI",
    "rai.communication.ros2.connectors.base.ROS2ServiceAPI",
    "rai.communication.ros2.connectors.base.ROS2ActionAPI",
    "rai.communication.ros2.connectors.base.Buffer",
    "rai.communication.ros2.connectors.base.TransformListener",
    "rai.communication.ros2.connectors.base.MultiThreadedExecutor",
    "threading.Thread",
]


# ---------------------------------------------------------------------------
# context=int  — connector creates and OWNS the context
# ---------------------------------------------------------------------------


def test_context_int_creates_rclpy_context():
    """Passing an int creates a new rclpy.Context instance."""
    conn = ROS2Connector(context=40)
    try:
        assert isinstance(conn._rclpy_context, rclpy.Context)
    finally:
        conn.shutdown()


def test_context_int_owns_context():
    """Passing an int sets _owns_context=True."""
    conn = ROS2Connector(context=41)
    try:
        assert conn._owns_context is True
    finally:
        conn.shutdown()


def test_context_int_context_is_initialized():
    """The created context is live (ok() returns True) during connector lifetime."""
    conn = ROS2Connector(context=42)
    try:
        assert conn._rclpy_context.ok() is True
    finally:
        conn.shutdown()


def test_context_int_node_participates_in_created_context():
    """The connector's Node is created with the owned context."""
    conn = ROS2Connector(context=43)
    try:
        assert conn.node.context is conn._rclpy_context
    finally:
        conn.shutdown()


def test_context_int_executor_participates_in_created_context():
    """The executor is created with the owned context."""
    conn = ROS2Connector(context=44)
    try:
        assert conn._executor._context is conn._rclpy_context
    finally:
        conn.shutdown()


def test_context_int_shutdown_destroys_context():
    """shutdown() shuts down the owned context when _owns_context=True."""
    conn = ROS2Connector(context=45)
    ctx = conn._rclpy_context
    assert ctx.ok() is True
    conn.shutdown()
    assert ctx.ok() is False


def test_context_int_multiple_domains_coexist():
    """Two connectors on different domain IDs hold independent contexts, both live."""
    conn_a = ROS2Connector(context=46)
    conn_b = ROS2Connector(context=47)
    try:
        assert conn_a._rclpy_context is not conn_b._rclpy_context
        assert conn_a._rclpy_context.ok() is True
        assert conn_b._rclpy_context.ok() is True
    finally:
        conn_a.shutdown()
        conn_b.shutdown()


def test_context_int_independent_shutdown():
    """Shutting down one connector does not affect the other's context."""
    conn_a = ROS2Connector(context=48)
    conn_b = ROS2Connector(context=49)
    ctx_a = conn_a._rclpy_context
    ctx_b = conn_b._rclpy_context
    try:
        conn_a.shutdown()
        assert ctx_a.ok() is False  # A's context gone
        assert ctx_b.ok() is True  # B's context untouched
    finally:
        conn_b.shutdown()


# ---------------------------------------------------------------------------
# context=rclpy.Context  — connector BORROWS the context
# ---------------------------------------------------------------------------


def test_context_object_stores_provided_context(external_context):
    """_rclpy_context is the exact object passed in."""
    conn = ROS2Connector(context=external_context)
    try:
        assert conn._rclpy_context is external_context
    finally:
        conn.shutdown()


def test_context_object_does_not_own_context(external_context):
    """Passing an rclpy.Context sets _owns_context=False."""
    conn = ROS2Connector(context=external_context)
    try:
        assert conn._owns_context is False
    finally:
        conn.shutdown()


def test_context_object_node_uses_provided_context(external_context):
    """The connector's Node is wired to the supplied context."""
    conn = ROS2Connector(context=external_context)
    try:
        assert conn.node.context is external_context
    finally:
        conn.shutdown()


def test_context_object_executor_uses_provided_context(external_context):
    """The executor is wired to the supplied context."""
    conn = ROS2Connector(context=external_context)
    try:
        assert conn._executor._context is external_context
    finally:
        conn.shutdown()


def test_context_object_shutdown_preserves_context(external_context):
    """shutdown() does NOT shut down a caller-owned context."""
    conn = ROS2Connector(context=external_context)
    conn.shutdown()
    assert external_context.ok() is True  # fixture-owned context is still live


def test_context_object_multiple_connectors_share_one_context():
    """Two connectors can share the same rclpy.Context; it survives both shutdowns."""
    ctx = rclpy.Context()
    rclpy.init(context=ctx, domain_id=61)
    conn_a = ROS2Connector(context=ctx)
    conn_b = ROS2Connector(context=ctx)
    try:
        assert conn_a._rclpy_context is ctx
        assert conn_b._rclpy_context is ctx
        assert conn_a.node.context is conn_b.node.context
        conn_a.shutdown()
        assert ctx.ok() is True  # context survives first shutdown
        conn_b.shutdown()
        assert ctx.ok() is True  # context survives second shutdown
    finally:
        if ctx.ok():
            rclpy.shutdown(context=ctx)


# ---------------------------------------------------------------------------
# context=None  — connector uses the global default context
# ---------------------------------------------------------------------------


def test_context_none_rclpy_context_is_none(ros_setup):
    """context=None leaves _rclpy_context as None (uses global default)."""
    conn = ROS2Connector()
    try:
        assert conn._rclpy_context is None
    finally:
        conn.shutdown()


def test_context_none_does_not_own_context(ros_setup):
    """context=None sets _owns_context=False."""
    conn = ROS2Connector()
    try:
        assert conn._owns_context is False
    finally:
        conn.shutdown()


def test_context_none_shutdown_does_not_affect_global_rclpy(ros_setup):
    """shutdown() must not call rclpy.shutdown() when the context is not owned."""
    conn = ROS2Connector()
    conn.shutdown()
    assert rclpy.ok() is True  # module-scoped rclpy still alive


# ---------------------------------------------------------------------------
# context=None  — auto-init path (rclpy not yet initialised)
#
# These tests mock all ROS2 infrastructure so they run without a live ROS2
# environment and can force the "rclpy not initialised" code path.
# ---------------------------------------------------------------------------


def test_context_none_auto_init_calls_rclpy_init():
    """When rclpy is not initialised, context=None triggers rclpy.init()."""
    with (
        patch.object(rclpy, "ok", return_value=False),
        patch.object(rclpy, "init") as mock_init,
    ):
        with patch.multiple(
            "rai.communication.ros2.connectors.base",
            **{p.split(".")[-1]: MagicMock() for p in _INFRA_PATCHES if "base" in p},
        ):
            with patch("threading.Thread"):
                ROS2Connector()
                mock_init.assert_called_once_with()


def test_context_none_auto_init_logs_warning(capsys):
    """When rclpy auto-initialises, a warning containing 'Auto-initializing' is written."""
    # The connector logger uses logging.lastResort (stderr) when no handlers are
    # configured, so we capture stderr rather than using caplog.
    with patch.object(rclpy, "ok", return_value=False), patch.object(rclpy, "init"):
        with patch.multiple(
            "rai.communication.ros2.connectors.base",
            **{p.split(".")[-1]: MagicMock() for p in _INFRA_PATCHES if "base" in p},
        ):
            with patch("threading.Thread"):
                ROS2Connector()

    captured = capsys.readouterr()
    assert "Auto-initializing" in captured.err, (
        f"Expected auto-init warning in stderr, got: {captured.err!r}"
    )

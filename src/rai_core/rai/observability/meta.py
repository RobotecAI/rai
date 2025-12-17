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

# Observability metaclass instrumentation.
#
# This module provides `ObservabilityMeta`, a metaclass that wraps a selected set
# of connector-like methods (send/receive/service/action) to emit lightweight
# observability events. The wrapping is centralized here so we get consistent
# timing/error signals across connectors without having to modify each method
# implementation.

import functools
import time
from typing import Any, Callable, Dict

from .correlation import get_observability_context
from .sink import NoOpSink

EVENT_SCHEMA_VERSION = "v1"
_OBS_WRAPPED_ATTR = "__rai_observability_wrapped__"


def _extract_target(
    fn_name: str, args: tuple[Any, ...], kwargs: dict[str, Any]
) -> dict[str, Any]:
    """Extract common fields like target/source from known method signatures."""
    fields: dict[str, Any] = {}
    if fn_name in {
        "send_message",
        "service_call",
        "call_service",
        "start_action",
        "terminate_action",
    }:
        # target is usually the second positional argument or a kwarg named target
        if "target" in kwargs:
            fields["target"] = kwargs["target"]
        elif len(args) >= 2:
            fields["target"] = args[1]
    if fn_name in {"receive_message"}:
        # source is usually the first positional argument or kwarg named source
        if "source" in kwargs:
            fields["source"] = kwargs["source"]
        elif len(args) >= 1:
            fields["source"] = args[0]
    if fn_name in {"create_service", "create_action"}:
        if "service_name" in kwargs:
            fields["target"] = kwargs["service_name"]
        elif "action_name" in kwargs:
            fields["target"] = kwargs["action_name"]
        elif len(args) >= 1:
            fields["target"] = args[0]
    return fields


def _timed_handler(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    # We capture both:
    # - wall clock time (useful for ordering events on a timeline)
    # - monotonic clock (safe for measuring latency even if wall clock jumps)
    started_wall = time.time()
    started_perf = time.perf_counter()
    status = "ok"
    error_type: str | None = None
    error_msg: str | None = None
    try:
        return fn(self, *args, **kwargs)
    except Exception as exc:
        # Record the error details, but keep behavior identical to original code:
        # we re-raise the exception to the caller.
        status = "error"
        error_type = type(exc).__name__
        error_msg = str(exc)
        raise
    finally:
        # Best-effort event emission: sink failures must never affect user code.
        sink = getattr(self, "observability_sink", None) or NoOpSink()
        connector_name = getattr(self, "connector_name", None)
        agent_name = getattr(self, "agent_name", None)
        try:
            event = {
                "schema_version": EVENT_SCHEMA_VERSION,
                "event_type": fn.__name__,
                "phase": "close",
                "timestamp_s": started_wall,
                "latency_ms": (time.perf_counter() - started_perf) * 1000.0,
                "status": status,
                "component": agent_name,
                "connector_name": connector_name,
            }
            if agent_name:
                event["agent_name"] = agent_name
            # Add lightweight routing hints for topology reconstruction.
            event.update(_extract_target(fn.__name__, args, kwargs))
            # Inject correlation IDs (run/job/task/request) stored in contextvars.
            event.update(get_observability_context())
            if status != "ok":
                event["error_type"] = error_type
                event["error_msg"] = error_msg
            sink.record(event)
        except Exception:
            # Best-effort: never raise into caller.
            pass


HANDLERS: Dict[str, Dict[str, Callable[..., Any]]] = {
    EVENT_SCHEMA_VERSION: {
        "send_message": _timed_handler,
        "receive_message": _timed_handler,
        "service_call": _timed_handler,
        "call_service": _timed_handler,
        "create_service": _timed_handler,
        "create_action": _timed_handler,
        "start_action": _timed_handler,
        "terminate_action": _timed_handler,
    }
}

DEFAULT_METHODS = tuple(HANDLERS[EVENT_SCHEMA_VERSION].keys())


class ObservabilityMeta(type):
    """Metaclass that wraps selected methods with observability handlers."""

    def __new__(mcls, name, bases, attrs):
        cls = super().__new__(mcls, name, bases, attrs)
        methods = getattr(cls, "__observability_methods__", DEFAULT_METHODS)
        schema_version = getattr(
            cls, "__observability_schema_version__", EVENT_SCHEMA_VERSION
        )
        handler_map = HANDLERS.get(schema_version, {})

        for method_name in methods:
            fn = getattr(cls, method_name, None)
            handler = handler_map.get(method_name)
            if not fn or not handler:
                continue
            # Avoid double-wrapping inherited methods. This can happen when a base
            # class is created with this metaclass and a subclass is created with
            # the same metaclass; wrapping twice increases overhead and can affect
            # timing-sensitive ROS2 tests.
            #
            # Root cause of a unit test failure we observed:
            # - `ROS2Connector` inherits from `BaseConnector` (metaclass=ObservabilityMeta).
            # - When `ROS2Connector` is created, this metaclass runs again and used to
            #   wrap methods that were already wrapped on the base class.
            # - That produced nested wrappers:
            #     wrapper -> _timed_handler -> wrapper -> _timed_handler -> real method
            # - The extra overhead/latency was enough to trip tight ROS2 timeouts,
            #   surfacing as "goal not accepted"/future timed out in action tests.
            #
            # Example failing test:
            # - `tests/tools/ros2/test_action_tools.py::test_action_call_tool_with_writable_action`
            #
            # Representative call stack excerpt:
            # - src/rai_core/rai/tools/ros2/generic/actions.py: StartROS2ActionTool._run
            # - src/rai_core/rai/observability/meta.py: wrapper
            # - src/rai_core/rai/observability/meta.py: _timed_handler
            # - src/rai_core/rai/observability/meta.py: wrapper   (second time)
            # - src/rai_core/rai/observability/meta.py: _timed_handler (second time)
            # - src/rai_core/rai/communication/ros2/connectors/action_mixin.py: ROS2Connector.start_action
            if getattr(fn, _OBS_WRAPPED_ATTR, False):
                continue

            @functools.wraps(fn)
            def wrapper(self, *args, __fn=fn, __handler=handler, **kwargs):
                # Delegate to the configured handler; by default it measures latency
                # and emits a single "close" event via `observability_sink`.
                return __handler(self, __fn, *args, **kwargs)

            setattr(wrapper, _OBS_WRAPPED_ATTR, True)
            setattr(cls, method_name, wrapper)
        return cls

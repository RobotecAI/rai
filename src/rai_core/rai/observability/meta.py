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

import functools
import time
from typing import Any, Callable, Dict

from .sink import NoOpSink

EVENT_SCHEMA_VERSION = "v1"


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
    started = time.time()
    try:
        return fn(self, *args, **kwargs)
    finally:
        sink = getattr(self, "observability_sink", None) or NoOpSink()
        connector_name = getattr(self, "connector_name", None)
        agent_name = getattr(self, "agent_name", None)
        try:
            event = {
                "schema_version": EVENT_SCHEMA_VERSION,
                "event_type": fn.__name__,
                "phase": "close",
                "latency_ms": (time.time() - started) * 1000.0,
                "component": agent_name,
                "connector_name": connector_name,
            }
            if agent_name:
                event["agent_name"] = agent_name
            event.update(_extract_target(fn.__name__, args, kwargs))
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

            @functools.wraps(fn)
            def wrapper(self, *args, __fn=fn, __handler=handler, **kwargs):
                return __handler(self, __fn, *args, **kwargs)

            setattr(cls, method_name, wrapper)
        return cls

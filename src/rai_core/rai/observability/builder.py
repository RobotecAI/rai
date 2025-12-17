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

import logging
import os
from typing import Callable, Mapping, Optional
from urllib.parse import urlparse

from .sink import (
    BufferedSink,
    LoggingSink,
    NoOpSink,
    ObservabilitySink,
    StdoutSink,
    default_buffer_size,
)

LOGGER = logging.getLogger("ObservabilityBuilder")


def _make_logging_sink(_endpoint: str) -> ObservabilitySink:
    return LoggingSink(logger=logging.getLogger("ObservabilityLoggingSink"))


def _make_stdout_sink(_endpoint: str) -> ObservabilitySink:
    return StdoutSink()


DEFAULT_FACTORY: Mapping[str, Callable[[str], ObservabilitySink]] = {
    "ws": _make_stdout_sink,  # visible by default for local debugging
    "wss": _make_logging_sink,
    "tcp": _make_logging_sink,
    "http": _make_logging_sink,
    "https": _make_logging_sink,
    "file": _make_logging_sink,
}


def build_sink_from_env(
    endpoint: Optional[str] = None,
    buffer_size: Optional[int] = None,
    factory: Mapping[str, Callable[[str], ObservabilitySink]] = DEFAULT_FACTORY,
) -> ObservabilitySink:
    """Build an observability sink from configuration.

    If no endpoint is provided or parsing fails, falls back to NoOpSink.
    """
    raw_target = endpoint or os.getenv("RAI_OBS_ENDPOINT")
    if not raw_target:
        return NoOpSink()

    # Accept scheme-less values like "ws" to mean "use the ws factory".
    parsed = urlparse(raw_target)
    if not parsed.scheme and raw_target in factory:
        target_scheme = raw_target
        target_full = raw_target
    else:
        target_scheme = parsed.scheme
        target_full = raw_target

    if not target_scheme:
        LOGGER.debug(
            "Observability endpoint missing scheme and not recognized: %s; using NoOpSink",
            raw_target,
        )
        return NoOpSink()

    factory_fn = factory.get(target_scheme)
    if not factory_fn:
        LOGGER.debug(
            "Observability endpoint scheme not recognized (%s), using NoOpSink",
            target_scheme,
        )
        return NoOpSink()

    try:
        sink = factory_fn(target_full)
    except Exception as exc:  # pragma: no cover - defensive
        LOGGER.debug("Failed to create sink for %s: %s", target_full, exc)
        return NoOpSink()

    buf_size = buffer_size if buffer_size is not None else default_buffer_size()
    if buf_size and buf_size > 0:
        wrapped = BufferedSink(
            sink, maxlen=buf_size, logger=logging.getLogger("BufferedSink")
        )
        wrapped.start()
        return wrapped
    if hasattr(sink, "start"):
        try:
            sink.start()
        except Exception:
            pass
    return sink

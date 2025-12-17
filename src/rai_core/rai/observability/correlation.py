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

from __future__ import annotations

import contextlib
from contextvars import ContextVar, Token
from typing import Any, Dict, Iterator, Optional

# Design summary:
# - Purpose: carry lightweight correlation identifiers (run/job/task/request) across
#   nested calls so observability events can be stitched into traces.
# - Mechanism: `contextvars` stores IDs per logical execution context (thread/task),
#   so concurrent runs do not overwrite each other.
# - API:
#   - `observability_context(...)`: preferred; scopes IDs to a `with` block and
#     reliably restores previous values on exit (even on exceptions).
#   - `set_observability_context(...)` / `reset_observability_context(...)`: lower-level
#     primitives for advanced cases; returns tokens that must be reset.
#   - `get_observability_context()`: best-effort snapshot of currently set IDs to
#     merge into emitted observability events.


# Module-level `ContextVar`s are intentional: they act as process-wide singletons
# (one canonical "slot" per ID) so values propagate per async-task/thread context
# and are shared consistently across all importing modules.
_run_id: ContextVar[Optional[str]] = ContextVar("rai_obs_run_id", default=None)
_job_id: ContextVar[Optional[str]] = ContextVar("rai_obs_job_id", default=None)
_task_id: ContextVar[Optional[str]] = ContextVar("rai_obs_task_id", default=None)
_request_id: ContextVar[Optional[str]] = ContextVar("rai_obs_request_id", default=None)


def get_observability_context() -> Dict[str, Any]:
    """Return the current correlation identifiers (best-effort)."""
    ctx: Dict[str, Any] = {}
    if (v := _run_id.get()) is not None:
        ctx["run_id"] = v
    if (v := _job_id.get()) is not None:
        ctx["job_id"] = v
    if (v := _task_id.get()) is not None:
        ctx["task_id"] = v
    if (v := _request_id.get()) is not None:
        ctx["request_id"] = v
    return ctx


def set_observability_context(
    *,
    run_id: Optional[str] = None,
    job_id: Optional[str] = None,
    task_id: Optional[str] = None,
    request_id: Optional[str] = None,
) -> Dict[str, Token[Optional[str]]]:
    """Set any provided IDs into contextvars. Returns tokens to restore."""
    tokens: Dict[str, Token[Optional[str]]] = {}
    if run_id is not None:
        tokens["run_id"] = _run_id.set(run_id)
    if job_id is not None:
        tokens["job_id"] = _job_id.set(job_id)
    if task_id is not None:
        tokens["task_id"] = _task_id.set(task_id)
    if request_id is not None:
        tokens["request_id"] = _request_id.set(request_id)
    return tokens


def reset_observability_context(tokens: Dict[str, Token[Optional[str]]]) -> None:
    """Reset contextvars using tokens from set_observability_context()."""
    tok = tokens.get("run_id")
    if tok is not None:
        _run_id.reset(tok)
    tok = tokens.get("job_id")
    if tok is not None:
        _job_id.reset(tok)
    tok = tokens.get("task_id")
    if tok is not None:
        _task_id.reset(tok)
    tok = tokens.get("request_id")
    if tok is not None:
        _request_id.reset(tok)


@contextlib.contextmanager
def observability_context(
    *,
    run_id: Optional[str] = None,
    job_id: Optional[str] = None,
    task_id: Optional[str] = None,
    request_id: Optional[str] = None,
) -> Iterator[None]:
    """Context manager to scope correlation IDs for emitted observability events."""
    tokens = set_observability_context(
        run_id=run_id, job_id=job_id, task_id=task_id, request_id=request_id
    )
    try:
        yield
    finally:
        reset_observability_context(tokens)

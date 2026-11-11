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


"""The client side of learned-policy (VLA) servers: one async protocol, observation
dict in, action chunk out, and the errors every client raises."""

import importlib
from collections.abc import Mapping
from types import ModuleType
from typing import Any, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

Observation = Mapping[str, Any]
"""Flat policy input: names to numpy arrays (images, states) or strings (prompt)."""

ActionChunk = NDArray[np.floating]
"""Shape (T, D): T future actions of D dimensions, executed in order."""


class PolicyError(RuntimeError):
    """The policy server failed, refused, or sent something that is not an action chunk."""


class PolicyConnectionError(PolicyError):
    """The policy server cannot be reached, or dropped the connection."""


class PolicyTimeoutError(PolicyError, TimeoutError):
    """The policy server did not answer in time."""


@runtime_checkable
class PolicyClient(Protocol):
    """A remote policy. Implementations reconnect by themselves: after an error or
    a cancelled call, the next `infer` opens a fresh connection."""

    async def infer(self, observation: Observation) -> ActionChunk:
        """One action chunk for one observation. Raises PolicyError."""
        ...

    async def reset(self) -> None:
        """Clear per-episode server state (no-op where the protocol has none)."""
        ...

    async def close(self) -> None: ...


def require(module: str) -> ModuleType:
    """Import an optional dependency of the `policies` extra."""
    try:
        return importlib.import_module(module)
    except ImportError as e:
        raise ImportError(
            f"{module} is missing: install the policies extra, "
            "e.g. pip install 'rai_core[policies]'"
        ) from e


def as_chunk(actions: Any) -> ActionChunk:
    """A (T, D) float array from a policy answer: (D,) becomes (1, D), a leading
    batch axis of 1 is dropped."""
    chunk = np.asarray(actions, dtype=np.float64)
    if chunk.ndim == 3 and chunk.shape[0] == 1:
        chunk = chunk[0]
    if chunk.ndim == 1:
        chunk = chunk[None, :]
    if chunk.ndim != 2 or chunk.shape[0] == 0:
        raise PolicyError(f"Expected an action chunk (T, D), got shape {chunk.shape}")
    if not np.all(np.isfinite(chunk)):
        raise PolicyError("The action chunk has NaN or infinite values")
    return chunk

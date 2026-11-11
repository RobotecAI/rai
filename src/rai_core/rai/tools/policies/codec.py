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


"""msgpack with numpy arrays, in the two encodings policy servers use."""

import io
from typing import Any

import numpy as np

from .client import require

# ---------------------------------------------------------------- openpi


def _openpi_default(obj: Any) -> Any:
    if isinstance(obj, np.ndarray | np.generic) and obj.dtype.kind in "VOc":
        raise ValueError(f"Unsupported dtype: {obj.dtype}")
    if isinstance(obj, np.ndarray):
        obj = np.ascontiguousarray(obj)
        return {
            b"__ndarray__": True,
            b"data": obj.tobytes(),
            b"dtype": obj.dtype.str,
            b"shape": obj.shape,
        }
    if isinstance(obj, np.generic):
        return {b"__npgeneric__": True, b"data": obj.item(), b"dtype": obj.dtype.str}
    raise TypeError(f"Cannot serialize {type(obj).__name__}")


def _openpi_hook(obj: dict[Any, Any]) -> Any:
    if b"__ndarray__" in obj:
        return np.ndarray(
            buffer=obj[b"data"], dtype=np.dtype(obj[b"dtype"]), shape=obj[b"shape"]
        )
    if b"__npgeneric__" in obj:
        return np.dtype(obj[b"dtype"]).type(obj[b"data"])
    return obj


def pack_openpi(obj: Any) -> bytes:
    """openpi_client.msgpack_numpy: arrays as {b"__ndarray__", b"data", b"dtype", b"shape"}."""
    return require("msgpack").packb(obj, default=_openpi_default)


def unpack_openpi(data: bytes) -> Any:
    return require("msgpack").unpackb(
        data, object_hook=_openpi_hook, strict_map_key=False
    )


# ---------------------------------------------------------------- GR00T


def _gr00t_default(obj: Any) -> Any:
    if isinstance(obj, np.ndarray | np.generic):
        if obj.dtype.kind == "O":
            raise ValueError("Object arrays cannot be sent")
        out = io.BytesIO()
        np.save(out, np.asarray(obj), allow_pickle=False)
        return {"__ndarray_class__": True, "as_npy": out.getvalue()}
    raise TypeError(f"Cannot serialize {type(obj).__name__}")


def _gr00t_hook(obj: dict[Any, Any]) -> Any:
    if obj.get("__ndarray_class__"):
        return np.load(io.BytesIO(obj["as_npy"]), allow_pickle=False)
    if obj.get(b"nd") is True:  # msgpack_numpy, chained behind it on N1.7
        dtype = obj[b"type"]
        dtype = np.dtype(dtype.decode() if isinstance(dtype, bytes) else dtype)
        return np.frombuffer(obj[b"data"], dtype=dtype).reshape(obj[b"shape"])
    return obj


def pack_gr00t(obj: Any) -> bytes:
    """Isaac-GR00T MsgSerializer: arrays as {"__ndarray_class__", "as_npy": np.save bytes}."""
    return require("msgpack").packb(obj, default=_gr00t_default)


def unpack_gr00t(data: bytes) -> Any:
    return require("msgpack").unpackb(data, object_hook=_gr00t_hook, raw=False)

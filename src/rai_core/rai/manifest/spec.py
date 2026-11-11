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

"""The static half of the manifest: a per-robot spec file, rai_whoami's output and
the `[manifest]` table of `rai.toml`."""

import json
import logging
import os
import re
import tomllib
from pathlib import Path
from typing import Annotated, Any

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, model_validator

from rai.settings import section

logger = logging.getLogger(__name__)

_NAME = re.compile(r"^[~/]?[A-Za-z_{][A-Za-z0-9_{}]*(/[A-Za-z_{][A-Za-z0-9_{}]*)*$")
_PATTERN = re.compile(r"^[A-Za-z0-9_/~{}*?\[\]!]+$")


def _ros_name(name: str) -> str:
    if not _NAME.match(name):
        raise ValueError(f"{name!r} is not a valid ROS 2 name")
    return name


def _pattern(pattern: str) -> str:
    if not _PATTERN.match(pattern):
        raise ValueError(f"{pattern!r} is not a name pattern (fnmatch syntax)")
    return pattern


def _interface_type(name: str) -> str:
    if not re.match(r"^\w+/(msg|srv|action)/\w+$", name):
        raise ValueError(f"{name!r} is not a 'pkg/msg|srv|action/Name' type")
    return name


RosName = Annotated[str, AfterValidator(_ros_name)]
NamePattern = Annotated[str, AfterValidator(_pattern)]
InterfaceType = Annotated[str, AfterValidator(_interface_type)]


class Capability(BaseModel):
    """Something the robot can do, and the interfaces that do it."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    description: str = ""
    topics: list[RosName] = []
    services: list[RosName] = []
    actions: list[RosName] = []
    examples: list[str] = []


class RobotSpec(BaseModel):
    """A per-robot spec file (TOML). Every field is optional."""

    model_config = ConfigDict(extra="forbid")

    name: str | None = None
    description: str | None = None
    # role -> frame, e.g. {base = "base_link", map = "map"}
    frames: dict[str, str] = {}
    # e.g. {max_linear_speed_mps = 0.22}; put the unit in the key
    limits: dict[str, float | int | str | bool] = {}
    safety: list[str] = []
    capabilities: list[Capability] = []
    # graph names to leave out of the manifest (fnmatch on resolved names)
    hide: list[NamePattern] = []
    # graph names the model must not use; advisory: enforce with tool
    # `forbidden` lists or egress hooks
    forbidden: list[NamePattern] = []
    # types that move hardware on this robot, on top of egress.MOTION
    motion_types: list[InterfaceType] = []

    @model_validator(mode="after")
    def _unique_capabilities(self) -> "RobotSpec":
        names = [c.name for c in self.capabilities]
        duplicates = sorted({n for n in names if names.count(n) > 1})
        if duplicates:
            raise ValueError(f"Duplicate capability names: {duplicates}")
        return self


def load_spec(path: str | os.PathLike[str]) -> RobotSpec:
    """Read and validate a spec file; raises on a missing or invalid file."""
    with Path(path).open("rb") as f:
        return RobotSpec.model_validate(tomllib.load(f))


class WhoamiInfo(BaseModel):
    """The text half of rai_whoami's `EmbodimentInfo` (images are left out)."""

    model_config = ConfigDict(extra="ignore")

    description: str | None = None
    rules: list[str] | None = None
    capabilities: list[str] | None = None
    behaviors: list[str] | None = None


def load_whoami(source: Any) -> WhoamiInfo | None:
    """rai_whoami's description, or None when there is none.

    `source` is an `EmbodimentInfo`, its `info.json`, or a rai_whoami
    directory (`<dir>/generated/info.json`). The file format is read directly,
    so rai_whoami need not be installed. A missing or broken source logs a
    warning and gives None: the manifest is built without it.
    """
    if source is None:
        return None
    if isinstance(source, WhoamiInfo):
        return source
    if hasattr(source, "model_dump"):
        return WhoamiInfo.model_validate(source.model_dump(exclude={"images"}))
    path = Path(source)
    if path.is_dir():
        candidates = [path / "generated" / "info.json", path / "info.json"]
        path = next((p for p in candidates if p.is_file()), candidates[0])
    try:
        return WhoamiInfo.model_validate(json.loads(path.read_text()))
    except (OSError, ValueError) as e:
        logger.warning("No rai_whoami description from %s: %s", source, e)
        return None


class ManifestSettings(BaseModel):
    """The `[manifest]` table of `rai.toml`."""

    model_config = ConfigDict(extra="forbid")

    spec: Path | None = None  # the robot spec file (TOML)
    whoami: Path | None = None  # rai_whoami directory or info.json
    max_chars: int = Field(6000, ge=200)  # rendered size bound, about 4 chars per token
    include: list[NamePattern] = []  # keep only matching graph names (empty: all)
    hide: list[NamePattern] = []  # added to the spec's hide
    nodes: bool = True  # list node names


def manifest_settings(path: str | os.PathLike[str] | None = None) -> ManifestSettings:
    """The `[manifest]` table; `RAI_MANIFEST__<FIELD>` env vars override it."""
    return section("manifest", ManifestSettings, path)

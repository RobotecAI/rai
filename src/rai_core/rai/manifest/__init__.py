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

"""The live robot capability manifest: the ROS 2 graph, a per-robot spec file and
rai_whoami's description, for the model's prompt and as an MCP resource.

`rai.manifest.mcp:setup` is a `rai mcp serve --setup` function (needs the
`rai-core[mcp]` extra).
"""

import importlib

_LAZY = {
    "Capability": "rai.manifest.spec",
    "ManifestSettings": "rai.manifest.spec",
    "RobotSpec": "rai.manifest.spec",
    "WhoamiInfo": "rai.manifest.spec",
    "load_spec": "rai.manifest.spec",
    "load_whoami": "rai.manifest.spec",
    "manifest_settings": "rai.manifest.spec",
    "RawGraph": "rai.manifest.graph",
    "is_noise": "rai.manifest.graph",
    "query_graph": "rai.manifest.graph",
    "CapabilityStatus": "rai.manifest.manifest",
    "Entry": "rai.manifest.manifest",
    "ManifestSource": "rai.manifest.manifest",
    "RobotManifest": "rai.manifest.manifest",
    "build_manifest": "rai.manifest.manifest",
    "GetRobotManifestTool": "rai.manifest.agent",
    "ManifestPrompt": "rai.manifest.agent",
    "with_manifest": "rai.manifest.agent",
}

__all__ = [*_LAZY]


def __getattr__(name: str):
    if name in _LAZY:
        return getattr(importlib.import_module(_LAZY[name]), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

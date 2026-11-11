# Copyright (C) 2026 Robotec.AI
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

"""TF health: the frame trees, stale transforms and missing links."""

import asyncio
from typing import Any, Dict, List, Optional, Type

from pydantic import BaseModel, Field, PrivateAttr

from rai.tools.hooks import Effect
from rai.tools.ros2.base import BaseROS2Tool
from rai.tools.ros2.diagnostics.common import bounded, now_s, seconds

MAX_FRAMES = 150


class GetROS2TFTreeInput(BaseModel):
    source_frame: Optional[str] = Field(
        None, description="With target_frame: check the chain between the two"
    )
    target_frame: Optional[str] = Field(None, description="See source_frame")
    stale_after_s: float = Field(
        1.0, gt=0, description="A dynamic transform older than this is STALE"
    )


class GetROS2TFTreeTool(BaseROS2Tool):
    """The TF listener starts at the first call (the first answer waits
    warmup_s for transforms to arrive). Static transforms never go stale."""

    default_effect = Effect.READ
    default_targets = ("/tf", "/tf_static")
    name: str = "get_ros2_tf_tree"
    description: str = (
        "TF health: the frame tree with each transform's rate and age, stale"
        " transforms, and for a frame pair the chain between them or the missing link"
    )
    args_schema: Type[GetROS2TFTreeInput] = GetROS2TFTreeInput
    warmup_s: float = 1.0

    _listening: bool = PrivateAttr(False)

    async def _acall(
        self,
        source_frame: Optional[str] = None,
        target_frame: Optional[str] = None,
        stale_after_s: float = 1.0,
    ) -> str:
        frames = self.connector.tf_frames()
        if not self._listening:
            self._listening = True
            await asyncio.sleep(self.warmup_s)
            frames = self.connector.tf_frames()
        now = now_s(self)
        parents: Dict[str, str] = {
            c: str(f.get("parent", "")) for c, f in frames.items()
        }
        known = set(parents) | set(parents.values())

        def describe(child: str) -> str:
            frame = frames[child]
            latest = float(frame.get("most_recent_transform", 0.0))
            if latest == 0.0:
                return "static"
            age = now - latest
            text = f"{frame.get('rate', 0):.1f} Hz, age {seconds(age)}"
            return text + (" STALE" if age > stale_after_s else "")

        def chain(frame: str) -> List[str]:
            path = [frame]
            while path[-1] in parents and parents[path[-1]] not in path:
                path.append(parents[path[-1]])
            return path

        roots = sorted({chain(f)[-1] for f in known})
        lines = [
            f"{len(parents)} transforms, {len(roots)} tree(s), roots: {', '.join(roots) or 'none'}"
        ]
        children: Dict[str, List[str]] = {}
        for child, parent in parents.items():
            children.setdefault(parent, []).append(child)

        tree: List[str] = []

        def walk(frame: str, depth: int) -> None:
            for child in sorted(children.get(frame, [])):
                tree.append(f"{'  ' * depth}{child} ({describe(child)})")
                walk(child, depth + 1)

        for root in roots:
            tree.append(root)
            walk(root, 1)
        lines += bounded(tree, MAX_FRAMES)
        stale = [c for c in sorted(parents) if describe(c).endswith("STALE")]
        if stale:
            lines.append(
                f"STALE (older than {seconds(stale_after_s)}): "
                + ", ".join(f"{parents[c]} -> {c}" for c in stale[:30])
            )
        if source_frame and target_frame:
            lines.append(self._pair(source_frame, target_frame, known, chain, describe))
        return "\n".join(lines)

    @staticmethod
    def _pair(source: str, target: str, known: set, chain: Any, describe: Any) -> str:
        missing = [f for f in (source, target) if f.lstrip("/") not in known]
        if missing:
            return f"MISSING: frame(s) {', '.join(missing)} not in the TF tree"
        source, target = source.lstrip("/"), target.lstrip("/")
        up, down = chain(source), chain(target)
        if up[-1] != down[-1]:
            return (
                f"NOT CONNECTED: {source} is in the tree of {up[-1]}, {target} in the"
                f" tree of {down[-1]}; no transform links these trees"
            )
        common = next(f for f in up if f in down)
        links = up[: up.index(common)] + down[: down.index(common)]
        path = up[: up.index(common) + 1] + down[: down.index(common)][::-1]
        details = ", ".join(f"{f}: {describe(f)}" for f in links)
        verdict = "STALE link on the chain" if "STALE" in details else "OK"
        return f"{source} -> {target}: {verdict}; chain {' - '.join(path)} ({details})"

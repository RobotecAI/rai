# Copyright (C) 2024 Robotec.AI
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
#

from dataclasses import dataclass
from typing import Optional, Tuple

from pydantic import UUID4

from rai_interfaces.action import Task


# ---------- Helpers ----------
class EMOJIS:
    human = "🧑‍💻"
    bot = "🤖"
    tool = "🛠️"
    unknown = "❓"
    success = "✅"
    failure = "❌"
    in_progress = "🕒"
    accepted = "👍"


@dataclass
class MissionMessage:
    uid: UUID4
    content: Optional[str] = ""
    avatar: Optional[str] = EMOJIS.unknown

    def __repr__(self):
        return f"{self.uid} | {self.content}"

    def render_steamlit(self) -> Tuple[str, str]:
        return self.avatar, str(self)


# TODO(boczekbartek): fix avatars
class MissionAcceptanceMessage(MissionMessage):
    avatar: Optional[str] = EMOJIS.accepted

    def __repr__(self):
        return f"Acceptance: {self.uid} | {self.content}"


class MissionFeedbackMessage(MissionMessage):
    avatar: Optional[str] = EMOJIS.in_progress

    def __repr__(self):
        return f"Feedback: {self.uid} | {self.content}"


class MissionDoneMessage(MissionMessage):
    def __init__(self, uid: UUID4, result: Task.Result):
        self.success = result.success
        self.report = result.report
        super().__init__(uid=uid, content=self.report)

    def __repr__(self):
        return f"{self.uid} | Success={self.success}\nReport={self.report}"

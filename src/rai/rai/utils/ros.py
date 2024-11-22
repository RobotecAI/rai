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


from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class NodeDiscovery:
    topics_and_types: Dict[str, str] = field(default_factory=dict)
    services_and_types: Dict[str, str] = field(default_factory=dict)
    actions_and_types: Dict[str, str] = field(default_factory=dict)
    allowlist: Optional[List[str]] = field(default_factory=list)

    def set(self, topics, services, actions):
        def to_dict(info: List[Tuple[str, List[str]]]) -> Dict[str, str]:
            return {k: v[0] for k, v in info}

        self.topics_and_types = to_dict(topics)
        self.services_and_types = to_dict(services)
        self.actions_and_types = to_dict(actions)
        if self.allowlist is not None:
            self.__filter(self.allowlist)

    def __filter(self, allowlist: List[str]):
        for d in [
            self.topics_and_types,
            self.services_and_types,
            self.actions_and_types,
        ]:
            to_remove = [k for k in d if k not in allowlist]
            for k in to_remove:
                d.pop(k)

    def dict(self):
        return {
            "topics_and_types": self.topics_and_types,
            "services_and_types": self.services_and_types,
            "actions_and_types": self.actions_and_types,
        }

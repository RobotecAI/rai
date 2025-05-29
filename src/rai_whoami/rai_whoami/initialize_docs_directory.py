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

import argparse
import os
from pathlib import Path

SUBDIRECTORIES = [
    "images",
    "documentation",
    "urdf",
]


def initialize_docs_directory(documentation_dir: Path) -> None:
    if documentation_dir.is_dir():
        print(
            f"Directory {documentation_dir} already exists. Remove it or use different folder."
        )
        return

    for subdirectory in SUBDIRECTORIES:
        (documentation_dir / subdirectory).mkdir(parents=True)
        print(
            f"Initialized subdirectory {subdirectory} at {documentation_dir / subdirectory}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("documentation_dir", type=Path)
    args = parser.parse_args()
    initialize_docs_directory(args)


if __name__ == "__main__":
    main()

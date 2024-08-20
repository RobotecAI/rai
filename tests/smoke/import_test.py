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

import importlib
import os
import pathlib
from types import ModuleType

import pytest


def test_can_import_all_modules_pathlib(rai_python_modules) -> None:

    def import_submodules(package: ModuleType) -> None:

        package_path = pathlib.Path(package.__file__).parent  # type: ignore

        importables = set()

        for path in package_path.rglob("*"):
            if path.is_dir() and path.name.startswith("__"):
                continue
            if path.is_file() and path.suffix != ".py" or path.name == "__init__.py":
                continue

            relative_path = str(path.relative_to(package_path))
            subpage_name = relative_path.replace(os.path.sep, ".").replace(".py", "")

            module_prefix = f"{package_path.name}.{subpage_name}"
            importables.add(module_prefix)

        for full_name in sorted(list(importables)):
            try:
                print(f"Importing {full_name}", end=" ")
                importlib.import_module(full_name)
                print("OK")

            except ImportError as e:
                print("FAIL")
                pytest.fail(f"Failed to import {full_name}: {str(e)}")

    for module in rai_python_modules:
        print(f"Checking {module}")
        import_submodules(module)

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


import glob
import importlib
import os
import pathlib
from types import ModuleType

import pytest


def rai_python_modules():
    ignore_packages = ["rai_finetune"]

    packages = glob.glob("src/rai*") + glob.glob("src/*/rai*")
    package_names = [
        os.path.basename(p)
        for p in packages
        if os.path.basename(p) not in ignore_packages
    ]
    ros2_python_packages = []
    for package_path, package_name in zip(packages, package_names):
        if os.path.isdir(f"{package_path}/{package_name}"):
            ros2_python_packages.append(package_name)
    return [importlib.import_module(p) for p in ros2_python_packages]


@pytest.mark.parametrize("module", rai_python_modules())
def test_can_import_all_modules_pathlib(module: ModuleType) -> None:
    def import_submodules(package: ModuleType) -> None:
        ignored_modules = ["rai_bench.experiments"]
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
            if any(module_prefix.startswith(ignored) for ignored in ignored_modules):
                print(f"Skipping {module_prefix} (in ignore list)")
                continue
            importables.add(module_prefix)

        for full_name in sorted(list(importables)):
            try:
                print(f"Importing {full_name}", end=" ")
                importlib.import_module(full_name)
                print("OK")

            except ImportError as e:
                print("FAIL")
                pytest.fail(f"Failed to import {full_name}: {str(e)}")

    import_submodules(module)

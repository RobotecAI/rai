import importlib
import os
import pkgutil
from types import ModuleType
from typing import List

import pytest


def test_can_import_all_modules() -> None:
    import rai

    EXCLUDED_DIRS: List[str] = ["tests", "examples", "docs", "logs"]

    def import_submodules(package: ModuleType) -> None:
        package_path: str = os.path.dirname(package.__file__)
        for root, dirs, files in os.walk(package_path):
            for dir_name in dirs:
                if dir_name.startswith("__"):
                    continue
                dir_path: str = os.path.join(root, dir_name)
                relative_path: str = os.path.relpath(dir_path, package_path)
                module_prefix: str = (
                    f"{package.__name__}.{relative_path.replace(os.path.sep, '.')}"
                )

                for loader, name, is_pkg in pkgutil.walk_packages([dir_path]):
                    full_name: str = f"{module_prefix}.{name}"

                    if any(
                        excluded in full_name.split(".") for excluded in EXCLUDED_DIRS
                    ):
                        continue

                    try:
                        importlib.import_module(full_name)
                    except ImportError as e:
                        pytest.fail(f"Failed to import {full_name}: {str(e)}")

            for file_name in files:
                if file_name.endswith(".py") and file_name != "__init__.py":
                    script_path: str = os.path.join(root, file_name)
                    relative_script_path: str = os.path.relpath(
                        script_path, package_path
                    )
                    module_name: str = (
                        f"{package.__name__}.{relative_script_path.replace(os.path.sep, '.')[:-3]}"  # Strip .py extension
                    )

                    if any(
                        excluded in module_name.split(".") for excluded in EXCLUDED_DIRS
                    ):
                        continue

                    try:
                        importlib.import_module(module_name)
                    except ImportError as e:
                        pytest.fail(f"Failed to import script {module_name}: {str(e)}")

    import_submodules(rai)

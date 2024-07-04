import importlib
import os
import pathlib
from types import ModuleType

import pytest


def test_can_import_all_modules_pathlib() -> None:
    import rai

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
                importlib.import_module(full_name)
            except ImportError as e:
                pytest.fail(f"Failed to import {full_name}: {str(e)}")

    import_submodules(rai)

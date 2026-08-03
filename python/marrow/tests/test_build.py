"""The wheel must contain every module `import marrow` reaches.

`build.py` force-includes files one by one. It used to list only
`__init__.py` and the shared library, while `__init__.py` ends with
`from . import compute` — so a built wheel raised `ImportError` on
`import marrow`, and nothing caught it because the test suite always runs
against the source tree.
"""

import importlib.util
import sys
import types
from pathlib import Path

import pytest

PACKAGE = Path(__file__).resolve().parents[1]
BUILD_PY = PACKAGE.parent / "build.py"


def _load_build_module():
    """Import `python/build.py` with `hatchling` stubbed.

    hatchling is a build-time-only dependency and is not in the dev
    environment, but the file's force-include logic is plain path handling.
    """
    for name in (
        "hatchling",
        "hatchling.builders",
        "hatchling.builders.hooks",
        "hatchling.builders.hooks.plugin",
        "hatchling.builders.hooks.plugin.interface",
    ):
        if name not in sys.modules:
            sys.modules[name] = types.ModuleType(name)
    sys.modules["hatchling.builders.hooks.plugin.interface"].BuildHookInterface = object

    spec = importlib.util.spec_from_file_location("_marrow_build", BUILD_PY)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def shipped():
    build = _load_build_module()
    hook = build.CustomBuildHook.__new__(build.CustomBuildHook)
    data = {"force_include": {}}
    # `initialize` would rebuild the .so if it is missing; the packaging half
    # is what we are asserting on.
    build.CustomBuildHook.initialize(hook, "standard", data)
    return {Path(src).name for src in data["force_include"]}


def test_every_package_module_ships(shipped):
    expected = {p.name for p in PACKAGE.glob("*.py")}
    assert expected <= shipped, f"missing from the wheel: {sorted(expected - shipped)}"


def test_compute_ships_because_init_imports_it(shipped):
    # the concrete failure: `marrow/__init__.py` does `from . import compute`
    assert "compute.py" in shipped

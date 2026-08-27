"""`import marrow` must work, and the wheel must contain every module it reaches.

Two sibling failures, both invisible to a suite that only ever runs one way.

`build.py` force-includes files one by one. It used to list only
`__init__.py` and the shared library, while `__init__.py` ends with
`from . import compute` — so a built wheel raised `ImportError` on
`import marrow`, and nothing caught it because the test suite always runs
against the source tree.

Then `ebd4c4c` renamed the Mojo package `expr` -> `exprold` and pointed
`__init__.py` at `.exprold` while the Python file was still `expr.py`, so
`import marrow` raised `ModuleNotFoundError` for nine commits. Nothing caught
*that* because the suite always runs from the repo root, where the Mojo source
directory `marrow/` shadows `python/marrow/` as an implicit namespace package —
`import marrow` then silently succeeds and resolves to a package with no
`__file__` and no names. `test_import_from_a_neutral_directory` is the guard:
it is the only test here that must not run from the repo root.
"""

import importlib.util
import os
import subprocess
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


def test_import_from_a_neutral_directory(tmp_path):
    """`import marrow` resolves the real package, run from outside the repo.

    `cwd=tmp_path` is the whole test. From the repo root the Mojo source
    directory `marrow/` wins as an implicit namespace package and hides a
    broken `python/marrow/__init__.py`; from anywhere else the real one is
    imported and any bad import in it raises.
    """
    result = subprocess.run(
        [sys.executable, "-c", "import marrow; print(marrow.__file__)"],
        cwd=tmp_path,
        env=os.environ | {"PYTHONPATH": str(PACKAGE.parent)},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert Path(result.stdout.strip()) == PACKAGE / "__init__.py"

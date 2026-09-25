"""The Mojo pin has one home, `pixi.toml`, and three copies that must agree.

A wheel is built by cibuildwheel, which never reads `pixi.toml`: its
`before-build` pip-installs the compiler and `max-core` by exact version.
`marrow compile` ships inside that wheel and names the nightly in its error
messages. Each copy drifted independently before this test existed -- three
different Mojo versions were pinned at once.
"""

import tomllib

from devkit.mojo import Repo
from devkit.wheel import compile_module

REPO = Repo.locate()
CATALOG = compile_module(REPO)
PYPROJECT = tomllib.loads(REPO.pyproject.read_text())
_PIXI = tomllib.loads((REPO.root / "pixi.toml").read_text())
MOJO = _PIXI["package"]["build-dependencies"]["mojo-compiler"].removeprefix("==")
MAX = _PIXI["dependencies"]["max"].removeprefix("==")


def test_compile_py_names_the_pixi_nightly():
    assert CATALOG.PINNED_NIGHTLY == MOJO


def test_cibuildwheel_installs_the_pixi_pins():
    before_build = PYPROJECT["tool"]["cibuildwheel"]["before-build"]
    assert f"mojo-compiler=={MOJO}" in before_build
    assert f"max-core=={MAX}" in before_build


def test_compile_extra_matches_the_minimum_version():
    major, minor, _ = CATALOG.MIN_VERSION.split(".")
    extra = PYPROJECT["project"]["optional-dependencies"]["compile"]
    assert extra == [f"mojo>={major}.{minor},<{CATALOG.MAX_VERSION}"]


def test_pyproject_is_not_a_second_pixi_manifest():
    assert "pixi" not in PYPROJECT.get("tool", {})

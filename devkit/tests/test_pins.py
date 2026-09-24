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


def _pixi_pins(repo):
    manifest = tomllib.loads((repo.root / "pixi.toml").read_text())
    mojo = manifest["package"]["build-dependencies"]["mojo-compiler"]
    max_ = manifest["dependencies"]["max"]
    return mojo.removeprefix("=="), max_.removeprefix("==")


def _pyproject(repo):
    return tomllib.loads((repo.python_dir / "pyproject.toml").read_text())


def test_compile_py_names_the_pixi_nightly():
    repo = Repo.locate()
    mojo, _ = _pixi_pins(repo)
    assert compile_module(repo).PINNED_NIGHTLY == mojo


def test_cibuildwheel_installs_the_pixi_pins():
    repo = Repo.locate()
    mojo, max_ = _pixi_pins(repo)
    before_build = _pyproject(repo)["tool"]["cibuildwheel"]["before-build"]
    assert f"mojo-compiler=={mojo}" in before_build
    assert f"max-core=={max_}" in before_build


def test_compile_extra_matches_the_minimum_version():
    repo = Repo.locate()
    minimum = compile_module(repo).MIN_VERSION
    major, minor, _ = minimum.split(".")
    extra = _pyproject(repo)["project"]["optional-dependencies"]["compile"]
    assert extra == [f"mojo>={major}.{minor},<{compile_module(repo).MAX_VERSION}"]


def test_pyproject_is_not_a_second_pixi_manifest():
    assert "pixi" not in _pyproject(Repo.locate()).get("tool", {})

"""Pins with one home in `pixi.toml`, and the copies that must agree with it.

A wheel is built by cibuildwheel, which never reads `pixi.toml`: its
`before-build` pip-installs the compiler and `max-core` by exact version.
`marrow compile` ships inside that wheel and names the nightly in its error
messages. Each copy drifted independently before this test existed -- three
different Mojo versions were pinned at once. cargo-about, which generates the
OpenDAL licence file, is pinned for CI too.
"""

import re
import tomllib

from devkit.tests import CATALOG, PYPROJECT, REPO

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


def test_ci_regenerates_opendal_licences_with_the_local_cargo_about():
    """opendal.yml diffs its regenerated file against the committed one; a
    different generator version would fail it, or pass on output nobody can
    reproduce with `pixi run -e opendal opendal_licenses`."""
    workflow = (REPO.root / ".github" / "workflows" / "opendal.yml").read_text()
    task = _PIXI["feature"]["opendal"]["tasks"]["install_cargo_about"]
    pin = re.compile(r"cargo-about@([\d.]+)")
    assert pin.search(workflow).group(1) == pin.search(task).group(1)

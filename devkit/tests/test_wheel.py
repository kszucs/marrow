"""`devkit wheel check`, against wheels written by hand.

A real wheel takes a `libmarrow` build and a `delocate` pass; what the check
reads is only the zip's names and its METADATA, so a synthetic one exercises
all of it. The catalog is the real `compile.py`, so these also pin the mapping
the check enforces.
"""

import zipfile

import pytest
from click.testing import CliRunner

from devkit.cli import Context, cli
from devkit.mojo import Repo
from devkit.wheel import check_wheel, compile_module

_DIST_INFO = "marrow-0.1.0.dist-info"

_MACOS_LIBS = (
    "marrow/libmarrow.cpython-314-darwin.so",
    "marrow/libzstd.1.dylib",
    "marrow/.dylibs/libKGENCompilerRTShared.dylib",
)
_TEXTS = (
    "LICENSE.txt",
    "NOTICE.txt",
    "licenses/zstd.txt",
    "licenses/modular-LICENSE.txt",
    "licenses/modular-Third-Party-Notices.txt",
)


def _write_wheel(tmp_path, libs=_MACOS_LIBS, texts=_TEXTS, expression=True):
    path = tmp_path / "marrow-0.1.0-cp314-cp314-macosx_13_0_arm64.whl"
    metadata = ["Metadata-Version: 2.4", "Name: marrow", "Version: 0.1.0"]
    if expression:
        metadata.append("License-Expression: Apache-2.0 AND LicenseRef-Modular")
    metadata += [f"License-File: {rel}" for rel in texts]
    with zipfile.ZipFile(path, "w") as wheel:
        wheel.writestr(f"{_DIST_INFO}/METADATA", "\n".join(metadata) + "\n")
        # hatchling writes directory entries too; they are not texts.
        wheel.writestr(f"{_DIST_INFO}/licenses/", "")
        wheel.writestr(f"{_DIST_INFO}/licenses/licenses/", "")
        for rel in texts:
            wheel.writestr(f"{_DIST_INFO}/licenses/{rel}", "text\n")
        for lib in libs:
            wheel.writestr(lib, b"\x00")
    return path


@pytest.fixture(scope="module")
def catalog():
    return compile_module(Repo.locate())


def test_a_consistent_wheel_passes(tmp_path, catalog):
    assert check_wheel(_write_wheel(tmp_path), catalog) == []


def test_an_unrecorded_library_fails(tmp_path, catalog):
    wheel = _write_wheel(tmp_path, libs=(*_MACOS_LIBS, "marrow/.dylibs/libmystery.dylib"))
    assert check_wheel(wheel, catalog) == [
        "marrow/.dylibs/libmystery.dylib has no LIBRARY_LICENSES entry"
    ]


def test_a_missing_text_names_the_library_that_needs_it(tmp_path, catalog):
    texts = tuple(t for t in _TEXTS if t != "licenses/zstd.txt")
    problems = check_wheel(_write_wheel(tmp_path, texts=texts), catalog)
    assert problems == [
        f"marrow/libzstd.1.dylib needs licenses/zstd.txt, which is not under "
        f"{_DIST_INFO}/licenses/"
    ]


def test_a_missing_license_expression_fails(tmp_path, catalog):
    problems = check_wheel(_write_wheel(tmp_path, expression=False), catalog)
    assert problems == ["METADATA has no License-Expression"]


@pytest.mark.parametrize(
    "lib",
    ["marrow/.dylibs/libMGPRT.dylib", "marrow.libs/libstdc++-0a1b2c3d.so.6"],
)
def test_a_forbidden_library_fails(tmp_path, catalog, lib):
    problems = check_wheel(_write_wheel(tmp_path, libs=(*_MACOS_LIBS, lib)), catalog)
    assert problems == [f"{lib} must not ship in a wheel"]


def test_auditwheel_renamed_libraries_resolve(tmp_path, catalog):
    libs = (
        "marrow/libmarrow.cpython-314-x86_64-linux-gnu.so",
        "marrow/libzstd.so.1",
        "marrow.libs/libKGENCompilerRTShared-0a1b2c3d.so",
    )
    assert check_wheel(_write_wheel(tmp_path, libs=libs), catalog) == []


def test_the_command_fails_listing_every_problem(tmp_path):
    good = _write_wheel(tmp_path)
    bad_dir = tmp_path / "bad"
    bad_dir.mkdir()
    bad = _write_wheel(bad_dir, expression=False)
    result = CliRunner().invoke(
        cli, ["wheel", "check", str(good), str(bad)], obj=Context(Repo.locate())
    )
    assert result.exit_code == 1
    assert f"{good.name}: ok" in result.output
    assert "METADATA has no License-Expression" in result.output

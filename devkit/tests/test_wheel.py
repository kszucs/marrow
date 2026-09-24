"""`devkit wheel check`, against wheels written by hand.

A real wheel takes a `libmarrow` build and a `delocate` pass; what the check
reads is only the zip's names and its METADATA, so a synthetic one exercises
all of it. The catalog is the real `compile.py`, so these also pin the mapping
and the tables the check enforces.
"""

import zipfile

import pytest
from click.testing import CliRunner

from devkit.cli import Context, cli
from devkit.mojo import Repo
from devkit.wheel import check_wheel, compile_module

_DIST_INFO = "marrow-0.1.0.dist-info"


@pytest.fixture(scope="module")
def catalog():
    return compile_module(Repo.locate())


def _libraries(catalog, *extra):
    """What a macOS wheel ships: the extension, one file per codec, and the
    Mojo runtime delocate grafts in."""
    codecs = [
        f"marrow/{names[0]}" for names in catalog._CODEC_LIB_CANDIDATES.values()
    ]
    return (
        "marrow/libmarrow.cpython-314-darwin.so",
        *codecs,
        "marrow/.dylibs/libKGENCompilerRTShared.dylib",
        *extra,
    )


def _texts(catalog, libraries):
    """Exactly the texts `libraries` need, as the wheel would carry them."""
    texts = {"LICENSE.txt", "NOTICE.txt"}
    for lib in libraries:
        texts.update(catalog.license_files(lib.rsplit("/", 1)[-1]) or ())
    return texts


def _write_wheel(tmp_path, libraries, texts, expression=True):
    path = tmp_path / "marrow-0.1.0-cp314-cp314-macosx_13_0_arm64.whl"
    metadata = ["Metadata-Version: 2.4", "Name: marrow", "Version: 0.1.0"]
    if expression:
        metadata.append("License-Expression: Apache-2.0 AND LicenseRef-Modular")
    metadata += [f"License-File: {rel}" for rel in sorted(texts)]
    with zipfile.ZipFile(path, "w") as wheel:
        wheel.writestr(f"{_DIST_INFO}/METADATA", "\n".join(metadata) + "\n")
        # hatchling writes directory entries too; they are not texts.
        wheel.writestr(f"{_DIST_INFO}/licenses/", "")
        wheel.writestr(f"{_DIST_INFO}/licenses/licenses/", "")
        for rel in texts:
            wheel.writestr(f"{_DIST_INFO}/licenses/{rel}", "text\n")
        for lib in libraries:
            wheel.writestr(lib, b"\x00")
    return path


def _consistent_wheel(tmp_path, catalog, *extra):
    libraries = _libraries(catalog, *extra)
    return _write_wheel(tmp_path, libraries, _texts(catalog, libraries))


def test_a_consistent_wheel_passes(tmp_path, catalog):
    assert check_wheel(_consistent_wheel(tmp_path, catalog), catalog) == []


def test_an_unrecorded_library_fails(tmp_path, catalog):
    libraries = _libraries(catalog, "marrow/.dylibs/libmystery.dylib")
    wheel = _write_wheel(tmp_path, libraries, _texts(catalog, libraries))
    assert check_wheel(wheel, catalog) == [
        "marrow/.dylibs/libmystery.dylib has no LIBRARY_LICENSES entry"
    ]


def test_a_missing_text_names_the_library_that_needs_it(tmp_path, catalog):
    libraries = _libraries(catalog)
    texts = _texts(catalog, libraries) - {"licenses/zstd.txt"}
    problems = check_wheel(_write_wheel(tmp_path, libraries, texts), catalog)
    zstd = next(lib for lib in libraries if "libzstd" in lib)
    assert problems == [
        f"{zstd} needs licenses/zstd.txt, which is not under {_DIST_INFO}/licenses/"
    ]


def test_a_missing_license_expression_fails(tmp_path, catalog):
    libraries = _libraries(catalog)
    wheel = _write_wheel(
        tmp_path, libraries, _texts(catalog, libraries), expression=False
    )
    assert check_wheel(wheel, catalog) == ["METADATA has no License-Expression"]


def test_a_missing_codec_fails(tmp_path, catalog):
    """The build only warns when a codec is absent; the check must not."""
    libraries = [lib for lib in _libraries(catalog) if "libsnappy" not in lib]
    wheel = _write_wheel(tmp_path, libraries, _texts(catalog, libraries))
    assert check_wheel(wheel, catalog) == ["no snappy library in the wheel"]


def test_a_required_optional_library_must_be_present(tmp_path, catalog):
    wheel = _consistent_wheel(tmp_path, catalog)
    assert check_wheel(wheel, catalog, require=["opendal"]) == [
        "no opendal library in the wheel"
    ]
    (tmp_path / "with").mkdir()
    with_opendal = _consistent_wheel(
        tmp_path / "with", catalog, "marrow/libopendal_c.dylib"
    )
    assert check_wheel(with_opendal, catalog, require=["opendal"]) == []


@pytest.mark.parametrize(
    "lib",
    ["marrow/.dylibs/libMGPRT.dylib", "marrow.libs/libstdc++-0a1b2c3d.so.6"],
)
def test_a_forbidden_library_fails(tmp_path, catalog, lib):
    libraries = _libraries(catalog, lib)
    wheel = _write_wheel(tmp_path, libraries, _texts(catalog, libraries))
    assert check_wheel(wheel, catalog) == [f"{lib} must not ship in a wheel"]


def test_auditwheel_renamed_libraries_resolve(tmp_path, catalog):
    codecs = [
        f"marrow/{names[-1]}" for names in catalog._CODEC_LIB_CANDIDATES.values()
    ]
    libraries = (
        "marrow/libmarrow.cpython-314-x86_64-linux-gnu.so",
        *codecs,
        "marrow.libs/libKGENCompilerRTShared-0a1b2c3d.so",
    )
    wheel = _write_wheel(tmp_path, libraries, _texts(catalog, libraries))
    assert check_wheel(wheel, catalog) == []


def test_the_command_fails_listing_every_problem(tmp_path, catalog):
    good = _consistent_wheel(tmp_path, catalog)
    bad_dir = tmp_path / "bad"
    bad_dir.mkdir()
    libraries = _libraries(catalog)
    bad = _write_wheel(
        bad_dir, libraries, _texts(catalog, libraries), expression=False
    )
    result = CliRunner().invoke(
        cli, ["wheel", "check", str(good), str(bad)], obj=Context(Repo.locate())
    )
    assert result.exit_code == 1
    assert f"{good.name}: ok" in result.output
    assert "METADATA has no License-Expression" in result.output

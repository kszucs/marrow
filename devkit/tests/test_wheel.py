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

CATALOG = compile_module(Repo.locate())
_DIST_INFO = "marrow-0.1.0.dist-info"
_ZSTD = f"marrow/{CATALOG._CODEC_LIB_CANDIDATES['zstd'][0]}"


def _wheel(directory, *extra, drop=(), expression=True, codecs=0):
    """A macOS-shaped wheel -- the extension, one file per codec (candidate
    `codecs` of each), the Mojo runtime delocate grafts in, and `extra` -- with
    exactly the texts those need, less any library or text in `drop`."""
    libraries = [
        "marrow/libmarrow.cpython-314-darwin.so",
        *(f"marrow/{names[codecs]}" for names in CATALOG._CODEC_LIB_CANDIDATES.values()),
        "marrow/.dylibs/libKGENCompilerRTShared.dylib",
        *extra,
    ]
    libraries = [lib for lib in libraries if lib not in drop]
    texts = {
        rel
        for lib in libraries
        for rel in CATALOG.license_files(lib.rsplit("/", 1)[-1]) or ()
    } - set(drop)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "marrow-0.1.0-cp314-cp314-macosx_13_0_arm64.whl"
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


def test_a_consistent_wheel_passes(tmp_path):
    assert check_wheel(_wheel(tmp_path), CATALOG) == []


def test_an_unrecorded_library_fails(tmp_path):
    wheel = _wheel(tmp_path, "marrow/.dylibs/libmystery.dylib")
    assert check_wheel(wheel, CATALOG) == [
        "marrow/.dylibs/libmystery.dylib has no LIBRARY_LICENSES entry"
    ]


def test_a_missing_text_names_the_library_that_needs_it(tmp_path):
    wheel = _wheel(tmp_path, drop={"licenses/zstd.txt"})
    assert check_wheel(wheel, CATALOG) == [
        f"{_ZSTD} needs licenses/zstd.txt, which is not under {_DIST_INFO}/licenses/"
    ]


def test_a_missing_license_expression_fails(tmp_path):
    wheel = _wheel(tmp_path, expression=False)
    assert check_wheel(wheel, CATALOG) == ["METADATA has no License-Expression"]


def test_a_missing_codec_fails(tmp_path):
    """The build refuses a missing codec; the check is the backstop."""
    snappy = f"marrow/{CATALOG._CODEC_LIB_CANDIDATES['snappy'][0]}"
    wheel = _wheel(tmp_path, drop={snappy})
    assert check_wheel(wheel, CATALOG) == ["no snappy library in the wheel"]


def test_a_required_optional_library_must_be_present(tmp_path):
    without = _wheel(tmp_path / "without")
    assert check_wheel(without, CATALOG, require=["opendal"]) == [
        "no opendal library in the wheel"
    ]
    with_opendal = _wheel(tmp_path / "with", "marrow/libopendal_c.dylib")
    assert check_wheel(with_opendal, CATALOG, require=["opendal"]) == []


@pytest.mark.parametrize(
    "lib",
    ["marrow/.dylibs/libMGPRT.dylib", "marrow.libs/libstdc++-0a1b2c3d.so.6"],
)
def test_a_forbidden_library_fails(tmp_path, lib):
    wheel = _wheel(tmp_path, lib)
    assert check_wheel(wheel, CATALOG) == [f"{lib} must not ship in a wheel"]


def test_a_library_both_staged_and_grafted_fails(tmp_path):
    """The broken Linux wheel: marrow staged brotlicommon under its real name,
    so auditwheel grafted the build image's older one beside it."""
    grafted = "marrow.libs/libbrotlicommon-97d45a34.so.1.0.9"
    wheel = _wheel(tmp_path, "marrow/libbrotlicommon.so.1.2.0", grafted)
    assert check_wheel(wheel, CATALOG) == [
        f"{grafted} was grafted beside marrow's own copy of the same library"
    ]


def test_auditwheel_renamed_libraries_resolve(tmp_path):
    wheel = _wheel(
        tmp_path,
        "marrow.libs/libMSupportGlobals-0a1b2c3d.so",
        codecs=-1,  # the `.so.1` spelling of each codec
    )
    assert check_wheel(wheel, CATALOG) == []


def test_the_command_fails_listing_every_problem(tmp_path):
    good = _wheel(tmp_path / "good")
    bad = _wheel(tmp_path / "bad", expression=False)
    result = CliRunner().invoke(
        cli, ["wheel", "check", str(good), str(bad)], obj=Context(Repo.locate())
    )
    assert result.exit_code == 1
    assert f"{good.name}: ok" in result.output
    assert "METADATA has no License-Expression" in result.output

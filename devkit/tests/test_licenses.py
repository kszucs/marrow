"""Every library marrow ships carries its licence, and the texts stay true.

`python/marrow/compile.py` owns `LIBRARY_LICENSES`; the wheel build and `marrow
compile --bundle` both stage libraries by its tables. These checks read text
files only -- no wheel is built -- so they run in the one-second selftest.
`devkit wheel check` asserts the same of a built wheel.
"""

import json
import re
import sys
import tomllib
from glob import glob
from pathlib import Path

import pytest

from devkit.mojo import Repo
from devkit.wheel import compile_module

REPO = Repo.locate()
CATALOG = compile_module(REPO)
PYPROJECT = tomllib.loads(REPO.pyproject.read_text())
#: Every text `LIBRARY_LICENSES` names, `python/`-relative.
MAPPED = {rel for files in CATALOG.LIBRARY_LICENSES.values() for rel in files}

# Texts copied verbatim from a conda package's `info/licenses/`: the file under
# `licenses/`, the package it came from, and the path inside `info/licenses/`.
_FROM_CONDA = {
    "zstd.txt": ("zstd", "LICENSE"),
    "snappy.txt": ("snappy", "COPYING"),
    "lz4.txt": ("lz4-c", "lib/LICENSE"),
    "brotli.txt": ("libbrotlicommon", "LICENSE"),
    "zlib.txt": ("libzlib", "LICENSE"),
    "libcxx.txt": ("libcxx", "libcxx/LICENSE.TXT"),
    "modular-LICENSE.txt": ("mojo-compiler", "LICENSE"),
    "modular-Third-Party-Notices.txt": ("mojo-compiler", "Third-Party-Notices"),
}


def test_every_staged_library_has_a_licence():
    """A codec or optional library `compile.py` can stage, under any of its
    candidate names, must map to a licence -- else the wheel ships it bare."""
    tables = CATALOG._CODEC_LIB_CANDIDATES | CATALOG._OPTIONAL_LIB_CANDIDATES
    unmapped = [
        name
        for names in tables.values()
        for name in names
        if CATALOG.license_files(name) is None
    ]
    assert not unmapped, f"no LIBRARY_LICENSES entry for {unmapped}"


@pytest.mark.parametrize(
    "filename, stem",
    [
        ("libzstd.1.dylib", "libzstd"),
        ("libzstd.so.1", "libzstd"),
        ("libKGENCompilerRTShared-0a1b2c3d.so", "libKGENCompilerRTShared"),
        ("libc++.1.dylib", "libc++"),
        ("libmarrow.cpython-314-darwin.so", "libmarrow"),
    ],
)
def test_license_files_accepts_every_spelling_of_a_library(filename, stem):
    assert CATALOG.license_files(filename) == CATALOG.LIBRARY_LICENSES[stem]


def test_every_mapped_text_exists_and_is_not_empty():
    missing = [rel for rel in MAPPED if not (REPO.python_dir / rel).is_file()]
    assert not missing, f"named in LIBRARY_LICENSES, absent from licenses/: {missing}"
    empty = [rel for rel in MAPPED if not (REPO.python_dir / rel).read_bytes().strip()]
    assert not empty, f"empty licence texts: {empty}"


def test_the_wheel_carries_every_text_but_the_bundle_only_ones():
    """`license-files` must cover every mapped text a wheel can hold, and none
    of `licenses/bundle-only/` -- those belong to libraries a wheel refuses."""
    covered = {
        Path(path).relative_to(REPO.python_dir).as_posix()
        for pattern in PYPROJECT["project"]["license-files"]
        for path in glob(str(REPO.python_dir / pattern))
        if Path(path).is_file()
    }
    bundle_only = {rel for rel in MAPPED if rel.startswith("licenses/bundle-only/")}
    assert MAPPED - bundle_only <= covered
    assert not bundle_only & covered


def test_notice_names_every_text():
    """NOTICE.txt is where a reader learns which component a text belongs to;
    a text it does not name is attribution nobody can place."""
    notice = (REPO.root / "NOTICE.txt").read_text()
    texts = [p.relative_to(REPO.root).as_posix() for p in REPO.licenses_dir.rglob("*.txt")]
    assert texts
    unnamed = [rel for rel in texts if rel not in notice]
    assert not unnamed, f"NOTICE.txt does not mention {unnamed}"


def test_opendal_licences_match_the_build_script():
    """The generated file must describe the TAG and FEATURES actually built."""
    script = (REPO.root / "scripts" / "build_opendal_c.sh").read_text()
    tag = re.search(r'^TAG="([^"]+)"', script, re.M).group(1)
    features = re.search(r'^FEATURES="([^"]+)"', script, re.M).group(1)
    header = (REPO.licenses_dir / "opendal-third-party.txt").read_text()
    assert header.splitlines()[0] == (
        f"Apache OpenDAL C binding {tag}, features: {features}"
    ), "run `pixi run -e opendal opendal_licenses` after changing TAG or FEATURES"


def test_every_opendal_licence_is_in_the_wheel_expression():
    """A crate licence the wheel's `License-Expression` does not name is a
    wheel that misstates its own licence."""
    text = (REPO.licenses_dir / "opendal-third-party.txt").read_text()
    overview = text.split("Licences used:", 1)[1].split("\n\n", 1)[0]
    used = set(re.findall(r"\(([A-Za-z0-9.+-]+)\): \d+ crates", overview))
    assert used, "no licence overview in opendal-third-party.txt"
    expression = set(re.split(r"\s+(?:AND|OR|WITH)\s+", PYPROJECT["project"]["license"]))
    assert used <= expression, f"missing from License-Expression: {used - expression}"


@pytest.mark.parametrize("name", sorted(_FROM_CONDA))
def test_copied_text_matches_its_conda_package(name):
    """The texts were copied from the packages the environment links; a
    version bump that changes one fails here until it is re-copied."""
    package, inner = _FROM_CONDA[name]
    records = list((Path(sys.prefix) / "conda-meta").glob(f"{package}-[0-9]*.json"))
    if not records:
        pytest.skip(f"{package} is not installed in {sys.prefix}")
    extracted = json.loads(records[0].read_text()).get("extracted_package_dir")
    source = Path(extracted or "") / "info" / "licenses" / inner
    if not source.is_file():
        pytest.skip(f"package cache for {package} is gone")
    assert (REPO.licenses_dir / name).read_bytes() == source.read_bytes(), (
        f"licenses/{name} differs from {package}'s; copy {source} over it"
    )

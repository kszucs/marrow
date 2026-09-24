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


def _mapped():
    """Every text `LIBRARY_LICENSES` names, `python/`-relative."""
    table = compile_module(Repo.locate()).LIBRARY_LICENSES
    return sorted({rel for files in table.values() for rel in files})


def _pyproject():
    return tomllib.loads((Repo.locate().python_dir / "pyproject.toml").read_text())


def test_every_staged_library_has_a_licence():
    """A codec or optional library `compile.py` can stage, under any of its
    candidate names, must map to a licence -- else the wheel ships it bare."""
    module = compile_module(Repo.locate())
    tables = module._CODEC_LIB_CANDIDATES | module._OPTIONAL_LIB_CANDIDATES
    unmapped = [
        name
        for names in tables.values()
        for name in names
        if module.license_files(name) is None
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
    module = compile_module(Repo.locate())
    assert module.license_files(filename) == module.LIBRARY_LICENSES[stem]


def test_every_mapped_text_exists_and_is_not_empty():
    python_dir = Repo.locate().python_dir
    missing = [rel for rel in _mapped() if not (python_dir / rel).is_file()]
    assert not missing, f"named in LIBRARY_LICENSES, absent from licenses/: {missing}"
    empty = [rel for rel in _mapped() if not (python_dir / rel).read_bytes().strip()]
    assert not empty, f"empty licence texts: {empty}"


def test_the_wheel_carries_every_text_but_the_bundle_only_ones():
    """`license-files` must cover every mapped text a wheel can hold, and none
    of `licenses/bundle-only/` -- those belong to libraries a wheel refuses."""
    python_dir = Repo.locate().python_dir
    covered = set()
    for pattern in _pyproject()["project"]["license-files"]:
        for path in glob(str(python_dir / pattern)):
            if Path(path).is_file():
                covered.add(Path(path).relative_to(python_dir).as_posix())
    bundle_only = {rel for rel in _mapped() if rel.startswith("licenses/bundle-only/")}
    assert set(_mapped()) - bundle_only <= covered
    assert not bundle_only & covered


def test_notice_names_every_text():
    """NOTICE.txt is where a reader learns which component a text belongs to;
    a text it does not name is attribution nobody can place."""
    root = Repo.locate().root
    notice = (root / "NOTICE.txt").read_text()
    texts = [p.relative_to(root).as_posix() for p in (root / "licenses").rglob("*.txt")]
    assert texts
    unnamed = [rel for rel in texts if rel not in notice]
    assert not unnamed, f"NOTICE.txt does not mention {unnamed}"


def test_opendal_licences_match_the_build_script():
    """The generated file must describe the TAG and FEATURES actually built."""
    repo = Repo.locate()
    script = (repo.root / "scripts" / "build_opendal_c.sh").read_text()
    tag = re.search(r'^TAG="([^"]+)"', script, re.M).group(1)
    features = re.search(r'^FEATURES="([^"]+)"', script, re.M).group(1)
    header = (repo.root / "licenses" / "opendal-third-party.txt").read_text()
    assert header.splitlines()[0] == (
        f"Apache OpenDAL C binding {tag}, features: {features}"
    ), "run `pixi run -e opendal opendal_licenses` after changing TAG or FEATURES"


def test_every_opendal_licence_is_in_the_wheel_expression():
    """A crate licence the wheel's `License-Expression` does not name is a
    wheel that misstates its own licence."""
    text = (Repo.locate().root / "licenses" / "opendal-third-party.txt").read_text()
    overview = text.split("Licences used:", 1)[1].split("\n\n", 1)[0]
    used = set(re.findall(r"\(([A-Za-z0-9.+-]+)\): \d+ crates", overview))
    assert used, "no licence overview in opendal-third-party.txt"
    expression = set(re.split(r"\s+(?:AND|OR|WITH)\s+", _pyproject()["project"]["license"]))
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
    committed = Repo.locate().root / "licenses" / name
    assert committed.read_bytes() == source.read_bytes(), (
        f"licenses/{name} differs from {package}'s; copy {source} over it"
    )

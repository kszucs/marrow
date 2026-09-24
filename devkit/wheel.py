"""The wheel: what it ships, and whether it says so.

`python/marrow/compile.py` owns the tables of libraries marrow stages into a
wheel or a `marrow compile --bundle` directory, and `LIBRARY_LICENSES`, the
texts each one needs. It ships inside the wheel and imports only the standard
library, so devkit reaches it by path rather than by `import marrow.compile` --
that import would run the package `__init__` and load `libmarrow.so`, which
nothing under `devkit/` may do at import time.

`check_wheel` is the last line: it reads a *repaired* wheel -- after `delocate`
or `auditwheel` grafted the libraries the extension links -- and reports every
shared library that travels without its licence.
"""

import importlib.util
import re
import zipfile
from email.parser import Parser

#: What a wheel must never carry. MAX's GPU runtime and engine: a CPU wheel
#: links neither, so one appearing means the build picked up a GPU flag. And the
#: two C++ runtimes manylinux guarantees: a conda copy would hide from
#: `auditwheel` whether the codecs fit the policy.
FORBIDDEN_IN_WHEEL = frozenset({"libMGPRT", "libmax", "libstdc++", "libgcc_s"})

_SHARED_LIBRARY = re.compile(r"\.(so|dylib)(\.\d+)*$")


def compile_module(repo):
    """`python/marrow/compile.py`, loaded by path from `repo`."""
    path = repo.python_dir / repo.PACKAGE / "compile.py"
    spec = importlib.util.spec_from_file_location("_marrow_compile", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def check_wheel(path, catalog, require=()):
    """Every way the wheel at `path` misstates what it ships, as messages.

    `catalog` is `compile.py` (see `compile_module`): its `library_stem` and
    `license_files` are the one definition of which texts a library needs, and
    its tables say what a wheel must carry -- every page codec always (the
    build only *warns* when one is missing, and a wheel without snappy cannot
    read most Parquet files), and each `_OPTIONAL_LIB_CANDIDATES` key named in
    `require`. An empty list means the wheel is consistent.
    """
    with zipfile.ZipFile(path) as wheel:
        names = wheel.namelist()
        dist_info = next(
            (top for top in (n.split("/", 1)[0] for n in names)
             if top.endswith(".dist-info")),
            None,
        )
        if dist_info is None:
            return ["no .dist-info directory"]
        metadata = Parser().parsestr(wheel.read(f"{dist_info}/METADATA").decode())

    prefix = f"{dist_info}/licenses/"
    # A zip may list directories too (`.../licenses/`); only files are texts.
    shipped = {
        n.removeprefix(prefix)
        for n in names
        if n.startswith(prefix) and not n.endswith("/")
    }
    declared = set(metadata.get_all("License-File") or [])

    problems = []
    if not metadata.get("License-Expression"):
        problems.append("METADATA has no License-Expression")
    if declared != shipped:
        problems.append(
            f"License-File entries {sorted(declared ^ shipped)} do not match "
            f"the files under {prefix}"
        )
    for rel in ("LICENSE.txt", "NOTICE.txt"):
        if rel not in shipped:
            problems.append(f"{rel} is not under {prefix}")

    libraries = [n for n in names if _SHARED_LIBRARY.search(n.rsplit("/", 1)[-1])]
    stems = {catalog.library_stem(n.rsplit("/", 1)[-1]) for n in libraries}
    expected = dict(catalog._CODEC_LIB_CANDIDATES)
    for key in require:
        if key in catalog._OPTIONAL_LIB_CANDIDATES:
            expected[key] = catalog._OPTIONAL_LIB_CANDIDATES[key]
        else:
            problems.append(f"cannot require {key!r}: not an optional library")
    for key, candidates in expected.items():
        if not stems & {catalog.library_stem(c) for c in candidates}:
            problems.append(f"no {key} library in the wheel")

    for name in libraries:
        base = name.rsplit("/", 1)[-1]
        if catalog.library_stem(base) in FORBIDDEN_IN_WHEEL:
            problems.append(f"{name} must not ship in a wheel")
            continue
        files = catalog.license_files(base)
        if files is None:
            problems.append(f"{name} has no LIBRARY_LICENSES entry")
            continue
        problems.extend(
            f"{name} needs {rel}, which is not under {prefix}"
            for rel in files
            if rel not in shipped
        )
    return problems

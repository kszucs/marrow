# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""The wheel: what it ships, and whether it says so.

`python/marrow/compile.py` owns the tables of libraries marrow stages into a
wheel or a `marrow compile --bundle` directory, and `LIBRARY_LICENSES`, the
texts each one needs; `compile_module` is the one way to reach it from here.

`check_wheel` is the last line: it reads a *repaired* wheel -- after `delocate`
or `auditwheel` grafted the libraries the extension links -- and reports every
shared library that travels without its licence.
"""

import importlib.util
import posixpath
import zipfile
from email.parser import Parser

#: What `python/build.py` never stages into a wheel, though a Linux `--bundle`
#: directory carries both: manylinux guarantees them, and a conda copy would
#: hide from auditwheel whether the codecs fit the policy's GLIBCXX.
WHEEL_EXCLUDED = frozenset({"libstdc++", "libgcc_s"})

#: MAX's GPU runtime and engine: a CPU wheel links neither, so one appearing
#: means the build picked up a GPU flag.
_NEVER_LINKED = frozenset({"libMGPRT", "libmax"})


def _grafted(name):
    """Whether the repair tool put `name` there: auditwheel grafts into
    `<package>.libs/`, delocate into `<package>/.dylibs/`."""
    return any(p.endswith(".libs") or p == ".dylibs" for p in name.split("/")[:-1])


def compile_module(repo):
    """`python/marrow/compile.py`, loaded by path from `repo`.

    Not `import marrow.compile`: that runs the package `__init__`, which loads
    `libmarrow.so` -- something nothing under `devkit/` may do at import time,
    and which the build hook cannot rely on, since under cross-compilation the
    building interpreter cannot load the extension it just built. `compile.py`
    imports only the standard library, so loading it alone is well defined.
    """
    spec = importlib.util.spec_from_file_location("_marrow_compile", repo.compile_py)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def check_wheel(path, catalog, require=()):
    """Every way the wheel at `path` misstates what it ships, as messages.

    `catalog` is `compile.py` (see `compile_module`): its `library_stem` and
    `LIBRARY_LICENSES` are the one definition of which texts a library needs,
    and its tables say what a wheel must carry -- every page codec, a backstop
    to the build that already refuses to leave one out, and each
    `_OPTIONAL_LIB_CANDIDATES` key named in `require`. An empty list means the
    wheel is consistent.
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

    libraries = {
        name: catalog.library_stem(posixpath.basename(name))
        for name in names
        if catalog.is_shared_library(name)
    }
    stems = set(libraries.values())
    expected = dict(catalog._CODEC_LIB_CANDIDATES)
    for key in require:
        if key in catalog._OPTIONAL_LIB_CANDIDATES:
            expected[key] = catalog._OPTIONAL_LIB_CANDIDATES[key]
        else:
            problems.append(f"cannot require {key!r}: not an optional library")
    for key, candidates in expected.items():
        if not stems & {catalog.library_stem(c) for c in candidates}:
            problems.append(f"no {key} library in the wheel")

    staged = {stem for name, stem in libraries.items() if not _grafted(name)}
    for name, stem in libraries.items():
        if stem in WHEEL_EXCLUDED | _NEVER_LINKED:
            problems.append(f"{name} must not ship in a wheel")
            continue
        # Grafted although marrow staged its own copy: the staged one was not
        # found under the name its dependent asks for, so the loader takes the
        # grafted one -- once the build image's older brotli.
        if _grafted(name) and stem in staged:
            problems.append(f"{name} was grafted beside marrow's own copy of the same library")
        files = catalog.LIBRARY_LICENSES.get(stem)
        if files is None:
            problems.append(f"{name} has no LIBRARY_LICENSES entry")
            continue
        problems.extend(
            f"{name} needs {rel}, which is not under {prefix}"
            for rel in files
            if rel not in shipped
        )
    return problems

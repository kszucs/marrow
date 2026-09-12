"""The soname tables exist twice, in two languages, and must not drift.

`marrow/**/*.mojo` declares each `dlopen`ed library as a `LibSpec`; the Mojo
side is the authority, because that is what actually opens them. But
`python/marrow/compile.py` has to know the same names without being able to
ask: it stages these libraries into the wheel and into a `marrow compile
--bundle` directory, and both run where no Mojo compiler is involved.

So the tables are duplicated by necessity, and this is what stops them
drifting. It lives in `devkit/tests/` rather than beside `compile.py` because
a selection under `python/` triggers a full `libmarrow.so` build
(`LaneSelector.needs_libmarrow`), and this check reads two text files.
"""

import importlib.util
import re

from devkit.mojo import Repo

# Anchored on the closing paren, with the trailing comma optional: `mojo
# format` writes a short declaration on one line (no trailing comma) and a
# long one split across lines (trailing comma), and an earlier version of this
# pattern silently matched only 5 of the 7 specs because of it.
_SPEC = re.compile(
    r'LibSpec\(\s*"([^"]+)"\s*,'
    r"\s*\[([^\]]*)\]\s*,"
    r"\s*(?:#[^\n]*\n\s*)*"  # the env list may carry a comment above it
    r"\[([^\]]*)\]",
    re.S,
)

_SOURCES = (
    ("utils", "compression.mojo"),
    ("io", "opendal.mojo"),
)


def _load_compile_module():
    """`compile.py` by path: importing `marrow` would pull in the extension."""
    path = Repo.locate().python_dir / Repo.PACKAGE / "compile.py"
    spec = importlib.util.spec_from_file_location("_marrow_compile", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _declared_specs():
    """Every `LibSpec` in the tree, as {name: (sonames, env)}.

    Parsed rather than imported -- there is no import that crosses from Mojo
    to Python. A regex suffices because a `LibSpec` has to be a literal to be
    a comptime value.
    """
    package = Repo.locate().package_dir
    out = {}
    for parts in _SOURCES:
        src = package.joinpath(*parts).read_text()
        for m in _SPEC.finditer(src):
            out[m.group(1)] = (
                re.findall(r'"([^"]+)"', m.group(2)),
                re.findall(r'"([^"]+)"', m.group(3)),
            )
    return out


def test_every_mojo_libspec_is_staged_by_compile_py():
    """A library Mojo can open that Python does not know to ship is a wheel
    that imports fine and fails on first use."""
    compile_module = _load_compile_module()
    declared = _declared_specs()
    assert declared, "no LibSpec declarations found -- did the syntax change?"

    staged = {
        tuple(v)
        for v in list(compile_module._CODEC_LIB_CANDIDATES.values())
        + list(compile_module._OPTIONAL_LIB_CANDIDATES.values())
    }
    missing = {
        name: sonames
        for name, (sonames, _) in declared.items()
        if tuple(sonames) not in staged
    }
    assert not missing, (
        f"declared in Mojo but not in compile.py's tables: {missing}. "
        "The wheel and `marrow compile --bundle` would omit them."
    )
    assert len(staged) == len(declared), (
        f"compile.py stages {len(staged)} libraries, Mojo declares "
        f"{len(declared)} -- one side has an entry the other does not"
    )


def test_opendal_env_overrides_match():
    """The env names are duplicated too, and were previously unguarded.

    `optional_lib_paths()` reads them in order to find a developer's own
    build; if Mojo grows a third override and Python does not, the wheel
    stages a different library than the one a run would open.
    """
    compile_module = _load_compile_module()
    declared = _declared_specs()
    assert declared["opendal_c"][1] == [
        "MARROW_OPENDAL_LIBRARY",
        "OPENDAL_C_LIBRARY",
    ]

    source = (Repo.locate().python_dir / Repo.PACKAGE / "compile.py").read_text()
    for env in declared["opendal_c"][1]:
        assert env in source, f"{env} is declared in Mojo but unknown to compile.py"
    _ = compile_module

# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""Every failure the library raises is an `ArrowError`.

`marrow/errors.mojo` defines the kinds -- `KeyError`, `CorruptError`, ... --
and each reaches Python as the PyArrow-named exception class of the same name.
A plain `Error(...)` reaches Python as a bare `Exception` that cannot be told
apart from any other failure, and they accumulate: 269 of them on 2026-09-04,
386 three weeks later. This check reads source text, so it runs in the devkit
suite rather than waiting on a compile.

Tests, `utils/testing.mojo` and the benchmark programs are exempt: what they
raise never reaches a user.
"""

import re

from devkit.mojo import Repo

# An `Error(` that is not the tail of a longer name (`KeyError(`, `CError(`) or
# an attribute (`x.Error(`).
_UNTYPED = re.compile(r"(?<![\w.])Error\(")


def _library_sources(repo):
    package = repo.package_dir
    for path in sorted(package.rglob("*.mojo")):
        rel = path.relative_to(package)
        if "tests" in rel.parts or rel.as_posix() in (
            "utils/testing.mojo",
            "errors.mojo",
        ):
            continue
        yield path
    yield from sorted((repo.python_dir / "bindings").rglob("*.mojo"))
    yield repo.golden_dir / "helpers.mojo"


def _code_lines(path):
    """(line number, text) outside comments and docstrings."""
    in_docstring = False
    for number, line in enumerate(path.read_text().splitlines(), 1):
        stripped = line.strip()
        quotes = stripped.count('"""')
        if in_docstring:
            if quotes % 2:
                in_docstring = False
            continue
        if quotes % 2:
            in_docstring = True
            continue
        if quotes or stripped.startswith("#"):
            continue
        yield number, line


def test_library_raises_only_arrow_errors():
    """Raise a kind -- `raise KeyError(t"...")` -- never a plain `Error`; see
    `marrow/errors.mojo` for which one."""
    repo = Repo.locate()
    offenders = [
        f"{path.relative_to(repo.root)}:{number}: {line.strip()}"
        for path in _library_sources(repo)
        for number, line in _code_lines(path)
        if _UNTYPED.search(line)
    ]
    assert not offenders, (
        "a plain Error(...) in library code -- raise one of the kinds in"
        " marrow/errors.mojo:\n" + "\n".join(offenders)
    )

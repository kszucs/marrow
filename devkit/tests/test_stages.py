# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""A stream stage calls register cores, never an array entry.

The stages and packers in `kernels/compose.mojo` and `kernels/packing.mojo`
compute a register at a time, through each kernel's `core`. An array entry
-- `apply` or `dispatch` -- takes and answers whole arrays, so a stage that
called one would leave the register loop it is composed into. This check
reads source text, so it runs in the devkit suite rather than waiting on a
compile.
"""

import re

from devkit.mojo import Repo
from devkit.tests.test_errors import code_lines

# `.apply(` or `.dispatch(`, and the explicit-parameter forms `.apply[T](`
# and `.dispatch[T](`, which call the same entries.
_ARRAY_ENTRY = re.compile(r"\.(?:apply|dispatch)[(\[]")

_STAGE_MODULES = ("kernels/compose.mojo", "kernels/packing.mojo")


def test_stream_stages_call_only_cores():
    """A stage in kernels/compose.mojo or kernels/packing.mojo calls register cores, never an
    array entry: `.apply` or `.dispatch` followed by `(` or `[`, outside comments and docstrings."""
    repo = Repo.locate()
    offenders = [
        f"{path.relative_to(repo.root)}:{number}: {line.strip()}"
        for path in (repo.package_dir / module for module in _STAGE_MODULES)
        for number, line in code_lines(path)
        if _ARRAY_ENTRY.search(line.split("#", 1)[0])
    ]
    assert not offenders, (
        "a stream stage calls an array entry -- call the kernel's `core`:\n"
        + "\n".join(offenders)
    )

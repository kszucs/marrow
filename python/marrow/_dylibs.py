"""Point marrow's `dlopen` search at the libraries shipped inside the package.

marrow opens its optional C libraries -- the Parquet page codecs and
`libopendal_c` -- with `dlopen` rather than linking them, so nothing in
`libmarrow.so`'s load commands mentions them. That is what keeps a missing
codec a recoverable error instead of an import failure, and it is also why the
dynamic loader has no idea where they are.

For an AOT binary marrow answers that from `argv[0]`. A wheel cannot: the
process is `python`, so `argv[0]` names the interpreter and never
`site-packages/marrow/`. This module supplies the missing directory through
`MARROW_DYLIB_DIR`, which `marrow/utils/dylib.mojo` consults ahead of the bare
soname.

Set before `libmarrow` is imported, and only when unset, so a user pointing
`MARROW_DYLIB_DIR` at their own build keeps that choice.
"""

import os
from pathlib import Path

DYLIB_DIR_ENV = "MARROW_DYLIB_DIR"


def configure() -> None:
    """Export `MARROW_DYLIB_DIR` as this package's directory, if unset."""
    os.environ.setdefault(DYLIB_DIR_ENV, str(Path(__file__).parent))

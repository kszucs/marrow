"""The wheel: what it ships, and whether it says so.

`python/marrow/compile.py` owns the tables of libraries marrow stages into a
wheel or a `marrow compile --bundle` directory. It ships inside the wheel and
imports only the standard library, so devkit reaches it by path rather than by
`import marrow.compile` -- that import would run the package `__init__` and
load `libmarrow.so`, which nothing under `devkit/` may do at import time.
"""

import importlib.util


def compile_module(repo):
    """`python/marrow/compile.py`, loaded by path from `repo`."""
    path = repo.python_dir / repo.PACKAGE / "compile.py"
    spec = importlib.util.spec_from_file_location("_marrow_compile", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

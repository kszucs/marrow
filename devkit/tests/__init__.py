# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""The devkit suite.

A package, unlike the repository's other test directories, because these modules
share a namespace with them under pytest's rootless import mode: this tree's
`test_arrays.py` and `python/marrow/tests/test_arrays.py` would otherwise
collide on a bare module name and abort collection.  Being inside a real
package makes them `devkit.tests.*` and immune to it.

It also holds what several modules read: the checkout, `compile.py`'s tables
(the wheel suites), and the parsed `python/pyproject.toml`.
"""

import tomllib

from devkit.mojo import Repo
from devkit.wheel import compile_module

REPO = Repo.locate()
CATALOG = compile_module(REPO)
PYPROJECT = tomllib.loads(REPO.pyproject.read_text())

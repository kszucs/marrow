"""Pytest wiring for the golden corpus.

The machinery lives in `devkit.golden`; this file exists only because pytest
requires hooks to be in a file called `conftest.py`.

`prepare()` runs at **import**, not from a hook, because the artefacts it writes
are what collection reads: `golden/generated/test_cases.mojo` has to exist
before `pytest_collect_file` reaches it, and a subdirectory conftest is imported
before the files beside it are collected.
"""

import sys
from pathlib import Path

import pytest

# `helpers` and `test_cases` import each other as top-level modules, and a
# traceback out of a case has to point at `golden/cases/<name>.mojo`.
sys.path.insert(0, str(Path(__file__).parent))

from devkit.golden import corpus  # noqa: E402 - follows the path insertion
from devkit.runner import RunnerOptions  # noqa: E402

corpus().prepare()


def pytest_collection_modifyitems(items):
    """Apply each case's `-- xfail` to *both* lanes.

    Done here rather than in `install_python_lane` because a Mojo case's item is
    built by the repository conftest's collector, which golden has no hand in.
    Marking by item name reaches both, and the name is identical in the two
    lanes by construction.
    """
    reasons = corpus().xfail_reasons()
    for item in items:
        reason = reasons.get(item.name)
        if reason is not None:
            item.add_marker(pytest.mark.xfail(reason=reason, strict=True))


def pytest_configure(config):
    # Through `RunnerOptions` rather than a second `getoption`, so the option
    # has exactly one reader and cannot drift from its declaration.
    corpus().num_threads = RunnerOptions.from_config(config).num_threads

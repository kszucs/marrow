"""Pytest wiring for the golden corpus.

A shim. Everything it does lives in `runner.py`; this file exists only because
pytest requires hooks to be in a file called `conftest.py`.

`prepare()` runs at **import**, not from a hook, because the artefacts it
writes are what collection reads: `golden/test_cases.mojo` has to exist before
`pytest_collect_file` reaches it, and a subdirectory conftest is imported
before the files beside it are collected.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import runner  # noqa: E402 — must follow the path insertion

runner.prepare()


def pytest_configure(config):
    runner.MORSEL_SIZE = config.getoption("--morsel-size")
    runner.NUM_THREADS = config.getoption("--num-threads")

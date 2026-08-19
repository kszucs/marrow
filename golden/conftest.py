"""Case discovery and comparison for the golden corpus.

The `golden` fixture is the only new machinery on the Python side: pytest
collects `golden/test_*.py` natively, and the AOT twins in `golden/test_*.mojo`
are collected by the Mojo harness in the root `conftest.py` with no changes at
all.

On import this module parses every `test_*.exp` and materialises one Arrow
IPC file per case under `golden/.exp/`, which is what the Mojo lane reads —
Mojo has no JSON library, and a typed IPC file needs no parser. The cache is
derived from the committed text, so it cannot drift from what a reviewer saw.
"""

import sys
from pathlib import Path

import pytest
import pyarrow as pa

sys.path.insert(0, str(Path(__file__).parent))
import fixtures  # noqa: E402
from expfmt import parse  # noqa: E402

import marrow

DIR = Path(__file__).parent
CACHE = DIR / ".exp"


def _load():
    tables = {}
    for path in sorted(DIR.glob("test_*.exp")):
        tables.update(parse(path.read_text()))
    return tables


EXPECTED = _load()


def _write_cache():
    """One Arrow IPC file per case, for the AOT lane to read."""
    CACHE.mkdir(exist_ok=True)
    for name, table in EXPECTED.items():
        # `write_table` emits *no* batch for a zero-row table, and an empty
        # result is a normal query outcome — so write the batches explicitly
        # and synthesise one empty batch when there are none. Otherwise the
        # Mojo lane reads a file with 0 batches and cannot tell "empty result"
        # from "expectation missing".
        batches = table.to_batches() or [
            pa.record_batch(
                [pa.array([], type=f.type) for f in table.schema],
                schema=table.schema,
            )
        ]
        with pa.ipc.new_file(CACHE / f"{name}.arrow", table.schema) as writer:
            for batch in batches:
                writer.write_batch(batch)


_write_cache()


class Golden:
    def __init__(self, request):
        self.name = request.node.name
        self.morsel_size = request.config.getoption("--morsel-size")
        self.num_threads = request.config.getoption("--num-threads")

    def table(self, name):
        """The fixture, as an in-memory source — never a file scan.

        What is under test is the engine, so the source is a memtable in every
        lane; Parquet and IPC keep their own suites.
        """
        batch = marrow.read_ipc_file(str(fixtures.path(name)))[0]
        return marrow.memtable(batch, morsel_size=self.morsel_size)

    def check(self, plan):
        if self.name not in EXPECTED:
            raise AssertionError(
                f"{self.name}: no expectation. Run "
                f"`pixi run -e bench python golden/regenerate.py`."
            )
        expected = EXPECTED[self.name]
        actual = pa.table(plan.to_pyarrow(num_threads=self.num_threads))
        if actual.equals(expected):
            return
        raise AssertionError(
            f"{self.name} does not match its expectation\n\n"
            f"--- expected (duckdb) ---\n{expected}\n"
            f"--- actual (marrow) ---\n{actual}\n"
        )


@pytest.fixture
def golden(request):
    return Golden(request)

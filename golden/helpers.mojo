"""The vocabulary a golden case body may use, on the Mojo side.

`helpers.py` is the same list for the runtime lane. Not named `test_*`, so the
harness never collects this file — `pytest_collect_file` only picks up
`test_*.mojo`, and the collectable module is the generated `test_cases.mojo`.

Imports are **absolute**: `golden/` sits outside the `marrow/` package and is
reached through the `-I .` the harness passes, exactly as `benchmarks/` is.
Paths are relative to the repository root, the working directory the generated
driver runs from.
"""

from marrow.arrays import DynArray
from marrow.expr.builders import table as _in_memory_table
from marrow.expr.logical import DynRelation
from marrow.ipc import read_ipc_file
from marrow.kernels.aggregate import AllKernel
from marrow.kernels.numeric import equal
from marrow.tabular import RecordBatch


def read_one(var path: String) raises -> RecordBatch:
    """The single batch in an IPC file, moved out rather than copied."""
    var batches = read_ipc_file(path)
    if len(batches) != 1:
        raise Error(String(path, ": expected 1 batch, got ", len(batches)))
    return batches.pop()


def table(var name: String) raises -> DynRelation:
    """A fixture as an in-memory source — never a file scan.

    What is under test is the engine, so the source is a memtable in every
    lane; Parquet and IPC keep their own suites.
    """
    return _in_memory_table(
        read_one(String("golden/fixtures/", name, ".arrow"))
    )


def values_equal(a: DynArray, b: DynArray) raises -> Bool:
    """Value equality, which is *not* what `==` on arrays means.

    `==` is structural on every array type — offset, buffers, whether a
    validity bitmap is present — so two columns holding identical values
    compare unequal when one was written by pyarrow into the expectation file
    and the other came out of a marrow kernel. `LIMIT ... OFFSET` shows it
    plainly: the result is a zero-copy window at a non-zero offset.

    So compare the way `assert_values_equal` does: the null pattern position
    by position, then the valid slots through the eq kernel, which raises on a
    dtype it has no equality for rather than answering something else.
    """
    if a.dtype() != b.dtype() or len(a) != len(b):
        return False
    for i in range(len(a)):
        if a.is_null(i) != b.is_null(i):
            return False
    return AllKernel.reduce(equal(a, b))


def check(var name: String, plan: DynRelation) raises:
    """Run the plan and hold it to the shared expectation.

    `plan` is **borrowed**, not owned: an owned parameter would make Mojo want
    `check(q^)` at the call site, and `^` has no Python reading, so the one
    spelling would stop being one spelling. `execute` borrows its receiver, so
    borrowing here gives nothing up. (`DynRelation` used to be
    `ImplicitlyCopyable`, which is why cases could write `var q = ...; return q`;
    it no longer is, so a case body inlines its plan into the `return` instead.)

    Schema, then row count, then columns — reported separately, because the
    three mean different things: a schema mismatch is a dtype or naming bug, a
    row-count mismatch is usually null semantics in a predicate, and a column
    mismatch is the arithmetic itself. `assert_true(a == b)` collapses all
    three into "condition was unexpectedly False".
    """
    var expected = read_one(String("golden/.exp/", name, ".arrow"))
    var actual = plan.execute()

    if actual.schema != expected.schema:
        raise Error(
            String(
                name,
                ": schema mismatch\n  expected ",
                expected.schema,
                "\n  actual   ",
                actual.schema,
            )
        )
    if actual.num_rows() != expected.num_rows():
        raise Error(
            String(
                name,
                ": row count ",
                actual.num_rows(),
                " != expected ",
                expected.num_rows(),
                "\n  expected ",
                expected,
                "\n  actual   ",
                actual,
            )
        )
    if len(actual.columns) != len(expected.columns):
        # Never index past the end: a bounds assert kills the whole test
        # binary, not one case, so an inconsistent batch here would mask every
        # other case in the unit. marrow currently returns an empty result as
        # a batch whose schema names its fields but whose column list is
        # empty, which is exactly that shape.
        raise Error(
            String(
                name,
                ": column count ",
                len(actual.columns),
                " != expected ",
                len(expected.columns),
                " (schema says ",
                len(actual.schema.fields),
                " fields)",
            )
        )
    for i in range(len(expected.columns)):
        if not values_equal(actual.columns[i], expected.columns[i]):
            raise Error(
                String(
                    name,
                    ": column '",
                    expected.schema.fields[i].name,
                    "' differs\n  expected ",
                    expected.columns[i],
                    "\n  actual   ",
                    actual.columns[i],
                )
            )

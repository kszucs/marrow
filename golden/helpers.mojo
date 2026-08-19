"""Shared machinery for the golden AOT cases.

Not named `test_*`, so the harness never collects it — `pytest_collect_file`
(`conftest.py:804`) only picks up `test_*.mojo`.

Imports are **absolute**: `golden/` sits outside the `marrow/` package and is
reached through the `-I .` the harness passes, exactly as `benchmarks/` is.
Paths are relative to the repository root, the working directory the generated
driver runs from.
"""

from marrow.arrays import DynArray
from marrow.dtypes import DynType, bool_, float64, int64
from marrow.expr.relations import DynRelation
from marrow.ipc import read_ipc_file
from marrow.tabular import RecordBatch


def read_one(var path: String) raises -> RecordBatch:
    """The single batch in an IPC file, moved out rather than copied."""
    var batches = read_ipc_file(path)
    if len(batches) != 1:
        raise Error(String(path, ": expected 1 batch, got ", len(batches)))
    return batches.pop()


def fixture(var name: String) raises -> RecordBatch:
    return read_one(String("golden/fixtures/", name, ".arrow"))


def values_equal(a: DynArray, b: DynArray) raises -> Bool:
    """Value equality, which is *not* what `DynArray.__eq__` means.

    `DynArray.__eq__` delegates to `ArrayData.__eq__`, which compares the
    physical layout — offset, padding, whether a validity bitmap is present.
    Two columns holding identical values compare unequal there when one was
    written by pyarrow into the expectation file and the other came out of a
    marrow kernel. The typed arrays' `__eq__` is the logical one ("same
    length, null pattern, and values", offset-aware), so narrow first.

    The ladder is closed on purpose: it covers the dtypes the corpus uses and
    raises on anything else, rather than quietly falling back to a comparison
    that means something different.
    """
    var dt = a.dtype()
    if dt != b.dtype():
        return False
    if dt.is_string_like():
        return a.as_string() == b.as_string()
    elif dt == DynType(int64):
        return a.as_int64() == b.as_int64()
    elif dt == DynType(float64):
        return a.as_float64() == b.as_float64()
    elif dt == DynType(bool_):
        return a.as_bool() == b.as_bool()
    else:
        raise Error(String("golden: no value comparison for dtype ", dt))


def check(var name: String, var plan: DynRelation) raises:
    """Run the plan and hold it to the shared expectation.

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

"""Late-bound parameters.

A parameter is a literal whose value arrives later, so these cases check the
two things that distinguish it from `NumericLiteral`: that binding reaches every use
site — including one nested inside a fused subtree — and that the value
belongs to an execution rather than to the plan.
"""

from std.testing import assert_equal, assert_raises, assert_true

from ...builders import array
from ...dtypes import DynType, Int64Type, field, float64, int64, string
from ...parquet.writer import write_table
from ...scalars import Int64Scalar, StringScalar
from ...schema import schema
from ...tabular import Table, record_batch
from ..builders import col, param, scan, table
from ..bindings import Bindings
from ..logical import DynRelation, DynValue, InMemoryTable
from ..optimizer import AllRules, ScanPruning
from ..`comptime`.leaves import NumericColumn
from ..`comptime`.boolean import Not
from ..`comptime`.numeric import Gt


def _table() raises -> DynRelation:
    return table(record_batch([array([1, 5, 9], int64).copy()], names=["a"]))


def test_a_value_is_supplied_per_execution() raises:
    """The rule this design exists to obey: a logical node is stateless.

    The plan holds no value. Two executions of the *same* plan with different
    bindings give different answers and cannot interfere — an earlier version
    kept the value in a shared cell inside the node, so `set()` reached into a
    built plan and changed what it computed.
    """
    var min_a = param("min-a", int64)
    var plan = _table().filter((col("a", int64) > min_a.copy()))

    var low = plan.execute(bindings={"min-a": Int64Scalar(4).to_dyn()})
    assert_true(low.columns[0].as_int64() == array([5, 9], int64))

    var high = plan.execute(bindings={"min-a": Int64Scalar(8).to_dyn()})
    assert_true(high.columns[0].as_int64() == array([9], int64))


def test_binding_reaches_a_nested_parameter() raises:
    """`bind` walks the whole tree, so depth does not matter.

    The parameter here is the right operand of a `Gt` that is itself the
    operand of a `Not` — two composites deep. `to_operator` copies a node
    without descending into it, so binding at that seam would leave this one
    unread; the per-batch walk is what reaches it.
    """
    var t = param("t", int64)
    var plan = _table().filter(Not((col("a", int64) > t.copy())))

    var got = plan.execute(bindings={"t": Int64Scalar(4).to_dyn()})
    assert_true(got.columns[0].as_int64() == array([1], int64))


def test_an_unbound_parameter_names_itself() raises:
    """The previous expression package's cell raises "parameter is not bound" without naming it,
    because a cell cannot know the name it is read through. Here the node is
    the parameter, so it can."""
    var missing = param("threshold", int64)
    var plan = _table().filter((col("a", int64) > missing.copy()))

    var raised = False
    try:
        _ = plan.execute()
    except e:
        raised = True
        assert_true("threshold" in String(e))
    assert_true(raised)


def test_a_default_is_used_until_something_binds() raises:
    var t = param("t", int64, default=Optional(Int64(4)))
    var plan = _table().filter((col("a", int64) > t.copy()))
    assert_true(plan.execute().columns[0].as_int64() == array([5, 9], int64))

    var bound = plan.execute(bindings={"t": Int64Scalar(8).to_dyn()})
    assert_true(bound.columns[0].as_int64() == array([9], int64))


# ---------------------------------------------------------------------------
# params() — what a plan declares
# ---------------------------------------------------------------------------


def _param_names(plan: DynRelation) raises -> List[String]:
    var specs = plan.params()
    var out = List[String]()
    for ref p in specs:
        out.append(p.name.copy())
    return out^


def test_params_are_distinct_by_name_in_first_seen_order() raises:
    """A name read twice is one parameter: `Bindings` is keyed by name, so
    both reads see the same value."""
    var t = param("t", int64)
    var plan = _table().filter(
        (col("a", int64) > t.copy())
        & (col("a", int64) < param("u", int64))
        & (col("a", int64) != t.copy())
    )
    assert_true(_param_names(plan) == ["t", "u"])


def test_one_name_read_as_two_types_is_refused_when_it_binds() raises:
    """`params()` reports the first declaration; the read that disagrees is
    refused where the value binds, naming the parameter."""
    var plan = _table().filter(
        (col("a", int64) > param("t", int64))
        & (col("a", int64).cast(float64, safe=False) < param("t", float64))
    )
    var specs = plan.params()
    assert_equal(len(specs), 1)
    assert_true(specs[0].dtype == DynType(int64))
    with assert_raises(contains="parameter 't'"):
        _ = plan.execute(bindings={"t": Int64Scalar(4).to_dyn()})


def test_a_scan_path_is_a_parameter() raises:
    var path = String("/tmp/marrow_expr_scan_path_param.parquet")
    var b = record_batch([array([1, 5, 9], int64).copy()], names=["a"])
    write_table(Table.from_batches(b.schema.copy(), [b.copy()]), path)

    var plan = scan(param("src", string), b.schema.copy()).filter(
        col("a", int64) > param("t", int64)
    )
    assert_true("ParquetScan(param(src))" in String(plan), String(plan))

    var specs = plan.params()
    assert_equal(len(specs), 2)
    assert_equal(specs[0].name, String("src"))
    assert_true(specs[0].dtype == DynType(string))
    assert_true(not specs[0].default)
    assert_true(True if specs[0].parse else False)

    var got = plan.execute(
        bindings={
            "src": StringScalar(path.copy()).to_dyn(),
            "t": Int64Scalar(4).to_dyn(),
        }
    )
    assert_true(got.columns[0].as_int64() == array([5, 9], int64))

    with assert_raises(contains="src"):
        _ = plan.execute(bindings={"t": Int64Scalar(4).to_dyn()})


def test_a_scan_path_parameter_survives_optimization() raises:
    """Every rewrite that rebuilds a scan carries its path, because the path is
    a `ScanPath` rather than a string a rewrite could copy on its own."""
    var sch = schema([field("a", int64), field("b", int64)])
    var pruned = (
        scan(param("src", string), sch.copy())
        .filter(col("a", int64) > param("t", int64))
        .optimize[ScanPruning]()
    )
    assert_true(
        "ParquetScan(param(src)) pruned by 1" in String(pruned), String(pruned)
    )
    assert_true(_param_names(pruned) == ["src", "t"])

    var narrowed = (
        scan(param("src", string), sch^)
        .filter(col("a", int64) > param("t", int64))
        .select(["a"])
        .optimize[AllRules]()
    )
    assert_true("param(src)" in String(narrowed), String(narrowed))
    assert_true(_param_names(narrowed) == ["src", "t"])

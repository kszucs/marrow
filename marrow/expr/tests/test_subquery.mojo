# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""Subqueries: every shape `Decorrelate` plans, answered against data.

`emp` has a NULL `dept` (eid 5) and a `dept` no department has (eid 4), the two
rows that tell `EXISTS` from `IN`, `NOT EXISTS` from `NOT IN`, and a left join's
null from `count(*)`'s 0.
"""

from std.os.path import join
from std.python import Python
from std.testing import assert_equal, assert_false, assert_raises, assert_true

from ...arrays import DynArray
from ...builders import Int64Builder, array
from ...dtypes import field, int64
from ...formats.parquet.reader import ParquetFile
from ...schema import schema
from ...utils.testing import ScratchDir
from ...scalars import Int64Scalar
from ...tabular import RecordBatch, record_batch
from ..builders import (
    col,
    count_star,
    exists,
    is_in,
    lit,
    outer,
    scalar,
    scan,
    table,
)
from ..index import Index
from ..logical import Bindings, DynRelation, DynValue
from ..optimizer import AllRules, SubqueryRules
from ..runtime.values import eq, ne, not_


def _emp() raises -> DynRelation:
    return table(
        record_batch(
            [
                array([1, 2, 3, 4, 5], int64).copy(),
                array([10, 20, 20, 99, None], int64).copy(),
            ],
            names=["eid", "dept"],
        )
    )


def _dept() raises -> DynRelation:
    return table(
        record_batch(
            [
                array([10, 20, 30], int64).copy(),
                array([100, 300, 50], int64).copy(),
            ],
            names=["did", "budget"],
        )
    )


def _eids(plan: DynRelation) raises -> List[Int]:
    """The `eid` column of `plan`'s answer, sorted, after `AllRules`."""
    var batch = (
        plan.optimize[AllRules]().sort_by([col("eid", int64)], [True]).execute()
    )
    var out = List[Int]()
    ref eids = batch.column(batch.schema.get_field_index("eid")).as_int64()
    for i in range(len(eids)):
        out.append(Int(eids[i].value()))
    return out^


def _lit(v: Int) raises -> RuntimeValue:
    return lit(Int64Scalar(Int64(v)).to_dyn())


from ..runtime.values import RuntimeValue


def test_subquery_uncorrelated_scalar_filter() raises:
    var n = _dept().aggregate([count_star().alias("n")])
    var plan = _emp().filter(col("eid") > scalar(n))
    assert_equal(_eids(plan), [4, 5])
    assert_true("ScalarBinding" in String(plan.optimize[AllRules]()))


def test_subquery_scalar_without_rows_is_null() raises:
    var none = _dept().filter(col("did") > _lit(1000)).select(["did"])
    var plan = _emp().filter(col("eid") > scalar(none))
    assert_equal(len(_eids(plan)), 0)


def test_subquery_scalar_with_two_rows_raises() raises:
    var two = _dept().filter(col("did") > _lit(10)).select(["did"])
    var plan = _emp().filter(col("eid") > scalar(two))
    with assert_raises(contains="at most one"):
        _ = _eids(plan)


def test_subquery_unplanned_raises() raises:
    var n = _dept().aggregate([count_star().alias("n")])
    var plan = _emp().filter(col("eid") > scalar(n))
    with assert_raises(contains="was not planned"):
        _ = plan.execute()


def test_subquery_in_uncorrelated() raises:
    var plan = _emp().filter(is_in(col("dept"), _dept().select(["did"])))
    assert_equal(_eids(plan), [1, 2, 3])
    var planned = String(plan.optimize[SubqueryRules]())
    assert_true("semi" in planned)


def test_subquery_exists_correlated() raises:
    var sub = _dept().filter(eq(col("did"), outer("dept")))
    assert_equal(_eids(_emp().filter(exists(sub))), [1, 2, 3])


def test_subquery_not_exists_keeps_null_key() raises:
    """Employee 5's dept is NULL, which matches nothing, so `NOT EXISTS`
    holds."""
    var sub = _dept().filter(eq(col("did"), outer("dept")))
    assert_equal(_eids(_emp().filter(not_(exists(sub)))), [4, 5])


def test_subquery_uncorrelated_exists_is_a_binding() raises:
    var some = _dept().filter(col("did") > _lit(20))
    var none = _dept().filter(col("did") > _lit(1000))
    assert_equal(len(_eids(_emp().filter(exists(some)))), 5)
    assert_equal(len(_eids(_emp().filter(exists(none)))), 0)
    assert_equal(len(_eids(_emp().filter(not_(exists(none))))), 5)


def test_subquery_correlated_scalar_count_is_zero_unmatched() raises:
    """`count(*)` over no rows is 0, not the null a left join leaves."""
    var sub = (
        _dept()
        .filter(eq(col("did"), outer("dept")))
        .aggregate([count_star().alias("n")])
    )
    var batch = (
        _emp()
        .with_columns(["n"], [scalar(sub)])
        .optimize[AllRules]()
        .sort_by([col("eid", int64)], [True])
        .execute()
    )
    ref n = batch.column(batch.schema.get_field_index("n")).as_int64()
    assert_equal(len(n), 5)
    var expected = [1, 1, 1, 0, 0]
    for i in range(5):
        assert_true(n.is_valid(i))
        assert_equal(Int(n[i].value()), expected[i])


def test_subquery_correlated_scalar_filter() raises:
    var budget = (
        _dept()
        .filter(eq(col("did"), outer("dept")))
        .aggregate([col("budget", int64).sum().alias("b")])
    )
    var plan = _emp().filter(scalar(budget) > _lit(200))
    assert_equal(_eids(plan), [2, 3])


def test_subquery_nested_in_inside_exists() raises:
    """`EXISTS (SELECT … WHERE did = e.dept AND did IN (SELECT did …))` —
    the inner subquery is planned inside the outer one first."""
    var cheap = _dept().filter(col("budget") < _lit(200)).select(["did"])
    var sub = (
        _dept()
        .filter(eq(col("did"), outer("dept")))
        .filter(is_in(col("did"), cheap))
    )
    assert_equal(_eids(_emp().filter(exists(sub))), [1])


def test_subquery_decorrelate_is_idempotent() raises:
    var sub = (
        _dept()
        .filter(eq(col("did"), outer("dept")))
        .aggregate([count_star().alias("n")])
    )
    var plan = _emp().filter(scalar(sub) > _lit(0))
    var once = plan.optimize[SubqueryRules]()
    var twice = once.optimize[SubqueryRules]()
    assert_equal(String(once), String(twice))


def _lines() raises -> DynRelation:
    """Line items of three orders: order 1 has two suppliers, order 2 one,
    order 3 two, of which supplier 9 was late."""
    return table(
        record_batch(
            [
                array([1, 1, 2, 3, 3], int64).copy(),
                array([7, 8, 7, 7, 9], int64).copy(),
                array([0, 0, 0, 0, 1], int64).copy(),
            ],
            names=["eid", "supp", "late"],
        )
    )


def test_subquery_exists_with_residual() raises:
    """TPC-H Q21's shape: another line of the same order from a different
    supplier. Order 2 has none."""
    var other = (
        _lines()
        .filter(eq(col("eid"), outer("eid")))
        .filter(ne(col("supp"), outer("supp")))
    )
    var plan = _lines().filter(exists(other))
    assert_equal(_eids(plan), [1, 1, 3, 3])
    assert_true("ResidualJoin" in String(plan.optimize[AllRules]()))


def test_subquery_not_exists_with_residual() raises:
    """No *late* line from another supplier: order 3's supplier 7 sees
    supplier 9's late line, supplier 9 sees none of its own kind."""
    var late = (
        _lines()
        .filter(eq(col("eid"), outer("eid")))
        .filter(ne(col("supp"), outer("supp")))
        .filter(eq(col("late"), _lit(1)))
    )
    assert_equal(_eids(_lines().filter(not_(exists(late)))), [1, 1, 2, 3])


def test_subquery_not_in_with_null_in_subquery_keeps_nothing() raises:
    """`x NOT IN (10, NULL)` is never true: unknown where x is not 10."""
    var depts = _emp().select(["dept"])
    var plan = _emp().filter(not_(is_in(col("eid"), depts)))
    assert_equal(len(_eids(plan)), 0)


def test_subquery_not_in_null_operand() raises:
    """A null operand is unknown against a non-empty subquery and true
    against an empty one."""
    var depts = _dept().select(["did"])
    var plan = _emp().filter(not_(is_in(col("dept"), depts)))
    assert_equal(_eids(plan), [4])
    var none = _dept().filter(col("did") > _lit(1000)).select(["did"])
    var everything = _emp().filter(not_(is_in(col("dept"), none)))
    assert_equal(_eids(everything), [1, 2, 3, 4, 5])


def test_subquery_correlated_aggregate_reads_only_asked_groups() raises:
    """TPC-H Q17's shape: two parts against forty line items. The aggregate
    is semi-joined with the parts first, and the answer is unchanged."""
    var keys = Int64Builder(capacity=40)
    var qty = Int64Builder(capacity=40)
    for i in range(40):
        keys.append(Int64(i % 20))
        qty.append(Int64(i))
    var lines = table(
        record_batch([keys.finish(), qty.finish()], names=["pk", "qty"])
    )
    var parts = table(
        record_batch(
            [array([3, 7], int64).copy(), array([1, 2], int64).copy()],
            names=["eid", "pk_of"],
        )
    )
    var total = lines.filter(eq(col("pk"), outer("eid"))).aggregate(
        [col("qty", int64).sum().alias("s")]
    )
    var plan = parts.filter(scalar(total) > _lit(30))
    # pk 3 holds qty 3 and 23 (26); pk 7 holds 7 and 27 (34).
    assert_equal(_eids(plan), [7])
    assert_true("semi" in String(plan.optimize[AllRules]()))


def test_subquery_uncorrelated_scalar_prunes_row_groups() raises:
    """`a > (SELECT max(...))` is a binding by the time the scan reads its
    footer, so it skips row groups exactly as a literal would — and the
    plan answers the same rows."""
    with ScratchDir() as dir:
        var path = join(dir, "marrow_subquery_banded.parquet")
        var pa = Python.import_module("pyarrow")
        var pq = Python.import_module("pyarrow.parquet")
        var a = Python.list()
        for i in range(200):
            a.append(i)
        pq.write_table(
            pa.table(Python.dict(a=pa.array(a))),
            path,
            row_group_size=50,
            compression="none",
        )
        var threshold = table(
            record_batch([array([150], int64).copy()], names=["t"])
        ).aggregate([col("t", int64).max().alias("m")])
        var value = col("a") > scalar(threshold)

        var sq = DynValue(value.copy()).subqueries()[0].copy()
        var bindings = Bindings()
        bindings[sq.slot()] = Int64Scalar(Int64(150)).to_dyn()
        var keep = Index.from_parquet(ParquetFile(path)).read_plan(
            [DynValue(value.copy())], bindings
        )
        assert_equal(len(keep), 1)
        assert_equal(keep[0], 3)

        var out = (
            scan(path, schema([field("a", int64)]))
            .filter(value^)
            .optimize[AllRules]()
            .execute()
        )
        assert_equal(out.num_rows(), 49)


def test_subquery_shared_relation_reads_once() raises:
    """A `WITH` query read twice — as a join input and inside a scalar
    subquery, TPC-H Q15's shape — answers as if inlined twice."""
    var revenue = (
        _lines()
        .aggregate([col("late", int64).sum().alias("n")], [col("supp")])
        .shared()
    )
    var top = revenue.aggregate([col("n", int64).max().alias("m")])
    var best = revenue.filter(col("n") >= scalar(top))
    var batch = best.optimize[AllRules]().execute()
    assert_equal(batch.num_rows(), 1)
    ref supp = batch.column(batch.schema.get_field_index("supp")).as_int64()
    assert_equal(Int(supp[0].value()), 9)
    assert_true("Shared#" in String(best.optimize[AllRules]()))

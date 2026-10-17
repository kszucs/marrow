# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""Subqueries spelled in the comptime lane: `scalar(rel, dtype)` and
`outer(name, dtype)` as typed leaves, planned by the same `Decorrelate` as the
runtime lane's, over the fixtures of `test_subquery.mojo`.
"""

from std.os.path import join
from std.python import Python
from std.testing import assert_equal, assert_false, assert_raises, assert_true

from ...builders import Date32Builder, Decimal128Builder, array
from ...dtypes import date32, decimal128, field, int32, int64, string
from ...formats.parquet.reader import ParquetFile
from ...scalars import Int64Scalar
from ...schema import schema
from ...tabular import record_batch
from ...utils.testing import ScratchDir
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
from ..logical import Bindings, DynRelation, DynValue, ParquetScan
from ..optimizer import AllRules, SubqueryRules
from ..runtime.values import not_


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
                array(["ten", "twenty", "thirty"]).copy(),
            ],
            names=["did", "budget", "dname"],
        )
    )


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


def _count() raises -> DynRelation:
    return _dept().aggregate([count_star().alias("n")])


# ---------------------------------------------------------------------------
# uncorrelated scalar
# ---------------------------------------------------------------------------
def test_comptime_subquery_uncorrelated_scalar_filter() raises:
    var plan = _emp().filter(col("eid", int64) > scalar(_count(), int64))
    assert_equal(_eids(plan), [4, 5])
    assert_true("ScalarBinding" in String(plan.optimize[AllRules]()))


def test_comptime_subquery_scalar_without_rows_is_null() raises:
    var none = (
        _dept().filter(col("did", int64) > lit(1000, int64)).select(["did"])
    )
    var plan = _emp().filter(col("eid", int64) > scalar(none, int64))
    assert_equal(len(_eids(plan)), 0)


def test_comptime_subquery_scalar_projected() raises:
    """Broadcast into a column, as a parameter is."""
    var batch = (
        _emp()
        .with_columns(["n"], [scalar(_count(), int64) + lit(1, int64)])
        .optimize[AllRules]()
        .execute()
    )
    ref n = batch.column(batch.schema.get_field_index("n")).as_int64()
    assert_equal(len(n), 5)
    for i in range(5):
        assert_equal(Int(n[i].value()), 4)


def test_comptime_subquery_unplanned_raises() raises:
    var plan = _emp().filter(col("eid", int64) > scalar(_count(), int64))
    with assert_raises(contains="was not planned"):
        _ = plan.execute()


def test_comptime_subquery_declared_type_is_checked() raises:
    var plan = _emp().filter(col("eid", int64) > scalar(_count(), int32))
    with assert_raises(contains="is declared int32"):
        _ = plan.optimize[AllRules]().execute()


def test_comptime_subquery_scalar_string() raises:
    var name = (
        _dept().filter(col("did", int64) == lit(20, int64)).select(["dname"])
    )
    var plan = _dept().filter(col("dname", string) == scalar(name, string))
    var batch = plan.optimize[AllRules]().execute()
    assert_equal(batch.num_rows(), 1)
    ref did = batch.column(batch.schema.get_field_index("did")).as_int64()
    assert_equal(Int(did[0].value()), 20)


def _days() raises -> DynRelation:
    var d = Date32Builder(date32(), 3)
    d.append(Scalar[DType.int32](19000))
    d.append(Scalar[DType.int32](19005))
    d.append(Scalar[DType.int32](19010))
    var p = Decimal128Builder(decimal128(10, 2), 3)
    p.append(Scalar[DType.int128](150))
    p.append(Scalar[DType.int128](250))
    p.append(Scalar[DType.int128](350))
    return table(
        record_batch(
            [
                array([1, 2, 3], int64).copy(),
                d.finish().to_dyn(),
                p.finish().to_dyn(),
            ],
            names=["k", "d", "p"],
        )
    )


def test_comptime_subquery_scalar_temporal_and_decimal() raises:
    var day = _days().filter(col("k", int64) == lit(2, int64)).select(["d"])
    var later = _days().filter(col("d", date32()) > scalar(day, date32()))
    var batch = later.optimize[AllRules]().execute()
    assert_equal(batch.num_rows(), 1)

    var price = _days().filter(col("k", int64) == lit(1, int64)).select(["p"])
    var dearer = _days().filter(
        col("p", decimal128(10, 2)) > scalar(price, decimal128(10, 2))
    )
    assert_equal(dearer.optimize[AllRules]().execute().num_rows(), 2)


# ---------------------------------------------------------------------------
# correlated: outer(name, dtype)
# ---------------------------------------------------------------------------
def _in_dept() raises -> DynRelation:
    return _dept().filter(col("did", int64) == outer("dept", int64))


def test_comptime_subquery_equality_is_a_correlation() raises:
    var eq = DynValue(col("did", int64) == outer("dept", int64))
    assert_true(eq.correlation())
    assert_equal(eq.correlation().value()[0], "did")
    assert_equal(eq.correlation().value()[1], "dept")
    var named = DynValue(col("dname", string) == outer("name", string))
    assert_true(named.correlation())
    var dated = DynValue(col("d", date32()) == outer("d", date32()))
    assert_true(dated.correlation())
    assert_false(
        DynValue(col("did", int64) > outer("dept", int64)).correlation()
    )
    assert_false(
        DynValue(
            (col("did", int64) + lit(1, int64)) == outer("dept", int64)
        ).correlation()
    )
    assert_false(DynValue(col("did", int64) == lit(1, int64)).correlation())


def test_comptime_subquery_exists_correlated() raises:
    var plan = _emp().filter(exists(_in_dept()))
    assert_equal(_eids(plan), [1, 2, 3])
    assert_true("semi" in String(plan.optimize[SubqueryRules]()))


def test_comptime_subquery_not_exists_keeps_null_key() raises:
    var plan = _emp().filter(not_(exists(_in_dept())))
    assert_equal(_eids(plan), [4, 5])
    assert_true("anti" in String(plan.optimize[SubqueryRules]()))


def test_comptime_subquery_in_and_not_in() raises:
    var depts = _dept().select(["did"])
    assert_equal(
        _eids(_emp().filter(is_in(col("dept", int64), depts))), [1, 2, 3]
    )
    assert_equal(
        _eids(_emp().filter(not_(is_in(col("dept", int64), depts)))), [4]
    )


def test_comptime_subquery_correlated_scalar_count_is_zero_unmatched() raises:
    var sub = _in_dept().aggregate([count_star().alias("n")])
    var batch = (
        _emp()
        .with_columns(["n"], [scalar(sub, int64)])
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


def test_comptime_subquery_correlated_scalar_filter() raises:
    var budget = _in_dept().aggregate([col("budget", int64).sum().alias("b")])
    var plan = _emp().filter(scalar(budget, int64) > lit(200, int64))
    assert_equal(_eids(plan), [2, 3])


def test_comptime_subquery_exists_with_residual() raises:
    """TPC-H Q21's shape, the residual a fused predicate reading the outer
    row's `supp`."""
    var other = (
        _lines()
        .filter(col("eid", int64) == outer("eid", int64))
        .filter(col("supp", int64) != outer("supp", int64))
    )
    var plan = _lines().filter(exists(other))
    assert_equal(_eids(plan), [1, 1, 3, 3])
    assert_true("ResidualJoin" in String(plan.optimize[AllRules]()))

    var late = other.filter(col("late", int64) == lit(1, int64))
    assert_equal(_eids(_lines().filter(not_(exists(late)))), [1, 1, 2, 3])


# ---------------------------------------------------------------------------
# pruning: an uncorrelated scalar prunes as a parameter does
# ---------------------------------------------------------------------------
def _write_banded(path: String, groups: Int, rows: Int) raises:
    """Group `g` holds `a` in `[g*rows, (g+1)*rows)`."""
    var pa = Python.import_module("pyarrow")
    var pq = Python.import_module("pyarrow.parquet")
    var a = Python.list()
    for g in range(groups):
        for i in range(rows):
            a.append(g * rows + i)
    var tbl = pa.table(Python.dict(a=pa.array(a)))
    pq.write_table(tbl, path, row_group_size=rows, compression="none")


def _pruners(plan: DynRelation) raises -> Optional[List[DynValue]]:
    """The predicates pushed into `plan`'s Parquet scan."""
    if plan.isa[ParquetScan]():
        return plan.get[ParquetScan]().pruners.copy()
    for ref child in plan.children():
        var found = _pruners(child)
        if found:
            return found^
    return None


def test_comptime_subquery_scalar_prunes_row_groups() raises:
    with ScratchDir() as dir:
        var path = join(dir, "marrow_subquery_banded.parquet")
        _write_banded(path, groups=4, rows=50)
        var threshold = table(
            record_batch([array([150], int64).copy()], names=["m"])
        )
        var plan = scan(path, schema([field("a", int64)])).filter(
            col("a", int64) > scalar(threshold, int64)
        )
        var optimized = plan.optimize[AllRules]()
        assert_true("ScalarBinding" in String(optimized))

        var out = optimized.execute()
        assert_equal(out.num_rows(), 49)
        ref a = out.column("a").as_int64()
        assert_equal(Int(a[0].value()), 151)

        # What the scan operator decides once the binding is in hand: the
        # pushed predicate reads the slot, and keeps one group of four.
        var pruners = _pruners(optimized).value().copy()
        assert_equal(len(pruners), 1)
        var slot = plan.subqueries()[0].slot()
        var idx = Index.from_parquet(ParquetFile(path))
        var bound = Bindings()
        bound[slot] = Int64Scalar(150).to_dyn()
        assert_equal(idx.read_plan(pruners, bound), [3])
        # Without the binding nothing is proven and every group is read.
        assert_equal(len(idx.read_plan(pruners)), 4)

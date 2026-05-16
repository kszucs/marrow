"""Tests for faszom static expression layer: ColumnRef, Pipeline, FilterPipeline."""

from std.testing import assert_equal, assert_true
from marrow.testing import TestSuite
from marrow.arrays import AnyArray
from marrow.builders import arange, array
from marrow.dtypes import Int32Type, int32, Field
from marrow.schema import Schema
from marrow.tabular import RecordBatch
from marrow.faszom import (
    Add,
    Column,
    ColumnRef,
    FilterExpr,
    FilterPipeline,
    GtExpr,
    Pipeline,
    col,
    execute,
    filter_pipeline,
)


def _make_batch(n: Int) raises -> RecordBatch:
    """Build a RecordBatch with columns a, b, c, data — all int32."""
    var schema = Schema(
        fields=[Field("a", int32), Field("b", int32), Field("c", int32), Field("data", int32)]
    )
    var ca: AnyArray = arange[Int32Type](0, n)
    var cb: AnyArray = arange[Int32Type](1, n + 1)
    var cc: AnyArray = arange[Int32Type](n // 2, n // 2 + n)
    var cd: AnyArray = arange[Int32Type](0, n)
    var columns = List[AnyArray]()
    columns.append(ca^)
    columns.append(cb^)
    columns.append(cc^)
    columns.append(cd^)
    return RecordBatch(schema=schema, columns=columns^)


def test_column_ref_same_result_as_column() raises:
    """ColumnRef bound via bind() produces the same exec_core output as Column."""
    var n = 16
    var batch = _make_batch(n)

    var a_arr = arange[Int32Type](0, n)
    var b_arr = arange[Int32Type](1, n + 1)
    var c_arr = arange[Int32Type](n // 2, n // 2 + n)

    # Column path — arrays embedded upfront
    var col_expr = Add(Column(a_arr.copy()), Column(b_arr.copy()))
    var col_result = execute(col_expr, n)

    # ColumnRef path — bound at execute time
    var ref_expr = Add(ColumnRef['a', Int32Type](), ColumnRef['b', Int32Type]())
    ref_expr.bind(batch)
    var ref_result = execute(ref_expr, n)

    assert_true(col_result == ref_result)


def test_col_convenience_same_as_column_ref() raises:
    """Col convenience function col['a'](int32) produces the same result as ColumnRef['a', Int32Type]()."""
    var n = 16
    var batch = _make_batch(n)

    var expr1 = Add(ColumnRef['a', Int32Type](), ColumnRef['b', Int32Type]())
    expr1.bind(batch)

    var expr2 = Add(col['a'](int32), col['b'](int32))
    expr2.bind(batch)

    assert_true(execute(expr1, n) == execute(expr2, n))


def test_filter_pipeline_matches_filter_expr() raises:
    """FilterPipeline output matches FilterExpr with arrays passed upfront."""
    var n = 100
    var batch = _make_batch(n)

    # FilterExpr with concrete arrays
    var a_arr = arange[Int32Type](0, n)
    var b_arr = arange[Int32Type](1, n + 1)
    var c_arr = arange[Int32Type](n // 2, n // 2 + n)
    var data_arr = arange[Int32Type](0, n)
    var direct = execute(
        FilterExpr(
            data_arr^,
            GtExpr(Add(Column(a_arr^), Column(b_arr^)), Column(c_arr^)),
        ),
        n,
    )

    # FilterPipeline via ColumnRef
    var pipeline = FilterPipeline['data', Int32Type, GtExpr[Add[ColumnRef['a', Int32Type], ColumnRef['b', Int32Type]], ColumnRef['c', Int32Type]]](
        GtExpr(Add(ColumnRef['a', Int32Type](), ColumnRef['b', Int32Type]()), ColumnRef['c', Int32Type]())
    )
    var piped = pipeline(batch)

    assert_true(direct == piped)


def test_filter_pipeline_convenience_syntax() raises:
    """FilterPipeline convenience function produces the same result as the bracket form."""
    var n = 100
    var batch = _make_batch(n)

    var p1 = FilterPipeline['data', Int32Type, GtExpr[Add[ColumnRef['a', Int32Type], ColumnRef['b', Int32Type]], ColumnRef['c', Int32Type]]](
        GtExpr(Add(ColumnRef['a', Int32Type](), ColumnRef['b', Int32Type]()), ColumnRef['c', Int32Type]())
    )

    var p2 = filter_pipeline['data'](
        GtExpr(Add(col['a'](int32), col['b'](int32)), col['c'](int32)),
        int32,
    )

    assert_true(p1(batch) == p2(batch))


def test_filter_pipeline_reuse() raises:
    """FilterPipeline called on two different batches gives correct results for each."""
    var n = 64

    var schema = Schema(
        fields=[Field("a", int32), Field("b", int32), Field("c", int32), Field("data", int32)]
    )

    # batch1: a=[0..n), b=[1..n+1), c=[n/2..3n/2), data=[0..n)
    var ca1: AnyArray = arange[Int32Type](0, n)
    var cb1: AnyArray = arange[Int32Type](1, n + 1)
    var cc1: AnyArray = arange[Int32Type](n // 2, n // 2 + n)
    var cd1: AnyArray = arange[Int32Type](0, n)
    var cols1 = List[AnyArray]()
    cols1.append(ca1^)
    cols1.append(cb1^)
    cols1.append(cc1^)
    cols1.append(cd1^)
    var batch1 = RecordBatch(schema=schema, columns=cols1^)

    # batch2: shifted by n
    var ca2: AnyArray = arange[Int32Type](n, 2 * n)
    var cb2: AnyArray = arange[Int32Type](n + 1, 2 * n + 1)
    var cc2: AnyArray = arange[Int32Type](3 * n // 2, 3 * n // 2 + n)
    var cd2: AnyArray = arange[Int32Type](n, 2 * n)
    var cols2 = List[AnyArray]()
    cols2.append(ca2^)
    cols2.append(cb2^)
    cols2.append(cc2^)
    cols2.append(cd2^)
    var batch2 = RecordBatch(schema=schema, columns=cols2^)

    var pipeline = filter_pipeline['data'](
        GtExpr(Add(col['a'](int32), col['b'](int32)), col['c'](int32)),
        int32,
    )
    var r1 = pipeline(batch1)
    var r2 = pipeline(batch2)

    # Compute reference results with FilterExpr
    var ref1 = execute(
        FilterExpr(
            arange[Int32Type](0, n),
            GtExpr(
                Add(Column(arange[Int32Type](0, n)), Column(arange[Int32Type](1, n + 1))),
                Column(arange[Int32Type](n // 2, n // 2 + n)),
            ),
        ),
        n,
    )
    var ref2 = execute(
        FilterExpr(
            arange[Int32Type](n, 2 * n),
            GtExpr(
                Add(Column(arange[Int32Type](n, 2 * n)), Column(arange[Int32Type](n + 1, 2 * n + 1))),
                Column(arange[Int32Type](3 * n // 2, 3 * n // 2 + n)),
            ),
        ),
        n,
    )

    assert_true(r1 == ref1)
    assert_true(r2 == ref2)


def test_pipeline_numeric_expr() raises:
    """Pipeline over Add(col['a'], col['b']) matches execute(Add(Column, Column))."""
    var n = 32
    var batch = _make_batch(n)

    var a_arr = arange[Int32Type](0, n)
    var b_arr = arange[Int32Type](1, n + 1)
    var expected = execute(Add(Column(a_arr^), Column(b_arr^)), n)

    var p = Pipeline(Add(col['a'](int32), col['b'](int32)))
    var result = p(batch)

    assert_true(expected == result)


def main() raises:
    TestSuite.run[__functions_in_module()]()

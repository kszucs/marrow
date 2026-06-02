"""Tests for faszom static expression layer: ColumnRef, FilterRel."""

from std.testing import assert_equal, assert_true
from marrow.testing import TestSuite
from marrow.arrays import AnyArray
from marrow.builders import arange, array
from marrow.dtypes import Int32Type, int32, Field as ArrowField
from marrow.schema import Schema as ArrowSchema
from marrow.tabular import RecordBatch
from marrow.faszom import (
    Add,
    Column,
    ColumnRef,
    Field,
    FilterRel,
    GtExpr,
    Schema,
    col,
    execute,
    lit,
)


def _make_batch(n: Int) raises -> RecordBatch:
    """Build a RecordBatch with columns a, b, c, data — all int32."""
    var schema = ArrowSchema(
        fields=[ArrowField("a", int32), ArrowField("b", int32), ArrowField("c", int32), ArrowField("data", int32)]
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


def test_filter_rel_basic() raises:
    """FilterRel.execute(batch) filters all batch columns and returns a RecordBatch."""
    var n = 10
    var batch = _make_batch(n)
    var t = Schema[Field['a', Int32Type], Field['b', Int32Type], Field['c', Int32Type], Field['data', Int32Type]]()
    # a[i]=i, b[i]=i+1, c[i]=n/2+i  =>  a+b > c: 2i+1 > n/2+i  =>  i >= n/2
    var result = t.filter(t.a + t.b > t.c).execute(batch)
    assert_true(result.column("data") == arange[Int32Type](n // 2, n))
    assert_true(result.column("a") == arange[Int32Type](n // 2, n))


def test_filter_rel_reuse() raises:
    """FilterRel.__call__ can be invoked on multiple batches and gives correct results."""
    var n = 64

    var schema_rt = ArrowSchema(
        fields=[ArrowField("a", int32), ArrowField("b", int32), ArrowField("c", int32), ArrowField("data", int32)]
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
    var batch1 = RecordBatch(schema=schema_rt, columns=cols1^)

    # batch2: shifted by n (condition always satisfied => all rows pass)
    var ca2: AnyArray = arange[Int32Type](n, 2 * n)
    var cb2: AnyArray = arange[Int32Type](n + 1, 2 * n + 1)
    var cc2: AnyArray = arange[Int32Type](3 * n // 2, 3 * n // 2 + n)
    var cd2: AnyArray = arange[Int32Type](n, 2 * n)
    var cols2 = List[AnyArray]()
    cols2.append(ca2^)
    cols2.append(cb2^)
    cols2.append(cc2^)
    cols2.append(cd2^)
    var batch2 = RecordBatch(schema=schema_rt, columns=cols2^)

    var t = Schema[Field['a', Int32Type], Field['b', Int32Type], Field['c', Int32Type], Field['data', Int32Type]]()
    var rel = t.filter(t.a + t.b > t.c)
    var r1 = rel.execute(batch1)
    var r2 = rel.execute(batch2)

    # batch1: 2i+1 > n/2+i => i >= n/2
    assert_true(r1.column("data") == arange[Int32Type](n // 2, n))
    # batch2: 2(n+i)+1 > 3n/2+i => n/2+i+1 > 0 (always) => all rows pass
    assert_true(r2.column("data") == arange[Int32Type](n, 2 * n))


def test_schema_filter_execute() raises:
    """Schema.filter(pred).execute(batch) filters and returns a RecordBatch."""
    var n = 10
    var schema_rt = ArrowSchema(fields=[ArrowField("a", int32), ArrowField("b", int32), ArrowField("data", int32)])
    var ca: AnyArray = arange[Int32Type](0, n)       # [0..9]
    var cb: AnyArray = arange[Int32Type](1, n + 1)   # [1..10]
    var cd: AnyArray = arange[Int32Type](0, n)       # [0..9]
    var cols = List[AnyArray]()
    cols.append(ca^)
    cols.append(cb^)
    cols.append(cd^)
    var batch = RecordBatch(schema=schema_rt, columns=cols^)

    # WHERE a + b > 10: a+b=[1,3,5,7,9,11,13,15,17,19] => indices 5..9 => data [5..9]
    var t = Schema[Field['a', Int32Type], Field['b', Int32Type], Field['data', Int32Type]]()
    var result = t.filter(t.a + t.b > lit(int32, 10)).execute(batch)
    assert_true(result.column("data") == arange[Int32Type](5, 10))


def test_filter_rel_with_projection() raises:
    """FilterRel outputs all batch columns; individual columns are accessible by name."""
    var n = 10
    var batch = _make_batch(n)
    var t = Schema[Field['a', Int32Type], Field['b', Int32Type], Field['c', Int32Type], Field['data', Int32Type]]()
    # a+b > c: i >= n/2; output has all 4 input columns, all filtered
    var result = t.filter(t.a + t.b > t.c).execute(batch)
    assert_equal(result.num_columns(), 4)
    assert_true(result.column("data") == arange[Int32Type](n // 2, n))
    assert_true(result.column("a") == arange[Int32Type](n // 2, n))


def test_boolean_operators() raises:
    """& (AND) and ~ (NOT) operators on BoolExpr produce correct filtered results."""
    var n = 10
    var schema_rt = ArrowSchema(fields=[ArrowField("a", int32), ArrowField("b", int32), ArrowField("data", int32)])
    var ca: AnyArray = arange[Int32Type](0, n)       # [0..9]
    var cb: AnyArray = arange[Int32Type](5, n + 5)   # [5..14]
    var cd: AnyArray = arange[Int32Type](0, n)       # [0..9]
    var cols = List[AnyArray]()
    cols.append(ca^)
    cols.append(cb^)
    cols.append(cd^)
    var batch = RecordBatch(schema=schema_rt, columns=cols^)

    var t = Schema[Field['a', Int32Type], Field['b', Int32Type], Field['data', Int32Type]]()

    # AND: a > 3 AND b < 12 => b[i]=a[i]+5 < 12 means a[i] < 7, combined a in {4,5,6}
    var r_and = t.filter(
        (t.a > lit(int32, 3)) & (t.b < lit(int32, 12))
    ).execute(batch)
    assert_true(r_and.column("data") == arange[Int32Type](4, 7))  # [4, 5, 6]

    # NOT: ~(a > 5) => a <= 5 => data [0..5]
    var r_not = t.filter(
        ~(t.a > lit(int32, 5))
    ).execute(batch)
    assert_true(r_not.column("data") == arange[Int32Type](0, 6))  # [0, 1, 2, 3, 4, 5]


def test_schema_filter_one_liner() raises:
    """Schema.filter(pred).execute(batch) gives the correct result."""
    var n = 10
    var batch = _make_batch(n)
    var t = Schema[Field['a', Int32Type], Field['b', Int32Type], Field['c', Int32Type], Field['data', Int32Type]]()
    # a[i]=i, b[i]=i+1, c[i]=n/2+i  =>  a+b > c: 2i+1 > n/2+i  =>  i >= n/2
    var result = t.filter(t.a + t.b > t.c).execute(batch)
    assert_true(result.column("data") == arange[Int32Type](n // 2, n))


def test_schema_filter_reuse() raises:
    """A FilterRel built from Schema can be called on multiple batches."""
    var n = 64
    var batch1 = _make_batch(n)
    var batch2 = _make_batch(2 * n)

    var t = Schema[Field['a', Int32Type], Field['b', Int32Type], Field['c', Int32Type], Field['data', Int32Type]]()
    var rel = t.filter(t.a + t.b > t.c)

    # _make_batch(n): a[i]=i, b[i]=i+1, c[i]=n/2+i  =>  a+b>c: 2i+1 > n/2+i  =>  i >= n/2
    assert_true(rel.execute(batch1).column("data") == arange[Int32Type](n // 2, n))
    assert_true(rel.execute(batch2).column("data") == arange[Int32Type](n, 2 * n))


def test_schema_getattr() raises:
    """Schema[Field[...]] t.col_name access returns ColumnRef and filters correctly."""
    var n = 10
    var batch = _make_batch(n)
    var t = Schema[Field['a', Int32Type], Field['b', Int32Type], Field['c', Int32Type], Field['data', Int32Type]]()
    # a[i]=i, b[i]=i+1, c[i]=n/2+i  =>  a+b > c: 2i+1 > n/2+i  =>  i >= n/2
    var result = t.filter(t.a + t.b > t.c).execute(batch)
    var expected: AnyArray = arange[Int32Type](n // 2, n)
    assert_true(result.column("data") == expected)


def test_schema_inferred() raises:
    """Schema(Field['a'](int32), ...) infers *Fields from value args."""
    var n = 10
    var batch = _make_batch(n)
    var t = Schema(
        Field['a'](int32),
        Field['b'](int32),
        Field['c'](int32),
        Field['data'](int32),
    )
    var result = t.filter(t.a + t.b > t.c).execute(batch)
    var expected: AnyArray = arange[Int32Type](n // 2, n)
    assert_true(result.column("data") == expected)


def test_col_name_dtype_properties() raises:
    """ColumnRef.col_name() returns the compile-time column name as a String."""
    assert_equal(col['price'](int32).col_name(), "price")
    assert_equal(col['qty'](int32).col_name(), "qty")


def main() raises:
    TestSuite.run[__functions_in_module()]()

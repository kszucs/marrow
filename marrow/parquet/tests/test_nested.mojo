from std.testing import assert_equal, assert_true
from std.python import Python, PythonObject
from std.os import remove
from marrow.testing import TestSuite
from marrow.parquet import read_table, write_table


def _pa_struct_table() raises -> PythonObject:
    var pa = Python.import_module("pyarrow")
    return pa.table(
        Python.dict(
            s=pa.array(
                Python.list(
                    Python.dict(a=1, x=1.5),
                    Python.dict(a=2, x=2.5),
                    Python.dict(a=3, x=3.5),
                ),
                type=pa.struct(
                    Python.list(
                        pa.field("a", pa.int64()),
                        pa.field("x", pa.float64()),
                    )
                ),
            ),
            k=pa.array(Python.list(10, 20, 30), type=pa.int64()),
        )
    )


def test_read_struct() raises:
    var pq = Python.import_module("pyarrow.parquet")
    var path = String("/tmp/marrow_nested_read.parquet")
    pq.write_table(_pa_struct_table(), path, compression="snappy")

    var t = read_table(path)
    assert_equal(t.num_rows(), 3)
    assert_equal(t.num_columns(), 2)
    assert_equal(t.column_names()[0], "s")

    var b = t.to_batches()[0].copy()
    var sc = b.columns[0].copy()
    ref st = sc.as_struct()
    assert_equal(len(st.children), 2)
    var ca = st.children[0].copy()
    ref col_a = ca.as_int64()
    assert_equal(col_a[0].value(), 1)
    assert_equal(col_a[2].value(), 3)
    var cx = st.children[1].copy()
    ref col_x = cx.as_float64()
    assert_true(col_x[1].value() == 2.5)
    remove(path)


def test_roundtrip_struct_via_pyarrow() raises:
    var pq = Python.import_module("pyarrow.parquet")
    var src = String("/tmp/marrow_nested_src.parquet")
    pq.write_table(_pa_struct_table(), src, compression="none")

    # marrow read -> marrow write -> pyarrow read
    var t = read_table(src)
    var dst = String("/tmp/marrow_nested_dst.parquet")
    write_table(t, dst)

    var back = pq.read_table(dst)
    assert_equal(Int(py=back.num_rows), 3)
    var s_col = back.column("s").to_pylist()
    assert_equal(Int(py=s_col[0]["a"]), 1)
    assert_true(Float64(py=s_col[2]["x"]) == 3.5)
    assert_equal(Int(py=back.column("k").to_pylist()[1]), 20)
    remove(src)
    remove(dst)


def main() raises:
    TestSuite.run[__functions_in_module()]()

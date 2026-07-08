from std.testing import assert_equal, assert_true
from std.python import Python
from std.pathlib import Path
from std.os import remove
from marrow.testing import TestSuite
from marrow.parquet.format import (
    read_footer,
    PT_INT64,
    PT_DOUBLE,
    PT_BYTE_ARRAY,
    REP_OPTIONAL,
)


def _write_pyarrow(path: String, compression: String) raises:
    var pa = Python.import_module("pyarrow")
    var pq = Python.import_module("pyarrow.parquet")
    var tbl = pa.table(
        Python.dict(
            x=pa.array(Python.list(1, 2, 3, 4), type=pa.int64()),
            y=pa.array(Python.list(1.5, 2.5, 3.5, 4.5), type=pa.float64()),
            z=pa.array(Python.list("a", "b", "c", "d")),
        )
    )
    pq.write_table(tbl, path, compression=compression)


def test_read_footer_metadata() raises:
    var path = String("/tmp/marrow_test_format.parquet")
    _write_pyarrow(path, "snappy")
    var data = Path(path).read_bytes()
    var meta = read_footer(Span(data))

    assert_equal(meta.num_rows, 4)
    assert_equal(len(meta.row_groups), 1)
    # schema[0] is the root group; then one leaf per column
    assert_equal(len(meta.schema), 4)
    assert_equal(meta.schema[0].num_children, 3)
    assert_equal(meta.schema[1].name, "x")
    assert_equal(meta.schema[1].type, PT_INT64)
    assert_equal(meta.schema[2].name, "y")
    assert_equal(meta.schema[2].type, PT_DOUBLE)
    assert_equal(meta.schema[3].name, "z")
    assert_equal(meta.schema[3].type, PT_BYTE_ARRAY)
    # pyarrow marks value columns optional (nullable)
    assert_equal(meta.schema[1].repetition_type, REP_OPTIONAL)

    ref rg = meta.row_groups[0]
    assert_equal(len(rg.columns), 3)
    assert_equal(rg.num_rows, 4)
    assert_equal(rg.columns[0].meta_data.path_in_schema[0], "x")
    assert_equal(rg.columns[0].meta_data.num_values, 4)
    assert_true(rg.columns[0].meta_data.data_page_offset >= 4)

    remove(path)


def main() raises:
    TestSuite.run[__functions_in_module()]()

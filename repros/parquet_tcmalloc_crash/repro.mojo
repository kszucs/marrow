# Minimal reproducer for a heap-state-dependent tcmalloc crash triggered by a
# specific sequence of Parquet reads followed by ParquetFile.statistics().
# See README.md in this directory for the full analysis.
#
# Run (crashes in a normal build):
#   pixi run -e dev mojo run -I . -D ASSERT=all repros/parquet_tcmalloc_crash/repro.mojo
#
# The same program is clean under AddressSanitizer (asan uses a different
# allocator), which is the whole puzzle.

from std.python import Python, PythonObject
from std.os import remove
from marrow.parquet import read_table, ParquetFile


def _flat_no_nulls() raises:
    var pa = Python.import_module("pyarrow")
    var pq = Python.import_module("pyarrow.parquet")
    var tbl = pa.table(
        Python.dict(
            i=pa.array(Python.list(1, 2, 3, 4, 5), type=pa.int64()),
            f=pa.array(Python.list(1.5, 2.5, 3.5, 4.5, 5.5), type=pa.float64()),
            s=pa.array(Python.list("apple", "banana", "cherry", "date", "elder")),
        )
    )
    var path = String("/tmp/repro_flat.parquet")
    pq.write_table(tbl, path, compression="snappy")
    var t = read_table(path)
    var b = t.to_batches()[0].copy()
    var ci = b.columns[0].copy()
    _ = ci.as_int64()[0].value()
    var cf = b.columns[1].copy()
    _ = cf.as_float64()[0].value()
    var cs = b.columns[2].copy()
    _ = String(cs.as_string()[0])
    remove(path)


def _many_pages_dict_string() raises:
    var pa = Python.import_module("pyarrow")
    var pq = Python.import_module("pyarrow.parquet")
    var np = Python.import_module("numpy")
    var colors = np.array(Python.list("red", "green", "blue"))
    var tbl = pa.table(Python.dict(s=pa.array(colors[np.arange(6000) % 3])))
    var path = String("/tmp/repro_dict.parquet")
    pq.write_table(
        tbl, path, data_page_size=128, use_dictionary=True, compression="snappy"
    )
    var t = read_table(path)
    var b = t.to_batches()[0].copy()
    ref s2 = b.columns[0].copy().as_string()
    _ = String(s2[0]); _ = String(s2[3001]); _ = String(s2[5999])
    remove(path)


def _wide_table() raises:
    var pa = Python.import_module("pyarrow")
    var pq = Python.import_module("pyarrow.parquet")
    var np = Python.import_module("numpy")
    var cols = Python.dict()
    var rows = np.arange(5) * 100
    for j in range(40):
        cols[Python.str("c") + String(j)] = pa.array(rows + j, type=pa.int64())
    var t_src = pa.table(cols)
    var path = String("/tmp/repro_wide.parquet")
    pq.write_table(t_src, path, compression="snappy")
    var t = read_table(path)
    var b = t.to_batches()[0].copy()
    var c0 = b.columns[0].copy()
    var c39 = b.columns[39].copy()
    _ = c0.as_int64()[3].value()
    _ = c39.as_int64()[4].value()
    remove(path)


def _parquet_file_statistics() raises:
    var pa = Python.import_module("pyarrow")
    var pq = Python.import_module("pyarrow.parquet")
    var tbl = pa.table(
        Python.dict(
            a=pa.array(Python.list(1, 2, 3, 4), type=pa.int64()),
            b=pa.array(Python.list("w", "x", "y", "z")),
        )
    )
    var path = String("/tmp/repro_pqfile.parquet")
    pq.write_table(tbl, path, row_group_size=2)
    var f = ParquetFile(path)
    _ = f.num_rows()
    _ = f.metadata().num_rows
    var stats = f.statistics()  # <- crash site in the suite
    print("statistics len:", len(stats))
    remove(path)


def main() raises:
    _flat_no_nulls()
    print("1/4 flat_no_nulls ok")
    _many_pages_dict_string()
    print("2/4 many_pages_dict_string ok")
    _wide_table()
    print("3/4 wide_table ok")
    _parquet_file_statistics()
    print("4/4 parquet_file statistics ok")
    print("DONE - no crash")

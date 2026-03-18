"""Parquet I/O via PyArrow.

Provides read_table and write_table functions that delegate to
pyarrow.parquet under the hood, using the Arrow C Stream Interface
for zero-copy data exchange between PyArrow and marrow.

This is a temporary bridge until a native Mojo Parquet implementation
is available.
"""

from std.python import Python, PythonObject
from .tabular import Table
from .c_data import CArrowArrayStream


def read_table(source: String) raises -> Table:
    """Read a Parquet file into a marrow Table.

    Args:
        source: Path to the Parquet file.

    Returns:
        A marrow Table.
    """
    var pq = Python.import_module("pyarrow.parquet")
    var pa_table = pq.read_table(source)
    var capsule = pa_table.__arrow_c_stream__(Python.none())
    var stream = CArrowArrayStream.from_pycapsule(capsule)
    return stream.to_table()


def write_table(table: Table, path: String) raises -> None:
    """Write a marrow Table to a Parquet file.

    Args:
        table: A marrow Table.
        path: Path to the output Parquet file.
    """
    var pa = Python.import_module("pyarrow")
    var pq = Python.import_module("pyarrow.parquet")

    var batches = table.to_batches()
    var fields = List(table.schema.fields)
    var capsule = CArrowArrayStream.from_batches(
        fields^, batches^
    ).to_pycapsule()

    # PyArrow's from_stream expects an object with __arrow_c_stream__,
    # not a raw PyCapsule. Wrap it in a minimal protocol object.
    var StreamHolder = Python.evaluate(
        "type('_S', (), {'__arrow_c_stream__':"
        " lambda self, requested_schema=None: self._cap})"
    )
    var holder = StreamHolder()
    holder._cap = capsule
    var reader = pa.RecordBatchReader.from_stream(holder)
    var pa_table = reader.read_all()

    pq.write_table(pa_table, path)

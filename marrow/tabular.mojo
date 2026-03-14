"""Tabular data structures: RecordBatch and Table.

RecordBatch holds a schema and a matching list of single-chunk Arrays.
Table holds a schema and a matching list of ChunkedArrays.

References:
- https://arrow.apache.org/docs/python/generated/pyarrow.RecordBatch.html
- https://arrow.apache.org/docs/python/generated/pyarrow.Table.html
"""
from .arrays import Array, ChunkedArray
from .schema import Schema
from .c_data import CArrowArray, CArrowSchema
from .dtypes import struct_, Field
from std.python import Python, PythonObject
from std.memory import alloc


struct RecordBatch(Copyable, Writable):
    """A schema together with a list of equal-length column arrays.

    Equivalent to PyArrow's `RecordBatch`.
    """

    var schema: Schema
    var columns: List[Array]

    fn __init__(out self, schema: Schema, var columns: List[Array]):
        self.schema = schema
        self.columns = columns^

    fn num_rows(self) -> Int:
        """Returns the number of rows (length of the first column, or 0)."""
        if len(self.columns) == 0:
            return 0
        return self.columns[0].length

    fn num_columns(self) -> Int:
        """Returns the number of columns."""
        return len(self.columns)

    fn column(self, index: Int) -> ref[self.columns] Array:
        """Returns the column at the given index."""
        return self.columns[index]

    fn column(self, name: String) raises -> ref[self.columns] Array:
        """Returns the column with the given name."""
        var idx = self.schema.get_field_index(name)
        if idx == -1:
            raise Error("Column '{}' not found.".format(name))
        return self.columns[idx]

    fn column_names(self) -> List[String]:
        """Returns the names of all columns (delegates to schema)."""
        return self.schema.names()

    fn slice(self, offset: Int, length: Int) -> RecordBatch:
        """Returns a zero-copy slice of this RecordBatch."""
        var sliced = List[Array]()
        for col in self.columns:
            sliced.append(col.slice(offset, length))
        return RecordBatch(schema=self.schema, columns=sliced^)

    fn slice(self, offset: Int) -> RecordBatch:
        """Returns a zero-copy slice from offset to the end."""
        return self.slice(offset, self.num_rows() - offset)

    @staticmethod
    fn from_pyarrow(pa_batch: PythonObject) raises -> RecordBatch:
        """Imports a RecordBatch from PyArrow via the C Data Interface."""
        var c_array_ptr = alloc[CArrowArray](1)
        var c_schema_ptr = alloc[CArrowSchema](1)
        pa_batch._export_to_c(Int(c_array_ptr), Int(c_schema_ptr))

        var c_schema = c_schema_ptr.take_pointee()
        c_schema_ptr.free()
        var schema = c_schema.to_schema()

        # A RecordBatch exports as a struct-formatted array.
        var struct_dtype = struct_(schema.fields)
        var c_array = c_array_ptr.take_pointee()
        c_array_ptr.free()
        var main_array = c_array^.to_array(struct_dtype)

        var columns = List[Array]()
        for col in main_array.children:
            columns.append(col.copy())
        return RecordBatch(schema=schema, columns=columns^)

    fn to_pyarrow(self) raises -> PythonObject:
        """Exports this RecordBatch to PyArrow via the C Data Interface."""
        var pa = Python.import_module("pyarrow")
        var pa_schema = self.schema.to_pyarrow()
        var pa_arrays = Python.list()
        for col in self.columns:
            pa_arrays.append(col.to_pyarrow())
        return pa.RecordBatch.from_arrays(pa_arrays, schema=pa_schema)

    fn write_to[W: Writer](self, mut writer: W):
        writer.write(
            "RecordBatch(num_rows=",
            self.num_rows(),
            ", schema=",
            self.schema,
            ")",
        )


struct Table(Copyable, Writable):
    """A schema together with a list of equal-length ChunkedArrays.

    Equivalent to PyArrow's `Table`.  Unlike RecordBatch, each column may
    consist of multiple chunks (a ChunkedArray).
    """

    var schema: Schema
    var columns: List[ChunkedArray]

    fn __init__(out self, schema: Schema, var columns: List[ChunkedArray]):
        self.schema = schema
        self.columns = columns^

    fn num_rows(self) -> Int:
        """Returns the number of rows (length of the first column, or 0)."""
        if len(self.columns) == 0:
            return 0
        return self.columns[0].length

    fn num_columns(self) -> Int:
        """Returns the number of columns."""
        return len(self.columns)

    fn column(self, index: Int) -> ref[self.columns] ChunkedArray:
        """Returns the column at the given index."""
        return self.columns[index]

    fn column(self, name: String) raises -> ref[self.columns] ChunkedArray:
        """Returns the column with the given name."""
        var idx = self.schema.get_field_index(name)
        if idx == -1:
            raise Error("Column '{}' not found.".format(name))
        return self.columns[idx]

    fn column_names(self) -> List[String]:
        """Returns the names of all columns (delegates to schema)."""
        return self.schema.names()

    @staticmethod
    fn from_batches(
        schema: Schema, batches: List[RecordBatch]
    ) -> Table:
        """Builds a Table from a list of RecordBatches sharing the same schema.

        Each column in the resulting Table is a ChunkedArray whose chunks are
        the corresponding columns from each RecordBatch.
        """
        var n_cols = schema.num_fields()
        var columns = List[ChunkedArray]()
        for col_idx in range(n_cols):
            var chunks = List[Array]()
            for batch in batches:
                chunks.append(batch.columns[col_idx].copy())
            columns.append(
                ChunkedArray(
                    dtype=schema.fields[col_idx].dtype.copy(),
                    chunks=chunks^,
                )
            )
        return Table(schema=schema, columns=columns^)

    fn write_to[W: Writer](self, mut writer: W):
        writer.write(
            "Table(num_rows=",
            self.num_rows(),
            ", num_columns=",
            self.num_columns(),
            ", schema=",
            self.schema,
            ")",
        )

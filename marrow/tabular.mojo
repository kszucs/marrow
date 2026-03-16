"""Tabular data structures: RecordBatch and Table.

RecordBatch holds a schema and a matching list of single-chunk Arrays.
Table holds a schema and a matching list of ChunkedArrays.

References:
- https://arrow.apache.org/docs/python/generated/pyarrow.RecordBatch.html
- https://arrow.apache.org/docs/python/generated/pyarrow.Table.html
"""
from std.python import PythonObject
from std.python.conversions import ConvertibleFromPython, ConvertibleToPython
from .arrays import Array, ChunkedArray, StructArray
from .schema import Schema
from .dtypes import struct_, Field


struct RecordBatch(
    ConvertibleFromPython, ConvertibleToPython, Copyable, Equatable, Writable
):
    """A schema together with a list of equal-length column arrays.

    Equivalent to PyArrow's `RecordBatch`.
    """

    var schema: Schema
    var columns: List[Array]

    fn __init__(out self, schema: Schema, var columns: List[Array]):
        self.schema = schema
        self.columns = columns^

    fn __init__(out self, *, copy: Self):
        self.schema = Schema(copy=copy.schema)
        var cols = List[Array]()
        for col in copy.columns:
            cols.append(col.copy())
        self.columns = cols^

    fn __init__(out self, *, py: PythonObject) raises:
        self = py.downcast_value_ptr[Self]()[].copy()

    fn copy(self) -> RecordBatch:
        """Returns a copy of this RecordBatch (O(1) Arc ref-count bumps)."""
        return RecordBatch(copy=self)

    fn to_python_object(var self) raises -> PythonObject:
        return PythonObject(alloc=self^)

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

    fn field(self, i: Int) raises -> Field:
        """Returns the Field at the given index (delegates to schema)."""
        return self.schema.field(index=i)

    fn __eq__(self, other: RecordBatch) -> Bool:
        """Returns True if the two RecordBatches have equal schema and columns.
        """
        if self.schema != other.schema:
            return False
        if len(self.columns) != len(other.columns):
            return False
        for i in range(len(self.columns)):
            if self.columns[i] != other.columns[i]:
                return False
        return True

    fn slice(self, offset: Int, length: Int) -> RecordBatch:
        """Returns a zero-copy slice of this RecordBatch."""
        var sliced = List[Array]()
        for col in self.columns:
            sliced.append(col.slice(offset, length))
        return RecordBatch(schema=self.schema, columns=sliced^)

    fn slice(self, offset: Int) -> RecordBatch:
        """Returns a zero-copy slice from offset to the end."""
        return self.slice(offset, self.num_rows() - offset)

    fn select(self, indices: List[Int]) -> RecordBatch:
        """Returns a new RecordBatch with only the columns at the given indices.
        """
        var new_cols = List[Array]()
        var new_fields = List[Field]()
        for i in indices:
            new_cols.append(self.columns[i].copy())
            new_fields.append(self.schema.fields[i])
        return RecordBatch(schema=Schema(fields=new_fields^), columns=new_cols^)

    fn select(self, names: List[String]) raises -> RecordBatch:
        """Returns a new RecordBatch with only the named columns."""
        var new_cols = List[Array]()
        var new_fields = List[Field]()
        for name in names:
            var idx = self.schema.get_field_index(name)
            if idx == -1:
                raise Error("Column '{}' not found.".format(name))
            new_cols.append(self.columns[idx].copy())
            new_fields.append(self.schema.fields[idx])
        return RecordBatch(schema=Schema(fields=new_fields^), columns=new_cols^)

    fn rename_columns(self, names: List[String]) raises -> RecordBatch:
        """Returns a new RecordBatch with columns renamed to `names`."""
        if len(names) != len(self.columns):
            raise Error(
                "rename_columns: expected {} names, got {}.".format(
                    len(self.columns), len(names)
                )
            )
        var new_fields = List[Field]()
        for i in range(len(names)):
            var f = self.schema.fields[i]
            new_fields.append(
                Field(name=names[i], dtype=f.dtype.copy(), nullable=f.nullable)
            )
        var cols = List[Array]()
        for col in self.columns:
            cols.append(col.copy())
        return RecordBatch(schema=Schema(fields=new_fields^), columns=cols^)

    fn add_column(self, i: Int, field: Field, column: Array) -> RecordBatch:
        """Returns a new RecordBatch with `column` inserted at position `i`."""
        var new_fields = List[Field]()
        var new_cols = List[Array]()
        for j in range(i):
            new_fields.append(self.schema.fields[j])
            new_cols.append(self.columns[j].copy())
        new_fields.append(field)
        new_cols.append(column.copy())
        for j in range(i, len(self.columns)):
            new_fields.append(self.schema.fields[j])
            new_cols.append(self.columns[j].copy())
        return RecordBatch(schema=Schema(fields=new_fields^), columns=new_cols^)

    fn append_column(self, field: Field, column: Array) -> RecordBatch:
        """Returns a new RecordBatch with `column` appended at the end."""
        return self.add_column(len(self.columns), field, column)

    fn remove_column(self, i: Int) -> RecordBatch:
        """Returns a new RecordBatch with the column at index `i` removed."""
        var new_fields = List[Field]()
        var new_cols = List[Array]()
        for j in range(len(self.columns)):
            if j != i:
                new_fields.append(self.schema.fields[j])
                new_cols.append(self.columns[j].copy())
        return RecordBatch(schema=Schema(fields=new_fields^), columns=new_cols^)

    fn set_column(self, i: Int, field: Field, column: Array) -> RecordBatch:
        """Returns a new RecordBatch with the column at index `i` replaced."""
        var new_fields = List[Field]()
        var new_cols = List[Array]()
        for j in range(len(self.columns)):
            if j == i:
                new_fields.append(field)
                new_cols.append(column.copy())
            else:
                new_fields.append(self.schema.fields[j])
                new_cols.append(self.columns[j].copy())
        return RecordBatch(schema=Schema(fields=new_fields^), columns=new_cols^)

    fn to_struct_array(self) -> StructArray:
        """Converts this RecordBatch to a StructArray (columns become fields).
        """
        var cols = List[Array]()
        for col in self.columns:
            cols.append(col.copy())
        return StructArray(
            dtype=struct_(self.schema.fields),
            length=self.num_rows(),
            nulls=0,
            bitmap=None,
            children=cols^,
        )

    fn __str__(self) -> String:
        return String.write(self)

    fn write_to[W: Writer](self, mut writer: W):
        writer.write(
            "RecordBatch(num_rows=",
            self.num_rows(),
            ", schema=",
            self.schema,
            ")",
        )

    fn write_repr_to[W: Writer](self, mut writer: W):
        self.write_to(writer)


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
    fn from_batches(schema: Schema, batches: List[RecordBatch]) -> Table:
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

"""Tabular data structures: RecordBatch and Table.

RecordBatch holds a schema and a matching list of single-chunk Arrays.
Table holds a schema and a matching list of ChunkedArrays.

References:
- https://arrow.apache.org/docs/python/generated/pyarrow.RecordBatch.html
- https://arrow.apache.org/docs/python/generated/pyarrow.Table.html
"""
from std.python import Python, PythonObject
from std.python.conversions import ConvertibleFromPython, ConvertibleToPython
from .arrays import AnyArray, ChunkedArray, StructArray
from .schema import Schema
from .dtypes import struct_, Field


struct RecordBatch(
    ConvertibleFromPython, ConvertibleToPython, Copyable, Equatable, Writable
):
    """A schema together with a list of equal-length column arrays.

    Equivalent to PyArrow's `RecordBatch`.
    """

    var schema: Schema
    var columns: List[AnyArray]

    def __init__(out self, schema: Schema, var columns: List[AnyArray]):
        self.schema = schema
        self.columns = columns^

    def __init__(out self, *, copy: Self):
        self.schema = Schema(copy=copy.schema)
        var cols = List[AnyArray]()
        for col in copy.columns:
            cols.append(col.copy())
        self.columns = cols^

    def __init__(out self, *, py: PythonObject) raises:
        from .c_data import CArrowSchema, CArrowArray

        # Try downcasting from a marrow Python object.
        try:
            self = py.downcast_value_ptr[Self]()[].copy()
            return
        except:
            pass
        # Fall back to Arrow C Data Interface for foreign objects.
        # Try __arrow_c_record_batch__ first, then __arrow_c_array__.
        var caps: PythonObject
        try:
            caps = py.__arrow_c_record_batch__()
        except:
            try:
                caps = py.__arrow_c_array__(Python.none())
            except:
                raise Error("cannot convert Python object to RecordBatch")
        var schema = CArrowSchema.from_pycapsule(caps[0]).to_schema()
        var struct_arr = CArrowArray.from_pycapsule(caps[1]).to_array(
            struct_(schema.fields)
        )
        var columns = List[AnyArray]()
        for child in struct_arr.children:
            columns.append(child.copy())
        self = RecordBatch(schema=schema, columns=columns^)

    def copy(self) -> RecordBatch:
        """Returns a copy of this RecordBatch (O(1) Arc ref-count bumps)."""
        return RecordBatch(copy=self)

    def to_python_object(var self) raises -> PythonObject:
        return PythonObject(alloc=self^)

    def num_rows(self) -> Int:
        """Returns the number of rows (length of the first column, or 0)."""
        if len(self.columns) == 0:
            return 0
        return self.columns[0].length

    def num_columns(self) -> Int:
        """Returns the number of columns."""
        return len(self.columns)

    def column(self, index: Int) -> ref[self.columns] AnyArray:
        """Returns the column at the given index."""
        return self.columns[index]

    def column(self, name: String) raises -> ref[self.columns] AnyArray:
        """Returns the column with the given name."""
        var idx = self.schema.get_field_index(name)
        if idx == -1:
            raise Error("Column '{}' not found.".format(name))
        return self.columns[idx]

    def column_names(self) -> List[String]:
        """Returns the names of all columns (delegates to schema)."""
        return self.schema.names()

    def field(self, i: Int) raises -> Field:
        """Returns the Field at the given index (delegates to schema)."""
        return self.schema.field(index=i)

    def __eq__(self, other: RecordBatch) -> Bool:
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

    def slice(self, offset: Int, length: Int) -> RecordBatch:
        """Returns a zero-copy slice of this RecordBatch."""
        var sliced = List[AnyArray]()
        for col in self.columns:
            sliced.append(col.slice(offset, length))
        return RecordBatch(schema=self.schema, columns=sliced^)

    def slice(self, offset: Int) -> RecordBatch:
        """Returns a zero-copy slice from offset to the end."""
        return self.slice(offset, self.num_rows() - offset)

    def select(self, indices: List[Int]) -> RecordBatch:
        """Returns a new RecordBatch with only the columns at the given indices.
        """
        var new_cols = List[AnyArray]()
        var new_fields = List[Field]()
        for i in indices:
            new_cols.append(self.columns[i].copy())
            new_fields.append(self.schema.fields[i])
        return RecordBatch(schema=Schema(fields=new_fields^), columns=new_cols^)

    def select(self, names: List[String]) raises -> RecordBatch:
        """Returns a new RecordBatch with only the named columns."""
        var new_cols = List[AnyArray]()
        var new_fields = List[Field]()
        for name in names:
            var idx = self.schema.get_field_index(name)
            if idx == -1:
                raise Error("Column '{}' not found.".format(name))
            new_cols.append(self.columns[idx].copy())
            new_fields.append(self.schema.fields[idx])
        return RecordBatch(schema=Schema(fields=new_fields^), columns=new_cols^)

    def rename_columns(self, names: List[String]) raises -> RecordBatch:
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
        var cols = List[AnyArray]()
        for col in self.columns:
            cols.append(col.copy())
        return RecordBatch(schema=Schema(fields=new_fields^), columns=cols^)

    def add_column(self, i: Int, field: Field, column: AnyArray) -> RecordBatch:
        """Returns a new RecordBatch with `column` inserted at position `i`."""
        var new_fields = List[Field]()
        var new_cols = List[AnyArray]()
        for j in range(i):
            new_fields.append(self.schema.fields[j])
            new_cols.append(self.columns[j].copy())
        new_fields.append(field)
        new_cols.append(column.copy())
        for j in range(i, len(self.columns)):
            new_fields.append(self.schema.fields[j])
            new_cols.append(self.columns[j].copy())
        return RecordBatch(schema=Schema(fields=new_fields^), columns=new_cols^)

    def append_column(self, field: Field, column: AnyArray) -> RecordBatch:
        """Returns a new RecordBatch with `column` appended at the end."""
        return self.add_column(len(self.columns), field, column)

    def remove_column(self, i: Int) -> RecordBatch:
        """Returns a new RecordBatch with the column at index `i` removed."""
        var new_fields = List[Field]()
        var new_cols = List[AnyArray]()
        for j in range(len(self.columns)):
            if j != i:
                new_fields.append(self.schema.fields[j])
                new_cols.append(self.columns[j].copy())
        return RecordBatch(schema=Schema(fields=new_fields^), columns=new_cols^)

    def set_column(self, i: Int, field: Field, column: AnyArray) -> RecordBatch:
        """Returns a new RecordBatch with the column at index `i` replaced."""
        var new_fields = List[Field]()
        var new_cols = List[AnyArray]()
        for j in range(len(self.columns)):
            if j == i:
                new_fields.append(field)
                new_cols.append(column.copy())
            else:
                new_fields.append(self.schema.fields[j])
                new_cols.append(self.columns[j].copy())
        return RecordBatch(schema=Schema(fields=new_fields^), columns=new_cols^)

    def to_struct_array(self) -> StructArray:
        """Converts this RecordBatch to a StructArray (columns become fields).
        """
        var cols = List[AnyArray]()
        for col in self.columns:
            cols.append(col.copy())
        return StructArray(
            dtype=struct_(self.schema.fields),
            length=self.num_rows(),
            nulls=0,
            bitmap=None,
            children=cols^,
        )

    def __str__(self) -> String:
        return String.write(self)

    def write_to[W: Writer](self, mut writer: W):
        writer.write(
            "RecordBatch(num_rows=",
            self.num_rows(),
            ", schema=",
            self.schema,
            ")",
        )

    def write_repr_to[W: Writer](self, mut writer: W):
        self.write_to(writer)


def record_batch(
    var columns: List[AnyArray], *, names: List[String]
) raises -> RecordBatch:
    """Construct a RecordBatch from a list of arrays and column names.

    Infers the schema from the column dtypes and the provided names.
    Raises if len(columns) != len(names).
    """
    if len(columns) != len(names):
        raise Error(
            "record_batch: len(columns) ("
            + String(len(columns))
            + ") != len(names) ("
            + String(len(names))
            + ")"
        )
    var fields = List[Field]()
    for i in range(len(columns)):
        fields.append(Field(names[i], columns[i].dtype))
    var schema = Schema(fields=fields^)
    return RecordBatch(schema=schema, columns=columns^)


struct Table(ConvertibleFromPython, ConvertibleToPython, Copyable, Writable):
    """A schema together with a list of equal-length ChunkedArrays.

    Equivalent to PyArrow's `Table`.  Unlike RecordBatch, each column may
    consist of multiple chunks (a ChunkedArray).
    """

    var schema: Schema
    var columns: List[ChunkedArray]

    def __init__(out self, schema: Schema, var columns: List[ChunkedArray]):
        self.schema = schema
        self.columns = columns^

    def __init__(out self, *, copy: Self):
        self.schema = Schema(copy=copy.schema)
        var cols = List[ChunkedArray]()
        for col in copy.columns:
            cols.append(
                ChunkedArray(dtype=col.dtype.copy(), chunks=List(col.chunks))
            )
        self.columns = cols^

    def __init__(out self, *, py: PythonObject) raises:
        from .c_data import CArrowArrayStream

        # Try downcasting from a marrow Python object.
        try:
            self = py.downcast_value_ptr[Self]()[].copy()
            return
        except:
            pass
        # Fall back to Arrow C Stream Interface for foreign objects.
        var capsule: PythonObject
        try:
            capsule = py.__arrow_c_stream__(Python.none())
        except:
            raise Error("cannot convert Python object to Table")
        self = CArrowArrayStream.from_pycapsule(capsule).to_table()

    def copy(self) -> Table:
        """Returns a copy of this Table (O(1) Arc ref-count bumps)."""
        return Table(copy=self)

    def to_python_object(var self) raises -> PythonObject:
        return PythonObject(alloc=self^)

    def get_schema(self) -> Schema:
        """Returns the schema."""
        return self.schema

    def num_rows(self) -> Int:
        """Returns the number of rows (length of the first column, or 0)."""
        if len(self.columns) == 0:
            return 0
        return self.columns[0].length

    def num_columns(self) -> Int:
        """Returns the number of columns."""
        return len(self.columns)

    def column(self, index: Int) -> ref[self.columns] ChunkedArray:
        """Returns the column at the given index."""
        return self.columns[index]

    def column(self, name: String) raises -> ref[self.columns] ChunkedArray:
        """Returns the column with the given name."""
        var idx = self.schema.get_field_index(name)
        if idx == -1:
            raise Error("Column '{}' not found.".format(name))
        return self.columns[idx]

    def combine_chunks(self) raises -> RecordBatch:
        """Combine all chunks in each column into a single RecordBatch."""
        var cols = List[AnyArray]()
        for col in self.columns:
            var ca = ChunkedArray(
                dtype=col.dtype.copy(), chunks=List(col.chunks)
            )
            cols.append(ca^.combine_chunks())
        return RecordBatch(schema=self.schema, columns=cols^)

    def column_names(self) -> List[String]:
        """Returns the names of all columns (delegates to schema)."""
        return self.schema.names()

    @staticmethod
    def from_batches(schema: Schema, batches: List[RecordBatch]) -> Table:
        """Builds a Table from a list of RecordBatches sharing the same schema.

        Each column in the resulting Table is a ChunkedArray whose chunks are
        the corresponding columns from each RecordBatch.
        """
        var n_cols = schema.num_fields()
        var columns = List[ChunkedArray]()
        for col_idx in range(n_cols):
            var chunks = List[AnyArray]()
            for batch in batches:
                chunks.append(batch.columns[col_idx].copy())
            columns.append(
                ChunkedArray(
                    dtype=schema.fields[col_idx].dtype.copy(),
                    chunks=chunks^,
                )
            )
        return Table(schema=schema, columns=columns^)

    def to_batches(self) raises -> List[RecordBatch]:
        """Convert this Table to a list of RecordBatches.

        Returns one RecordBatch per chunk. If columns have different chunk
        counts the result aligns on the first column's chunk boundaries
        (single-batch fallback when chunk counts differ).
        """
        if len(self.columns) == 0:
            return List[RecordBatch]()

        # Check if all columns have the same number of chunks.
        var n_chunks = len(self.columns[0].chunks)
        var aligned = True
        for col in self.columns:
            if len(col.chunks) != n_chunks:
                aligned = False
                break

        if aligned and n_chunks > 0:
            var batches = List[RecordBatch]()
            for chunk_idx in range(n_chunks):
                var cols = List[AnyArray]()
                for col in self.columns:
                    cols.append(col.chunks[chunk_idx].copy())
                batches.append(RecordBatch(schema=self.schema, columns=cols^))
            return batches^

        # Fallback: combine chunks into a single batch.
        from .kernels.concat import concat

        var cols = List[AnyArray]()
        for col in self.columns:
            if len(col.chunks) == 1:
                cols.append(col.chunks[0].copy())
            else:
                var ca = ChunkedArray(
                    dtype=col.dtype.copy(), chunks=List(col.chunks)
                )
                cols.append(ca^.combine_chunks())
        var batches = List[RecordBatch]()
        batches.append(RecordBatch(schema=self.schema, columns=cols^))
        return batches^

    def field(self, i: Int) raises -> Field:
        """Returns the Field at the given index (delegates to schema)."""
        return self.schema.field(index=i)

    def __eq__(self, other: Table) -> Bool:
        """Returns True if the two Tables have equal schema and columns."""
        if self.schema != other.schema:
            return False
        if len(self.columns) != len(other.columns):
            return False
        for i in range(len(self.columns)):
            if len(self.columns[i].chunks) != len(other.columns[i].chunks):
                return False
            for j in range(len(self.columns[i].chunks)):
                if self.columns[i].chunks[j] != other.columns[i].chunks[j]:
                    return False
        return True

    def __str__(self) -> String:
        return String.write(self)

    def write_to[W: Writer](self, mut writer: W):
        writer.write(
            "Table(num_rows=",
            self.num_rows(),
            ", num_columns=",
            self.num_columns(),
            ", schema=",
            self.schema,
            ")",
        )

    def write_repr_to[W: Writer](self, mut writer: W):
        self.write_to(writer)

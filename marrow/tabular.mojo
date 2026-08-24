"""Tabular data structures: RecordBatch and Table.

RecordBatch holds a schema and a matching list of single-chunk Arrays.
Table holds a schema and a matching list of ChunkedArrays.

References:
- https://arrow.apache.org/docs/python/generated/pyarrow.RecordBatch.html
- https://arrow.apache.org/docs/python/generated/pyarrow.Table.html
"""

from std.python import Python, PythonObject
from std.python.conversions import ConvertibleFromPython, ConvertibleToPython
from .arrays import DynArray, ChunkedArray, StructArray
from .builders import array
from .schema import Schema
from .dtypes import struct_, Field
from .kernels.join import (
    hash_join,
    JoinKind,
)
from .execution import ExecContext
from .kernels.groupby import GroupBy
from .exprold.aggregates import FoldedAggregates
from .kernels.sort import sort


struct RecordBatch(
    ConvertibleFromPython, ConvertibleToPython, Copyable, Equatable, Writable
):
    """A schema together with a list of equal-length column arrays.

    Equivalent to PyArrow's `RecordBatch`.
    """

    var schema: Schema
    var columns: List[DynArray]

    def __init__(out self, schema: Schema, var columns: List[DynArray]):
        self.schema = schema
        self.columns = columns^

    def __init__(out self, *, copy: Self):
        self.schema = Schema(copy=copy.schema)
        var cols = List[DynArray]()
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
            struct_(schema.fields.copy())
        )
        var columns = List[DynArray]()
        for child in struct_arr.as_struct().children:
            columns.append(child.copy())
        self = RecordBatch(schema=schema, columns=columns^)

    def to_python_object(var self) raises -> PythonObject:
        return PythonObject(alloc=self^)

    def num_rows(self) -> Int:
        """Returns the number of rows (length of the first column, or 0)."""
        if len(self.columns) == 0:
            return 0
        return self.columns[0].length()

    def num_columns(self) -> Int:
        """Returns the number of columns."""
        return len(self.columns)

    @staticmethod
    def empty(schema: Schema) raises -> RecordBatch:
        """Create a 0-row RecordBatch for the given schema."""
        var cols = List[DynArray]()
        for f in schema.fields:
            cols.append(array(f.dtype))
        return RecordBatch(schema=schema, columns=cols^)

    def column(self, index: Int) -> ref[self.columns[index]] DynArray:
        """Returns the column at the given index."""
        return self.columns[index]

    def column(self, name: String) raises -> ref[self.columns[0]] DynArray:
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
        return self.schema.field(index=i).copy()

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

    def slice(self, offset: Int, length: Int) raises -> RecordBatch:
        """Returns a zero-copy slice of this RecordBatch."""
        var sliced = List[DynArray]()
        for col in self.columns:
            sliced.append(col.slice(offset, length))
        return RecordBatch(schema=self.schema, columns=sliced^)

    def slice(self, offset: Int) raises -> RecordBatch:
        """Returns a zero-copy slice from offset to the end."""
        return self.slice(offset, self.num_rows() - offset)

    def select(self, indices: List[Int]) -> RecordBatch:
        """Returns a new RecordBatch with only the columns at the given indices.
        """
        var new_cols = List[DynArray]()
        var new_fields = List[Field]()
        for i in indices:
            new_cols.append(self.columns[i].copy())
            new_fields.append(self.schema.fields[i].copy())
        return RecordBatch(schema=Schema(fields=new_fields^), columns=new_cols^)

    def select(self, names: List[String]) raises -> RecordBatch:
        """Returns a new RecordBatch with only the named columns."""
        var new_cols = List[DynArray]()
        var new_fields = List[Field]()
        for name in names:
            var idx = self.schema.get_field_index(name)
            if idx == -1:
                raise Error("Column '{}' not found.".format(name))
            new_cols.append(self.columns[idx].copy())
            new_fields.append(self.schema.fields[idx].copy())
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
            ref f = self.schema.fields[i]
            new_fields.append(
                Field(name=names[i], dtype=f.dtype.copy(), nullable=f.nullable)
            )
        var cols = List[DynArray]()
        for col in self.columns:
            cols.append(col.copy())
        return RecordBatch(schema=Schema(fields=new_fields^), columns=cols^)

    def add_column(self, i: Int, field: Field, column: DynArray) -> RecordBatch:
        """Returns a new RecordBatch with `column` inserted at position `i`."""
        var new_fields = List[Field]()
        var new_cols = List[DynArray]()
        for j in range(i):
            new_fields.append(self.schema.fields[j].copy())
            new_cols.append(self.columns[j].copy())
        new_fields.append(field.copy())
        new_cols.append(column.copy())
        for j in range(i, len(self.columns)):
            new_fields.append(self.schema.fields[j].copy())
            new_cols.append(self.columns[j].copy())
        return RecordBatch(schema=Schema(fields=new_fields^), columns=new_cols^)

    def append_column(self, field: Field, column: DynArray) -> RecordBatch:
        """Returns a new RecordBatch with `column` appended at the end."""
        return self.add_column(len(self.columns), field, column)

    def remove_column(self, i: Int) -> RecordBatch:
        """Returns a new RecordBatch with the column at index `i` removed."""
        var new_fields = List[Field]()
        var new_cols = List[DynArray]()
        for j in range(len(self.columns)):
            if j != i:
                new_fields.append(self.schema.fields[j].copy())
                new_cols.append(self.columns[j].copy())
        return RecordBatch(schema=Schema(fields=new_fields^), columns=new_cols^)

    def set_column(self, i: Int, field: Field, column: DynArray) -> RecordBatch:
        """Returns a new RecordBatch with the column at index `i` replaced."""
        var new_fields = List[Field]()
        var new_cols = List[DynArray]()
        for j in range(len(self.columns)):
            if j == i:
                new_fields.append(field.copy())
                new_cols.append(column.copy())
            else:
                new_fields.append(self.schema.fields[j].copy())
                new_cols.append(self.columns[j].copy())
        return RecordBatch(schema=Schema(fields=new_fields^), columns=new_cols^)

    def _key_indices(
        self, names: List[String], side: String
    ) raises -> List[Int]:
        """Resolve key column names to positions, naming the side on failure."""
        var out = List[Int](capacity=len(names))
        for ref n in names:
            var i = self.schema.get_field_index(n)
            if i == -1:
                raise Error(side, " key column '", n, "' not found")
            out.append(i)
        return out^

    def join(
        self,
        right: RecordBatch,
        keys: List[String],
        right_keys: List[String],
        how: String = "inner",
        ctx: ExecContext = ExecContext.auto(),
    ) raises -> RecordBatch:
        """Equi-join two batches on key column *names*.

        `right_keys` empty means "same names as `keys`". `how` is PyArrow's
        spelling — `inner`, `left outer`, `right outer`, `full outer`,
        `left semi`, `left anti`, with the short forms also accepted.

        This lived in `python/bindings/tabular.mojo`: name resolution, join-kind
        parsing and result assembly existed **only** for Python callers, and the
        binding imported `marrow.exprold.relations` inside a function body to reach
        the kind constants. Joining two batches is core behaviour, so it lives
        with the type; the binding now just marshals Python values."""
        var left_on = self._key_indices(keys, "Left")
        var right_on = right._key_indices(
            right_keys if right_keys else keys, "Right"
        )

        var kind = JoinKind.parse(how)

        var joined = hash_join(
            self.to_struct_array(),
            right.to_struct_array(),
            left_on,
            right_on,
            kind,
            ctx=ctx,
        )
        return RecordBatch.from_struct_array(joined^)

    def _agg_columns(
        self, values: List[String], funcs: List[String], who: String
    ) raises -> Tuple[List[DynArray], FoldedAggregates, List[String]]:
        """Resolve `(value column, aggregate)` pairs and their output names.

        `<value>_<func>` matches PyArrow's naming. Shared by `group_by` and
        `aggregate`, which differ only in whether a key grouping runs."""
        var cols = List[DynArray]()
        var aggs = FoldedAggregates()
        var names = List[String]()
        for j in range(len(funcs)):
            var vname = values[j]
            var vidx = self.schema.get_field_index(vname)
            if vidx == -1:
                raise Error(who, ": column '", vname, "' not found")
            cols.append(self.column(vidx).copy())
            aggs.append(funcs[j], self.column(vidx).dtype())
            names.append(vname + "_" + funcs[j])
        return (cols^, aggs^, names^)

    def group_by(
        self,
        keys: List[String],
        values: List[String],
        funcs: List[String],
        ctx: ExecContext = ExecContext.auto(),
    ) raises -> RecordBatch:
        """`GROUP BY keys` with one output column per `(value, func)` pair.

        The keys are grouped once and every aggregate rides that pass. Output is
        the unique key columns followed by `<value>_<func>` columns."""
        var key_indices = self._key_indices(keys, "group_by")
        var key_struct = self.select(key_indices).to_struct_array()
        var resolved = self._agg_columns(values, funcs, "group_by")

        var gb = GroupBy(key_struct, ctx)
        var res = resolved[1].grouped(gb, resolved[0])

        # `res` is [key columns..., aggregate columns...]; name the aggregates.
        var n_keys = len(res.columns) - len(resolved[1])
        var fields = List[Field]()
        var columns = List[DynArray]()
        for c in range(n_keys):
            fields.append(res.schema.fields[c].copy())
            columns.append(res.columns[c].copy())
        for j in range(len(resolved[1])):
            fields.append(
                Field(
                    resolved[2][j], res.schema.fields[n_keys + j].dtype.copy()
                )
            )
            columns.append(res.columns[n_keys + j].copy())
        return RecordBatch(schema=Schema(fields=fields^), columns=columns^)

    def aggregate(
        self, values: List[String], funcs: List[String]
    ) raises -> RecordBatch:
        """Whole-table aggregation — one row, a `<value>_<func>` column each.

        `count` of a non-null column is `COUNT(*)`."""
        var resolved = self._agg_columns(values, funcs, "aggregate")
        var res = resolved[1].whole(resolved[0])
        var fields = List[Field]()
        var columns = List[DynArray]()
        for j in range(len(resolved[1])):
            fields.append(
                Field(resolved[2][j], res.schema.fields[j].dtype.copy())
            )
            columns.append(res.columns[j].copy())
        return RecordBatch(schema=Schema(fields=fields^), columns=columns^)

    def sort_by(
        self,
        keys: List[String],
        ascending: List[Bool],
        nulls_first: Bool = True,
        ctx: ExecContext = ExecContext.auto(),
    ) raises -> RecordBatch:
        """Sort by one or more key columns, most-significant first.

        `ascending` is parallel to `keys`. The Python binding accepts PyArrow's
        `"name"` / `[(name, "descending"), ...]` spellings and flattens them to
        these two lists."""
        var indices = self._key_indices(keys, "sort_by")
        var sorted_sa = sort(
            self.to_struct_array(),
            indices,
            ascending,
            nulls_first,
            ctx=ctx,
        )
        var fields = List[Field]()
        for ref f in sorted_sa.dtype.as_struct().fields:
            fields.append(f.copy())
        return RecordBatch(
            schema=Schema(fields=fields^), columns=sorted_sa.children.copy()
        )

    @staticmethod
    def from_struct_array(var array: StructArray) raises -> RecordBatch:
        """Wrap a struct array as a batch — the inverse of `to_struct_array`.

        Cheap: the children move across and the schema is read off the struct
        dtype, which already carries one `Field` per child. This is the shim
        the execution layer uses at its boundary, so operators can work in
        struct arrays and still hand back a `RecordBatch`.
        """
        return RecordBatch(
            schema=Schema.from_dtype(array.dtype),
            columns=array.children.copy(),
        )

    def to_struct_array(self) -> StructArray:
        """Converts this RecordBatch to a StructArray (columns become fields).
        """
        var cols = List[DynArray]()
        for col in self.columns:
            cols.append(col.copy())
        return StructArray(
            dtype=struct_(self.schema.fields.copy()),
            length=self.num_rows(),
            nulls=0,
            offset=0,
            bitmap=None,
            children=cols^,
        )

    def __str__(self) -> String:
        return String(self)

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
    var columns: List[DynArray], *, names: List[String]
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
        fields.append(Field(names[i], columns[i].dtype()))
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
            cols.append(ChunkedArray(dtype=col.dtype, chunks=List(col.chunks)))
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

    def column(self, index: Int) -> ref[self.columns[index]] ChunkedArray:
        """Returns the column at the given index."""
        return self.columns[index]

    def column(self, name: String) raises -> ref[self.columns[0]] ChunkedArray:
        """Returns the column with the given name."""
        var idx = self.schema.get_field_index(name)
        if idx == -1:
            raise Error("Column '{}' not found.".format(name))
        return self.columns[idx]

    def combine_chunks(self) raises -> RecordBatch:
        """Combine all chunks in each column into a single RecordBatch."""
        var cols = List[DynArray]()
        for col in self.columns:
            var ca = ChunkedArray(dtype=col.dtype, chunks=List(col.chunks))
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
            var chunks = List[DynArray]()
            for batch in batches:
                chunks.append(batch.columns[col_idx].copy())
            columns.append(
                ChunkedArray(
                    dtype=schema.fields[col_idx].dtype,
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
                var cols = List[DynArray]()
                for col in self.columns:
                    cols.append(col.chunks[chunk_idx].copy())
                batches.append(RecordBatch(schema=self.schema, columns=cols^))
            return batches^

        # TODO: this in the middle import it not nice
        # Fallback: combine chunks into a single batch.
        from .kernels.concat import concat

        var cols = List[DynArray]()
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
        return self.schema.field(index=i).copy()

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
        return String(self)

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

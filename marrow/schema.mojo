"""Define the Mojo representation of the Arrow Schema.

[Reference](https://arrow.apache.org/docs/python/generated/pyarrow.Schema.html#pyarrow.Schema)
"""
from std.python import PythonObject
from std.python.conversions import ConvertibleFromPython, ConvertibleToPython
from std.reflection import reflect
from .dtypes import DataType, Field


def _construct_default[D: Defaultable & DataType]() -> D:
    """Construct a reflected field's DataType marker.

    A bare ``FieldT()`` call inside a ``comptime for`` over
    ``reflect[T].field_at[i].T`` fails to resolve — the compiler only sees
    ``FieldT`` as an opaque type during generic-mode checking of the enclosing
    generic function, with no constructor visible. Routing the construction
    through this separately-instantiated generic function (bound on
    ``Defaultable``) makes the zero-arg constructor visible via the trait
    witness instead.
    """
    return D()


struct Schema(
    ConvertibleFromPython,
    ConvertibleToPython,
    ImplicitlyCopyable,
    Movable,
    Sized,
    Writable,
):
    var fields: List[Field]
    var metadata: Dict[String, String]

    def __init__(
        out self,
        *,
        var fields: List[Field] = [],
        var metadata: Dict[String, String] = {},
    ):
        """Initializes a schema with the given fields, if provided."""
        self.fields = fields^
        self.metadata = metadata^

    def __init__(out self, *, copy: Self):
        self.fields = List[Field](copy=copy.fields)
        self.metadata = Dict[String, String](copy=copy.metadata)

    def __init__(out self, *, deinit move: Self):
        self.fields = move.fields^
        self.metadata = move.metadata^

    def __init__(out self, *, py: PythonObject) raises:
        from .c_data import CArrowSchema

        # Try downcasting from a marrow Python object.
        try:
            self = py.downcast_value_ptr[Self]()[].copy()
            return
        except:
            pass

        # Try the Arrow C Schema Interface for foreign objects.
        try:
            var capsule = py.__arrow_c_schema__()
            self = CArrowSchema.from_pycapsule(capsule).to_schema()
            return
        except:
            pass

        # Fall back to iterating as a sequence of Field objects.
        var fields = List[Field]()
        for f in py:
            fields.append(f.downcast_value_ptr[Field]()[].copy())
        self = Schema(fields=fields^)

    @staticmethod
    def from_struct[T: AnyType]() -> Schema:
        """Derive a Schema from a struct's fields via compile-time reflection.

        Each field's declared type must conform to `DataType` (e.g.
        `Int32Type`, `StringType` from `marrow.dtypes`) — the struct declares
        its schema by naming its fields after columns and typing them with
        marrow's zero-size Arrow type markers. Fields are always non-nullable;
        there is no notion of an optional field yet.

        Example:
            struct Orders:
                var a: Int32Type
                var b: StringType

            var s = Schema.from_struct[Orders]()
            # equivalent to schema([field("a", int32, nullable=False),
            #                       field("b", string, nullable=False)])
        """
        comptime r = reflect[T]
        comptime assert r.is_struct(), "Schema.from_struct[T] requires a struct"
        var fields = List[Field]()
        comptime for i in range(r.field_count()):
            comptime FieldT = r.field_at[i].T
            comptime assert conforms_to(
                FieldT, DataType
            ), "Schema.from_struct: every field must implement DataType"
            comptime assert conforms_to(
                FieldT, Defaultable
            ), "Schema.from_struct: every field must implement Defaultable"
            var dt = _construct_default[FieldT]()
            fields.append(
                Field(String(r.field_names()[i]), dt^, nullable=False)
            )
        return Schema(fields=fields^)

    def append(mut self, var field: Field):
        """Appends a field to the schema."""
        self.fields.append(field^)

    def __len__(self) -> Int:
        """Returns the number of fields in the schema."""
        return len(self.fields)

    def num_fields(self) -> Int:
        """Returns the number of fields in the schema."""
        return len(self.fields)

    def names(self) -> List[String]:
        """Returns the names of the fields in the schema."""
        return [field.name for field in self.fields]

    def field(self, *, index: Int) raises -> ref[self.fields[index]] Field:
        """Returns the field at the given index."""
        return self.fields[index]

    def field(self, *, name: StringSlice) raises -> ref[self.fields] Field:
        """Returns the field with the given name."""
        for field in self.fields:
            if field.name == name:
                return field
        raise Error(t"Field with name `{name}` not found.")

    def get_field_index(self, name: String) -> Int:
        """Returns the index of the field with the given name, or -1 if not found.
        """
        for i in range(len(self.fields)):
            if self.fields[i].name == name:
                return i
        return -1

    def __ne__(self, other: Schema) -> Bool:
        return not self.__eq__(other)

    def __eq__(self, other: Schema) -> Bool:
        """Returns True if the schemas have equal fields (metadata ignored)."""
        if len(self.fields) != len(other.fields):
            return False
        for i in range(len(self.fields)):
            if self.fields[i] != other.fields[i]:
                return False
        return True

    def to_python_object(var self) raises -> PythonObject:
        return PythonObject(alloc=self^)

    def write_to[W: Writer](self, mut writer: W):
        """Writes the schema to a writer."""
        writer.write("Schema(fields=[")
        for i in range(len(self.fields)):
            if i > 0:
                writer.write(", ")
            writer.write(self.fields[i])
        writer.write("])")


# TODO: add an overload with support for schema({"field1": int32, "field2": int16}) syntax
def schema(var fields: List[Field]) -> Schema:
    """Construct a Schema from a list of fields.

    Equivalent to PyArrow's `pa.schema()`.

    Example:
        schema([field("x", int32), field("y", float64)])
    """
    return Schema(fields=fields^)

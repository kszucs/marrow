"""Define the Mojo representation of the Arrow Schema.

[Reference](https://arrow.apache.org/docs/python/generated/pyarrow.Schema.html#pyarrow.Schema)
"""
from std.python import PythonObject
from std.python.conversions import ConvertibleFromPython, ConvertibleToPython
from .dtypes import DataType, Field


struct Schema(
    ConvertibleFromPython, ConvertibleToPython, ImplicitlyCopyable, Sized, Writable
):
    var fields: List[Field]
    var metadata: Dict[String, String]

    fn __init__(
        out self,
        *,
        var fields: List[Field] = [],
        var metadata: Dict[String, String] = {},
    ):
        """Initializes a schema with the given fields, if provided."""
        self.fields = fields^
        self.metadata = metadata^

    fn __init__(out self, *, copy: Self):
        self.fields = List[Field](copy=copy.fields)
        self.metadata = Dict[String, String](copy=copy.metadata)

    fn __init__(out self, *, py: PythonObject) raises:
        from .c_data import CArrowSchema

        # Try downcasting from a marrow Python object.
        try:
            self = py.downcast_value_ptr[Self]()[].copy()
            return
        except:
            pass

        # Fall back to the Arrow C Schema Interface for foreign objects.
        var capsule: PythonObject
        try:
            capsule = py.__arrow_c_schema__()
        except:
            raise Error("cannot convert Python object to Schema")
        self = CArrowSchema.from_pycapsule(capsule).to_schema()

    fn append(mut self, var field: Field):
        """Appends a field to the schema."""
        self.fields.append(field^)

    fn __len__(self) -> Int:
        """Returns the number of fields in the schema."""
        return len(self.fields)

    fn num_fields(self) -> Int:
        """Returns the number of fields in the schema."""
        return len(self.fields)

    fn names(self) -> List[String]:
        """Returns the names of the fields in the schema."""
        return [field.name for field in self.fields]

    fn field(self, *, index: Int) raises -> ref[self.fields] Field:
        """Returns the field at the given index."""
        return self.fields[index]

    fn field(self, *, name: StringSlice) raises -> ref[self.fields] Field:
        """Returns the field with the given name."""
        for field in self.fields:
            if field.name == name:
                return field
        raise Error(t"Field with name `{name}` not found.")

    fn get_field_index(self, name: String) -> Int:
        """Returns the index of the field with the given name, or -1 if not found.
        """
        for i in range(len(self.fields)):
            if self.fields[i].name == name:
                return i
        return -1

    fn __ne__(self, other: Schema) -> Bool:
        return not self.__eq__(other)

    fn __eq__(self, other: Schema) -> Bool:
        """Returns True if the schemas have equal fields (metadata ignored)."""
        if len(self.fields) != len(other.fields):
            return False
        for i in range(len(self.fields)):
            if self.fields[i] != other.fields[i]:
                return False
        return True

    fn to_python_object(var self) raises -> PythonObject:
        return PythonObject(alloc=self^)

    fn write_to[W: Writer](self, mut writer: W):
        """Writes the schema to a writer."""
        writer.write("Schema(fields=[")
        for i in range(len(self.fields)):
            if i > 0:
                writer.write(", ")
            writer.write(self.fields[i])
        writer.write("])")

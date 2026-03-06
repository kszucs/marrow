"""Define the Mojo representation of the Arrow Schema.

[Reference](https://arrow.apache.org/docs/python/generated/pyarrow.Schema.html#pyarrow.Schema)
"""
from .dtypes import Field
from .c_data import CArrowSchema


struct Schema(Copyable):
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

    @staticmethod
    fn from_c(c_arrow_schema: CArrowSchema) raises -> Schema:
        """Initializes a schema from a CArrowSchema."""
        var fields = List[Field]()
        for i in range(c_arrow_schema.n_children):
            var child = c_arrow_schema.children[i]
            var field = child[].to_field()
            fields.append(field^)

        return Schema(fields=fields^)

    fn append(mut self, var field: Field):
        """Appends a field to the schema."""
        self.fields.append(field^)

    fn names(self) -> List[String]:
        """Returns the names of the fields in the schema."""
        return [field.name for field in self.fields]

    fn field(
        self,
        *,
        index: Optional[Int] = None,
        name: Optional[StringSlice[origin=ImmutAnyOrigin]] = None,
    ) raises -> ref[self.fields] Field:
        """Returns the field at the given index or with the given name."""
        if index and name:
            raise Error("Either an index or a name must be provided, not both.")
        if index:
            return self.fields[index.value()]
        if not name:
            raise Error("Either an index or a name must be provided.")
        for field in self.fields:
            if StringSlice(field.name) == name.value():
                return field
        raise Error("Field with name `{}` not found.".format(name.value()))

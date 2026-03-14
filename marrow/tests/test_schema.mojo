"""Test the schema.mojo file."""
from std.testing import assert_equal, assert_true, assert_false, TestSuite
from std.python import Python, PythonObject
from marrow.schema import Schema
from marrow.c_data import CArrowSchema
from marrow.dtypes import (
    int8,
    int16,
    int32,
    int64,
    uint8,
    uint16,
    uint32,
    uint64,
)
from marrow.dtypes import float16, float32, float64, binary, string, list_
from marrow.dtypes import Field



def test_schema_primitive_fields() raises:
    """Test the schema with primitive fields."""

    # Create a schema with different data types
    fields = [
        Field("field1", int8),
        Field("field2", int16),
        Field("field3", int32),
        Field("field4", int64),
        Field("field5", uint8),
        Field("field6", uint16),
        Field("field7", uint32),
        Field("field8", uint64),
        Field("field9", float16),
        Field("field10", float32),
        Field("field11", float64),
        Field("field12", binary),
        Field("field13", string),
    ]
    var nb_fields = len(fields)

    var schema = Schema(fields=fields^)

    # Check the number of fields in the schema
    assert_equal(len(schema.fields), nb_fields)

    # Check the names of the fields in the schema
    for i in range(nb_fields):
        assert_equal(schema.field(index=i).name, String("field", i + 1))


def test_schema_names() raises -> None:
    fields = [
        Field("field1", int8, False),
        Field("field2", int16, False),
    ]

    var schema = Schema(fields=fields^)
    assert_equal(
        schema.names(),
        List[String]([String("field", i + 1) for i in range(2)]),
    )

    schema.append(Field("field3", int32))
    assert_equal(
        schema.names(),
        List[String]([String("field", i + 1) for i in range(3)]),
    )


def test_from_c_schema() raises -> None:
    var pa = Python.import_module("pyarrow")
    var pa_schema = pa.schema(
        Python.list(
            pa.field("field1", pa.list_(pa.int32())),
            pa.field(
                "field2",
                pa.`struct`(
                    Python.list(
                        pa.field("field_a", pa.int32()),
                        pa.field("field_b", pa.float64()),
                    )
                ),
            ),
        )
    )

    var schema = CArrowSchema.from_pycapsule(pa_schema.__arrow_c_schema__()).to_schema()

    assert_equal(len(schema.fields), 2)

    # Test first field.
    ref field_0 = schema.field(index=0)
    assert_true(field_0.dtype.is_list())
    assert_true(field_0.dtype.fields[0].dtype.is_integer())

    # Test second field.
    ref field_1 = schema.field(index=1)
    assert_true(field_1.dtype.is_struct())
    assert_equal(field_1.dtype.fields[0].name, "field_a")
    assert_equal(field_1.dtype.fields[1].name, "field_b")


def test_schema_len() raises:
    """Test Schema.__len__ and num_fields."""
    var schema = Schema(
        fields=[Field("a", int32), Field("b", float64), Field("c", string)]
    )
    assert_equal(len(schema), 3)
    assert_equal(schema.num_fields(), 3)

    var empty = Schema()
    assert_equal(len(empty), 0)


def test_schema_equality() raises:
    """Test Schema.__eq__ and __ne__."""
    var s1 = Schema(fields=[Field("x", int32), Field("y", float64)])
    var s2 = Schema(fields=[Field("x", int32), Field("y", float64)])
    var s3 = Schema(fields=[Field("x", int32), Field("z", float64)])
    var s4 = Schema(fields=[Field("x", int32)])

    assert_true(s1 == s2)
    assert_false(s1 == s3)
    assert_false(s1 == s4)
    assert_true(s1 != s3)
    assert_false(s1 != s2)


def test_schema_get_field_index() raises:
    """Test Schema.get_field_index."""
    var schema = Schema(
        fields=[Field("a", int32), Field("b", float64), Field("c", string)]
    )
    assert_equal(schema.get_field_index("a"), 0)
    assert_equal(schema.get_field_index("b"), 1)
    assert_equal(schema.get_field_index("c"), 2)
    assert_equal(schema.get_field_index("missing"), -1)


def test_schema_from_pyarrow() raises:
    """Test Schema.from_pyarrow convenience method."""
    var pa = Python.import_module("pyarrow")
    var pa_schema = pa.schema(
        Python.list(pa.field("x", pa.int32()), pa.field("y", pa.float64()))
    )
    var schema = CArrowSchema.from_pycapsule(pa_schema.__arrow_c_schema__()).to_schema()
    assert_equal(len(schema), 2)
    assert_equal(schema.field(index=0).name, "x")
    assert_equal(schema.field(index=0).dtype, int32)
    assert_equal(schema.field(index=1).name, "y")
    assert_equal(schema.field(index=1).dtype, float64)



def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()

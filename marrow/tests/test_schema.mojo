"""Test the schema.mojo file."""
from std.testing import assert_equal, assert_true, assert_false, TestSuite
from std.python import Python, PythonObject
from marrow.schema import Schema, schema
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
from marrow.dtypes import Field, field


def test_schema_primitive_fields() raises:
    """Test the schema with primitive fields."""
    var s = schema(
        [
            field("field1", int8),
            field("field2", int16),
            field("field3", int32),
            field("field4", int64),
            field("field5", uint8),
            field("field6", uint16),
            field("field7", uint32),
            field("field8", uint64),
            field("field9", float16),
            field("field10", float32),
            field("field11", float64),
            field("field12", binary),
            field("field13", string),
        ]
    )

    assert_equal(len(s.fields), 13)
    for i in range(13):
        assert_equal(s.field(index=i).name, String("field", i + 1))


def test_schema_names() raises -> None:
    var s = schema([field("field1", int8), field("field2", int16)])
    assert_equal(
        s.names(),
        List[String]([String("field", i + 1) for i in range(2)]),
    )

    s.append(field("field3", int32))
    assert_equal(
        s.names(),
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

    var s = CArrowSchema.from_pycapsule(
        pa_schema.__arrow_c_schema__()
    ).to_schema()

    assert_equal(len(s.fields), 2)

    # Test first field.
    ref field_0 = s.field(index=0)
    assert_true(field_0.dtype.is_list())
    assert_true(field_0.dtype.as_list_type().value_type().is_integer())

    # Test second field.
    ref field_1 = s.field(index=1)
    assert_true(field_1.dtype.is_struct())
    var f1_fields = field_1.dtype.as_struct_type().fields.copy()
    assert_equal(f1_fields[0].name, "field_a")
    assert_equal(f1_fields[1].name, "field_b")


def test_schema_len() raises:
    """Test Schema.__len__ and num_fields."""
    var s = schema([field("a", int32), field("b", float64), field("c", string)])
    assert_equal(len(s), 3)
    assert_equal(s.num_fields(), 3)

    var empty = Schema()
    assert_equal(len(empty), 0)


def test_schema_equality() raises:
    """Test Schema.__eq__ and __ne__."""
    var s1 = schema([field("x", int32), field("y", float64)])
    var s2 = schema([field("x", int32), field("y", float64)])
    var s3 = schema([field("x", int32), field("z", float64)])
    var s4 = schema([field("x", int32)])

    assert_true(s1 == s2)
    assert_false(s1 == s3)
    assert_false(s1 == s4)
    assert_true(s1 != s3)
    assert_false(s1 != s2)


def test_schema_get_field_index() raises:
    """Test Schema.get_field_index."""
    var s = schema([field("a", int32), field("b", float64), field("c", string)])
    assert_equal(s.get_field_index("a"), 0)
    assert_equal(s.get_field_index("b"), 1)
    assert_equal(s.get_field_index("c"), 2)
    assert_equal(s.get_field_index("missing"), -1)


def test_schema_from_pyarrow() raises:
    """Test Schema.from_pyarrow convenience method."""
    var pa = Python.import_module("pyarrow")
    var pa_schema = pa.schema(
        Python.list(pa.field("x", pa.int32()), pa.field("y", pa.float64()))
    )
    var s = CArrowSchema.from_pycapsule(
        pa_schema.__arrow_c_schema__()
    ).to_schema()
    assert_equal(len(s), 2)
    assert_equal(s.field(index=0).name, "x")
    assert_equal(s.field(index=0).dtype, int32)
    assert_equal(s.field(index=1).name, "y")
    assert_equal(s.field(index=1).dtype, float64)


# ---------------------------------------------------------------------------
# schema() convenience function
# ---------------------------------------------------------------------------


def test_schema_fn_empty() raises:
    """Schema([]) produces an empty schema."""
    var s = schema([])
    assert_equal(len(s), 0)


def test_schema_fn_single_field() raises:
    """Schema([field(...)]) produces a one-field schema."""
    var s = schema([field("x", int32)])
    assert_equal(len(s), 1)
    assert_equal(s.field(index=0).name, "x")
    assert_equal(s.field(index=0).dtype, int32)


def test_schema_fn_multiple_fields() raises:
    """Schema([...]) with multiple fields preserves order and types."""
    var s = schema([field("a", int64), field("b", float32), field("c", string)])
    assert_equal(len(s), 3)
    assert_equal(s.field(index=0).name, "a")
    assert_equal(s.field(index=0).dtype, int64)
    assert_equal(s.field(index=1).name, "b")
    assert_equal(s.field(index=1).dtype, float32)
    assert_equal(s.field(index=2).name, "c")
    assert_equal(s.field(index=2).dtype, string)


def test_schema_fn_equals_struct_init() raises:
    """Schema([...]) produces the same result as Schema(fields=[...])."""
    var s1 = schema(
        [field("x", int32, nullable=False), field("y", float64, nullable=False)]
    )
    var s2 = Schema(
        fields=[Field("x", int32, False), Field("y", float64, False)]
    )
    assert_true(s1 == s2)


# ---------------------------------------------------------------------------
# field() convenience function
# ---------------------------------------------------------------------------


def test_field_fn_name_and_type() raises:
    """Field() sets name and dtype correctly."""
    var f = field("x", int32)
    assert_equal(f.name, "x")
    assert_equal(f.dtype, int32)


def test_field_fn_nullable_default_true() raises:
    """Field() defaults to nullable=True, matching PyArrow."""
    var f = field("x", int32)
    assert_true(f.nullable)


def test_field_fn_nullable_false() raises:
    """Field(..., nullable=False) marks the field as non-nullable."""
    var f = field("x", int32, nullable=False)
    assert_false(f.nullable)


def test_field_fn_equals_field_struct() raises:
    """Field() and Field() with the same args produce equal fields."""
    var f1 = field("x", int32, nullable=False)
    var f2 = Field("x", int32, False)
    assert_true(f1 == f2)


def test_field_fn_various_types() raises:
    """Field() works for all primitive types."""
    assert_equal(field("int8", int8).dtype, int8)
    assert_equal(field("int64", int64).dtype, int64)
    assert_equal(field("float32", float32).dtype, float32)
    assert_equal(field("float64", float64).dtype, float64)
    assert_equal(field("string", string).dtype, string)
    assert_equal(field("binary", binary).dtype, binary)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()

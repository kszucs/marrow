"""Test Schema Python binding roundtrip with PyArrow."""
import pyarrow as pa
import marrow as ma


def test_schema_to_pyarrow():
    schema = ma.schema([
        ma.field("x", ma.int32()),
        ma.field("y", ma.float64()),
        ma.field("s", ma.string()),
    ])
    pa_schema = pa.schema(schema)
    assert len(pa_schema) == 3
    assert pa_schema.field("x").type == pa.int32()
    assert pa_schema.field("y").type == pa.float64()
    assert pa_schema.field("s").type == pa.string()


def test_schema_to_pyarrow_nested():
    schema = ma.schema([
        ma.field("lst", ma.list_(ma.int32())),
        ma.field("st", ma.struct([
            ma.field("a", ma.int32()),
            ma.field("b", ma.float64()),
        ])),
    ])
    pa_schema = pa.schema(schema)
    assert len(pa_schema) == 2
    assert str(pa_schema.field("lst").type) == "list<item: int32>"
    assert pa_schema.field("st").type == pa.struct([
        pa.field("a", pa.int32(), nullable=False),
        pa.field("b", pa.float64(), nullable=False),
    ])


def test_schema_from_pyarrow():
    pa_schema = pa.schema([pa.field("x", pa.int32()), pa.field("y", pa.float64())])
    schema = ma.schema(pa_schema)
    assert pa.schema(schema).equals(pa_schema)

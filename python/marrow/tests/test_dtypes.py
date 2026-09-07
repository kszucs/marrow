"""Test the DataType Python api."""

import marrow as ma


def test_factory_functions() -> None:
    """Test that all DataType factory functions work and return DataType."""
    assert isinstance(ma.null(), ma.DataType)
    assert isinstance(ma.bool_(), ma.DataType)
    assert isinstance(ma.int8(), ma.DataType)
    assert isinstance(ma.int16(), ma.DataType)
    assert isinstance(ma.int32(), ma.DataType)
    assert isinstance(ma.int64(), ma.DataType)
    assert isinstance(ma.uint8(), ma.DataType)
    assert isinstance(ma.uint16(), ma.DataType)
    assert isinstance(ma.uint32(), ma.DataType)
    assert isinstance(ma.uint64(), ma.DataType)
    assert isinstance(ma.float16(), ma.DataType)
    assert isinstance(ma.float32(), ma.DataType)
    assert isinstance(ma.float64(), ma.DataType)
    assert isinstance(ma.string(), ma.DataType)
    assert isinstance(ma.binary(), ma.DataType)
    assert isinstance(ma.year_month_interval(), ma.DataType)
    assert isinstance(ma.day_time_interval(), ma.DataType)
    assert isinstance(ma.month_day_nano_interval(), ma.DataType)


def test_field_factory() -> None:
    """Test that field factory function works and returns Field."""
    f = ma.field("my_field", ma.int32(), False, {})
    assert isinstance(f, ma.Field)
    assert f.name == "my_field"
    assert f.nullable is False
    assert isinstance(f.type, ma.DataType)


# ── equality and repr ───────────────────────────────────────────────────────
#
# `DynType` had no registered methods at all, so `ma.int32() == ma.int32()` was
# an identity comparison between two distinct binding objects and answered
# False. The wrapper is what makes a dtype behave like a value.


def test_datatypes_compare_by_value():
    assert ma.int32() == ma.int32()
    assert ma.int32() != ma.int64()
    assert ma.list_(ma.int64()) == ma.list_(ma.int64())
    assert ma.list_(ma.int64()) != ma.list_(ma.int32())


def test_datatypes_are_hashable_by_value():
    assert len({ma.int32(), ma.int32(), ma.int64()}) == 2
    assert {ma.int32(): "a"}[ma.int32()] == "a"


def test_datatype_str_and_repr():
    assert str(ma.int32()) == "int32"
    assert str(ma.list_(ma.int64())) == "list<int64>"
    assert repr(ma.int32()) == "DataType(int32)"


def test_datatype_byte_width():
    assert ma.int32().byte_width == 4
    assert ma.float64().byte_width == 8


def test_timestamp_takes_one_argument():
    """The binding drops its declared `tz=None`; the wrapper supplies it."""
    assert str(ma.timestamp("us")) == "timestamp[us]"
    assert "UTC" in str(ma.timestamp("us", "UTC"))


def test_fields_compare_by_value():
    assert ma.field("a", ma.int32()) == ma.field("a", ma.int32())
    assert ma.field("a", ma.int32()) != ma.field("b", ma.int32())
    assert ma.field("a", ma.int32(), nullable=False) != ma.field("a", ma.int32())


def test_schema_surface():
    s = ma.schema([ma.field("a", ma.int32()), ma.field("b", ma.string())])
    assert len(s) == 2
    assert s.names == ["a", "b"]
    assert s.types == [ma.int32(), ma.string()]
    assert s.field(0).name == "a"
    assert s.field("b").type == ma.string()
    assert s.get_field_index("b") == 1
    assert s.get_field_index("nope") == -1
    assert s == ma.schema([ma.field("a", ma.int32()), ma.field("b", ma.string())])
    assert [f.name for f in s] == ["a", "b"]

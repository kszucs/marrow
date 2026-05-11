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
    assert f.name() == "my_field"
    assert isinstance(f.type(), ma.DataType)

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

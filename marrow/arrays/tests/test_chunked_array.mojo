"""Test the chunked array implementation."""

from std.testing import assert_equal, TestSuite
from std.memory import ArcPointer

from marrow.arrays.base import ArrayData
from marrow.arrays.chunked_array import ChunkedArray
from marrow.buffers import Buffer, Bitmap
from marrow.dtypes import int8
from marrow.test_fixtures.arrays import build_array_data, assert_bitmap_set


def test_chunked_array() raises:
    var first_array_data = build_array_data(1, 0)
    var arrays = List[ArrayData]()
    arrays.append(first_array_data^)

    var second_array_data = build_array_data(2, 0)
    arrays.append(second_array_data^)

    var chunked_array = ChunkedArray(materialize[int8](), arrays^)
    assert_equal(chunked_array.length, 3)

    assert_equal(chunked_array.chunk(0).length, 1)
    var second_chunk = chunked_array.chunk(1).copy().as_uint8()
    assert_equal(second_chunk.data.length, 2)
    assert_equal(second_chunk.unsafe_get(0), 0)
    assert_equal(second_chunk.unsafe_get(1), 1)


def test_combine_chunked_array() raises:
    var first_array_data = build_array_data(1, 0)
    var arrays = List[ArrayData]()
    arrays.append(first_array_data^)

    var second_array_data = build_array_data(2, 0)
    arrays.append(second_array_data^)

    var chunked_array = ChunkedArray(materialize[int8](), arrays^)
    assert_equal(chunked_array.length, 3)
    assert_equal(len(chunked_array.chunks), 2)
    assert_equal(chunked_array.chunk(1).copy().as_uint8().unsafe_get(1), 1)

    var combined_array = chunked_array^.combine_chunks()
    assert_equal(combined_array.length, 3)
    assert_equal(combined_array.dtype, materialize[int8]())
    # Ensure that the last element of the last buffer has the expected value.
    assert_equal(combined_array.buffers[1][].unsafe_get(1), 1)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()

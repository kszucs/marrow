from std.testing import assert_equal, TestSuite

from marrow.arrays import *
from marrow.builders import array
from marrow.dtypes import *
from marrow.visitor import ArrayVisitor


struct ElementCounter(ArrayVisitor):
    """Demonstrates custom visitor: counts valid elements across array kinds."""

    var count: Int

    def __init__(out self):
        self.count = 0

    def visit[T: PrimitiveType](mut self, array: PrimitiveArray[T]) raises:
        self.count += array.null_count() * -1 + array.length

    def visit(mut self, array: StringArray) raises:
        self.count += array.length

    def visit(mut self, array: ListArray) raises:
        self.count += array.length

    def visit(mut self, array: StructArray) raises:
        self.count += array.length


def test_custom_visitor() raises:
    var a = array[Int64Type]([10, 20, 30, 40])
    var counter = ElementCounter()
    counter.visit((a^).to_any())
    assert_equal(counter.count, 4)


def test_chunked_array_default_dispatch() raises:
    """ChunkedArray.visit default delegates to visit(AnyArray) for each chunk.
    """
    var chunks: List[AnyArray] = [
        array[Int64Type]([1, 2, 3]).to_any(),
        array[Int64Type]([4, 5]).to_any(),
    ]
    var chunked = ChunkedArray(int64, chunks^)
    var counter = ElementCounter()
    counter.visit(chunked)
    assert_equal(counter.count, 5)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()

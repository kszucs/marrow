from std.testing import assert_equal, assert_true, assert_false, TestSuite
from std.gpu.host import DeviceContext

from marrow.builders import array, arange
from marrow.dtypes import int32, float32
from marrow.kernels.compare import (
    equal,
    not_equal,
    less,
    less_equal,
    greater,
    greater_equal,
)


def test_equal_gpu() raises:
    var ctx = DeviceContext()
    var a = array[int32]([1, 2, 3, 4]).to_device(ctx)
    var b = array[int32]([1, 0, 3, 0]).to_device(ctx)
    var result = equal[int32](a, b, ctx)
    assert_equal(len(result), 4)
    assert_true(result[0])
    assert_false(result[1])
    assert_true(result[2])
    assert_false(result[3])


def test_not_equal_gpu() raises:
    var ctx = DeviceContext()
    var a = array[int32]([1, 2, 3, 4]).to_device(ctx)
    var b = array[int32]([1, 0, 3, 0]).to_device(ctx)
    var result = not_equal[int32](a, b, ctx)
    assert_false(result[0])
    assert_true(result[1])
    assert_false(result[2])
    assert_true(result[3])


def test_less_gpu() raises:
    var ctx = DeviceContext()
    var a = array[int32]([1, 5, 3, 4]).to_device(ctx)
    var b = array[int32]([2, 3, 3, 8]).to_device(ctx)
    var result = less[int32](a, b, ctx)
    assert_true(result[0])
    assert_false(result[1])
    assert_false(result[2])
    assert_true(result[3])


def test_less_equal_gpu() raises:
    var ctx = DeviceContext()
    var a = array[int32]([1, 5, 3, 4]).to_device(ctx)
    var b = array[int32]([2, 3, 3, 8]).to_device(ctx)
    var result = less_equal[int32](a, b, ctx)
    assert_true(result[0])
    assert_false(result[1])
    assert_true(result[2])
    assert_true(result[3])


def test_greater_gpu() raises:
    var ctx = DeviceContext()
    var a = array[int32]([1, 5, 3, 4]).to_device(ctx)
    var b = array[int32]([2, 3, 3, 8]).to_device(ctx)
    var result = greater[int32](a, b, ctx)
    assert_false(result[0])
    assert_true(result[1])
    assert_false(result[2])
    assert_false(result[3])


def test_greater_equal_gpu() raises:
    var ctx = DeviceContext()
    var a = array[int32]([1, 5, 3, 4]).to_device(ctx)
    var b = array[int32]([2, 3, 3, 8]).to_device(ctx)
    var result = greater_equal[int32](a, b, ctx)
    assert_false(result[0])
    assert_true(result[1])
    assert_true(result[2])
    assert_false(result[3])


def test_equal_gpu_large() raises:
    var ctx = DeviceContext()
    var a = arange[int32](0, 10000).to_device(ctx)
    var b = arange[int32](0, 10000).to_device(ctx)
    var result = equal[int32](a, b, ctx)
    assert_equal(len(result), 10000)
    assert_true(result[0])
    assert_true(result[4999])
    assert_true(result[9999])


def test_less_gpu_float32() raises:
    var ctx = DeviceContext()
    var a = array[float32]([1, 2, 3, 4]).to_device(ctx)
    var b = array[float32]([4, 3, 2, 1]).to_device(ctx)
    var result = less[float32](a, b, ctx)
    assert_true(result[0])
    assert_true(result[1])
    assert_false(result[2])
    assert_false(result[3])


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()

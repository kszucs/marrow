from std.testing import assert_equal, assert_true, assert_false, TestSuite
from std.gpu.host import DeviceContext

from marrow.builders import array, arange
from marrow.dtypes import int32, float32, Int32Type, Float32Type
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
    var a = array[Int32Type]([1, 2, 3, 4]).to_device(ctx)
    var b = array[Int32Type]([1, 0, 3, 0]).to_device(ctx)
    var result = equal[Int32Type](a, b, ctx).to_cpu(ctx)
    assert_equal(len(result), 4)
    assert_true(result[0].value())
    assert_false(result[1].value())
    assert_true(result[2].value())
    assert_false(result[3].value())


def test_not_equal_gpu() raises:
    var ctx = DeviceContext()
    var a = array[Int32Type]([1, 2, 3, 4]).to_device(ctx)
    var b = array[Int32Type]([1, 0, 3, 0]).to_device(ctx)
    var result = not_equal[Int32Type](a, b, ctx).to_cpu(ctx)
    assert_false(result[0].value())
    assert_true(result[1].value())
    assert_false(result[2].value())
    assert_true(result[3].value())


def test_less_gpu() raises:
    var ctx = DeviceContext()
    var a = array[Int32Type]([1, 5, 3, 4]).to_device(ctx)
    var b = array[Int32Type]([2, 3, 3, 8]).to_device(ctx)
    var result = less[Int32Type](a, b, ctx).to_cpu(ctx)
    assert_true(result[0].value())
    assert_false(result[1].value())
    assert_false(result[2].value())
    assert_true(result[3].value())


def test_less_equal_gpu() raises:
    var ctx = DeviceContext()
    var a = array[Int32Type]([1, 5, 3, 4]).to_device(ctx)
    var b = array[Int32Type]([2, 3, 3, 8]).to_device(ctx)
    var result = less_equal[Int32Type](a, b, ctx).to_cpu(ctx)
    assert_true(result[0].value())
    assert_false(result[1].value())
    assert_true(result[2].value())
    assert_true(result[3].value())


def test_greater_gpu() raises:
    var ctx = DeviceContext()
    var a = array[Int32Type]([1, 5, 3, 4]).to_device(ctx)
    var b = array[Int32Type]([2, 3, 3, 8]).to_device(ctx)
    var result = greater[Int32Type](a, b, ctx).to_cpu(ctx)
    assert_false(result[0].value())
    assert_true(result[1].value())
    assert_false(result[2].value())
    assert_false(result[3].value())


def test_greater_equal_gpu() raises:
    var ctx = DeviceContext()
    var a = array[Int32Type]([1, 5, 3, 4]).to_device(ctx)
    var b = array[Int32Type]([2, 3, 3, 8]).to_device(ctx)
    var result = greater_equal[Int32Type](a, b, ctx).to_cpu(ctx)
    assert_false(result[0].value())
    assert_true(result[1].value())
    assert_true(result[2].value())
    assert_false(result[3].value())


def test_equal_gpu_large() raises:
    var ctx = DeviceContext()
    var a = arange[Int32Type](0, 10000).to_device(ctx)
    var b = arange[Int32Type](0, 10000).to_device(ctx)
    var result = equal[Int32Type](a, b, ctx).to_cpu(ctx)
    assert_equal(len(result), 10000)
    assert_true(result[0].value())
    assert_true(result[4999].value())
    assert_true(result[9999].value())


def test_less_gpu_float32() raises:
    var ctx = DeviceContext()
    var a = array[Float32Type]([1, 2, 3, 4]).to_device(ctx)
    var b = array[Float32Type]([4, 3, 2, 1]).to_device(ctx)
    var result = less[Float32Type](a, b, ctx).to_cpu(ctx)
    assert_true(result[0].value())
    assert_true(result[1].value())
    assert_false(result[2].value())
    assert_false(result[3].value())


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()

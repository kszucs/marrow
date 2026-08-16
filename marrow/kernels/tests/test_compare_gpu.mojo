from std.testing import assert_equal, assert_true, assert_false
from max.gpu.host import DeviceContext

from ...builders import array, arange
from ...dtypes import int32, float32, Int32Type, Float32Type
from ...kernels.numeric import (
    EqKernel,
    NeKernel,
    LtKernel,
    LeKernel,
    GtKernel,
    GeKernel,
)


def test_equal_gpu() raises:
    var ctx = DeviceContext()
    var a = array([1, 2, 3, 4], int32).to_device(ctx)
    var b = array([1, 0, 3, 0], int32).to_device(ctx)
    var result = EqKernel.apply[Int32Type](a, b, ctx).to_cpu(ctx)
    assert_equal(len(result), 4)
    assert_true(result[0].value())
    assert_false(result[1].value())
    assert_true(result[2].value())
    assert_false(result[3].value())


def test_not_equal_gpu() raises:
    var ctx = DeviceContext()
    var a = array([1, 2, 3, 4], int32).to_device(ctx)
    var b = array([1, 0, 3, 0], int32).to_device(ctx)
    var result = NeKernel.apply[Int32Type](a, b, ctx).to_cpu(ctx)
    assert_false(result[0].value())
    assert_true(result[1].value())
    assert_false(result[2].value())
    assert_true(result[3].value())


def test_less_gpu() raises:
    var ctx = DeviceContext()
    var a = array([1, 5, 3, 4], int32).to_device(ctx)
    var b = array([2, 3, 3, 8], int32).to_device(ctx)
    var result = LtKernel.apply[Int32Type](a, b, ctx).to_cpu(ctx)
    assert_true(result[0].value())
    assert_false(result[1].value())
    assert_false(result[2].value())
    assert_true(result[3].value())


def test_less_equal_gpu() raises:
    var ctx = DeviceContext()
    var a = array([1, 5, 3, 4], int32).to_device(ctx)
    var b = array([2, 3, 3, 8], int32).to_device(ctx)
    var result = LeKernel.apply[Int32Type](a, b, ctx).to_cpu(ctx)
    assert_true(result[0].value())
    assert_false(result[1].value())
    assert_true(result[2].value())
    assert_true(result[3].value())


def test_greater_gpu() raises:
    var ctx = DeviceContext()
    var a = array([1, 5, 3, 4], int32).to_device(ctx)
    var b = array([2, 3, 3, 8], int32).to_device(ctx)
    var result = GtKernel.apply[Int32Type](a, b, ctx).to_cpu(ctx)
    assert_false(result[0].value())
    assert_true(result[1].value())
    assert_false(result[2].value())
    assert_false(result[3].value())


def test_greater_equal_gpu() raises:
    var ctx = DeviceContext()
    var a = array([1, 5, 3, 4], int32).to_device(ctx)
    var b = array([2, 3, 3, 8], int32).to_device(ctx)
    var result = GeKernel.apply[Int32Type](a, b, ctx).to_cpu(ctx)
    assert_false(result[0].value())
    assert_true(result[1].value())
    assert_true(result[2].value())
    assert_false(result[3].value())


def test_equal_gpu_large() raises:
    var ctx = DeviceContext()
    var a = arange[Int32Type](0, 10000).to_device(ctx)
    var b = arange[Int32Type](0, 10000).to_device(ctx)
    var result = EqKernel.apply[Int32Type](a, b, ctx).to_cpu(ctx)
    assert_equal(len(result), 10000)
    assert_true(result[0].value())
    assert_true(result[4999].value())
    assert_true(result[9999].value())


def test_less_gpu_float32() raises:
    var ctx = DeviceContext()
    var a = array([1, 2, 3, 4], float32).to_device(ctx)
    var b = array([4, 3, 2, 1], float32).to_device(ctx)
    var result = LtKernel.apply[Float32Type](a, b, ctx).to_cpu(ctx)
    assert_true(result[0].value())
    assert_true(result[1].value())
    assert_false(result[2].value())
    assert_false(result[3].value())

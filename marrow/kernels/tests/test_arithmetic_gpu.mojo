from std.testing import assert_equal, assert_false, assert_true, TestSuite
from std.gpu.host import DeviceContext
from std.sys.info import CompilationTarget

from marrow.builders import array, arange
from marrow.dtypes import int32, float32, Int32Type, Float32Type
from marrow.kernels.arithmetic import add


def test_add_gpu() raises:
    """Element-wise add on GPU with small int32 arrays."""
    var ctx = DeviceContext()
    var a = array[Int32Type]([1, 2, 3, 4]).to_device(ctx)
    var b = array[Int32Type]([10, 20, 30, 40]).to_device(ctx)
    var result = add[Int32Type](a, b, ctx).to_cpu(ctx)
    assert_equal(len(result), 4)
    assert_equal(result.unsafe_get(0), 11)
    assert_equal(result.unsafe_get(1), 22)
    assert_equal(result.unsafe_get(2), 33)
    assert_equal(result.unsafe_get(3), 44)


def test_add_gpu_large() raises:
    """Exercise GPU add with a large array (10k elements)."""
    var ctx = DeviceContext()
    var a = arange[Int32Type](0, 10000).to_device(ctx)
    var b = arange[Int32Type](0, 10000).to_device(ctx)
    var result = add[Int32Type](a, b, ctx).to_cpu(ctx)
    assert_equal(len(result), 10000)
    assert_equal(result.unsafe_get(0), 0)
    assert_equal(result.unsafe_get(4999), 9998)
    assert_equal(result.unsafe_get(9999), 19998)


def test_add_gpu_float32() raises:
    """GPU add with float32 arrays."""
    var ctx = DeviceContext()
    var a = array[Float32Type]([1, 2, 3, 4]).to_device(ctx)
    var b = array[Float32Type]([10, 20, 30, 40]).to_device(ctx)
    var result = add[Float32Type](a, b, ctx).to_cpu(ctx)
    assert_equal(len(result), 4)
    assert_true(result.unsafe_get(0) == 11)
    assert_true(result.unsafe_get(1) == 22)
    assert_true(result.unsafe_get(2) == 33)
    assert_true(result.unsafe_get(3) == 44)


def test_device_round_trip() raises:
    """Upload array to GPU, download back, verify values."""
    var ctx = DeviceContext()
    var a = arange[Int32Type](0, 1000)

    var on_device = a.to_device(ctx)
    assert_true(on_device.buffer.is_device())

    var on_host = on_device.to_cpu(ctx)
    assert_equal(len(on_host), 1000)
    assert_equal(on_host.unsafe_get(0), 0)
    assert_equal(on_host.unsafe_get(999), 999)


def test_chained_gpu_add() raises:
    """Chained GPU adds: (a + b) + c with device-resident intermediates."""
    var ctx = DeviceContext()
    var a = arange[Int32Type](0, 1000).to_device(ctx)
    var b = arange[Int32Type](0, 1000).to_device(ctx)
    var c = arange[Int32Type](0, 1000).to_device(ctx)

    var ab = add[Int32Type](a, b, ctx)
    assert_true(ab.buffer.is_device())

    var abc = add[Int32Type](ab, c, ctx).to_cpu(ctx)
    assert_equal(len(abc), 1000)
    assert_equal(abc.unsafe_get(0), 0)
    assert_equal(abc.unsafe_get(1), 3)
    assert_equal(abc.unsafe_get(999), 2997)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()

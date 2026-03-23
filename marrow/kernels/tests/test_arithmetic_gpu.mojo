from std.testing import assert_equal, assert_true, TestSuite
from std.gpu.host import DeviceContext

from marrow.builders import array, arange
from marrow.dtypes import int32, float32
from marrow.kernels.arithmetic import add


def test_add_gpu() raises:
    """Element-wise add on GPU with small int32 arrays."""
    var ctx = DeviceContext()
    var a = array[int32]([1, 2, 3, 4]).to_device(ctx)
    var b = array[int32]([10, 20, 30, 40]).to_device(ctx)
    var result = add[int32](a, b, ctx).to_cpu(ctx)
    assert_equal(len(result), 4)
    assert_equal(result.unsafe_get(0), 11)
    assert_equal(result.unsafe_get(1), 22)
    assert_equal(result.unsafe_get(2), 33)
    assert_equal(result.unsafe_get(3), 44)


def test_add_gpu_large() raises:
    """Exercise GPU add with a large array (10k elements)."""
    var ctx = DeviceContext()
    var a = arange[int32](0, 10000).to_device(ctx)
    var b = arange[int32](0, 10000).to_device(ctx)
    var result = add[int32](a, b, ctx).to_cpu(ctx)
    assert_equal(len(result), 10000)
    assert_equal(result.unsafe_get(0), 0)
    assert_equal(result.unsafe_get(4999), 9998)
    assert_equal(result.unsafe_get(9999), 19998)


def test_add_gpu_float32() raises:
    """GPU add with float32 arrays."""
    var ctx = DeviceContext()
    var a = array[float32]([1, 2, 3, 4]).to_device(ctx)
    var b = array[float32]([10, 20, 30, 40]).to_device(ctx)
    var result = add[float32](a, b, ctx).to_cpu(ctx)
    assert_equal(len(result), 4)
    assert_true(result.unsafe_get(0) == 11)
    assert_true(result.unsafe_get(1) == 22)
    assert_true(result.unsafe_get(2) == 33)
    assert_true(result.unsafe_get(3) == 44)


def test_device_round_trip() raises:
    """Upload array to GPU, download back, verify values."""
    var ctx = DeviceContext()
    var a = arange[int32](0, 1000)

    var on_device = a.to_device(ctx)
    assert_true(on_device.buffer.is_device())

    var on_host = on_device.to_cpu(ctx)
    assert_equal(len(on_host), 1000)
    assert_equal(on_host.unsafe_get(0), 0)
    assert_equal(on_host.unsafe_get(999), 999)


def test_chained_gpu_add() raises:
    """Chained GPU adds: (a + b) + c with device-resident intermediates."""
    var ctx = DeviceContext()
    var a = arange[int32](0, 1000).to_device(ctx)
    var b = arange[int32](0, 1000).to_device(ctx)
    var c = arange[int32](0, 1000).to_device(ctx)

    var ab = add[int32](a, b, ctx)
    assert_true(ab.buffer.is_device())

    var abc = add[int32](ab, c, ctx).to_cpu(ctx)
    assert_equal(len(abc), 1000)
    assert_equal(abc.unsafe_get(0), 0)
    assert_equal(abc.unsafe_get(1), 3)
    assert_equal(abc.unsafe_get(999), 2997)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()

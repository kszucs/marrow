import std.math as math
from std.testing import assert_equal, assert_true, assert_false, TestSuite
from std.gpu.host import DeviceContext

from marrow.buffers import Buffer, BufferBuilder, DeviceType


def test_buffer_device_kind() raises:
    """GPU DEVICE buffers are not CPU-accessible."""
    var ctx = DeviceContext()
    var dev = ctx.enqueue_create_buffer[DType.uint8](64)
    var buf = Buffer.from_device(dev, 64)
    assert_true(buf.is_device())
    assert_false(buf.is_cpu())
    assert_false(buf.is_host())

    var api = ctx.api()
    var expected_dev_type: Int32
    if api == "cuda":
        expected_dev_type = DeviceType.CUDA
    elif api == "hip":
        expected_dev_type = DeviceType.ROCM
    else:
        expected_dev_type = DeviceType.METAL
    assert_equal(buf.device_type(), expected_dev_type)
    assert_equal(buf.device_id(), Int64(0))


def test_buffer_host_kind() raises:
    """HOST (pinned) buffers are CPU-accessible with a valid ptr."""
    var ctx = DeviceContext()
    var host = ctx.enqueue_create_host_buffer[DType.uint8](64)
    var buf = Buffer.from_host(host)
    assert_true(buf.is_host())
    assert_true(buf.is_cpu())
    assert_false(buf.is_device())

    var api = ctx.api()
    if api == "cuda":
        assert_equal(buf.device_type(), DeviceType.CUDA_HOST)
    elif api == "hip":
        assert_equal(buf.device_type(), DeviceType.ROCM_HOST)
    assert_equal(buf.device_id(), Int64(0))
    assert_true(Int(buf.ptr) != 0)


def test_buffer_host_builder() raises:
    """BufferBuilder.alloc_host + finish produce a valid HOST buffer."""
    var ctx = DeviceContext()
    var b = BufferBuilder.alloc_host[DType.uint8](ctx, 64)
    b.unsafe_set(0, 7)
    b.unsafe_set(1, 13)
    var buf = b.finish()
    assert_true(buf.is_host())
    assert_true(buf.is_cpu())
    assert_false(buf.is_device())

    var api = ctx.api()
    if api == "cuda":
        assert_equal(buf.device_type(), DeviceType.CUDA_HOST)
    elif api == "hip":
        assert_equal(buf.device_type(), DeviceType.ROCM_HOST)
    assert_equal(buf.device_id(), Int64(0))
    assert_equal(buf.unsafe_get(0), UInt8(7))
    assert_equal(buf.unsafe_get(1), UInt8(13))
    assert_true(Int(buf.ptr) != 0)


def test_buffer_to_cpu_round_trip() raises:
    """Upload a CPU buffer to GPU then download back; data is preserved."""
    var ctx = DeviceContext()
    var builder = BufferBuilder.alloc_zeroed[DType.uint8](64)
    builder.unsafe_set(0, 42)
    builder.unsafe_set(1, 99)
    var cpu_buf = builder.finish()

    var dev_buf = cpu_buf.to_device(ctx)
    assert_true(dev_buf.is_device())

    var back = dev_buf.to_cpu(ctx)
    assert_true(back.is_cpu())
    assert_equal(back.unsafe_get(0), 42)
    assert_equal(back.unsafe_get(1), 99)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()

"""GPU tests for rapidhash kernel.

Verifies that GPU-dispatched rapidhash produces identical results to CPU SIMD.
"""

from std.testing import assert_equal, assert_true, TestSuite
from std.gpu.host import DeviceContext

from marrow.arrays import BoolArray, PrimitiveArray
from marrow.builders import array, arange, BoolBuilder, PrimitiveBuilder
from marrow.dtypes import bool_, int32, int64, float32, uint64
from marrow.kernels.hashing import rapidhash, NULL_HASH_SENTINEL


def test_rapidhash_gpu_int32() raises:
    """GPU rapidhash on small int32 array matches CPU."""
    var ctx = DeviceContext()
    var arr = array[int32]([1, 2, 3, 4, 5])
    var cpu_hashes = rapidhash[int32](arr)
    var gpu_hashes = rapidhash[int32](arr.to_device(ctx), ctx).to_cpu(ctx)
    assert_equal(len(gpu_hashes), len(cpu_hashes))
    assert_true(cpu_hashes == gpu_hashes)


def test_rapidhash_gpu_int64() raises:
    """GPU rapidhash on small int64 array matches CPU."""
    var ctx = DeviceContext()
    var arr = array[int64]([10, 20, 30, 40, 50])
    var cpu_hashes = rapidhash[int64](arr)
    var gpu_hashes = rapidhash[int64](arr.to_device(ctx), ctx).to_cpu(ctx)
    assert_equal(len(gpu_hashes), len(cpu_hashes))
    assert_true(cpu_hashes == gpu_hashes)


def test_rapidhash_gpu_float32() raises:
    """GPU rapidhash on float32 array matches CPU."""
    var ctx = DeviceContext()
    var arr = array[float32]([1.0, 2.5, 3.14, 0.0, -1.0])
    var cpu_hashes = rapidhash[float32](arr)
    var gpu_hashes = rapidhash[float32](arr.to_device(ctx), ctx).to_cpu(ctx)
    assert_equal(len(gpu_hashes), len(cpu_hashes))
    assert_true(cpu_hashes == gpu_hashes)


def test_rapidhash_gpu_large() raises:
    """GPU rapidhash on 10k int32 array, spot-check positions match CPU."""
    var ctx = DeviceContext()
    var arr = arange[int32](0, 10000)
    var cpu_hashes = rapidhash[int32](arr)
    var gpu_hashes = rapidhash[int32](arr.to_device(ctx), ctx).to_cpu(ctx)
    assert_equal(len(gpu_hashes), 10000)
    assert_equal(gpu_hashes.unsafe_get(0), cpu_hashes.unsafe_get(0))
    assert_equal(gpu_hashes.unsafe_get(4999), cpu_hashes.unsafe_get(4999))
    assert_equal(gpu_hashes.unsafe_get(9999), cpu_hashes.unsafe_get(9999))


def test_rapidhash_gpu_nulls() raises:
    """GPU rapidhash with nulls produces sentinel values matching CPU."""
    var ctx = DeviceContext()
    var b = PrimitiveBuilder[int32](capacity=5)
    b.append(Scalar[int32.native](1))
    b.append_null()
    b.append(Scalar[int32.native](3))
    b.append_null()
    b.append(Scalar[int32.native](5))
    var arr = b.finish()
    var cpu_hashes = rapidhash[int32](arr)
    var gpu_hashes = rapidhash[int32](arr.to_device(ctx), ctx).to_cpu(ctx)
    assert_equal(len(gpu_hashes), 5)
    # Valid positions should match
    assert_equal(gpu_hashes.unsafe_get(0), cpu_hashes.unsafe_get(0))
    assert_equal(gpu_hashes.unsafe_get(2), cpu_hashes.unsafe_get(2))
    assert_equal(gpu_hashes.unsafe_get(4), cpu_hashes.unsafe_get(4))
    # Null positions should be sentinel
    assert_equal(UInt64(gpu_hashes.unsafe_get(1)), NULL_HASH_SENTINEL)
    assert_equal(UInt64(gpu_hashes.unsafe_get(3)), NULL_HASH_SENTINEL)


def test_rapidhash_gpu_bool() raises:
    """GPU rapidhash on bool array matches CPU."""
    var ctx = DeviceContext()
    var b = BoolBuilder(capacity=6)
    b.append(True)
    b.append(False)
    b.append(True)
    b.append_null()
    b.append(False)
    b.append(True)
    var arr = b.finish()
    var cpu_hashes = rapidhash(arr)
    var gpu_hashes = rapidhash(arr.to_device(ctx), ctx).to_cpu(ctx)
    assert_equal(len(gpu_hashes), 6)
    assert_true(cpu_hashes == gpu_hashes)
    assert_equal(UInt64(gpu_hashes.unsafe_get(3)), NULL_HASH_SENTINEL)


def test_rapidhash_gpu_device_resident() raises:
    """Verify GPU result is device-resident before to_cpu()."""
    var ctx = DeviceContext()
    var arr = array[int32]([1, 2, 3]).to_device(ctx)
    var result = rapidhash[int32](arr, ctx)
    assert_true(result.buffer.is_device())
    var on_cpu = result.to_cpu(ctx)
    assert_equal(len(on_cpu), 3)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()

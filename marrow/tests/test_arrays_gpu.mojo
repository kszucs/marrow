"""Device round-trips for sliced arrays.

`to_device` uploads the whole values buffer — the device copy mirrors the host
buffer byte for byte — but then rebuilt the array with `offset=0` while keeping
`length`, so a sliced array silently became the *parent's* first `length`
elements. `FixedSizeListArray.to_device` preserved the offset and was correct,
which is what made the inconsistency visible.
"""

from std.testing import assert_equal, assert_true
from max.gpu.host import DeviceContext

from ..arrays import BoolArray, Int64Array
from ..builders import array, BoolBuilder, Int64Builder
from ..dtypes import int64


def test_sliced_primitive_array_survives_device_roundtrip() raises:
    var ctx = DeviceContext()
    var full = array([10, 20, 30, 40, 50], int64)
    var sliced = full.slice(2, 3)  # [30, 40, 50]

    var back = sliced.to_device(ctx).to_cpu(ctx)
    assert_equal(len(back), 3)
    assert_equal(Int(back[0].value()), 30)
    assert_equal(Int(back[1].value()), 40)
    assert_equal(Int(back[2].value()), 50)


def test_unsliced_primitive_array_device_roundtrip() raises:
    var ctx = DeviceContext()
    var a = array([1, 2, 3], int64)
    var back = a.to_device(ctx).to_cpu(ctx)
    assert_equal(len(back), 3)
    assert_equal(Int(back[0].value()), 1)
    assert_equal(Int(back[2].value()), 3)


def test_sliced_primitive_array_with_nulls_device_roundtrip() raises:
    var ctx = DeviceContext()
    var b = Int64Builder(capacity=5)
    b.append(Int64(10))
    b.append_null()
    b.append(Int64(30))
    b.append_null()
    b.append(Int64(50))
    var sliced = b.finish().slice(1, 3)  # [null, 30, null]

    var back = sliced.to_device(ctx).to_cpu(ctx)
    assert_equal(len(back), 3)
    assert_true(not back.is_valid(0))
    assert_true(back.is_valid(1))
    assert_equal(Int(back[1].value()), 30)
    assert_true(not back.is_valid(2))


def test_sliced_bool_array_survives_device_roundtrip() raises:
    var ctx = DeviceContext()
    var bb = BoolBuilder(capacity=5)
    bb.append(True)
    bb.append(False)
    bb.append(True)
    bb.append(True)
    bb.append(False)
    var sliced = bb.finish().slice(1, 3)  # [False, True, True]

    var back = sliced.to_device(ctx).to_cpu(ctx)
    assert_equal(len(back), 3)
    assert_true(not back[0].value())
    assert_true(back[1].value())
    assert_true(back[2].value())

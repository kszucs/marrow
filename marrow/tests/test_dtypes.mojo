from std.testing import assert_equal, assert_true, assert_false, TestSuite
import marrow.dtypes as dt
from marrow.dtypes import ArrowType, Field


def test_bool_type() raises:
    assert_true(ArrowType(dt.bool_) == ArrowType(dt.bool_))
    assert_false(ArrowType(dt.bool_) == ArrowType(dt.int64))


def test_list_type() raises:
    assert_true(dt.list_(dt.int64) == dt.list_(dt.int64))
    assert_false(dt.list_(dt.int64) == dt.list_(dt.int32))
    t1 = dt.list_(dt.list_(dt.int64))
    assert_equal(String(t1), "list<list<int64>>")


def test_field() raises:
    var field = dt.Field("a", dt.int64, False)
    var expected = "a: int64"
    assert_equal(String(field), expected)


def test_struct_type() raises:
    s1 = dt.struct_(
        dt.Field("a", dt.int64, False),
        dt.Field("b", dt.int32, False),
    )
    s2 = dt.struct_(
        dt.Field("a", dt.int64, False),
        dt.Field("b", dt.int32, False),
    )
    s3 = dt.struct_(
        dt.Field("a", dt.int64, False),
        dt.Field("b", dt.int32, False),
        dt.Field("c", dt.int8, False),
    )
    assert_true(s1 == s2)
    assert_false(s1 == s3)


def test_is_integer() raises:
    assert_true(ArrowType(dt.int8).is_integer())
    assert_true(ArrowType(dt.int16).is_integer())
    assert_true(ArrowType(dt.int32).is_integer())
    assert_true(ArrowType(dt.int64).is_integer())
    assert_true(ArrowType(dt.uint8).is_integer())
    assert_true(ArrowType(dt.uint16).is_integer())
    assert_true(ArrowType(dt.uint32).is_integer())
    assert_true(ArrowType(dt.uint64).is_integer())
    assert_false(ArrowType(dt.bool_).is_integer())
    assert_false(ArrowType(dt.float32).is_integer())
    assert_false(ArrowType(dt.float64).is_integer())
    assert_false(dt.list_(dt.int64).is_integer())


def test_is_signed_integer() raises:
    assert_true(ArrowType(dt.int8).is_signed_integer())
    assert_true(ArrowType(dt.int16).is_signed_integer())
    assert_true(ArrowType(dt.int32).is_signed_integer())
    assert_true(ArrowType(dt.int64).is_signed_integer())
    assert_false(ArrowType(dt.uint8).is_signed_integer())
    assert_false(ArrowType(dt.uint16).is_signed_integer())
    assert_false(ArrowType(dt.uint32).is_signed_integer())
    assert_false(ArrowType(dt.uint64).is_signed_integer())
    assert_false(ArrowType(dt.bool_).is_signed_integer())
    assert_false(ArrowType(dt.float32).is_signed_integer())
    assert_false(ArrowType(dt.float64).is_signed_integer())


def test_is_unsigned_integer() raises:
    assert_false(ArrowType(dt.int8).is_unsigned_integer())
    assert_false(ArrowType(dt.int16).is_unsigned_integer())
    assert_false(ArrowType(dt.int32).is_unsigned_integer())
    assert_false(ArrowType(dt.int64).is_unsigned_integer())
    assert_true(ArrowType(dt.uint8).is_unsigned_integer())
    assert_true(ArrowType(dt.uint16).is_unsigned_integer())
    assert_true(ArrowType(dt.uint32).is_unsigned_integer())
    assert_true(ArrowType(dt.uint64).is_unsigned_integer())
    assert_false(ArrowType(dt.bool_).is_unsigned_integer())
    assert_false(ArrowType(dt.float32).is_unsigned_integer())
    assert_false(ArrowType(dt.float64).is_unsigned_integer())


def test_is_floating_point() raises:
    assert_false(ArrowType(dt.int8).is_floating_point())
    assert_false(ArrowType(dt.int16).is_floating_point())
    assert_false(ArrowType(dt.int32).is_floating_point())
    assert_false(ArrowType(dt.int64).is_floating_point())
    assert_false(ArrowType(dt.uint8).is_floating_point())
    assert_false(ArrowType(dt.uint16).is_floating_point())
    assert_false(ArrowType(dt.uint32).is_floating_point())
    assert_false(ArrowType(dt.uint64).is_floating_point())
    assert_false(ArrowType(dt.bool_).is_floating_point())
    assert_true(ArrowType(dt.float32).is_floating_point())
    assert_true(ArrowType(dt.float64).is_floating_point())


def test_bit_width() raises:
    assert_equal(ArrowType(dt.int8).bit_width(), 8)
    assert_equal(ArrowType(dt.int16).bit_width(), 16)
    assert_equal(ArrowType(dt.int32).bit_width(), 32)
    assert_equal(ArrowType(dt.int64).bit_width(), 64)
    assert_equal(ArrowType(dt.uint8).bit_width(), 8)
    assert_equal(ArrowType(dt.uint16).bit_width(), 16)
    assert_equal(ArrowType(dt.uint32).bit_width(), 32)
    assert_equal(ArrowType(dt.uint64).bit_width(), 64)
    assert_equal(ArrowType(dt.bool_).bit_width(), 1)
    assert_equal(ArrowType(dt.float32).bit_width(), 32)
    assert_equal(ArrowType(dt.float64).bit_width(), 64)
    assert_equal(dt.list_(dt.int64).bit_width(), 0)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()

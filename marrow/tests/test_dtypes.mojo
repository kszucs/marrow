from std.testing import assert_equal, assert_true, assert_false, TestSuite

import marrow.dtypes as dt


def test_bool_type() raises:
    assert_true(materialize[dt.bool_]() == materialize[dt.bool_]())
    assert_false(materialize[dt.bool_]() == materialize[dt.int64]())
    assert_true(materialize[dt.bool_]() is materialize[dt.bool_]())
    assert_false(materialize[dt.bool_]() is materialize[dt.int64]())


def test_list_type() raises:
    assert_true(
        dt.list_(materialize[dt.int64]()) == dt.list_(materialize[dt.int64]())
    )
    assert_false(
        dt.list_(materialize[dt.int64]()) == dt.list_(materialize[dt.int32]())
    )


def test_field() raises:
    var field = dt.Field("a", materialize[dt.int64](), False)
    var writer = String()
    writer.write(field)
    var expected = (
        'Field(name="a", dtype=DataType(code=int64), nullable=False, )'
    )
    assert_equal(writer, expected)
    assert_equal(String(field), expected)


def test_struct_type() raises:
    s1 = dt.struct_(
        dt.Field("a", materialize[dt.int64](), False),
        dt.Field("b", materialize[dt.int32](), False),
    )
    s2 = dt.struct_(
        dt.Field("a", materialize[dt.int64](), False),
        dt.Field("b", materialize[dt.int32](), False),
    )
    s3 = dt.struct_(
        dt.Field("a", materialize[dt.int64](), False),
        dt.Field("b", materialize[dt.int32](), False),
        dt.Field("c", materialize[dt.int8](), False),
    )
    assert_true(s1 == s2)
    assert_false(s1 == s3)


def test_is_integer() raises:
    assert_true(materialize[dt.int8]().is_integer())
    assert_true(materialize[dt.int16]().is_integer())
    assert_true(materialize[dt.int32]().is_integer())
    assert_true(materialize[dt.int64]().is_integer())
    assert_true(materialize[dt.uint8]().is_integer())
    assert_true(materialize[dt.uint16]().is_integer())
    assert_true(materialize[dt.uint32]().is_integer())
    assert_true(materialize[dt.uint64]().is_integer())
    assert_false(materialize[dt.bool_]().is_integer())
    assert_false(materialize[dt.float32]().is_integer())
    assert_false(materialize[dt.float64]().is_integer())
    assert_false(dt.list_(materialize[dt.int64]()).is_integer())


def test_is_signed_integer() raises:
    assert_true(materialize[dt.int8]().is_signed_integer())
    assert_true(materialize[dt.int16]().is_signed_integer())
    assert_true(materialize[dt.int32]().is_signed_integer())
    assert_true(materialize[dt.int64]().is_signed_integer())
    assert_false(materialize[dt.uint8]().is_signed_integer())
    assert_false(materialize[dt.uint16]().is_signed_integer())
    assert_false(materialize[dt.uint32]().is_signed_integer())
    assert_false(materialize[dt.uint64]().is_signed_integer())
    assert_false(materialize[dt.bool_]().is_signed_integer())
    assert_false(materialize[dt.float32]().is_signed_integer())
    assert_false(materialize[dt.float64]().is_signed_integer())


def test_is_unsigned_integer() raises:
    assert_false(materialize[dt.int8]().is_unsigned_integer())
    assert_false(materialize[dt.int16]().is_unsigned_integer())
    assert_false(materialize[dt.int32]().is_unsigned_integer())
    assert_false(materialize[dt.int64]().is_unsigned_integer())
    assert_true(materialize[dt.uint8]().is_unsigned_integer())
    assert_true(materialize[dt.uint16]().is_unsigned_integer())
    assert_true(materialize[dt.uint32]().is_unsigned_integer())
    assert_true(materialize[dt.uint64]().is_unsigned_integer())
    assert_false(materialize[dt.bool_]().is_unsigned_integer())
    assert_false(materialize[dt.float32]().is_unsigned_integer())
    assert_false(materialize[dt.float64]().is_unsigned_integer())


def test_is_floating_point() raises:
    assert_false(materialize[dt.int8]().is_floating_point())
    assert_false(materialize[dt.int16]().is_floating_point())
    assert_false(materialize[dt.int32]().is_floating_point())
    assert_false(materialize[dt.int64]().is_floating_point())
    assert_false(materialize[dt.uint8]().is_floating_point())
    assert_false(materialize[dt.uint16]().is_floating_point())
    assert_false(materialize[dt.uint32]().is_floating_point())
    assert_false(materialize[dt.uint64]().is_floating_point())
    assert_false(materialize[dt.bool_]().is_floating_point())
    assert_true(materialize[dt.float32]().is_floating_point())
    assert_true(materialize[dt.float64]().is_floating_point())


def test_bitwidth() raises:
    assert_equal(materialize[dt.int8]().bitwidth(), 8)
    assert_equal(materialize[dt.int16]().bitwidth(), 16)
    assert_equal(materialize[dt.int32]().bitwidth(), 32)
    assert_equal(materialize[dt.int64]().bitwidth(), 64)
    assert_equal(materialize[dt.uint8]().bitwidth(), 8)
    assert_equal(materialize[dt.uint16]().bitwidth(), 16)
    assert_equal(materialize[dt.uint32]().bitwidth(), 32)
    assert_equal(materialize[dt.uint64]().bitwidth(), 64)
    assert_equal(materialize[dt.bool_]().bitwidth(), 1)
    assert_equal(materialize[dt.float32]().bitwidth(), 32)
    assert_equal(materialize[dt.float64]().bitwidth(), 64)
    assert_equal(dt.list_(materialize[dt.int64]()).bitwidth(), 0)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()

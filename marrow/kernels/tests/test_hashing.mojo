from std.testing import assert_equal, assert_true, assert_false, TestSuite

from marrow.arrays import AnyArray, PrimitiveArray, StringArray
from marrow.builders import array, PrimitiveBuilder, StringBuilder
from marrow.dtypes import int32, int64, uint8, uint64, float64, bool_
from marrow.arrays import StructArray
from marrow.dtypes import Field, struct_
from marrow.kernels.hashing import rapidhash, NULL_HASH_SENTINEL


def _children(ref a: AnyArray, ref b: AnyArray) -> List[AnyArray]:
    var c = List[AnyArray]()
    c.append(a.copy())
    c.append(b.copy())
    return c^


def _children1(ref a: AnyArray) -> List[AnyArray]:
    var c = List[AnyArray]()
    c.append(a.copy())
    return c^


# ---------------------------------------------------------------------------
# hash_ — primitive
# ---------------------------------------------------------------------------


def test_hash__int32_deterministic() raises:
    """Same values produce same hashes."""
    var a = array[Int32Type]([1, 2, 3, 1, 2])
    var h = rapidhash(a)
    assert_equal(len(h), 5)
    assert_equal(h[0], h[3])  # both are value 1
    assert_equal(h[1], h[4])  # both are value 2


def test_hash__int32_distinct() raises:
    """Different values produce different hashes (probabilistic)."""
    var a = array[Int32Type]([1, 2, 3])
    var h = rapidhash(a)
    assert_true(h[0] != h[1])
    assert_true(h[1] != h[2])


def test_hash__int32_nulls() raises:
    """Null elements hash to NULL_HASH_SENTINEL."""
    var a = array[Int32Type]([1, None, 2, None])
    var h = rapidhash(a)
    assert_equal(h[1], Scalar[uint64.native](NULL_HASH_SENTINEL))
    assert_equal(h[3], Scalar[uint64.native](NULL_HASH_SENTINEL))
    assert_true(h[0] != Scalar[uint64.native](NULL_HASH_SENTINEL))


def test_hash__empty() raises:
    var a = array[Int32Type]()
    var h = rapidhash(a)
    assert_equal(len(h), 0)


def test_hash__float64() raises:
    var a = array[Float64Type]([1.5, 2.5, 1.5])
    var h = rapidhash(a)
    assert_equal(h[0], h[2])


# ---------------------------------------------------------------------------
# hash_ — string
# ---------------------------------------------------------------------------


def test_hash__string() raises:
    var b = StringBuilder(4)
    b.append("foo")
    b.append("bar")
    b.append("foo")
    b.append("baz")
    var keys = b.finish()

    var h = rapidhash(keys)
    assert_equal(len(h), 4)
    assert_equal(h[0], h[2])  # both "foo"
    assert_true(h[0] != h[1])  # "foo" != "bar"


def test_hash__string_nulls() raises:
    var b = StringBuilder(3)
    b.append("a")
    b.append_null()
    b.append("b")
    var keys = b.finish()

    var h = rapidhash(keys)
    assert_equal(h[1], Scalar[uint64.native](NULL_HASH_SENTINEL))


# ---------------------------------------------------------------------------
# hash_ — type-erased dispatch
# ---------------------------------------------------------------------------


def test_hash__dispatch() raises:
    var a = AnyArray(array[Int32Type]([1, 2, 1]))
    var h = rapidhash(a)
    assert_equal(len(h), 3)
    assert_equal(h[0], h[2])


def test_hash__dispatch_string() raises:
    var b = StringBuilder(2)
    b.append("x")
    b.append("x")
    var a = AnyArray(b.finish())

    var h = rapidhash(a)
    assert_equal(h[0], h[1])


# ---------------------------------------------------------------------------
# hash_ — struct array (multi-column)
# ---------------------------------------------------------------------------


def test_hash_struct_two_fields() raises:
    """StructArray hashing combines per-field hashes."""
    var a = AnyArray(array[Int32Type]([1, 1, 2, 2]))
    var b = AnyArray(array[Int32Type]([10, 20, 10, 20]))
    var sa = StructArray(
        dtype=struct_(
            Field("a", a.dtype().copy()), Field("b", b.dtype().copy())
        ),
        length=4,
        nulls=0,
        offset=0,
        bitmap=None,
        children=_children(a, b),
    )
    var h = rapidhash(sa)
    assert_equal(len(h), 4)
    # (1,10) != (1,20)
    assert_true(h[0] != h[1])
    # (1,10) != (2,10)
    assert_true(h[0] != h[2])
    # (2,10) != (2,20)
    assert_true(h[2] != h[3])


def test_hash_struct_single_field() raises:
    """Single-field struct matches direct array hash."""
    var a = array[Int32Type]([1, 2, 3])
    var h1 = rapidhash(a)

    var arr = AnyArray(a^)
    var sa = StructArray(
        dtype=struct_(Field("a", arr.dtype().copy())),
        length=3,
        nulls=0,
        offset=0,
        bitmap=None,
        children=_children1(arr),
    )
    var h2 = rapidhash(sa)
    assert_equal(h1[0], h2[0])
    assert_equal(h1[1], h2[1])
    assert_equal(h1[2], h2[2])


def test_hash_dispatch_struct() raises:
    """Type-erased dispatch to struct hash."""
    var a = AnyArray(array[Int32Type]([1, 2, 1]))
    var b = AnyArray(array[Int32Type]([3, 3, 3]))
    var sa = StructArray(
        dtype=struct_(
            Field("a", a.dtype().copy()), Field("b", b.dtype().copy())
        ),
        length=3,
        nulls=0,
        offset=0,
        bitmap=None,
        children=_children(a, b),
    )
    var h = rapidhash(AnyArray(sa^))
    assert_equal(len(h), 3)
    # (1,3) == (1,3) but row 0 and 2 same
    assert_equal(h[0], h[2])


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()

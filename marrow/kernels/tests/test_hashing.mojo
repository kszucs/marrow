from std.testing import assert_equal, assert_true, assert_false


from ...arrays import DynArray, DictionaryArray, PrimitiveArray, StringArray
from ...builders import (
    Int8Builder,
    Int16Builder,
    array,
    Date32Builder,
    Decimal128Builder,
    Int32Builder,
    LargeStringBuilder,
    PrimitiveBuilder,
    StringBuilder,
    TimestampBuilder,
)
from ...dtypes import (
    int32,
    int64,
    uint8,
    uint64,
    float64,
    date32,
    decimal128,
    microsecond,
    timestamp,
    Int32Type,
    Int64Type,
    UInt8Type,
    UInt64Type,
    Float64Type,
)
from ...arrays import StructArray
from ...dtypes import Field, struct_
from ...kernels.aggregate import Fold, SumKernel
from ...kernels.groupby import GroupBy

from ...arrays import UInt64Array
from ...builders import Int64Builder
from ...utils import RapidHash64
from ...kernels.hashing import RapidHashKernel, NULL_HASH_SENTINEL


def _children(ref a: DynArray, ref b: DynArray) -> List[DynArray]:
    var c = List[DynArray]()
    c.append(a.copy())
    c.append(b.copy())
    return c^


def _children1(ref a: DynArray) -> List[DynArray]:
    var c = List[DynArray]()
    c.append(a.copy())
    return c^


# ---------------------------------------------------------------------------
# hash_ — primitive
# ---------------------------------------------------------------------------


def test_hash__int32_deterministic() raises:
    """Same values produce same hashes."""
    var a = array([1, 2, 3, 1, 2], int32)
    var h = RapidHashKernel.apply(a)
    assert_equal(len(h), 5)
    assert_equal(h[0], h[3])  # both are value 1
    assert_equal(h[1], h[4])  # both are value 2


def test_hash__int32_distinct() raises:
    """Different values produce different hashes (probabilistic)."""
    var a = array([1, 2, 3], int32)
    var h = RapidHashKernel.apply(a)
    assert_true(h[0] != h[1])
    assert_true(h[1] != h[2])


def test_hash__int32_nulls() raises:
    """Null elements hash to NULL_HASH_SENTINEL."""
    var a = array([1, None, 2, None], int32)
    var h = RapidHashKernel.apply(a)
    assert_equal(h[1].value(), Scalar[uint64.native](NULL_HASH_SENTINEL))
    assert_equal(h[3].value(), Scalar[uint64.native](NULL_HASH_SENTINEL))
    assert_true(h[0].value() != Scalar[uint64.native](NULL_HASH_SENTINEL))


def test_hash__empty() raises:
    var a = array(int32)
    var h = RapidHashKernel.apply(a)
    assert_equal(len(h), 0)


def test_hash__float64() raises:
    var a = array([1.5, 2.5, 1.5], float64)
    var h = RapidHashKernel.apply(a)
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

    var h = RapidHashKernel.apply(keys)
    assert_equal(len(h), 4)
    assert_equal(h[0], h[2])  # both "foo"
    assert_true(h[0] != h[1])  # "foo" != "bar"


def test_hash__string_nulls() raises:
    var b = StringBuilder(3)
    b.append("a")
    b.append_null()
    b.append("b")
    var keys = b.finish()

    var h = RapidHashKernel.apply(keys)
    assert_equal(h[1].value(), Scalar[uint64.native](NULL_HASH_SENTINEL))


# ---------------------------------------------------------------------------
# hash_ — type-erased dispatch
# ---------------------------------------------------------------------------


def test_hash__dispatch() raises:
    var a: DynArray = array([1, 2, 1], int32)
    var h = RapidHashKernel.dispatch(a)
    assert_equal(len(h), 3)
    assert_equal(h[0], h[2])


def test_hash__dispatch_string() raises:
    var b = StringBuilder(2)
    b.append("x")
    b.append("x")
    var a: DynArray = b.finish()

    var h = RapidHashKernel.dispatch(a)
    assert_equal(h[0], h[1])


# ---------------------------------------------------------------------------
# hash_ — struct array (multi-column)
# ---------------------------------------------------------------------------


def test_hash_struct_two_fields() raises:
    """StructArray hashing combines per-field hashes."""
    var a: DynArray = array([1, 1, 2, 2], int32)
    var b: DynArray = array([10, 20, 10, 20], int32)
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
    var h = RapidHashKernel.apply(sa)
    assert_equal(len(h), 4)
    # (1,10) != (1,20)
    assert_true(h[0] != h[1])
    # (1,10) != (2,10)
    assert_true(h[0] != h[2])
    # (2,10) != (2,20)
    assert_true(h[2] != h[3])


def test_hash_struct_single_field() raises:
    """Single-field struct matches direct array hash."""
    var a = array([1, 2, 3], int32)
    var h1 = RapidHashKernel.apply(a)

    var arr: DynArray = a^
    var sa = StructArray(
        dtype=struct_(Field("a", arr.dtype().copy())),
        length=3,
        nulls=0,
        offset=0,
        bitmap=None,
        children=_children1(arr),
    )
    var h2 = RapidHashKernel.apply(sa)
    assert_equal(h1[0], h2[0])
    assert_equal(h1[1], h2[1])
    assert_equal(h1[2], h2[2])


def test_hash_dispatch_struct() raises:
    """Type-erased dispatch to struct hash."""
    var a: DynArray = array([1, 2, 1], int32)
    var b: DynArray = array([3, 3, 3], int32)
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
    var h = RapidHashKernel.apply(sa^)
    assert_equal(len(h), 3)
    # (1,3) == (1,3) but row 0 and 2 same
    assert_equal(h[0], h[2])


# ---------------------------------------------------------------------------
# hash_ — temporal, large_string, decimal, dictionary
#
# These dtypes used to fall off the end of the dispatch ladder and raise, which
# broke `GROUP BY` on any temporal key. Fixed by dispatching on the widest
# family the typed leaf accepts; see CLAUDE.md, "Dispatch on the widest family".
# ---------------------------------------------------------------------------


def _date32(var days: List[Int]) raises -> DynArray:
    var b = Date32Builder(date32(), len(days))
    for d in days:
        b.append(Scalar[int32.native](d))
    return b.finish()


def _timestamp(var micros: List[Int]) raises -> DynArray:
    var b = TimestampBuilder(timestamp(microsecond, "UTC"), len(micros))
    for m in micros:
        b.append(Scalar[int64.native](m))
    return b.finish()


def test_hash_date32() raises:
    """A date32 column hashes as its int32 storage type."""
    var d = _date32([19000, 18500, 19000])
    var h = RapidHashKernel.dispatch(d)
    assert_equal(len(h), 3)
    assert_equal(h[0], h[2])
    assert_true(h[0] != h[1])
    # Identical to hashing the same days as plain int32 — the reinterpret path
    # must not perturb the hash.
    var i = array([19000, 18500, 19000], int32)
    assert_true(RapidHashKernel.apply(i) == h)


def test_hash_timestamp() raises:
    """A timestamp column (int64 storage) hashes and dedups by value."""
    var t = _timestamp([1_700_000_000_000_000, 1_600_000_000_000_000])
    var h = RapidHashKernel.dispatch(t)
    assert_equal(len(h), 2)
    assert_true(h[0] != h[1])


def test_hash_timestamp_nulls() raises:
    var b = TimestampBuilder(timestamp(microsecond), 3)
    b.append(Scalar[int64.native](10))
    b.append_null()
    b.append(Scalar[int64.native](10))
    var t: DynArray = b.finish()
    var h = RapidHashKernel.dispatch(t)
    assert_equal(h[1].value(), Scalar[uint64.native](NULL_HASH_SENTINEL))
    assert_equal(h[0], h[2])


def test_hash_large_string() raises:
    """`large_string` hashes identically to string for the same bytes."""
    var lb = LargeStringBuilder(3)
    lb.append("foo")
    lb.append("bar")
    lb.append("foo")
    var large: DynArray = lb.finish()

    var sb = StringBuilder(3)
    sb.append("foo")
    sb.append("bar")
    sb.append("foo")
    var small: DynArray = sb.finish()

    var h = RapidHashKernel.dispatch(large)
    assert_equal(h[0], h[2])
    assert_true(RapidHashKernel.dispatch(small) == h)


def test_hash_decimal128_high_bits() raises:
    """`decimal128` folds both 64-bit limbs — values that differ only above bit 63
    must not collide (group-by buckets on the hash alone)."""
    var b = Decimal128Builder(decimal128(38, 0), 3)
    b.append(Scalar[DType.int128](1))
    b.append(Scalar[DType.int128](1) << Scalar[DType.int128](70))
    b.append(Scalar[DType.int128](1))
    var d: DynArray = b.finish()
    var h = RapidHashKernel.dispatch(d)
    assert_equal(h[0], h[2])
    assert_true(h[0] != h[1])


def test_hash_dictionary_matches_decoded() raises:
    """A dictionary column hashes like the plain column it encodes."""
    var values = StringBuilder(2)
    values.append("red")
    values.append("blue")
    var ib = Int32Builder(4)
    for i in [0, 1, 0, 1]:
        ib.append(Int32(i))
    var dict_arr: DynArray = DictionaryArray.from_arrays(
        ib.finish(), values.finish()
    )

    var plain = StringBuilder(4)
    for s in ["red", "blue", "red", "blue"]:
        plain.append(s)
    var plain_arr: DynArray = plain.finish()

    var h = RapidHashKernel.dispatch(dict_arr)
    assert_equal(len(h), 4)
    assert_true(RapidHashKernel.dispatch(plain_arr) == h)


def test_hash_dictionary_with_narrow_indices() raises:
    """The same, with int8 indices.

    Arrow allows any integer type as a dictionary index. The int32 case above
    takes a fast path that returns the indices as-is; every other width goes
    through the widening in `_indices_as_int32`, which replaced the `cast` call
    that used to make `kernels.cast` reachable from every hashing binary (Q4.7).
    Without this the widening branch has no coverage at all.
    """
    var values = StringBuilder(2)
    values.append("red")
    values.append("blue")
    var ib = Int8Builder(4)
    for i in [0, 1, 0, 1]:
        ib.append(Int8(i))
    var dict_arr: DynArray = DictionaryArray.from_arrays(
        ib.finish(), values.finish()
    )

    var plain = StringBuilder(4)
    for s in ["red", "blue", "red", "blue"]:
        plain.append(s)
    var plain_arr: DynArray = plain.finish()

    assert_true(
        RapidHashKernel.dispatch(plain_arr)
        == RapidHashKernel.dispatch(dict_arr)
    )


def test_hash_dictionary_with_null_index() raises:
    """A null index must survive the widening as a null, not as index 0."""
    var values = StringBuilder(2)
    values.append("red")
    values.append("blue")
    var ib = Int16Builder(3)
    ib.append(Int16(1))
    ib.append_null()
    ib.append(Int16(0))
    var dict_arr: DynArray = DictionaryArray.from_arrays(
        ib.finish(), values.finish()
    )
    var h = RapidHashKernel.dispatch(dict_arr)
    assert_equal(len(h), 3)
    # The null row must not hash as "red" (index 0).
    assert_true(h[1].value() != h[2].value())


# ---------------------------------------------------------------------------
# GROUP BY over a temporal key — the M1 blocker this dispatch gap caused
# (ClickBench Q19/35/36/40/43 all group on EventDate/EventTime).
# ---------------------------------------------------------------------------


def test_groupby_date32_key() raises:
    var keys = _date32([19000, 18500, 19000, 18500, 19000])
    var vals: DynArray = array([1, 2, 3, 4, 5], int32)
    var result = GroupBy(keys).aggregate[Fold[SumKernel, Int32Type]](vals)

    assert_equal(result.num_rows(), 2)
    ref k = result.keys[0].as_date32()
    assert_equal(k[0].value(), 19000)
    assert_equal(k[1].value(), 18500)
    ref s = result.aggregates[0].as_int64()
    assert_equal(s[0].value(), 9)  # 1 + 3 + 5
    assert_equal(s[1].value(), 6)  # 2 + 4


def test_groupby_timestamp_key() raises:
    var keys = _timestamp([1_000, 2_000, 1_000])
    var vals: DynArray = array([10, 20, 30], int32)
    var result = GroupBy(keys).aggregate[Fold[SumKernel, Int32Type]](vals)

    assert_equal(result.num_rows(), 2)
    ref s = result.aggregates[0].as_int64()
    assert_equal(s[0].value(), 40)
    assert_equal(s[1].value(), 20)


def test_groupby_large_string_key() raises:
    var lb = LargeStringBuilder(4)
    for s in ["a", "b", "a", "b"]:
        lb.append(s)
    var keys: DynArray = lb.finish()
    var vals: DynArray = array([1, 2, 3, 4], int32)
    var result = GroupBy(keys).aggregate[Fold[SumKernel, Int32Type]](vals)

    assert_equal(result.num_rows(), 2)
    ref s = result.aggregates[0].as_int64()
    assert_equal(s[0].value(), 4)
    assert_equal(s[1].value(), 6)


# ===========================================================================
# Reference vectors — folded in from the former test_rapidhash.mojo
#
# Generated by compiling and running rapidhash.h (v3) with the default seed and
# secrets: https://github.com/Nicoshev/rapidhash
# ===========================================================================

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _assert_hash_eq(h: UInt64Array, idx: Int, expected: UInt64) raises:
    var got = UInt64(h.unsafe_get(idx))
    if got != expected:
        raise Error(
            "rapidhash mismatch at index "
            + String(idx)
            + ": got 0x"
            + hex(got)
            + " expected 0x"
            + hex(expected)
        )


# ---------------------------------------------------------------------------
# int32 test vectors (4 bytes, seed=0)
# ---------------------------------------------------------------------------


def test_rapidhash_int32() raises:
    """Validate vectorized rapidhash[Int32Type] against C reference."""
    var b = Int32Builder(capacity=7)
    b.append(Scalar[int32.native](0))
    b.append(Scalar[int32.native](1))
    b.append(Scalar[int32.native](-1))
    b.append(Scalar[int32.native](42))
    b.append(Scalar[int32.native](1000000))
    b.append(Scalar[int32.native](2147483647))
    b.append(Scalar[int32.native](-2147483648))
    var arr = b.finish()
    var h = RapidHashKernel.apply(arr)

    _assert_hash_eq(h, 0, 0xBA8945AAEC02CEE2)
    _assert_hash_eq(h, 1, 0x2B281F986C8B17D6)
    _assert_hash_eq(h, 2, 0x60486C15260D35CD)
    _assert_hash_eq(h, 3, 0x5EBDF38ED638F7A4)
    _assert_hash_eq(h, 4, 0xA0069A3A9E41DC88)
    _assert_hash_eq(h, 5, 0x90D336B991118909)
    _assert_hash_eq(h, 6, 0x13890D5C06B3EEB0)


# ---------------------------------------------------------------------------
# int64 test vectors (8 bytes, seed=0)
# ---------------------------------------------------------------------------


def test_rapidhash_int64() raises:
    """Validate vectorized rapidhash[Int64Type] against C reference."""
    var b = Int64Builder(capacity=7)
    b.append(Scalar[int64.native](0))
    b.append(Scalar[int64.native](1))
    b.append(Scalar[int64.native](-1))
    b.append(Scalar[int64.native](42))
    b.append(Scalar[int64.native](1000000))
    b.append(Scalar[int64.native](9223372036854775807))
    b.append(Scalar[int64.native](-9223372036854775808))
    var arr = b.finish()
    var h = RapidHashKernel.apply(arr)

    _assert_hash_eq(h, 0, 0x9EFC171AEBCEA1F3)
    _assert_hash_eq(h, 1, 0xBEC178309F44AFBC)
    _assert_hash_eq(h, 2, 0x8A185FBD8550916D)
    _assert_hash_eq(h, 3, 0x46007CEF671E1CDF)
    _assert_hash_eq(h, 4, 0xBAE276403A287B35)
    _assert_hash_eq(h, 5, 0x5A45FC0CE02F9D32)
    _assert_hash_eq(h, 6, 0x772E91684DA5B145)


# ---------------------------------------------------------------------------
# Scalar matches vectorized
# ---------------------------------------------------------------------------


def test_scalar_matches_vectorized() raises:
    """Scalar _rapidhash_fixed produces same output as vectorized rapidhash."""
    var b = Int64Builder(capacity=5)
    b.append(Scalar[int64.native](0))
    b.append(Scalar[int64.native](42))
    b.append(Scalar[int64.native](-1))
    b.append(Scalar[int64.native](999999))
    b.append(Scalar[int64.native](123456789))
    var arr = b.finish()
    var h = RapidHashKernel.apply(arr)

    for i in range(5):
        var scalar_h = RapidHash64.hash_lanes[8, 1](
            SIMD[DType.uint64, 1](UInt64(arr.unsafe_get(i)))
        )[0]
        var vec_h = UInt64(h.unsafe_get(i))
        if scalar_h != vec_h:
            raise Error("scalar/vectorized mismatch at index " + String(i))


# ---------------------------------------------------------------------------
# Null sentinel
# ---------------------------------------------------------------------------


def test_null_produces_sentinel() raises:
    """Null elements hash to NULL_HASH_SENTINEL."""
    var b = Int64Builder(capacity=3)
    b.append(Scalar[int64.native](42))
    b.append_null()
    b.append(Scalar[int64.native](7))
    var arr = b.finish()
    var h = RapidHashKernel.apply(arr)

    assert_true(UInt64(h.unsafe_get(0)) != NULL_HASH_SENTINEL)
    assert_equal(UInt64(h.unsafe_get(1)), NULL_HASH_SENTINEL)
    assert_true(UInt64(h.unsafe_get(2)) != NULL_HASH_SENTINEL)

# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""The laws behind the kernel property traits, one generic checker each.

A property trait in `kernels/core.mojo` is a claim: the compiler proves that a
kernel has the members, and the checker here proves that the members keep the
law. Every kernel that claims a property is instantiated against its law.
"""

from std.math import isnan
from std.memory import bitcast
from std.sys import size_of
from std.testing import assert_equal, assert_false, assert_raises, assert_true
from std.utils.numerics import inf, nan, neg_inf

from ...arrays import Int32Array, PrimitiveArray
from ...builders import Date32Builder, PrimitiveBuilder, array
from ...dtypes import (
    Date32Type,
    Decimal128Type,
    Decimal256Type,
    DecimalType,
    DynType,
    Float16Type,
    Float64Type,
    Int8Type,
    Int32Type,
    Int64Type,
    NumericType,
    PrimitiveType,
    UInt64Type,
    date32,
    decimal128,
    decimal256,
    float64,
    int32,
    int64,
)
from ...kernels.aggregate import (
    BitAndFold,
    BitOrFold,
    BitXorFold,
    CountFold,
    DecimalSumFold,
    FoldKernel,
    FoldLane,
    MaxFold,
    MeanFold,
    MinFold,
    ProductFold,
    SumFold,
)
from ...kernels.bitwise import BitAndKernel, BitOrKernel, BitXorKernel
from ...kernels.boolean import AndKernel, IsNanKernel
from ...kernels.cast import CastKernel
from ...kernels.core import (
    Associative,
    BinaryKernel,
    Commutative,
    Identity,
    Partial,
    Prunable,
    UnaryKernel,
)
from ...kernels.decimal import DecimalDivKernel
from ...kernels.numeric import (
    AddKernel,
    EqKernel,
    FloordivKernel,
    GeKernel,
    GtKernel,
    LeKernel,
    LtKernel,
    MaxKernel,
    MinKernel,
    ModKernel,
    MulKernel,
    NegKernel,
    NumericCompareKernel,
    SubKernel,
)


comptime _REPEATS = 24
"""How many times a column repeats its pattern: enough rows to fill the widest
SIMD register more than once and still leave a tail."""


def _same_bits[T: DType](a: Scalar[T], b: Scalar[T]) -> Bool:
    """Whether `a` and `b` are one value bit for bit, so a NaN equals itself
    and `-0.0` differs from `0.0`."""
    comptime n = size_of[Scalar[T]]()
    comptime if n == 2:
        # One 16-bit lane: `bitcast` refuses to split a float16 into bytes.
        return bitcast[DType.uint16, 1](a) == bitcast[DType.uint16, 1](b)
    else:
        return bitcast[DType.uint8, n](a) == bitcast[DType.uint8, n](b)


def _column[
    A: PrimitiveType
](
    dtype: A, row: List[Scalar[A.native]], nullable: Bool = True
) raises -> PrimitiveArray[A]:
    """`row` repeated. When `nullable`, each repetition is followed by two
    nulls, one over a zero and one over a defined value, so the law tells a
    null from an undefined row; otherwise the column has no validity bitmap.
    Read at the wrong offset, the validity disagrees with the values."""
    var b = PrimitiveBuilder[A](dtype)
    for _ in range(_REPEATS):
        for v in row:
            b.append(v)
        if nullable:
            b.append_null()
            b.append_null()
            b.unsafe_set(b.length() - 1, Scalar[A.native](1))
    return b.finish()


def _least_positive[T: DType]() -> Scalar[T]:
    """The least positive value of the float type `T`: the subnormal whose
    bit pattern is 1."""
    comptime n = size_of[Scalar[T]]()
    comptime U = DType.uint16 if n == 2 else (
        DType.uint32 if n == 4 else DType.uint64
    )
    return bitcast[T, 1](Scalar[U](1))


def _edge_values[T: DType]() -> List[Scalar[T]]:
    """Zero, one, minus one where the type has it and both ends of the range
    -- and for floats `-0.0`, NaN, both infinities and the least positive
    value."""
    comptime S = Scalar[T]
    var row: List[S] = [S(0), S(1), S.MIN_FINITE, S.MAX_FINITE]
    comptime if T.is_signed():
        row.append(S(-1))
    comptime if T.is_floating_point():
        row.append(S(-0.0))
        row.append(nan[T]())
        row.append(inf[T]())
        row.append(neg_inf[T]())
        row.append(_least_positive[T]())
    return row^


def _edges[A: NumericType](nullable: Bool = True) raises -> PrimitiveArray[A]:
    """`_edge_values` as a column of `A`."""
    return _column(A(), _edge_values[A.native](), nullable)


def _decimal_edges() raises -> PrimitiveArray[Decimal128Type]:
    """`decimal128(38, 2)` by its unscaled integers, the lanes `defined`
    reads: zero, the smallest non-zero values (`±0.01`), one (`100`), minus
    one and both ends of the range."""
    comptime I = Scalar[DType.int128]
    comptime top = I(99999999999999999999999999999999999999)
    var row: List[I] = [0, 1, -1, 100, -100, top, -top]
    return _column(decimal128(38, 2), row)


def _decimal256_edges(scale: Int) raises -> PrimitiveArray[Decimal256Type]:
    """`decimal256(76, scale)` by its unscaled integers, as `_decimal_edges`:
    zero, the smallest non-zero values, `±100` and both ends of the range."""
    comptime I = Scalar[DType.int256]
    comptime top = I(
        9999999999999999999999999999999999999999999999999999999999999999999999999999
    )
    var row: List[I] = [0, 1, -1, 100, -100, top, -top]
    return _column(decimal256(76, scale), row)


def check_partial[
    K: Partial, A: PrimitiveType
](var values: PrimitiveArray[A], offset: Int) raises:
    """The `Partial` law: after `null_undefined`, a row is valid exactly when
    it was valid and `defined` holds; valid values are unchanged; and at least
    one row is nulled by `defined` alone and one kept, so a rule that rejects
    everything fails as surely as one that accepts everything."""
    var sliced = values.slice(offset, len(values) - offset)
    var out = K.null_undefined(sliced)
    assert_equal(len(out), len(sliced))
    var nulls = 0
    var nulled = 0
    var kept = 0
    for i in range(len(sliced)):
        var valid = (
            sliced.is_valid(i) and K.defined[A.native, 1](sliced[i].value())[0]
        )
        assert_equal(out.is_valid(i), valid, String(t"row {i}: validity"))
        if valid:
            kept += 1
            assert_true(
                _same_bits(out[i].value(), sliced[i].value()),
                String(t"row {i}: value changed"),
            )
        else:
            nulls += 1
            if sliced.is_valid(i):
                nulled += 1
    assert_equal(out.null_count(), nulls)
    assert_true(nulled > 0, "no row was undefined: the law is vacuous here")
    assert_true(kept > 0, "no row was defined: the rule rejects everything")


def test_partial_law_floordiv_int8() raises:
    check_partial[FloordivKernel, Int8Type](_edges[Int8Type](), 3)


def test_partial_law_floordiv_int32() raises:
    check_partial[FloordivKernel, Int32Type](_edges[Int32Type](), 3)


def test_partial_law_floordiv_int64() raises:
    check_partial[FloordivKernel, Int64Type](_edges[Int64Type](), 3)


def test_partial_law_floordiv_uint64() raises:
    check_partial[FloordivKernel, UInt64Type](_edges[UInt64Type](), 3)


def test_partial_law_floordiv_float64() raises:
    check_partial[FloordivKernel, Float64Type](_edges[Float64Type](), 3)


def test_partial_law_floordiv_float16() raises:
    check_partial[FloordivKernel, Float16Type](_edges[Float16Type](), 3)


def test_partial_law_floordiv_int64_without_nulls() raises:
    check_partial[FloordivKernel, Int64Type](_edges[Int64Type](False), 3)


def test_partial_law_floordiv_float64_without_nulls() raises:
    check_partial[FloordivKernel, Float64Type](_edges[Float64Type](False), 3)


def test_partial_law_mod_int8() raises:
    check_partial[ModKernel, Int8Type](_edges[Int8Type](), 3)


def test_partial_law_mod_int32() raises:
    check_partial[ModKernel, Int32Type](_edges[Int32Type](), 3)


def test_partial_law_mod_int64() raises:
    check_partial[ModKernel, Int64Type](_edges[Int64Type](), 3)


def test_partial_law_mod_uint64() raises:
    check_partial[ModKernel, UInt64Type](_edges[UInt64Type](), 3)


def test_partial_law_mod_float64() raises:
    check_partial[ModKernel, Float64Type](_edges[Float64Type](), 3)


def test_partial_law_decimal_div_decimal128() raises:
    check_partial[DecimalDivKernel, Decimal128Type](_decimal_edges(), 3)


def test_partial_law_decimal_div_decimal256() raises:
    check_partial[DecimalDivKernel, Decimal256Type](_decimal256_edges(2), 3)


def test_partial_null_undefined_nothing_to_null_is_unchanged() raises:
    """With no valid row to null, `null_undefined` answers its argument,
    offset and all -- here with a null over a zero, which is already null."""
    var values = array([4, None, 2, 5, -3, 7], int32).slice(1)
    assert_true(FloordivKernel.null_undefined(values) == values)


def test_partial_null_undefined_without_validity_all_defined() raises:
    """No null and no zero: `null_undefined` answers `b` itself, still with
    no validity bitmap -- here sliced, so at an offset."""
    var row: List[Scalar[DType.int64]] = [4, -3, 7, Int64.MAX]
    var values = _column(int64, row, nullable=False).slice(1)
    assert_true(not values.bitmap)
    var out = FloordivKernel.null_undefined(values)
    assert_true(out == values)
    assert_true(not out.bitmap)


def test_partial_null_undefined_empty() raises:
    """A zero-length divisor answers a zero-length array, typed and erased."""
    var empty = Int32Array.empty(int32)
    assert_equal(len(FloordivKernel.null_undefined(empty)), 0)
    assert_equal(len(FloordivKernel.null_undefined(empty.copy().to_dyn())), 0)


def test_partial_null_undefined_erased() raises:
    """The erased form answers what the typed one does, through its numeric
    arm and its decimal arm, and refuses any other dtype."""
    var ints = _edges[Int32Type]().slice(3)
    var erased_ints = FloordivKernel.null_undefined(ints.copy().to_dyn())
    assert_true(erased_ints.as_int32() == FloordivKernel.null_undefined(ints))
    var decimals = _decimal_edges().slice(3)
    var erased_decimals = DecimalDivKernel.null_undefined(
        decimals.copy().to_dyn()
    )
    assert_true(
        erased_decimals.as_decimal128()
        == DecimalDivKernel.null_undefined(decimals)
    )
    with assert_raises(contains="not numeric"):
        _ = ModKernel.null_undefined(array(["a"]).to_dyn())


def test_partial_defined_is_a_non_zero_divisor() raises:
    """The rule the law is checked against: every zero is undefined, `-0.0`
    included, and every other divisor -- NaN, an infinity, the smallest
    non-zero decimal -- is defined."""
    var floats = SIMD[DType.float64, 4](
        0.0, -0.0, nan[DType.float64](), inf[DType.float64]()
    )
    var expected = SIMD[DType.bool, 4](False, False, True, True)
    assert_true(FloordivKernel.defined[DType.float64, 4](floats) == expected)
    assert_true(ModKernel.defined[DType.float64, 4](floats) == expected)
    var unscaled = SIMD[DType.int128, 2](0, 1)
    assert_true(
        DecimalDivKernel.defined[DType.int128, 2](unscaled)
        == SIMD[DType.bool, 2](False, True)
    )


def check_promotion[K: Partial, A: DType, *Ps: DType]() raises:
    """`Partial`'s promotion requirement, for the widenings from `A` to each
    of `Ps`: `defined` answers the same for every edge value of `A` before
    and after the cast."""
    comptime for P in Ps:
        for v in _edge_values[A]():
            assert_equal(
                K.defined[A, 1](v)[0],
                K.defined[P, 1](v.cast[P]())[0],
                String(t"{v}: `defined` changes from {A} to {P}"),
            )


def _check_numeric_promotions[K: Partial]() raises:
    """Every widening a divisor of `//` or `%` meets before `core`: `promote`
    in the comptime lane, `_common_type` in the runtime one. An integer goes
    to any integer at least as wide, of either sign, or to any float; a float
    to a wider float."""
    comptime f16 = DType.float16
    comptime f32 = DType.float32
    comptime f64 = DType.float64
    check_promotion[
        K,
        DType.int8,
        DType.uint8,
        DType.int16,
        DType.uint16,
        DType.int32,
        DType.uint32,
        DType.int64,
        DType.uint64,
        f16,
        f32,
        f64,
    ]()
    check_promotion[
        K,
        DType.uint8,
        DType.int8,
        DType.int16,
        DType.uint16,
        DType.int32,
        DType.uint32,
        DType.int64,
        DType.uint64,
        f16,
        f32,
        f64,
    ]()
    check_promotion[
        K,
        DType.int16,
        DType.uint16,
        DType.int32,
        DType.uint32,
        DType.int64,
        DType.uint64,
        f16,
        f32,
        f64,
    ]()
    check_promotion[
        K,
        DType.uint16,
        DType.int16,
        DType.int32,
        DType.uint32,
        DType.int64,
        DType.uint64,
        f16,
        f32,
        f64,
    ]()
    check_promotion[
        K, DType.int32, DType.uint32, DType.int64, DType.uint64, f16, f32, f64
    ]()
    check_promotion[
        K, DType.uint32, DType.int32, DType.int64, DType.uint64, f16, f32, f64
    ]()
    check_promotion[K, DType.int64, DType.uint64, f16, f32, f64]()
    check_promotion[K, DType.uint64, DType.int64, f16, f32, f64]()
    check_promotion[K, f16, f32, f64]()
    check_promotion[K, f32, f64]()


def test_partial_promotion_floordiv() raises:
    _check_numeric_promotions[FloordivKernel]()


def test_partial_promotion_mod() raises:
    _check_numeric_promotions[ModKernel]()


def test_partial_promotion_decimal_div() raises:
    """`DecimalDivKernel`'s divisor widens to the result's unscaled width, 128
    or 256 bits: a narrower decimal's unscaled integer in both lanes, and in
    the runtime lane an integer too, as a decimal of scale 0 whose unscaled
    value is the integer itself."""
    comptime i128 = DType.int128
    comptime i256 = DType.int256
    check_promotion[DecimalDivKernel, DType.int8, i128, i256]()
    check_promotion[DecimalDivKernel, DType.uint8, i128, i256]()
    check_promotion[DecimalDivKernel, DType.int16, i128, i256]()
    check_promotion[DecimalDivKernel, DType.uint16, i128, i256]()
    check_promotion[DecimalDivKernel, DType.int32, i128, i256]()
    check_promotion[DecimalDivKernel, DType.uint32, i128, i256]()
    check_promotion[DecimalDivKernel, DType.int64, i128, i256]()
    check_promotion[DecimalDivKernel, DType.uint64, i128, i256]()
    check_promotion[DecimalDivKernel, i128, i256]()


def check_promotion_to_float64[
    K: Partial, D: DecimalType
](decimals: PrimitiveArray[D]) raises:
    """`Partial`'s promotion requirement for the runtime lane's one widening
    that is not a cast of the lane: a decimal divisor against a float
    dividend, which becomes `float64` through `CastKernel` before `//` or
    `%`."""
    var promoted = CastKernel.dispatch(
        decimals.copy().to_dyn(), DynType(float64)
    )
    ref floats = promoted.as_float64()
    for i in range(len(decimals)):
        if decimals.is_valid(i):
            var v = decimals[i].value()
            assert_equal(
                K.defined[D.native, 1](v)[0],
                K.defined[DType.float64, 1](floats[i].value())[0],
                String(t"{v}: `defined` changes from {D.native} to float64"),
            )


def test_partial_promotion_decimal_to_float64() raises:
    """The least non-zero decimal at the widest scale stays non-zero."""
    check_promotion_to_float64[FloordivKernel](_decimal_edges())
    check_promotion_to_float64[FloordivKernel](_decimal256_edges(76))
    check_promotion_to_float64[ModKernel](_decimal_edges())
    check_promotion_to_float64[ModKernel](_decimal256_edges(76))


def _ordered[T: DType](a: Scalar[T], b: Scalar[T]) -> Bool:
    """`a <= b` under the total order, where a NaN is the greatest value."""
    return LeKernel[nan_safe=True].core[T, 1](a, b)[0]


def _some_pair_holds[
    K: NumericCompareKernel, T: DType
](
    domain: List[Scalar[T]],
    lmin: Scalar[T],
    lmax: Scalar[T],
    rmin: Scalar[T],
    rmax: Scalar[T],
) -> Bool:
    """Whether some `a` in `[lmin, lmax]` and `b` in `[rmin, rmax]`, both
    drawn from `domain`, has `K.core(a, b)`."""
    for a in domain:
        if _ordered(lmin, a) and _ordered(a, lmax):
            for b in domain:
                if _ordered(rmin, b) and _ordered(b, rmax):
                    if K.bool_core[T, 1](a, b):
                        return True
    return False


def check_prunable[
    K: NumericCompareKernel & Prunable, T: DType
](domain: List[Scalar[T]]) raises:
    """The `Prunable` law, by brute force over every pair of intervals
    [lmin, lmax], [rmin, rmax] drawn from `domain`, each ordered by
    `LeKernel[nan_safe=True]`:
      if some a in [lmin, lmax] and b in [rmin, rmax] has K.core(a, b),
      then K.possible(lmin, lmax, rmin, rmax) is true  (soundness);
    and some interval pair gives false (the rule prunes something).

    Drawing the witnesses from `domain` too loses nothing: whenever some pair
    satisfies the comparison, one does whose members are among the four
    bounds, or `inf` where an IEEE comparison meets a NaN upper bound, and
    `domain` holds both."""
    var lows = List[Scalar[T]]()
    var highs = List[Scalar[T]]()
    for lo in domain:
        for hi in domain:
            if _ordered(lo, hi):
                lows.append(lo)
                highs.append(hi)
    var pruned = 0
    for i in range(len(lows)):
        for j in range(len(lows)):
            if not K.possible[T, 1](lows[i], highs[i], lows[j], highs[j])[0]:
                pruned += 1
                assert_false(
                    _some_pair_holds[K](
                        domain, lows[i], highs[i], lows[j], highs[j]
                    ),
                    String(
                        t"[{lows[i]}, {highs[i]}] against [{lows[j]},"
                        t" {highs[j]}] is pruned, but a pair inside matches"
                    ),
                )
    assert_true(pruned > 0, "no interval pair was pruned: the rule is vacuous")


def _int8_domain() -> List[Scalar[DType.int8]]:
    """Every value from -3 to 3: each interval holds all its integers."""
    return [Scalar[DType.int8](v) for v in range(-3, 4)]


def _float64_domain() -> List[Scalar[DType.float64]]:
    """Both infinities, both zeros, one either side of them, and NaN, the
    greatest value under the total order."""
    return [
        neg_inf[DType.float64](),
        -1.0,
        -0.0,
        0.0,
        1.0,
        inf[DType.float64](),
        nan[DType.float64](),
    ]


def test_prunable_law_eq_int8() raises:
    check_prunable[EqKernel[nan_safe=False], DType.int8](_int8_domain())
    check_prunable[EqKernel[nan_safe=True], DType.int8](_int8_domain())


def test_prunable_law_lt_int8() raises:
    check_prunable[LtKernel[nan_safe=False], DType.int8](_int8_domain())
    check_prunable[LtKernel[nan_safe=True], DType.int8](_int8_domain())


def test_prunable_law_le_int8() raises:
    check_prunable[LeKernel[nan_safe=False], DType.int8](_int8_domain())
    check_prunable[LeKernel[nan_safe=True], DType.int8](_int8_domain())


def test_prunable_law_gt_int8() raises:
    check_prunable[GtKernel[nan_safe=False], DType.int8](_int8_domain())
    check_prunable[GtKernel[nan_safe=True], DType.int8](_int8_domain())


def test_prunable_law_ge_int8() raises:
    check_prunable[GeKernel[nan_safe=False], DType.int8](_int8_domain())
    check_prunable[GeKernel[nan_safe=True], DType.int8](_int8_domain())


def test_prunable_law_eq_float64_total() raises:
    check_prunable[EqKernel[nan_safe=True], DType.float64](_float64_domain())


def test_prunable_law_lt_float64_total() raises:
    check_prunable[LtKernel[nan_safe=True], DType.float64](_float64_domain())


def test_prunable_law_le_float64_total() raises:
    check_prunable[LeKernel[nan_safe=True], DType.float64](_float64_domain())


def test_prunable_law_gt_float64_total() raises:
    check_prunable[GtKernel[nan_safe=True], DType.float64](_float64_domain())


def test_prunable_law_ge_float64_total() raises:
    check_prunable[GeKernel[nan_safe=True], DType.float64](_float64_domain())


def test_prunable_law_eq_float64_ieee() raises:
    check_prunable[EqKernel[nan_safe=False], DType.float64](_float64_domain())


def test_prunable_law_lt_float64_ieee() raises:
    check_prunable[LtKernel[nan_safe=False], DType.float64](_float64_domain())


def test_prunable_law_le_float64_ieee() raises:
    check_prunable[LeKernel[nan_safe=False], DType.float64](_float64_domain())


def test_prunable_law_gt_float64_ieee() raises:
    check_prunable[GtKernel[nan_safe=False], DType.float64](_float64_domain())


def test_prunable_law_ge_float64_ieee() raises:
    check_prunable[GeKernel[nan_safe=False], DType.float64](_float64_domain())


def test_possible_segments_keeps_a_segment_without_statistics() raises:
    """A null bound is an absent statistic: `possible_segments` answers null
    at that segment, whichever of the four bounds is missing, and the Kleene
    AND that combines a scan's predicates reads it as keep -- while the
    segment the bounds rule out is pruned. Sliced, so every bound is read at
    an offset."""
    var lmin = array([9, 0, None, 0, 0, 0, 0], int64).slice(1)
    var lmax = array([9, 2, 2, None, 2, 2, 2], int64).slice(1)
    var rmin = array([9, 1, 1, 1, None, 1, 5], int64).slice(1)
    var rmax = array([9, 1, 1, 1, 1, None, 6], int64).slice(1)
    var possible = EqKernel.possible_segments(lmin, lmax, rmin, rmax)
    assert_true(possible == array([True, None, None, None, None, False]))
    var conjunction = AndKernel.apply(
        array([True, True, True, True, True, True]), possible
    )
    var keep = List[Bool]()
    for i in range(len(conjunction)):
        keep.append(conjunction.is_null(i) or conjunction[i].value())
    assert_true(keep == [True, True, True, True, True, False])


def _date32_bounds(values: List[Int]) raises -> PrimitiveArray[Date32Type]:
    var b = Date32Builder(date32(), len(values))
    for v in values:
        b.append(Scalar[Date32Type.native](v))
    return b.finish()


def test_possible_segments_erased() raises:
    """The erased form answers what the typed one does -- over `date32`
    bounds too, which is what lets a temporal predicate prune in the
    runtime lane -- and refuses bounds of two dtypes, or of no primitive
    one."""
    var lmin = _date32_bounds([19000, 19000])
    var lmax = _date32_bounds([19100, 19100])
    var rmin = _date32_bounds([18000, 19200])
    var rmax = _date32_bounds([18500, 19300])
    var erased = GtKernel.possible_segments(
        lmin.copy().to_dyn(),
        lmax.copy().to_dyn(),
        rmin.copy().to_dyn(),
        rmax.copy().to_dyn(),
    )
    assert_true(erased == GtKernel.possible_segments(lmin, lmax, rmin, rmax))
    assert_true(erased == array([True, False]))
    var ints = array([1, 2], int64).to_dyn()
    with assert_raises(contains="share one dtype"):
        _ = GtKernel.possible_segments(
            ints, ints, ints, array([1, 2], int32).to_dyn()
        )
    var strings = array(["a", "b"]).to_dyn()
    with assert_raises(contains="not primitive"):
        _ = GtKernel.possible_segments(strings, strings, strings, strings)


def _same_value[T: DType](a: Scalar[T], b: Scalar[T]) -> Bool:
    """Equal as values: `==`, or both NaN. `-0.0` equals `0.0`."""
    comptime if T.is_floating_point():
        return a == b or (isnan(a) and isnan(b))
    else:
        return a == b


def check_identity[
    K: BinaryKernel, T: DType
](e: Scalar[T], domain: List[Scalar[T]]) raises:
    """The `Identity` law: `core(e, x)` and `core(x, e)` are both `x`, for
    every `x` in `domain`."""
    for x in domain:
        var left = K.same_core[T, 1](e, x)
        assert_true(
            _same_value(left, x),
            String(t"identity: core({e}, {x}) is {left}, not {x}"),
        )
        var right = K.same_core[T, 1](x, e)
        assert_true(
            _same_value(right, x),
            String(t"identity: core({x}, {e}) is {right}, not {x}"),
        )


def check_associative[
    K: BinaryKernel, T: DType
](domain: List[Scalar[T]]) raises:
    """The `Associative` law: `core(core(a, b), c) == core(a, core(b, c))` for
    every triple drawn from `domain`."""
    for a in domain:
        for b in domain:
            for c in domain:
                var ab = K.same_core[T, 1](a, b)
                var bc = K.same_core[T, 1](b, c)
                var left = K.same_core[T, 1](ab, c)
                var right = K.same_core[T, 1](a, bc)
                assert_true(
                    _same_value(left, right),
                    String(
                        t"associative: ({a}, {b}), {c} gives {left} but {a},"
                        t" ({b}, {c}) gives {right}"
                    ),
                )


def check_commutative[
    K: BinaryKernel, T: DType
](domain: List[Scalar[T]]) raises:
    """The `Commutative` law: `core(a, b) == core(b, a)` for every pair drawn
    from `domain`."""
    for a in domain:
        for b in domain:
            var left = K.same_core[T, 1](a, b)
            var right = K.same_core[T, 1](b, a)
            assert_true(
                _same_value(left, right),
                String(
                    t"commutative: core({a}, {b}) is {left} but core({b}, {a})"
                    t" is {right}"
                ),
            )


def _claims_identity[
    K: BinaryKernel & Identity, T: DType
](domain: List[Scalar[T]]) raises:
    """Check the law against the identity the kernel itself declares."""
    check_identity[K, T](K.identity[T](), domain)


def _claims_associative[
    K: BinaryKernel & Associative, T: DType
](domain: List[Scalar[T]]) raises:
    check_associative[K, T](domain)


def _claims_commutative[
    K: BinaryKernel & Commutative, T: DType
](domain: List[Scalar[T]]) raises:
    check_commutative[K, T](domain)


def _int8_every_value() -> List[Scalar[DType.int8]]:
    """Every `int8`."""
    var domain = List[Scalar[DType.int8]]()
    for v in range(-128, 128):
        domain.append(Scalar[DType.int8](v))
    return domain^


def _int8_triples() -> List[Scalar[DType.int8]]:
    """Both ends of the range, either side of zero and a few between: the
    full cube of `int8` is too slow."""
    return [-128, -127, -2, -1, 0, 1, 2, 3, 64, 126, 127]


def _uint8_domain() -> List[Scalar[DType.uint8]]:
    return [0, 1, 2, 127, 128, 254, 255]


def _int64_domain() -> List[Scalar[DType.int64]]:
    return [Int64.MIN, Int64.MIN + 1, -2, -1, 0, 1, 2, Int64.MAX - 1, Int64.MAX]


def _float64_exact_domain() -> List[Scalar[DType.float64]]:
    """Both infinities, both zeros, NaN and values every sum and product of
    which is exact, so `+` and `*` are checked where they are exact."""
    return [
        neg_inf[DType.float64](),
        -2.0,
        -1.0,
        -0.5,
        -0.0,
        0.0,
        0.5,
        1.0,
        2.0,
        inf[DType.float64](),
        nan[DType.float64](),
    ]


def test_identity_law_add_int8() raises:
    _claims_identity[AddKernel](_int8_every_value())


def test_identity_law_add_uint8() raises:
    _claims_identity[AddKernel](_uint8_domain())


def test_identity_law_add_int64() raises:
    _claims_identity[AddKernel](_int64_domain())


def test_identity_law_add_float64() raises:
    _claims_identity[AddKernel](_float64_exact_domain())


def test_identity_law_mul_int8() raises:
    _claims_identity[MulKernel](_int8_every_value())


def test_identity_law_mul_uint8() raises:
    _claims_identity[MulKernel](_uint8_domain())


def test_identity_law_mul_int64() raises:
    _claims_identity[MulKernel](_int64_domain())


def test_identity_law_mul_float64() raises:
    _claims_identity[MulKernel](_float64_exact_domain())


def test_identity_law_min_int8() raises:
    _claims_identity[MinKernel](_int8_every_value())


def test_identity_law_min_uint8() raises:
    _claims_identity[MinKernel](_uint8_domain())


def test_identity_law_min_int64() raises:
    _claims_identity[MinKernel](_int64_domain())


def test_identity_law_min_float64() raises:
    _claims_identity[MinKernel](_float64_exact_domain())


def test_identity_law_max_int8() raises:
    _claims_identity[MaxKernel](_int8_every_value())


def test_identity_law_max_uint8() raises:
    _claims_identity[MaxKernel](_uint8_domain())


def test_identity_law_max_int64() raises:
    _claims_identity[MaxKernel](_int64_domain())


def test_identity_law_max_float64() raises:
    _claims_identity[MaxKernel](_float64_exact_domain())


def test_associative_law_add_int8() raises:
    _claims_associative[AddKernel](_int8_triples())


def test_associative_law_add_uint8() raises:
    _claims_associative[AddKernel](_uint8_domain())


def test_associative_law_add_int64() raises:
    _claims_associative[AddKernel](_int64_domain())


def test_associative_law_add_float64() raises:
    _claims_associative[AddKernel](_float64_exact_domain())


def test_associative_law_mul_int8() raises:
    _claims_associative[MulKernel](_int8_triples())


def test_associative_law_mul_uint8() raises:
    _claims_associative[MulKernel](_uint8_domain())


def test_associative_law_mul_int64() raises:
    _claims_associative[MulKernel](_int64_domain())


def test_associative_law_mul_float64() raises:
    _claims_associative[MulKernel](_float64_exact_domain())


def test_associative_law_min_int8() raises:
    _claims_associative[MinKernel](_int8_triples())


def test_associative_law_min_uint8() raises:
    _claims_associative[MinKernel](_uint8_domain())


def test_associative_law_min_int64() raises:
    _claims_associative[MinKernel](_int64_domain())


def test_associative_law_min_float64() raises:
    _claims_associative[MinKernel](_float64_exact_domain())


def test_associative_law_max_int8() raises:
    _claims_associative[MaxKernel](_int8_triples())


def test_associative_law_max_uint8() raises:
    _claims_associative[MaxKernel](_uint8_domain())


def test_associative_law_max_int64() raises:
    _claims_associative[MaxKernel](_int64_domain())


def test_associative_law_max_float64() raises:
    _claims_associative[MaxKernel](_float64_exact_domain())


def test_commutative_law_add_int8() raises:
    _claims_commutative[AddKernel](_int8_every_value())


def test_commutative_law_add_uint8() raises:
    _claims_commutative[AddKernel](_uint8_domain())


def test_commutative_law_add_int64() raises:
    _claims_commutative[AddKernel](_int64_domain())


def test_commutative_law_add_float64() raises:
    _claims_commutative[AddKernel](_float64_exact_domain())


def test_commutative_law_mul_int8() raises:
    _claims_commutative[MulKernel](_int8_every_value())


def test_commutative_law_mul_uint8() raises:
    _claims_commutative[MulKernel](_uint8_domain())


def test_commutative_law_mul_int64() raises:
    _claims_commutative[MulKernel](_int64_domain())


def test_commutative_law_mul_float64() raises:
    _claims_commutative[MulKernel](_float64_exact_domain())


def test_commutative_law_min_int8() raises:
    _claims_commutative[MinKernel](_int8_every_value())


def test_commutative_law_min_uint8() raises:
    _claims_commutative[MinKernel](_uint8_domain())


def test_commutative_law_min_int64() raises:
    _claims_commutative[MinKernel](_int64_domain())


def test_commutative_law_min_float64() raises:
    _claims_commutative[MinKernel](_float64_exact_domain())


def test_commutative_law_max_int8() raises:
    _claims_commutative[MaxKernel](_int8_every_value())


def test_commutative_law_max_uint8() raises:
    _claims_commutative[MaxKernel](_uint8_domain())


def test_commutative_law_max_int64() raises:
    _claims_commutative[MaxKernel](_int64_domain())


def test_commutative_law_max_float64() raises:
    _claims_commutative[MaxKernel](_float64_exact_domain())


def test_identity_law_bit_and_int8() raises:
    _claims_identity[BitAndKernel](_int8_every_value())


def test_identity_law_bit_and_uint8() raises:
    _claims_identity[BitAndKernel](_uint8_domain())


def test_identity_law_bit_and_int64() raises:
    _claims_identity[BitAndKernel](_int64_domain())


def test_identity_law_bit_or_int8() raises:
    _claims_identity[BitOrKernel](_int8_every_value())


def test_identity_law_bit_or_uint8() raises:
    _claims_identity[BitOrKernel](_uint8_domain())


def test_identity_law_bit_or_int64() raises:
    _claims_identity[BitOrKernel](_int64_domain())


def test_identity_law_bit_xor_int8() raises:
    _claims_identity[BitXorKernel](_int8_every_value())


def test_identity_law_bit_xor_uint8() raises:
    _claims_identity[BitXorKernel](_uint8_domain())


def test_identity_law_bit_xor_int64() raises:
    _claims_identity[BitXorKernel](_int64_domain())


def test_associative_law_bit_and_int8() raises:
    _claims_associative[BitAndKernel](_int8_triples())


def test_associative_law_bit_and_uint8() raises:
    _claims_associative[BitAndKernel](_uint8_domain())


def test_associative_law_bit_and_int64() raises:
    _claims_associative[BitAndKernel](_int64_domain())


def test_associative_law_bit_or_int8() raises:
    _claims_associative[BitOrKernel](_int8_triples())


def test_associative_law_bit_or_uint8() raises:
    _claims_associative[BitOrKernel](_uint8_domain())


def test_associative_law_bit_or_int64() raises:
    _claims_associative[BitOrKernel](_int64_domain())


def test_associative_law_bit_xor_int8() raises:
    _claims_associative[BitXorKernel](_int8_triples())


def test_associative_law_bit_xor_uint8() raises:
    _claims_associative[BitXorKernel](_uint8_domain())


def test_associative_law_bit_xor_int64() raises:
    _claims_associative[BitXorKernel](_int64_domain())


def test_commutative_law_bit_and_int8() raises:
    _claims_commutative[BitAndKernel](_int8_every_value())


def test_commutative_law_bit_and_uint8() raises:
    _claims_commutative[BitAndKernel](_uint8_domain())


def test_commutative_law_bit_and_int64() raises:
    _claims_commutative[BitAndKernel](_int64_domain())


def test_commutative_law_bit_or_int8() raises:
    _claims_commutative[BitOrKernel](_int8_every_value())


def test_commutative_law_bit_or_uint8() raises:
    _claims_commutative[BitOrKernel](_uint8_domain())


def test_commutative_law_bit_or_int64() raises:
    _claims_commutative[BitOrKernel](_int64_domain())


def test_commutative_law_bit_xor_int8() raises:
    _claims_commutative[BitXorKernel](_int8_every_value())


def test_commutative_law_bit_xor_uint8() raises:
    _claims_commutative[BitXorKernel](_uint8_domain())


def test_commutative_law_bit_xor_int64() raises:
    _claims_commutative[BitXorKernel](_int64_domain())


def test_identity_law_rejects_a_non_identity() raises:
    """`0` is not subtraction's identity: `0 - x` is `-x`."""
    with assert_raises(contains="identity:"):
        check_identity[SubKernel, DType.int64](
            Scalar[DType.int64](0), _int64_domain()
        )


def test_associative_law_rejects_sub() raises:
    with assert_raises(contains="associative:"):
        check_associative[SubKernel, DType.int64](_int64_domain())


def test_commutative_law_rejects_sub() raises:
    with assert_raises(contains="commutative:"):
        check_commutative[SubKernel, DType.int64](_int64_domain())


def test_identity_law_rejects_inf_for_min() raises:
    """`+inf` looks like `min`'s identity but is not: `min(inf, NaN)` is
    `inf`, which is why the identity of a float `min` is NaN."""
    with assert_raises(contains="identity:"):
        check_identity[MinKernel, DType.float64](
            inf[DType.float64](), _float64_exact_domain()
        )


# ---------------------------------------------------------------------------
# fold_lanes: one register folded across its lanes is the left fold of `core`
# ---------------------------------------------------------------------------


def _registers[T: DType, W: Int](domain: List[Scalar[T]]) -> List[SIMD[T, W]]:
    """64 registers whose lanes are drawn from `domain`, deterministically."""
    var out = List[SIMD[T, W]]()
    var state = UInt64(0x9E3779B97F4A7C15)
    for _ in range(64):
        var v = SIMD[T, W]()
        for j in range(W):
            state = state * 6364136223846793005 + 1442695040888963407
            v[j] = domain[Int(state >> 33) % len(domain)]
        out.append(v)
    return out^


def check_fold_lanes[
    K: BinaryKernel, T: DType, W: Int
](vectors: List[SIMD[T, W]]) raises:
    """The `fold_lanes` law: `K.fold_lanes(v)` is the left fold of `K.core`
    over `v`'s lanes, as a value."""
    for v in vectors:
        var expected = v[0]
        for k in range(1, W):
            expected = K.same_core[T, 1](expected, v[k])
        var got = K.fold_lanes[T, W](v)
        assert_true(
            _same_value(got, expected),
            String(t"fold_lanes: {v} gives {got}, the left fold {expected}"),
        )


def _fold_lanes_at_every_width[
    K: BinaryKernel, T: DType
](domain: List[Scalar[T]]) raises:
    check_fold_lanes[K, T, 1](_registers[T, 1](domain))
    check_fold_lanes[K, T, 2](_registers[T, 2](domain))
    check_fold_lanes[K, T, 4](_registers[T, 4](domain))
    check_fold_lanes[K, T, 8](_registers[T, 8](domain))
    check_fold_lanes[K, T, 16](_registers[T, 16](domain))


def test_fold_lanes_law_add_int8() raises:
    _fold_lanes_at_every_width[AddKernel](_int8_every_value())


def test_fold_lanes_law_add_uint8() raises:
    _fold_lanes_at_every_width[AddKernel](_uint8_domain())


def test_fold_lanes_law_add_int64() raises:
    _fold_lanes_at_every_width[AddKernel](_int64_domain())


def test_fold_lanes_law_add_float64() raises:
    _fold_lanes_at_every_width[AddKernel](_float64_exact_domain())


def test_fold_lanes_law_mul_int8() raises:
    _fold_lanes_at_every_width[MulKernel](_int8_every_value())


def test_fold_lanes_law_mul_uint8() raises:
    _fold_lanes_at_every_width[MulKernel](_uint8_domain())


def test_fold_lanes_law_mul_int64() raises:
    _fold_lanes_at_every_width[MulKernel](_int64_domain())


def test_fold_lanes_law_mul_float64() raises:
    _fold_lanes_at_every_width[MulKernel](_float64_exact_domain())


def test_fold_lanes_law_min_int8() raises:
    _fold_lanes_at_every_width[MinKernel](_int8_every_value())


def test_fold_lanes_law_min_uint8() raises:
    _fold_lanes_at_every_width[MinKernel](_uint8_domain())


def test_fold_lanes_law_min_int64() raises:
    _fold_lanes_at_every_width[MinKernel](_int64_domain())


def test_fold_lanes_law_min_float64() raises:
    _fold_lanes_at_every_width[MinKernel](_float64_exact_domain())


def test_fold_lanes_law_max_int8() raises:
    _fold_lanes_at_every_width[MaxKernel](_int8_every_value())


def test_fold_lanes_law_max_uint8() raises:
    _fold_lanes_at_every_width[MaxKernel](_uint8_domain())


def test_fold_lanes_law_max_int64() raises:
    _fold_lanes_at_every_width[MaxKernel](_int64_domain())


def test_fold_lanes_law_max_float64() raises:
    _fold_lanes_at_every_width[MaxKernel](_float64_exact_domain())


def check_fold_lanes_of[
    F: FoldKernel, T: DType, W: Int
](vectors: List[SIMD[T, W]]) raises:
    """The same law for a fold: `F.fold_lanes(v)` is the left fold of
    `F.combine` over `v`'s lanes."""
    for v in vectors:
        var expected = v[0]
        for k in range(1, W):
            expected = F.combine[T, 1](expected, v[k])
        var got = F.fold_lanes[T, W](v)
        assert_true(
            _same_value(got, expected),
            String(
                t"{F.name} fold_lanes: {v} gives {got}, the left fold"
                t" {expected}"
            ),
        )


def _fold_lanes_of_at_every_width[
    F: FoldKernel, T: DType
](domain: List[Scalar[T]]) raises:
    check_fold_lanes_of[F, T, 1](_registers[T, 1](domain))
    check_fold_lanes_of[F, T, 2](_registers[T, 2](domain))
    check_fold_lanes_of[F, T, 4](_registers[T, 4](domain))
    check_fold_lanes_of[F, T, 8](_registers[T, 8](domain))


def _int128_domain() -> List[Scalar[DType.int128]]:
    var domain = List[Scalar[DType.int128]]()
    for v in _int64_domain():
        domain.append(v.cast[DType.int128]())
    return domain^


def test_fold_lanes_law_folds() raises:
    _fold_lanes_of_at_every_width[SumFold](_int64_domain())
    _fold_lanes_of_at_every_width[SumFold](_float64_exact_domain())
    _fold_lanes_of_at_every_width[ProductFold](_int64_domain())
    _fold_lanes_of_at_every_width[ProductFold](_float64_exact_domain())
    _fold_lanes_of_at_every_width[MinFold](_int64_domain())
    _fold_lanes_of_at_every_width[MinFold](_float64_exact_domain())
    _fold_lanes_of_at_every_width[MaxFold](_int64_domain())
    _fold_lanes_of_at_every_width[MaxFold](_float64_exact_domain())
    _fold_lanes_of_at_every_width[MeanFold](_float64_exact_domain())
    _fold_lanes_of_at_every_width[CountFold](_int64_domain())
    _fold_lanes_of_at_every_width[DecimalSumFold](_int128_domain())


struct _FirstLane(BinaryKernel):
    """Addition with a wrong `fold_lanes`: it answers lane 0."""

    comptime name = "first_lane"

    @always_inline
    @staticmethod
    def core[T: DType, W: Int](a: SIMD[T, W], b: SIMD[T, W]) -> SIMD[T, W]:
        return a + b

    @always_inline
    @staticmethod
    def fold_lanes[T: DType, W: Int](v: SIMD[T, W]) -> Scalar[T]:
        return v[0]


def test_fold_lanes_law_rejects_a_wrong_override() raises:
    with assert_raises(contains="fold_lanes:"):
        check_fold_lanes[_FirstLane, DType.int64, 4](
            _registers[DType.int64, 4](_int64_domain())
        )


# ---------------------------------------------------------------------------
# FoldLane: every fold as a binary kernel. CountFold's combine keeps its left
# operand and claims no law; its lane is never read.
# ---------------------------------------------------------------------------


def test_identity_law_fold_lane() raises:
    _claims_identity[FoldLane[SumFold]](_int64_domain())
    _claims_identity[FoldLane[SumFold]](_float64_exact_domain())
    _claims_identity[FoldLane[ProductFold]](_int64_domain())
    _claims_identity[FoldLane[ProductFold]](_float64_exact_domain())
    _claims_identity[FoldLane[MinFold]](_int64_domain())
    _claims_identity[FoldLane[MinFold]](_float64_exact_domain())
    _claims_identity[FoldLane[MaxFold]](_int64_domain())
    _claims_identity[FoldLane[MaxFold]](_float64_exact_domain())
    _claims_identity[FoldLane[MeanFold]](_float64_exact_domain())
    _claims_identity[FoldLane[BitAndFold]](_int64_domain())
    _claims_identity[FoldLane[BitOrFold]](_int64_domain())
    _claims_identity[FoldLane[BitXorFold]](_int64_domain())
    _claims_identity[FoldLane[DecimalSumFold]](_int128_domain())


def test_associative_law_fold_lane() raises:
    _claims_associative[FoldLane[SumFold]](_int64_domain())
    _claims_associative[FoldLane[SumFold]](_float64_exact_domain())
    _claims_associative[FoldLane[ProductFold]](_int64_domain())
    _claims_associative[FoldLane[ProductFold]](_float64_exact_domain())
    _claims_associative[FoldLane[MinFold]](_int64_domain())
    _claims_associative[FoldLane[MinFold]](_float64_exact_domain())
    _claims_associative[FoldLane[MaxFold]](_int64_domain())
    _claims_associative[FoldLane[MaxFold]](_float64_exact_domain())
    _claims_associative[FoldLane[MeanFold]](_float64_exact_domain())
    _claims_associative[FoldLane[BitAndFold]](_int64_domain())
    _claims_associative[FoldLane[BitOrFold]](_int64_domain())
    _claims_associative[FoldLane[BitXorFold]](_int64_domain())
    _claims_associative[FoldLane[DecimalSumFold]](_int128_domain())


def test_commutative_law_fold_lane() raises:
    _claims_commutative[FoldLane[SumFold]](_int64_domain())
    _claims_commutative[FoldLane[SumFold]](_float64_exact_domain())
    _claims_commutative[FoldLane[ProductFold]](_int64_domain())
    _claims_commutative[FoldLane[ProductFold]](_float64_exact_domain())
    _claims_commutative[FoldLane[MinFold]](_int64_domain())
    _claims_commutative[FoldLane[MinFold]](_float64_exact_domain())
    _claims_commutative[FoldLane[MaxFold]](_int64_domain())
    _claims_commutative[FoldLane[MaxFold]](_float64_exact_domain())
    _claims_commutative[FoldLane[MeanFold]](_float64_exact_domain())
    _claims_commutative[FoldLane[BitAndFold]](_int64_domain())
    _claims_commutative[FoldLane[BitOrFold]](_int64_domain())
    _claims_commutative[FoldLane[BitXorFold]](_int64_domain())
    _claims_commutative[FoldLane[DecimalSumFold]](_int128_domain())


def _binary_out[K: BinaryKernel, T: DType]() -> DType:
    """The dtype `K.core` answers for values of `T`, read through the bound."""
    return K.Out[T]


def _unary_out[K: UnaryKernel, T: DType]() -> DType:
    """The dtype `K.core` answers for values of `T`, read through the bound."""
    return K.Out[T]


def test_comparisons_and_predicates_are_kernels_of_bool() raises:
    """A comparison is a `BinaryKernel` and a value predicate a `UnaryKernel`
    whose `core` answers `bool`; arithmetic keeps its operands' dtype."""
    assert_true(_binary_out[LtKernel[], DType.int64]() == DType.bool)
    assert_true(
        _binary_out[EqKernel[nan_safe=True], DType.float64]() == DType.bool
    )
    assert_true(_binary_out[AddKernel, DType.int32]() == DType.int32)
    assert_true(_unary_out[IsNanKernel, DType.float32]() == DType.bool)
    assert_true(_unary_out[NegKernel, DType.float64]() == DType.float64)

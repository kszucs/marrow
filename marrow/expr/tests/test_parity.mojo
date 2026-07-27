"""Cross-driver parity harness for the expression system.

Every stable op is expressible two ways: as a fused comptime ``Value`` tree
(``values.mojo``) and as a runtime tag-based ``DynValue`` tree (``dynamic.mojo``).
Both box into the shared ``AnyValue`` and expose ``execute(batch) -> AnyArray``.
This suite asserts the two drivers agree element-for-element on the same input,
so the runtime interpreter can never silently diverge from the fused algebra it
mirrors.

``assert_parity`` is the reusable primitive: hand it a fused ``Value`` and an
equivalent ``DynValue`` (each implicitly boxed into ``AnyValue``) plus a
``RecordBatch``; it runs both and asserts the resulting arrays are equal. The
fused column leaves resolve by name (``col("a", int64)``) and the ``DynValue``
leaves by position (``col(0)``), so a batch whose columns are named ``a``/``b``
lets the two trees reference the same data.

Seeded with STABLE ops that exist on both drivers: arithmetic
(``+ - * mod floordiv``), numeric comparisons, ``cast``, and ``if_else``. There
is no fused ``if_else`` node, so its fused reference is the equivalent branch-free
mask identity ``a*c + b*(1-c)`` (with ``c = (a > b)`` promoted to int via
``BoolToNum``) — a genuinely different fused path that must still match the
runtime ``select``.
"""

from std.testing import assert_true

from ...arrays import AnyArray, Int64Array
from ...builders import array, PrimitiveBuilder
from ...dtypes import (
    int64,
    int32,
    float64,
    string,
    Int64Type,
    Float64Type,
    TimestampType,
    timestamp,
    second,
)
from ...tabular import RecordBatch, record_batch

# Fused comptime algebra (values.mojo)
from ...expr.values import (
    AnyValue,
    col as fcol,
    lit as flit,
    NumericCast,
    BoolToNum,
    IsNull,
    StrLt,
    StrLe,
    StrGt,
    StrGe,
    Like,
    ILike,
    IsIn,
    Coalesce,
    Nullif,
    CaseWhen,
    Gt,
    Lt,
    Any,
    All,
    Year,
    DateTrunc,
    Hour,
)

# Runtime tag interpreter (dynamic.mojo)
from ...expr.dynamic import (
    DynValue,
    col as dcol,
    lit as dlit,
    if_else,
)


def assert_fused(
    var fused: AnyValue, expected: AnyArray, batch: RecordBatch
) raises:
    """Assert a fused node matches an expected array. Used for ops the runtime
    ``DynValue`` interpreter does not yet expose — their cross-driver parity case
    is PENDING T2.2 (which wires the same ops into ``dynamic.mojo``); until then
    we pin the fused result against the kernel's expected output."""
    var actual = fused.execute(batch)
    assert_true(actual == expected)


def assert_parity(
    var fused: AnyValue, var dyn: AnyValue, batch: RecordBatch
) raises:
    """Execute both drivers against *batch* and assert the arrays are equal."""
    var expected = fused.execute(batch)
    var actual = dyn.execute(batch)
    assert_true(expected == actual)


def _ab_batch() raises -> RecordBatch:
    """A two-column int64 batch (columns ``a`` and ``b``) shared by the cases.
    """
    var a = array([1, 5, 3, 10, 7, 2], int64)
    var b = array([9, 2, 3, 1, 4, 8], int64)
    return record_batch([a^, b^], names=["a", "b"])


# ---------------------------------------------------------------------------
# Arithmetic: + - * mod floordiv
# ---------------------------------------------------------------------------


def test_parity_add() raises:
    assert_parity(
        fcol("a", int64) + fcol("b", int64), dcol(0) + dcol(1), _ab_batch()
    )


def test_parity_sub() raises:
    assert_parity(
        fcol("a", int64) - fcol("b", int64), dcol(0) - dcol(1), _ab_batch()
    )


def test_parity_mul() raises:
    assert_parity(
        fcol("a", int64) * fcol("b", int64), dcol(0) * dcol(1), _ab_batch()
    )


def test_parity_mod() raises:
    assert_parity(
        fcol("a", int64) % fcol("b", int64), dcol(0) % dcol(1), _ab_batch()
    )


def test_parity_floordiv() raises:
    assert_parity(
        fcol("a", int64) // fcol("b", int64), dcol(0) // dcol(1), _ab_batch()
    )


# ---------------------------------------------------------------------------
# Numeric comparisons
# ---------------------------------------------------------------------------


def test_parity_gt() raises:
    assert_parity(
        fcol("a", int64) > fcol("b", int64), dcol(0) > dcol(1), _ab_batch()
    )


def test_parity_lt() raises:
    assert_parity(
        fcol("a", int64) < fcol("b", int64), dcol(0) < dcol(1), _ab_batch()
    )


def test_parity_eq() raises:
    assert_parity(
        fcol("a", int64) == fcol("b", int64), dcol(0) == dcol(1), _ab_batch()
    )


# ---------------------------------------------------------------------------
# Mixed-width comparison. The runtime `LT/GT` tags route through `compare.mojo`'s
# `dispatch`, which *rejects* a dtype mismatch outright, so there is no
# `DynValue` counterpart here — pin the fused result instead. `NumericCompare`
# widens both operands to `promote[L, R]` exactly like `NumericBinary`; casting
# only the right one into the left's type truncated (D4).
# ---------------------------------------------------------------------------


def _mixed_width_batch() raises -> RecordBatch:
    var a = array([1, 2, 3], int32)
    # 2**32 and -2**32 both truncate to 0 in int32.
    var b = array([4294967296, -4294967296, 5], int64)
    return record_batch([a^, b^], names=["a", "b"])


def test_parity_mixed_width_gt() raises:
    assert_fused(
        Gt(fcol("a", int32), fcol("b", int64)),
        array([False, True, False]).to_any(),
        _mixed_width_batch(),
    )


# ---------------------------------------------------------------------------
# Cast (there is a fused NumericCast node)
# ---------------------------------------------------------------------------


def test_parity_cast() raises:
    assert_parity(
        NumericCast[Float64Type](fcol("a", int64)),
        dcol(0).cast(float64),
        _ab_batch(),
    )


# ---------------------------------------------------------------------------
# if_else — no fused node; reference is the equivalent mask identity
#   if_else(a > b, a, b) == a*c + b*(1 - c),  c = (a > b) as int64
# ---------------------------------------------------------------------------


def test_parity_if_else() raises:
    var cnum = BoolToNum[Int64Type](fcol("a", int64) > fcol("b", int64))
    var one = flit(1, int64)
    var fused_ref = fcol("a", int64) * cnum.copy() + fcol("b", int64) * (
        one - cnum.copy()
    )
    var dyn_expr = if_else(dcol(0) > dcol(1), dcol(0), dcol(1))
    assert_parity(fused_ref, dyn_expr, _ab_batch())


# ---------------------------------------------------------------------------
# Null propagation through the fused lane (T0.7). A shared nullable batch: `a`
# and `b` each carry nulls in different rows, so an AND-combine of their
# validities is a non-trivial pattern the fused lane must reproduce.
# ---------------------------------------------------------------------------


def _nullable_ab_batch() raises -> RecordBatch:
    var a = array([1, None, 3, None, 7, 2], int64)
    var b = array([9, 2, None, 1, None, 8], int64)
    return record_batch([a^, b^], names=["a", "b"])


def test_parity_add_nulls() raises:
    # a + b nulls where either operand is null — fused AND-combine == dynamic.
    assert_parity(
        fcol("a", int64) + fcol("b", int64),
        dcol(0) + dcol(1),
        _nullable_ab_batch(),
    )


def test_parity_mul_nulls() raises:
    assert_parity(
        fcol("a", int64) * fcol("b", int64),
        dcol(0) * dcol(1),
        _nullable_ab_batch(),
    )


def test_parity_gt_nulls() raises:
    # (a > b) is valid only where both operands are valid.
    assert_parity(
        fcol("a", int64) > fcol("b", int64),
        dcol(0) > dcol(1),
        _nullable_ab_batch(),
    )


def test_parity_cast_nulls() raises:
    # cast preserves the operand's validity.
    assert_parity(
        NumericCast[Float64Type](fcol("a", int64)),
        dcol(0).cast(float64),
        _nullable_ab_batch(),
    )


def test_parity_isnull_never_null() raises:
    # an IS NULL result is itself always valid (no null bit set).
    assert_parity(
        IsNull(fcol("a", int64)), dcol(0).is_null(), _nullable_ab_batch()
    )


# ---------------------------------------------------------------------------
# Kleene 3-valued and_/or_ over nullable masks (T0.7). The fused `BoolValue` lane
# now tracks validity, reusing the null-correct `AndKernel`/`OrKernel` (Kleene,
# fixed in T0.1) that the runtime `DynValue` path already routes through, so the
# two drivers agree element-for-element — including where a known-false operand
# forces a valid AND result and a known-true operand forces a valid OR result.
# ---------------------------------------------------------------------------


def test_parity_and_kleene() raises:
    var fused = (fcol("a", int64) > flit(0, int64)) & (
        fcol("b", int64) > flit(0, int64)
    )
    var dyn = (dcol(0) > dlit[Int64Type](0)) & (dcol(1) > dlit[Int64Type](0))
    assert_parity(fused, dyn, _nullable_ab_batch())


def test_parity_or_kleene() raises:
    var fused = (fcol("a", int64) > flit(0, int64)) | (
        fcol("b", int64) > flit(0, int64)
    )
    var dyn = (dcol(0) > dlit[Int64Type](0)) | (dcol(1) > dlit[Int64Type](0))
    assert_parity(fused, dyn, _nullable_ab_batch())


# ---------------------------------------------------------------------------
# any/all over a mask whose NULL slots carry SET data bits (D3). The fused
# comparison writes a mask bit for every row, valid or not, so a reduction that
# popcounts the raw buffer answers from data the row does not own. The runtime
# interpreter has no ANY/ALL tag, so pin the fused result.
# ---------------------------------------------------------------------------


def _null_bits_batch() raises -> RecordBatch:
    # A null slot keeps its zero-filled data, so `x < 1` sets the bit there.
    var a = array([None, 5, 7], int64)  # `<1` -> bits [T,F,F], valid [F,T,T]
    var b = array([None, 0, 0], int64)  # `<1` -> bits [T,T,T], valid [F,T,T]
    return record_batch([a^, b^], names=["a", "b"])


def test_parity_any_ignores_null_bits() raises:
    # the one set bit belongs to a null row -> False (a raw popcount says True)
    assert_fused(
        Any(Lt(fcol("a", int64), flit(1, int64))),
        array([False, False, False]).to_any(),
        _null_bits_batch(),
    )


def test_parity_all_ignores_null_bits() raises:
    # every VALID row is true -> True (popcount==valid-count says False)
    assert_fused(
        All(Lt(fcol("b", int64), flit(1, int64))),
        array([True, True, True]).to_any(),
        _null_bits_batch(),
    )


# ---------------------------------------------------------------------------
# String ordering comparisons — the runtime `LT/LE/GT/GE` tags route through the
# same `compare.mojo` kernels' `dispatch`, which handles string operands
# (lexicographic byte order). So these ARE cross-driver parity cases today: the
# fused `StrLt` breaker must agree with `dcol < dcol` over string columns.
# ---------------------------------------------------------------------------


def _spair_batch() raises -> RecordBatch:
    var s = array(["apple", "banana", "cherry", "date"])
    var p = array(["apple", "apricot", "date", "cab"])
    return record_batch([s^, p^], names=["s", "p"])


def test_parity_string_lt() raises:
    assert_parity(
        StrLt(fcol("s", string), fcol("p", string)),
        dcol(0) < dcol(1),
        _spair_batch(),
    )


def test_parity_string_le() raises:
    assert_parity(
        StrLe(fcol("s", string), fcol("p", string)),
        dcol(0) <= dcol(1),
        _spair_batch(),
    )


def test_parity_string_gt() raises:
    assert_parity(
        StrGt(fcol("s", string), fcol("p", string)),
        dcol(0) > dcol(1),
        _spair_batch(),
    )


def test_parity_string_ge() raises:
    assert_parity(
        StrGe(fcol("s", string), fcol("p", string)),
        dcol(0) >= dcol(1),
        _spair_batch(),
    )


# ---------------------------------------------------------------------------
# Ops the runtime `DynValue` interpreter does not yet expose — like/ilike, is_in,
# coalesce, nullif, case_when, temporal. Their cross-driver parity is PENDING
# T2.2 (which adds these tags to dynamic.mojo). Until then, pin the fused result
# against the kernel's expected output (`assert_fused`).
# ---------------------------------------------------------------------------


def _like_batch() raises -> RecordBatch:
    var s = array(["apple", "banana", "cherry"])
    var pat = array(["a%", "b%", "x%"])
    return record_batch([s^, pat^], names=["s", "pat"])


def test_parity_like() raises:
    assert_fused(
        Like(fcol("s", string), fcol("pat", string)),
        array([True, True, False]).to_any(),
        _like_batch(),
    )


def test_parity_ilike() raises:
    var s = array(["APPLE", "Banana", "x"])
    var pat = array(["a%", "b%", "y%"])
    var b = record_batch([s^, pat^], names=["s", "pat"])
    assert_fused(
        ILike(fcol("s", string), fcol("pat", string)),
        array([True, True, False]).to_any(),
        b,
    )


def test_parity_is_in() raises:
    # a=[1,5,3,10,7,2] IN {3,7} -> [F,F,T,F,T,F]
    assert_fused(
        IsIn(fcol("a", int64), array([3, 7], int64)),
        array([False, False, True, False, True, False]).to_any(),
        _ab_batch(),
    )


def test_parity_coalesce() raises:
    var a = array([1, None, None, 4], int64)
    var b = array([10, 20, None, 40], int64)
    var batch = record_batch([a^, b^], names=["a", "b"])
    assert_fused(
        Coalesce(fcol("a", int64), fcol("b", int64)),
        array([1, 20, None, 4], int64).to_any(),
        batch,
    )


def test_parity_nullif() raises:
    var a = array([1, 2, 3, 4], int64)
    var b = array([9, 2, 3, 9], int64)
    var batch = record_batch([a^, b^], names=["a", "b"])
    assert_fused(
        Nullif(fcol("a", int64), fcol("b", int64)),
        array([1, None, None, 4], int64).to_any(),
        batch,
    )


def test_parity_case_when() raises:
    # CASE WHEN a>b THEN a ELSE b == max(a,b), over the shared _ab_batch
    assert_fused(
        CaseWhen(
            Gt(fcol("a", int64), fcol("b", int64)),
            fcol("a", int64),
            fcol("b", int64),
        ),
        array([9, 5, 3, 10, 7, 8], int64).to_any(),
        _ab_batch(),
    )


def _ts_batch() raises -> RecordBatch:
    # 2019-06-15 12:30:45 UTC ; 2020-02-29 00:00:00 UTC
    var b = PrimitiveBuilder[TimestampType](timestamp(second), capacity=2)
    b.append(Int64(1_560_601_845))
    b.append(Int64(1_582_934_400))
    return record_batch([b.finish()], names=["ts"])


def test_parity_year() raises:
    assert_fused(
        Year(fcol("ts", timestamp(second))),
        array([2019, 2020], int32).to_any(),
        _ts_batch(),
    )


def test_parity_date_trunc() raises:
    # hour(date_trunc(ts, "day")) == 0 for every row
    assert_fused(
        Hour(DateTrunc(fcol("ts", timestamp(second)), "day")),
        array([0, 0], int32).to_any(),
        _ts_batch(),
    )

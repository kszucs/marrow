"""Cross-driver parity harness for the expression system.

Every stable op is expressible two ways: as a fused comptime ``Value`` tree
(``values.mojo``) and as a runtime tag-based ``DynValue`` tree (``dynamic.mojo``).
The two lanes share **no node types** — that separation is what makes each one
honest about its operands — so nothing but this suite holds them to the same
arithmetic. A divergence here is the failure mode the split trades for.

``assert_parity`` is the reusable primitive: hand it a fused ``Value`` and an
equivalent ``DynValue`` (each implicitly boxed into ``BoxedValue``, the one type
both lanes erase into) plus a ``RecordBatch``; it runs both and asserts the
resulting arrays are equal. Both lanes' column leaves resolve by name, so a batch
whose columns are named ``a``/``b`` lets the two trees reference the same data.

Seeded with STABLE ops that exist on both drivers: arithmetic
(``+ - * mod floordiv``), numeric comparisons, ``cast``, and ``if_else``. There
is no fused ``if_else`` node, so its fused reference is the equivalent branch-free
mask identity ``a*c + b*(1-c)`` (with ``c = (a > b)`` promoted to int via
``BoolToNum``) — a genuinely different fused path that must still match the
runtime ``select``.
"""

from std.testing import assert_true

from ...arrays import DynArray, Int64Array
from ...builders import array, PrimitiveBuilder
from ...dtypes import (
    DynType,
    NumericType,
    int8,
    int16,
    uint8,
    uint16,
    uint32,
    uint64,
    float16,
    float32,
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
from ...schema import schema
from ...dtypes import field

from ...kernels.temporal import unit_day

# Fused comptime algebra (values.mojo)
from ...expr.values import (
    _rank,
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
    Le,
    Ge,
    Ne,
    Pow,
    Xor,
    Neg,
    Abs,
    Sign,
    Floor,
    Ceil,
    Round,
    Sqrt,
    Exp,
    Ln,
    Upper,
    Lower,
    Strip,
    LStrip,
    RStrip,
    Reverse,
    Capitalize,
    StringLength,
    StartsWith,
    EndsWith,
    StrContains,
    Any,
    All,
    Year,
    DateTrunc,
    Hour,
)

from ...kernels.core import Kernel
from ...kernels.numeric import (
    AddKernel,
    SubKernel,
    MulKernel,
    DivKernel,
    ModKernel,
    FloordivKernel,
    PowKernel,
    LtKernel,
    LeKernel,
    GtKernel,
    GeKernel,
    EqKernel,
    NeKernel,
    NegKernel,
    AbsKernel,
    SignKernel,
    FloorKernel,
    CeilKernel,
    RoundKernel,
    SqrtKernel,
    ExpKernel,
    LogKernel,
)
from ...kernels.boolean import AndKernel, OrKernel, XorKernel, NotKernel
from ...kernels.string import StartsWithKernel, EndsWithKernel, ContainsKernel
from ...kernels.temporal import (
    YearKernel,
    MonthKernel,
    DayKernel,
    HourKernel,
    MinuteKernel,
    SecondKernel,
    QuarterKernel,
    DayOfWeekKernel,
    DayOfYearKernel,
)
from ...kernels.conditional import CoalesceKernel, NullifKernel

from ...expr.dynamic import DynValue, _numeric_rank
from ...expr.relations import BoxedValue

# The runtime lane's own leaves — these build `DynValue` tag nodes
from ...expr.values import (
    col as dcol,
    lit as dlit,
    if_else,
)


def assert_fused(
    var fused: BoxedValue, expected: DynArray, batch: RecordBatch
) raises:
    """Assert a fused node matches an expected array. Used for ops the runtime
    ``DynValue`` interpreter does not yet expose — their cross-driver parity case
    is PENDING T2.2 (which wires the same ops into ``dynamic.mojo``); until then
    we pin the fused result against the kernel's expected output."""
    var actual = fused.execute(batch)
    assert_true(actual == expected)


def assert_parity(
    var fused: BoxedValue, var dyn: BoxedValue, batch: RecordBatch
) raises:
    """Execute both lanes against *batch* and assert the arrays are equal.

    ``BoxedValue`` is the one type both lanes erase into — a fused node from
    ``values.mojo`` on the left, a ``DynValue`` tag tree on the right — so this
    is where the two are held to the same answer. They share no node types, and
    this suite is what stops them diverging."""
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
        fcol("a", int64) + fcol("b", int64), dcol("a") + dcol("b"), _ab_batch()
    )


def test_parity_sub() raises:
    assert_parity(
        fcol("a", int64) - fcol("b", int64), dcol("a") - dcol("b"), _ab_batch()
    )


def test_parity_mul() raises:
    assert_parity(
        fcol("a", int64) * fcol("b", int64), dcol("a") * dcol("b"), _ab_batch()
    )


def test_parity_mod() raises:
    assert_parity(
        fcol("a", int64) % fcol("b", int64), dcol("a") % dcol("b"), _ab_batch()
    )


def test_parity_floordiv() raises:
    assert_parity(
        fcol("a", int64) // fcol("b", int64),
        dcol("a") // dcol("b"),
        _ab_batch(),
    )


# ---------------------------------------------------------------------------
# Numeric comparisons
# ---------------------------------------------------------------------------


def test_parity_gt() raises:
    assert_parity(
        fcol("a", int64) > fcol("b", int64), dcol("a") > dcol("b"), _ab_batch()
    )


def test_parity_lt() raises:
    assert_parity(
        fcol("a", int64) < fcol("b", int64), dcol("a") < dcol("b"), _ab_batch()
    )


def test_parity_eq() raises:
    assert_parity(
        fcol("a", int64) == fcol("b", int64),
        dcol("a") == dcol("b"),
        _ab_batch(),
    )


# ---------------------------------------------------------------------------
# Mixed operand types. Both lanes promote to the wider operand — the fused one
# at comptime (`promote[L, R]`), the interpreted one in `_promote_operands`
# before it reaches a kernel — so the pairings a fused tree accepts are exactly
# the ones the interpreter accepts, with the same result. `NumericCompare`
# widens *both* operands; casting only the right one into the left's type
# truncated (D4), which is what the int32/int64 case below pins.
#
# Parity coverage here is keyed on the *fused* lane's accepted domain, not on
# the intersection of the two — an operand pairing only one lane accepts is
# invisible to a test that can only build what both allow, which is how the
# int64/float64 divergence (Q0.4) survived this suite.
# ---------------------------------------------------------------------------


def _mixed_width_batch() raises -> RecordBatch:
    var a = array([1, 2, 3], int32)
    # 2**32 and -2**32 both truncate to 0 in int32.
    var b = array([4294967296, -4294967296, 5], int64)
    return record_batch([a^, b^], names=["a", "b"])


def test_parity_mixed_width_gt() raises:
    """`int32 > int64` compares in the wider operand's domain, both lanes."""
    assert_fused(
        Gt(fcol("a", int32), fcol("b", int64)),
        array([False, True, False]).to_dyn(),
        _mixed_width_batch(),
    )
    assert_parity(
        Gt(fcol("a", int32), fcol("b", int64)),
        dcol("a") > dcol("b"),
        _mixed_width_batch(),
    )


def _int_float_batch() raises -> RecordBatch:
    var a = array([1, 5, 3], int64)
    var b = array([0.5, 2.25, 3.0], float64)
    return record_batch([a^, b^], names=["a", "b"])


def test_parity_int_float_add() raises:
    """`int64 + float64`: a float outranks every integer, so both lanes compute
    in float64. This is the pairing the interpreted lane used to reject with
    `add: dtype mismatch: int64 vs float64` while the fused lane executed it."""
    assert_fused(
        fcol("a", int64) + fcol("b", float64),
        array([1.5, 7.25, 6.0], float64).to_dyn(),
        _int_float_batch(),
    )
    assert_parity(
        fcol("a", int64) + fcol("b", float64),
        dcol("a") + dcol("b"),
        _int_float_batch(),
    )


def test_parity_int_float_gt() raises:
    """`int64 > float64` compares in float64, not by truncating the float."""
    assert_fused(
        Gt(fcol("a", int64), fcol("b", float64)),
        array([True, True, False]).to_dyn(),
        _int_float_batch(),
    )
    assert_parity(
        Gt(fcol("a", int64), fcol("b", float64)),
        dcol("a") > dcol("b"),
        _int_float_batch(),
    )


# ---------------------------------------------------------------------------
# Cast (there is a fused NumericCast node)
# ---------------------------------------------------------------------------


def test_parity_cast() raises:
    assert_parity(
        NumericCast[Float64Type](fcol("a", int64)),
        dcol("a").cast(float64),
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
    var cond: DynValue = dcol("a") > dcol("b")
    var then_: DynValue = dcol("a")
    var else_: DynValue = dcol("b")
    var dyn_expr = if_else(cond, then_, else_)
    assert_parity(fused_ref, dyn_expr^, _ab_batch())


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
        dcol("a") + dcol("b"),
        _nullable_ab_batch(),
    )


def test_parity_mul_nulls() raises:
    assert_parity(
        fcol("a", int64) * fcol("b", int64),
        dcol("a") * dcol("b"),
        _nullable_ab_batch(),
    )


def test_parity_gt_nulls() raises:
    # (a > b) is valid only where both operands are valid.
    assert_parity(
        fcol("a", int64) > fcol("b", int64),
        dcol("a") > dcol("b"),
        _nullable_ab_batch(),
    )


def test_parity_cast_nulls() raises:
    # cast preserves the operand's validity.
    assert_parity(
        NumericCast[Float64Type](fcol("a", int64)),
        dcol("a").cast(float64),
        _nullable_ab_batch(),
    )


def test_parity_isnull_never_null() raises:
    # an IS NULL result is itself always valid (no null bit set).
    assert_parity(
        IsNull(fcol("a", int64)), dcol("a").isnull(), _nullable_ab_batch()
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
    var dyn = (dcol("a") > dlit[Int64Type](0)) & (
        dcol("b") > dlit[Int64Type](0)
    )
    assert_parity(fused, dyn^, _nullable_ab_batch())


def test_parity_or_kleene() raises:
    var fused = (fcol("a", int64) > flit(0, int64)) | (
        fcol("b", int64) > flit(0, int64)
    )
    var dyn = (dcol("a") > dlit[Int64Type](0)) | (
        dcol("b") > dlit[Int64Type](0)
    )
    assert_parity(fused, dyn^, _nullable_ab_batch())


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
        array([False, False, False]).to_dyn(),
        _null_bits_batch(),
    )


def test_parity_all_ignores_null_bits() raises:
    # every VALID row is true -> True (popcount==valid-count says False)
    assert_fused(
        All(Lt(fcol("b", int64), flit(1, int64))),
        array([True, True, True]).to_dyn(),
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
        dcol("s") < dcol("p"),
        _spair_batch(),
    )


def test_parity_string_le() raises:
    assert_parity(
        StrLe(fcol("s", string), fcol("p", string)),
        dcol("s") <= dcol("p"),
        _spair_batch(),
    )


def test_parity_string_gt() raises:
    assert_parity(
        StrGt(fcol("s", string), fcol("p", string)),
        dcol("s") > dcol("p"),
        _spair_batch(),
    )


def test_parity_string_ge() raises:
    assert_parity(
        StrGe(fcol("s", string), fcol("p", string)),
        dcol("s") >= dcol("p"),
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
        array([True, True, False]).to_dyn(),
        _like_batch(),
    )


def test_parity_ilike() raises:
    var s = array(["APPLE", "Banana", "x"])
    var pat = array(["a%", "b%", "y%"])
    var b = record_batch([s^, pat^], names=["s", "pat"])
    assert_fused(
        ILike(fcol("s", string), fcol("pat", string)),
        array([True, True, False]).to_dyn(),
        b,
    )


def test_parity_is_in() raises:
    # a=[1,5,3,10,7,2] IN {3,7} -> [F,F,T,F,T,F]
    assert_fused(
        IsIn(fcol("a", int64), array([3, 7], int64)),
        array([False, False, True, False, True, False]).to_dyn(),
        _ab_batch(),
    )


def test_parity_coalesce() raises:
    var a = array([1, None, None, 4], int64)
    var b = array([10, 20, None, 40], int64)
    var batch = record_batch([a^, b^], names=["a", "b"])
    assert_fused(
        Coalesce(fcol("a", int64), fcol("b", int64)),
        array([1, 20, None, 4], int64).to_dyn(),
        batch,
    )


def test_parity_nullif() raises:
    var a = array([1, 2, 3, 4], int64)
    var b = array([9, 2, 3, 9], int64)
    var batch = record_batch([a^, b^], names=["a", "b"])
    assert_fused(
        Nullif(fcol("a", int64), fcol("b", int64)),
        array([1, None, None, 4], int64).to_dyn(),
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
        array([9, 5, 3, 10, 7, 8], int64).to_dyn(),
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
        array([2019, 2020], int32).to_dyn(),
        _ts_batch(),
    )


def test_parity_date_trunc() raises:
    # hour(date_trunc(ts, "day")) == 0 for every row
    assert_fused(
        Hour(DateTrunc(fcol("ts", timestamp(second)), unit_day)),
        array([0, 0], int32).to_dyn(),
        _ts_batch(),
    )


# ---------------------------------------------------------------------------
# Rank agreement between the two lanes
# ---------------------------------------------------------------------------


def test_numeric_rank_agrees_across_lanes() raises:
    """`values._rank[T]()` and `dynamic._numeric_rank(dt)` must return the same
    rank for every numeric type.

    They cannot be one function: the fused lane needs a comptime `Int` (it feeds
    `comptime promote[L, R]`) and the interpreter needs a runtime one from an
    erased `DynType`. So the invariant is that the two agree, and until now
    nothing checked it — `dynamic.mojo` documents the requirement in a comment
    and stops there. A drift would not fail loudly; it would make `a + b` pick a
    different result dtype in the two lanes for the same operand pair, which the
    end-to-end parity cases only catch for the pairings they happen to name.

    `dispatch_numeric` is what makes this checkable at all: it resolves each
    runtime dtype to the comptime type its fused counterpart would use, which is
    exactly the bridge the two functions sit on either side of.

    Reaching for the underscore-private functions is deliberate. The public
    alternative is a fused instantiation per ordered pair — 121 of them — to
    observe a property both functions state directly.
    """
    var numerics = [
        DynType(int8),
        DynType(int16),
        DynType(int32),
        DynType(int64),
        DynType(uint8),
        DynType(uint16),
        DynType(uint32),
        DynType(uint64),
        DynType(float16),
        DynType(float32),
        DynType(float64),
    ]

    for i in range(len(numerics)):
        ref dt = numerics[i]
        var runtime_rank = _numeric_rank(dt)

        @__parameter
        def check[T: NumericType](d: T) raises -> Bool:
            return _rank[T]() == runtime_rank

        assert_true(
            dt.dispatch_numeric[check](),
            String("rank disagreement for ") + String(dt),
        )


# ---------------------------------------------------------------------------
# The runtime lane's own nodes, held against the fused tree
# ---------------------------------------------------------------------------


def test_erased_binary_matches_the_fused_tree() raises:
    """`a + b` on two erased operands builds a tag node, and it must compute
    what the fused `Add` computes.

    The two lanes stopped sharing node types when the box stopped claiming the
    family traits: `DynValue` supplies `__add__` itself and yields another
    `DynValue`, so nothing here is parameterised on an erased operand and no
    node has to ask whether its operand has a lane. What used to be a *type*
    guarantee of agreement is now this assertion.
    """
    var batch = _ab_batch()

    var lhs: DynValue = dcol("a")
    var rhs: DynValue = dcol("b")
    var dyn = lhs + rhs

    assert_parity(fcol("a", int64) + fcol("b", int64), dyn^, batch)


def test_erased_binary_nests() raises:
    """`(a + b) * a` — an erased subtree under another erased node.

    Nesting used to be the hazard: a composite over erased operands had to
    propagate `IsErased` by hand or its fused arm elaborated
    `SIMD[DynType.native, W]` and failed to *instantiate*. A tag node has no
    fused arm to elaborate, so the failure mode is gone; what remains to check
    is only that the answer still matches.
    """
    var batch = _ab_batch()

    var lhs: DynValue = dcol("a")
    var rhs: DynValue = dcol("b")
    var again: DynValue = dcol("a")
    var dyn = (lhs + rhs) * again

    assert_parity(
        (fcol("a", int64) + fcol("b", int64)) * fcol("a", int64),
        dyn^,
        batch,
    )


def test_erased_add_concatenates_strings() raises:
    """`+` on two erased operands means *concatenate* when they turn out to be
    strings, and *add* when they turn out to be numbers.

    The choice cannot be made when the tree is built: an erased column's dtype
    is only known once a schema is applied, so `col("x") + col("y")` has no dtype
    to branch on at construction. It is made in the `"add"` arm instead, against
    the materialized operands — the same shape the comparison arms use, where a
    tag names a pair of kernels and the runtime dtype picks one.

    The numeric half of this is `test_erased_binary_matches_the_fused_tree`;
    both go through the same tag.
    """
    var x = array(["a", "c", "e"])
    var y = array(["b", "d", "f"])
    var batch = record_batch([x^, y^], names=["x", "y"])

    var lhs: DynValue = dcol("x")
    var rhs: DynValue = dcol("y")
    var joined: DynValue = lhs + rhs

    var got = joined.execute(batch)
    var want: DynArray = array(["ab", "cd", "ef"])
    assert_true(got == want)


def test_erased_lane_covers_the_regular_operators() raises:
    """Every regular operator over erased operands agrees with the fused tree:
    arithmetic, division, unary, comparison and boolean logic.

    One case per fused node family — `NumericBinary`, `FloatBinary`,
    `NumericUnary`, `NumericCompare`, `BoolBinary`, `BoolUnary` — because each
    has a separate tag arm on the other side, and a wrong kernel in any of them
    is a wrong answer rather than a build failure.
    """
    var batch = _ab_batch()

    var a: DynValue = dcol("a")
    var b: DynValue = dcol("b")

    # NumericBinary
    assert_parity(fcol("a", int64) - fcol("b", int64), a - b, batch)
    assert_parity(fcol("a", int64) * fcol("b", int64), a * b, batch)
    # FloatBinary (true division promotes to float64)
    assert_parity(fcol("a", int64) / fcol("b", int64), a / b, batch)
    # NumericUnary
    assert_parity(-fcol("a", int64), -a, batch)
    # NumericCompare
    assert_parity(fcol("a", int64) > fcol("b", int64), a > b, batch)
    assert_parity(fcol("a", int64) == fcol("b", int64), a == b, batch)
    # BoolBinary / BoolUnary over the compare results
    assert_parity(
        (fcol("a", int64) > fcol("b", int64))
        & (fcol("a", int64) < fcol("b", int64)),
        (a > b) & (a < b),
        batch,
    )
    assert_parity(~(fcol("a", int64) > fcol("b", int64)), ~(a > b), batch)


def test_erased_compare_compares_strings() raises:
    """`<` on erased string operands takes the string kernel, not the numeric
    one — the tag arm carries both and the runtime dtype picks."""
    var x = array(["a", "d", "c"])
    var y = array(["b", "b", "c"])
    var batch = record_batch([x^, y^], names=["x", "y"])

    var lhs: DynValue = dcol("x")
    var rhs: DynValue = dcol("y")
    var lt: DynValue = lhs < rhs

    var got = lt.execute(batch)
    var want: DynArray = array([True, False, False])
    assert_true(got == want)


def test_erased_payload_nodes() raises:
    """The ops whose fused counterparts carry a payload beyond their children.

    These are the ones whose fused twins are pipeline *breakers*: `Value.execute`
    routes a breaker through `prepare` and never calls `materialize`. The erased
    lane has no fused loop to break, so it computes each of these in a single
    dispatch — a different mechanism reaching the same column, which is exactly
    what wants asserting.

    Covers `StringLength`, the string comparison family, and `ConditionalBinary`
    (coalesce/nullif); `TemporalExtract` is covered by `test_parity_year`
    alongside its fused twin.
    """
    var s0 = array(["ab", "cde", "f"])
    var s1 = array(["ab", "xy", "f"])
    var sbatch = record_batch([s0^, s1^], names=["s", "t"])

    var s: DynValue = dcol("s")
    var t: DynValue = dcol("t")

    # StringLength — breaker when fused, single dispatch when erased
    var lens: DynValue = s.length()
    var want_len: DynArray = array([2, 3, 1], int32)
    assert_true(lens.execute(sbatch) == want_len)

    # StringPredicate — the string comparison family
    var eq: DynValue = s == t
    var want_eq: DynArray = array([True, False, True])
    assert_true(eq.execute(sbatch) == want_eq)

    # ConditionalBinary — nullif over erased numeric operands
    var nbatch = _ab_batch()
    var a: DynValue = dcol("a")
    var b: DynValue = dcol("b")
    var nl: DynValue = a.nullif(b)
    var got = nl.execute(nbatch)
    # a == b only at index 2 (3 == 3), which nullif turns into a null
    assert_true(got.null_count() == 1)


def test_erased_cast_isin_casewhen() raises:
    """The last three payload-carrying ops: `cast`, `is_in` and `case_when`.

    Each carries something beyond its children — a target dtype, a value-set
    array, a third branch — and the tag node holds all three in the same
    `DynPayload` variant, so this is the one place that shape is exercised end
    to end. The fused lane spells these `NumericCast[To]`, `IsIn` and
    `CaseWhen`; here they are methods on the box.
    """
    var batch = _ab_batch()
    var a: DynValue = dcol("a")
    var b: DynValue = dcol("b")

    # cast — payload is the target dtype
    var casted: DynValue = a.cast(DynType(Float64Type()))
    var got_cast = casted.execute(batch)
    assert_true(got_cast.dtype() == DynType(Float64Type()))
    assert_true(got_cast.length() == 6)

    # is_in — payload is a value-set array
    var member: DynValue = a.isin(array([1, 3, 7], int64))
    var want_in: DynArray = array([True, False, True, False, True, False])
    assert_true(member.execute(batch) == want_in)

    # if_else — three children, no payload
    var picked: DynValue = if_else(a > b, a.copy(), b.copy())
    var want_pick: DynArray = array([9, 5, 3, 10, 7, 8], int64)
    assert_true(picked.execute(batch) == want_pick)


def test_erased_temporal_extract() raises:
    """`year()` on an erased column.

    The fused twin is `TemporalExtract`, whose operand is bound on
    `TemporalValue` — a bound the box can no longer satisfy and no longer needs
    to, since it extracts through its own tag arm.
    """
    var ts = array([0, 86_400, 31_536_000], int64)
    var batch = record_batch([ts^], names=["t"])

    var col: DynValue = dcol("t").cast(timestamp(second))
    var yr: DynValue = col.year()

    var want: DynArray = array([1970, 1970, 1971], int32)
    assert_true(yr.execute(batch) == want)


# ---------------------------------------------------------------------------
# The runtime lane, entered through the untyped factories
# ---------------------------------------------------------------------------


def test_runtime_factories_build_tag_nodes() raises:
    """`col("a") + col("b")` — the one-argument `col`, with no dtype — builds a
    tag tree, and it agrees with the fused tree built from the two-argument
    `col("a", int64)`.

    The two frontends differ in what they *know*: one is handed a dtype, the
    other finds it on the batch. That is the whole distinction between the lanes.
    """
    var batch = _ab_batch()
    var a = fcol("a")
    var b = fcol("b")

    assert_parity(fcol("a", int64) + fcol("b", int64), a + b, batch)
    assert_parity(fcol("a", int64) > fcol("b", int64), a > b, batch)
    assert_parity(
        (fcol("a", int64) + fcol("b", int64)) * fcol("a", int64),
        (a + b) * a,
        batch,
    )


def test_runtime_factories_resolve_by_name() raises:
    """A named erased column resolves against the batch schema at execute time,
    so a plan can be built before a schema is known."""
    var batch = _ab_batch()
    assert_parity(
        fcol("a", int64) - fcol("b", int64),
        fcol("a") - fcol("b"),
        batch,
    )


def test_runtime_bound_column_replaces_tag_inspection() raises:
    """`bound_column` is how the relational layer identifies a join/group key.

    It replaces reaching into the interpreter for `kind() == LOAD` and then
    `kind_data()`. A bare column answers with its position; anything computed
    answers -1.
    """
    var sch = schema([field("a", int64), field("b", int64)])
    assert_true(fcol("a").bound_column(sch) == 0)
    assert_true(fcol("b").bound_column(sch) == 1)
    assert_true((fcol("a") + fcol("b")).bound_column(sch) == -1)


def test_runtime_literal_and_cast() raises:
    var batch = _ab_batch()
    var casted = fcol("a").cast(DynType(Float64Type()))
    var out = casted.execute(batch)
    assert_true(out.dtype() == DynType(Float64Type()))

    assert_parity(
        fcol("a", int64) + flit(3, int64),
        fcol("a") + flit[Int64Type](3),
        batch,
    )


# ---------------------------------------------------------------------------
# A5 — op-name parity, asserted mechanically rather than by hand-written literal
# ---------------------------------------------------------------------------
def _tag_mismatch[K: Kernel](rendered: String) -> String:
    """Empty when a runtime node renders under its fused kernel's own `name`.

    The expected string is never written down here — it is read off the kernel
    the runtime factory already names. That is the whole point: `prune`
    correctness keys on these strings against two hand-maintained sets, and a
    divergence between them fails *silently*, as a `Interval.unknown()`
    fall-through that disables row-group skipping rather than raising.

    Returns a description rather than asserting so one run reports *every*
    divergence; stopping at the first would hide the rest.
    """
    if rendered.startswith(String(K.name) + "("):
        return String()
    return (
        String("\n  `")
        + rendered
        + "` should render under `"
        + String(K.name)
        + "`"
    )


def test_op_names_agree_across_lanes() raises:
    """Every shared op must render under one name in both lanes.

    The runtime factories already name their kernel — `Self("modulo",
    Self._binary[ModKernel], ...)` — so the tag string is redundant with
    `ModKernel.name` and can only drift away from it. This enumerates the op
    set once and reads every expected name off the kernel.
    """
    var a = dcol("a")
    var b = dcol("b")
    var bad = String()

    # arithmetic
    bad += _tag_mismatch[AddKernel]((a + b).render())
    bad += _tag_mismatch[SubKernel]((a - b).render())
    bad += _tag_mismatch[MulKernel]((a * b).render())
    bad += _tag_mismatch[DivKernel]((a / b).render())
    bad += _tag_mismatch[ModKernel]((a % b).render())
    bad += _tag_mismatch[FloordivKernel]((a // b).render())
    bad += _tag_mismatch[PowKernel]((a**b).render())

    # comparison
    bad += _tag_mismatch[LtKernel]((a < b).render())
    bad += _tag_mismatch[LeKernel]((a <= b).render())
    bad += _tag_mismatch[GtKernel]((a > b).render())
    bad += _tag_mismatch[GeKernel]((a >= b).render())
    bad += _tag_mismatch[EqKernel]((a == b).render())
    bad += _tag_mismatch[NeKernel]((a != b).render())

    # boolean logic
    bad += _tag_mismatch[AndKernel]((a & b).render())
    bad += _tag_mismatch[OrKernel]((a | b).render())
    bad += _tag_mismatch[XorKernel]((a ^ b).render())
    bad += _tag_mismatch[NotKernel]((~a).render())

    # numeric unaries
    bad += _tag_mismatch[NegKernel]((-a).render())
    bad += _tag_mismatch[AbsKernel](a.abs().render())
    bad += _tag_mismatch[SignKernel](a.sign().render())
    bad += _tag_mismatch[FloorKernel](a.floor().render())
    bad += _tag_mismatch[CeilKernel](a.ceil().render())
    bad += _tag_mismatch[RoundKernel](a.round().render())

    # float unaries
    bad += _tag_mismatch[SqrtKernel](a.sqrt().render())
    bad += _tag_mismatch[ExpKernel](a.exp().render())
    bad += _tag_mismatch[LogKernel](a.ln().render())

    # string predicates and temporal extracts — these already agreed, and the
    # test is what keeps them agreeing
    bad += _tag_mismatch[StartsWithKernel](a.startswith(b).render())
    bad += _tag_mismatch[EndsWithKernel](a.endswith(b).render())
    bad += _tag_mismatch[ContainsKernel](a.contains(b).render())
    bad += _tag_mismatch[YearKernel](a.year().render())
    bad += _tag_mismatch[MonthKernel](a.month().render())
    bad += _tag_mismatch[DayKernel](a.day().render())
    bad += _tag_mismatch[HourKernel](a.hour().render())
    bad += _tag_mismatch[MinuteKernel](a.minute().render())
    bad += _tag_mismatch[SecondKernel](a.second().render())
    bad += _tag_mismatch[QuarterKernel](a.quarter().render())
    bad += _tag_mismatch[DayOfWeekKernel](a.day_of_week().render())
    bad += _tag_mismatch[DayOfYearKernel](a.day_of_year().render())
    bad += _tag_mismatch[CoalesceKernel](a.coalesce(b).render())
    bad += _tag_mismatch[NullifKernel](a.nullif(b).render())

    assert_true(
        not bad,
        String("op names diverge between the fused and runtime lanes:") + bad,
    )


# ---------------------------------------------------------------------------
# A5 — value parity for the ops the hand-written cases never reached
#
# Naming and pruning are held mechanically above. This is the third axis: do
# the two lanes compute the same *answer*? Each case is one op, both lanes,
# against real data — the ops below had no cross-lane assertion at all, which
# is how the op-name drift survived long enough to be found by a different
# test.
# ---------------------------------------------------------------------------
def _unary_batch() raises -> RecordBatch:
    """Signed, fractional and integral float64 — enough to tell `floor`,
    `ceil`, `round`, `abs` and `sign` apart from each other and from identity.
    """
    var a = array([-2.5, -0.5, 0.0, 1.5, 3.25], float64)
    return record_batch([a^], names=["a"])


def _positive_batch() raises -> RecordBatch:
    """Strictly positive float64, so `sqrt`/`ln` are defined everywhere."""
    var a = array([0.25, 1.0, 2.0, 7.5], float64)
    return record_batch([a^], names=["a"])


def _smaps_batch() raises -> RecordBatch:
    """Mixed case with leading and trailing whitespace, so the seven string
    maps produce visibly different results from one another."""
    var s = array(["  hello  ", "WORLD  ", "  MiXeD", "x"])
    return record_batch([s^], names=["s"])


# --- comparisons the hand-written cases skipped: <= >= != ------------------


def test_parity_le() raises:
    var b = _ab_batch()
    assert_parity(
        Le(fcol("a", int64), fcol("b", int64)), dcol("a") <= dcol("b"), b
    )


def test_parity_ge() raises:
    var b = _ab_batch()
    assert_parity(
        Ge(fcol("a", int64), fcol("b", int64)), dcol("a") >= dcol("b"), b
    )


def test_parity_ne() raises:
    var b = _ab_batch()
    assert_parity(
        Ne(fcol("a", int64), fcol("b", int64)), dcol("a") != dcol("b"), b
    )


# --- pow and xor ------------------------------------------------------------


def test_parity_pow() raises:
    var b = _ab_batch()
    assert_parity(
        Pow(fcol("a", int64), fcol("b", int64)), dcol("a") ** dcol("b"), b
    )


def test_parity_xor() raises:
    """`(a < b) ^ (a > b)` — true wherever the two differ."""
    var b = _ab_batch()
    assert_parity(
        Xor(
            Lt(fcol("a", int64), fcol("b", int64)),
            Gt(fcol("a", int64), fcol("b", int64)),
        ),
        (dcol("a") < dcol("b")) ^ (dcol("a") > dcol("b")),
        b,
    )


# --- numeric unaries --------------------------------------------------------


def test_parity_neg() raises:
    var b = _unary_batch()
    assert_parity(Neg(fcol("a", float64)), -dcol("a"), b)


def test_parity_abs() raises:
    var b = _unary_batch()
    assert_parity(Abs(fcol("a", float64)), dcol("a").abs(), b)


def test_parity_sign() raises:
    var b = _unary_batch()
    assert_parity(Sign(fcol("a", float64)), dcol("a").sign(), b)


def test_parity_floor() raises:
    var b = _unary_batch()
    assert_parity(Floor(fcol("a", float64)), dcol("a").floor(), b)


def test_parity_ceil() raises:
    var b = _unary_batch()
    assert_parity(Ceil(fcol("a", float64)), dcol("a").ceil(), b)


def test_parity_round() raises:
    var b = _unary_batch()
    assert_parity(Round(fcol("a", float64)), dcol("a").round(), b)


# --- float unaries ----------------------------------------------------------


def test_parity_sqrt() raises:
    var b = _positive_batch()
    assert_parity(Sqrt(fcol("a", float64)), dcol("a").sqrt(), b)


def test_parity_exp() raises:
    var b = _positive_batch()
    assert_parity(Exp(fcol("a", float64)), dcol("a").exp(), b)


def test_parity_ln() raises:
    var b = _positive_batch()
    assert_parity(Ln(fcol("a", float64)), dcol("a").ln(), b)


# --- string maps ------------------------------------------------------------


def test_parity_upper() raises:
    var b = _smaps_batch()
    assert_parity(Upper(fcol("s", string)), dcol("s").upper(), b)


def test_parity_lower() raises:
    var b = _smaps_batch()
    assert_parity(Lower(fcol("s", string)), dcol("s").lower(), b)


def test_parity_strip() raises:
    var b = _smaps_batch()
    assert_parity(Strip(fcol("s", string)), dcol("s").strip(), b)


def test_parity_lstrip() raises:
    var b = _smaps_batch()
    assert_parity(LStrip(fcol("s", string)), dcol("s").lstrip(), b)


def test_parity_rstrip() raises:
    var b = _smaps_batch()
    assert_parity(RStrip(fcol("s", string)), dcol("s").rstrip(), b)


def test_parity_reverse() raises:
    var b = _smaps_batch()
    assert_parity(Reverse(fcol("s", string)), dcol("s").reverse(), b)


def test_parity_capitalize() raises:
    var b = _smaps_batch()
    assert_parity(Capitalize(fcol("s", string)), dcol("s").capitalize(), b)


# --- string -> numeric, and the remaining string predicates -----------------


def test_parity_length() raises:
    var b = _smaps_batch()
    assert_parity(StringLength(fcol("s", string)), dcol("s").length(), b)


def test_parity_startswith() raises:
    var b = _spair_batch()
    assert_parity(
        StartsWith(fcol("s", string), fcol("p", string)),
        dcol("s").startswith(dcol("p")),
        b,
    )


def test_parity_endswith() raises:
    var b = _spair_batch()
    assert_parity(
        EndsWith(fcol("s", string), fcol("p", string)),
        dcol("s").endswith(dcol("p")),
        b,
    )


def test_parity_contains() raises:
    var b = _spair_batch()
    assert_parity(
        StrContains(fcol("s", string), fcol("p", string)),
        dcol("s").contains(dcol("p")),
        b,
    )

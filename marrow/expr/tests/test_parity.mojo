"""Cross-driver parity harness for the expression system.

Every stable op is expressible two ways: as a fused comptime ``Value`` tree
(``values.mojo``) and as a runtime tag-based ``TagValue`` tree (``dynamic.mojo``).
Both box into the shared ``DynValue`` and expose ``execute(batch) -> DynArray``.
This suite asserts the two drivers agree element-for-element on the same input,
so the runtime interpreter can never silently diverge from the fused algebra it
mirrors.

``assert_parity`` is the reusable primitive: hand it a fused ``Value`` and an
equivalent ``TagValue`` (each implicitly boxed into ``DynValue``) plus a
``RecordBatch``; it runs both and asserts the resulting arrays are equal. The
fused column leaves resolve by name (``col("a", int64)``) and the ``TagValue``
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

from ...kernels.temporal import unit_day

# Fused comptime algebra (values.mojo)
from ...expr.values import (
    _rank,
    DynValue,
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
    _numeric_rank,
    TagValue,
    col as dcol,
    lit as dlit,
    if_else,
)


def assert_fused(
    var fused: DynValue, expected: DynArray, batch: RecordBatch
) raises:
    """Assert a fused node matches an expected array. Used for ops the runtime
    ``TagValue`` interpreter does not yet expose — their cross-driver parity case
    is PENDING T2.2 (which wires the same ops into ``dynamic.mojo``); until then
    we pin the fused result against the kernel's expected output."""
    var actual = fused.execute(batch)
    assert_true(actual == expected)


def assert_parity(
    var fused: DynValue, var dyn: DynValue, batch: RecordBatch
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
        dcol(0) > dcol(1),
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
        dcol(0) + dcol(1),
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
        dcol(0) > dcol(1),
        _int_float_batch(),
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
# fixed in T0.1) that the runtime `TagValue` path already routes through, so the
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
# Ops the runtime `TagValue` interpreter does not yet expose — like/ilike, is_in,
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

        @parameter
        def check[T: NumericType](d: T) raises -> Bool:
            return _rank[T]() == runtime_rank

        assert_true(
            dt.dispatch_numeric[check](),
            String("rank disagreement for ") + String(dt),
        )


# ---------------------------------------------------------------------------
# One node set: the shared fused nodes, instantiated over erased operands
# ---------------------------------------------------------------------------


def test_shared_node_over_erased_operands() raises:
    """`a + b` on two erased operands builds `Add[DynValue, DynValue]` — the
    *same* `Add` the fused lane builds, not a runtime tag.

    This is the whole point of Step 3. `DynValue` conforms to `NumericValue`, so
    `NumericValue.__add__` applies to it unchanged and yields `Add[Self, Rhs]`
    with both parameters erased. The node then takes its dispatch arm because
    `IsErased` propagates from its operands.

    Asserted against the fused tree over the same batch: one node type, two
    lanes, same answer.
    """
    var batch = _ab_batch()

    var lhs: DynValue = dcol(0)
    var rhs: DynValue = dcol(1)
    var shared = lhs + rhs  # Add[DynValue, DynValue]

    assert_parity(fcol("a", int64) + fcol("b", int64), shared^, batch)


def test_shared_node_nests_over_erased_operands() raises:
    """`(a + b) * a` — the composite's own `IsErased` propagates, so an erased
    subtree under another shared node still takes the dispatch arm rather than
    trying to fuse a lane that does not exist.

    Phase 0 found this the hard way: without propagation the outer node fails to
    *instantiate*, because its fused arm elaborates `SIMD[DynType.native, W]`.
    """
    var batch = _ab_batch()

    var lhs: DynValue = dcol(0)
    var rhs: DynValue = dcol(1)
    var again: DynValue = dcol(0)
    var shared = (lhs + rhs) * again

    assert_parity(
        (fcol("a", int64) + fcol("b", int64)) * fcol("a", int64),
        shared^,
        batch,
    )


def test_shared_add_node_concatenates_erased_strings() raises:
    """`+` on two erased operands means *concatenate* when they turn out to be
    strings, and *add* when they turn out to be numbers.

    The choice cannot be made when the tree is built: an erased column's dtype
    is only known once a schema is applied, so `col(0) + col(1)` has no dtype to
    branch on at construction. It is made in the node's erased arm instead,
    against the materialized operands — the same shape `TagValue._compare` uses,
    where an operator names a pair of kernels and the dtype picks one.

    The numeric half of this is `test_shared_node_over_erased_operands`; both go
    through the same `Add[DynValue, DynValue]`.
    """
    var x = array(["a", "c", "e"])
    var y = array(["b", "d", "f"])
    var batch = record_batch([x^, y^], names=["x", "y"])

    var lhs: DynValue = dcol(0)
    var rhs: DynValue = dcol(1)
    var joined: DynValue = lhs + rhs

    var got = joined.execute(batch)
    var want: DynArray = array(["ab", "cd", "ef"])
    assert_true(got == want)


def test_shared_nodes_cover_the_regular_operators() raises:
    """Every regular operator over erased operands goes through the shared node
    and agrees with the fused tree: arithmetic, division, unary, comparison and
    boolean logic.

    One case per node type — `NumericBinary`, `FloatBinary`, `NumericUnary`,
    `NumericCompare`, `BoolBinary`, `BoolUnary` — since each gained its erased
    arm independently and a missing `IsErased` propagation in any of them is a
    build failure rather than a wrong answer.
    """
    var batch = _ab_batch()

    var a: DynValue = dcol(0)
    var b: DynValue = dcol(1)

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


def test_shared_compare_node_compares_erased_strings() raises:
    """`<` on erased string operands takes the string kernel, not the numeric
    one — the node carries both and the runtime dtype picks."""
    var x = array(["a", "d", "c"])
    var y = array(["b", "b", "c"])
    var batch = record_batch([x^, y^], names=["x", "y"])

    var lhs: DynValue = dcol(0)
    var rhs: DynValue = dcol(1)
    var lt: DynValue = lhs < rhs

    var got = lt.execute(batch)
    var want: DynArray = array([True, False, False])
    assert_true(got == want)


def test_shared_payload_nodes_over_erased_operands() raises:
    """The payload-carrying nodes also serve both lanes.

    These are the ones the isolated gates' favourable assumption did not cover:
    they are pipeline *breakers* when fused, so `Value.execute` routes them
    through `prepare` and never calls `materialize` — which is where the erased
    arm lives. Each therefore makes `IsBreaker` follow `IsErased`: an erased node
    has no fused loop to break, it computes the column in one dispatch.

    Covers `StringLength`, `StringPredicate` (like) and `ConditionalBinary`
    (coalesce/nullif); `TemporalExtract` is covered by `test_parity_year`
    alongside its fused twin.
    """
    var s0 = array(["ab", "cde", "f"])
    var s1 = array(["ab", "xy", "f"])
    var sbatch = record_batch([s0^, s1^], names=["s", "t"])

    var s: DynValue = dcol(0)
    var t: DynValue = dcol(1)

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
    var a: DynValue = dcol(0)
    var b: DynValue = dcol(1)
    var nl: DynValue = Nullif(a.copy(), b.copy())
    var got = nl.execute(nbatch)
    # a == b only at index 2 (3 == 3), which nullif turns into a null
    assert_true(got.null_count() == 1)


def test_shared_cast_isin_casewhen_over_erased_operands() raises:
    """The last three payload nodes: `cast`, `is_in` and `case_when`.

    `NumericCast` is the only node whose payload survives erasure — `To` stays a
    comptime type because just the *operand* is erased, so the target dtype is
    known and the runtime cast router does the rest. `IsIn` carries a value-set
    array and `CaseWhen` three child values, and both already worked in
    `DynArray` internally, so their erased arms are the existing helper.
    """
    var batch = _ab_batch()
    var a: DynValue = dcol(0)
    var b: DynValue = dcol(1)

    # NumericCast — erased operand, comptime target dtype
    var casted: DynValue = NumericCast[Float64Type](a.copy())
    var got_cast = casted.execute(batch)
    assert_true(got_cast.dtype() == DynType(Float64Type()))
    assert_true(got_cast.length() == 6)

    # IsIn — payload is a value-set array
    var member: DynValue = IsIn(a.copy(), array([1, 3, 7], int64))
    var want_in: DynArray = array([True, False, True, False, True, False])
    assert_true(member.execute(batch) == want_in)

    # CaseWhen — three erased children
    var picked: DynValue = CaseWhen(a > b, a.copy(), b.copy())
    var want_pick: DynArray = array([9, 5, 3, 10, 7, 8], int64)
    assert_true(picked.execute(batch) == want_pick)


def test_shared_temporal_extract_over_erased_operand() raises:
    """`year()` on an erased column goes through `TemporalExtract` with an
    erased operand.

    This node's arm was written before it could be reached — its operand is
    bound on `TemporalValue`, which `DynValue` could not conform to until
    `DynAgg` took a `DynValue`. It is asserted here now that it can be.
    """
    var ts = array([0, 86_400, 31_536_000], int64)
    var batch = record_batch([ts^], names=["t"])

    var col: DynValue = dcol(0).cast(timestamp(second))
    var yr: DynValue = col.year()

    var want: DynArray = array([1970, 1970, 1971], int32)
    assert_true(yr.execute(batch) == want)

"""The interval algebra: `Ord`, `Bounds` and the six comparison readings.

Two kinds of case here, and the second is the one that matters.

The table-driven cases pin the *stated* rules — the null rule, the unknown
rule, NaN, and the non-monotone cast — each of which is a place where a
plausible implementation is silently wrong rather than obviously wrong.

`test_bounds_readings_are_exact_over_small_integers` then brute-forces the
whole algebra: over a small integer domain an interval is dense, so "could
`x op y` hold for some pair" is decidable by enumeration, and the reading must
agree **exactly** — not merely conservatively. Asserting equality rather than
one-sidedness is what makes the test able to fail: a kernel that answered
`True` unconditionally would pass a one-sided check and is caught here.
"""

from std.testing import assert_equal, assert_false, assert_true

from ..bounds import (
    Bounds,
    EqBounds,
    GeBounds,
    GtBounds,
    LeBounds,
    LtBounds,
    NeBounds,
    Ord,
)


comptime I64 = DType.int64
comptime F64 = DType.float64


def _r(lo: Int, hi: Int) -> Bounds[I64]:
    return Bounds[I64].range(Scalar[I64](lo), Scalar[I64](hi))


def _p(v: Int) -> Bounds[I64]:
    return Bounds[I64].point(Scalar[I64](v))


# ---------------------------------------------------------------------------
# Ord
# ---------------------------------------------------------------------------
def test_bounds_ord_reports_the_three_orderings() raises:
    assert_true(Ord.of(Scalar[I64](1), Scalar[I64](2)) == Ord.lt)
    assert_true(Ord.of(Scalar[I64](2), Scalar[I64](2)) == Ord.eq)
    assert_true(Ord.of(Scalar[I64](3), Scalar[I64](2)) == Ord.gt)


def test_bounds_ord_reports_nan_as_unknown_not_equal() raises:
    """The defect `Interval._three_way` still carries: a three-way written as
    `-1 if x < y else (1 if x > y else 0)` reports an IEEE-unordered pair as
    *equal*, and an "equal" bound is what makes `x < 5` prune a matching group.
    """
    var nan = Scalar[F64](0.0) / Scalar[F64](0.0)
    assert_true(Ord.of(nan, Scalar[F64](1.0)) == Ord.unknown)
    assert_true(Ord.of(Scalar[F64](1.0), nan) == Ord.unknown)
    assert_true(Ord.of(nan, nan) == Ord.unknown)


def test_bounds_ord_unknown_is_permissive_but_never_equal() raises:
    """`unknown` answers yes to all four ordering questions and no to
    equality — the asymmetry that keeps `!=` from pruning on a NaN."""
    assert_true(Ord.unknown.maybe_lt())
    assert_true(Ord.unknown.maybe_le())
    assert_true(Ord.unknown.maybe_gt())
    assert_true(Ord.unknown.maybe_ge())
    assert_false(Ord.unknown.is_eq())


# ---------------------------------------------------------------------------
# Bounds construction
# ---------------------------------------------------------------------------
def test_bounds_range_rejects_an_inverted_interval() raises:
    assert_true(_r(0, 9).known)
    assert_false(_r(9, 0).known)


def test_bounds_range_rejects_a_nan_endpoint() raises:
    """`lo <= hi` is false for any NaN endpoint, so the single well-formedness
    test doubles as the NaN guard and no comparison ever sees one."""
    var nan = Scalar[F64](0.0) / Scalar[F64](0.0)
    assert_false(Bounds[F64].range(nan, Scalar[F64](1.0)).known)
    assert_false(Bounds[F64].range(Scalar[F64](0.0), nan).known)


def test_bounds_unknown_and_null_are_different_answers() raises:
    """`unknown` proves nothing; `null` proves `never`. Collapsing the two is
    how a pruner either stops pruning or starts dropping rows."""
    assert_false(Bounds[I64].unknown().known)
    assert_false(Bounds[I64].unknown().all_null)
    assert_false(Bounds[I64].null().known)
    assert_true(Bounds[I64].null().all_null)


# ---------------------------------------------------------------------------
# The two guards, which every kernel inherits
# ---------------------------------------------------------------------------
def test_bounds_all_null_proves_never_for_every_comparison() raises:
    """A NULL operand makes a comparison NULL, and a filter keeps only rows
    whose mask bit is valid *and* true. So no row of an all-null granule can
    survive any comparison — the one exactly provable prune that needs no
    bound at all."""
    var n = Bounds[I64].null()
    var v = _r(0, 100)
    assert_false(LtBounds.maybe[I64](n, v))
    assert_false(LeBounds.maybe[I64](n, v))
    assert_false(GtBounds.maybe[I64](n, v))
    assert_false(GeBounds.maybe[I64](n, v))
    assert_false(EqBounds.maybe[I64](n, v))
    assert_false(NeBounds.maybe[I64](n, v))
    # and symmetrically on the right
    assert_false(GtBounds.maybe[I64](v, n))
    assert_false(NeBounds.maybe[I64](v, n))


def test_bounds_an_unknown_side_prunes_nothing() raises:
    """Absence of a statistic must never be read as a zero, an empty range or
    a sentinel. Every reading answers `maybe`."""
    var u = Bounds[I64].unknown()
    var v = _r(0, 100)
    assert_true(LtBounds.maybe[I64](u, v))
    assert_true(LeBounds.maybe[I64](u, v))
    assert_true(GtBounds.maybe[I64](u, v))
    assert_true(GeBounds.maybe[I64](u, v))
    assert_true(EqBounds.maybe[I64](u, v))
    assert_true(NeBounds.maybe[I64](u, v))
    assert_true(GtBounds.maybe[I64](v, u))


# ---------------------------------------------------------------------------
# The readings themselves
# ---------------------------------------------------------------------------
def test_bounds_greater_reads_the_upper_bound() raises:
    """The ClickBench shape: `x > 5` over `[0, 3]` is provably empty, over
    `[0, 9]` is not."""
    assert_false(GtBounds.maybe[I64](_r(0, 3), _p(5)))
    assert_true(GtBounds.maybe[I64](_r(0, 9), _p(5)))
    # the boundary: `x > 5` needs max strictly above 5
    assert_false(GtBounds.maybe[I64](_r(0, 5), _p(5)))
    assert_true(GeBounds.maybe[I64](_r(0, 5), _p(5)))


def test_bounds_less_reads_the_lower_bound() raises:
    assert_false(LtBounds.maybe[I64](_r(7, 9), _p(5)))
    assert_true(LtBounds.maybe[I64](_r(1, 9), _p(5)))
    assert_false(LtBounds.maybe[I64](_r(5, 9), _p(5)))
    assert_true(LeBounds.maybe[I64](_r(5, 9), _p(5)))


def test_bounds_equal_is_interval_overlap() raises:
    assert_true(EqBounds.maybe[I64](_r(0, 9), _p(5)))
    assert_false(EqBounds.maybe[I64](_r(0, 4), _p(5)))
    assert_false(EqBounds.maybe[I64](_r(6, 9), _p(5)))
    assert_true(EqBounds.maybe[I64](_r(0, 9), _r(9, 20)))
    assert_false(EqBounds.maybe[I64](_r(0, 9), _r(10, 20)))


def test_bounds_not_equal_prunes_only_two_identical_points() raises:
    """`interval.mojo` answered `True` unconditionally here. The point case is
    two `Ord` reads and is exactly sound, and it is what makes `!=` a real
    reading rather than a placeholder that no test can distinguish from a bug.
    """
    assert_false(NeBounds.maybe[I64](_p(5), _p(5)))
    assert_true(NeBounds.maybe[I64](_p(5), _p(6)))
    assert_true(NeBounds.maybe[I64](_r(5, 6), _p(5)))
    assert_true(NeBounds.maybe[I64](_p(5), _r(5, 6)))


# ---------------------------------------------------------------------------
# cast — where a plausible implementation is unsound
# ---------------------------------------------------------------------------
def test_bounds_cast_widens_monotonically() raises:
    var b = Bounds[DType.int32].range(
        Scalar[DType.int32](-5), Scalar[DType.int32](7)
    )
    var w = b.cast[I64]()
    assert_true(w.known)
    assert_equal(Int(w.lo), -5)
    assert_equal(Int(w.hi), 7)


def test_bounds_cast_refuses_a_non_monotone_reinterpret() raises:
    """`promote[Int64Type, UInt64Type]` resolves to `Int64Type`, so a `uint64`
    operand is *reinterpreted* rather than widened — and that cast is not
    monotone.

    A column holding `{1, 2**63}` has bounds `[1, 2**63]`. Cast to `int64` they
    become `[1, -2**63]`, an inverted interval, and reading `x > 0` off it
    would answer `hi > 0` = `False` and skip a group whose first row matches.
    Routing the cast result through `range` catches it, and exactly: the
    rotation inverts the interval **iff** it straddles `2**63`.
    """
    var straddles = Bounds[DType.uint64].range(
        Scalar[DType.uint64](1), Scalar[DType.uint64](1 << 63)
    )
    assert_true(straddles.known)
    assert_false(straddles.cast[I64]().known)
    assert_true(GtBounds.maybe[I64](straddles.cast[I64](), _p(0)))

    # Entirely below 2**63: the cast is order-preserving there, so the bound
    # survives and still prunes.
    var below = Bounds[DType.uint64].range(
        Scalar[DType.uint64](1), Scalar[DType.uint64](9)
    )
    assert_true(below.cast[I64]().known)
    assert_false(GtBounds.maybe[I64](below.cast[I64](), _p(100)))


def test_bounds_cast_preserves_the_two_special_answers() raises:
    assert_true(Bounds[DType.int32].null().cast[I64]().all_null)
    assert_false(Bounds[DType.int32].unknown().cast[I64]().known)
    assert_false(Bounds[DType.int32].unknown().cast[I64]().all_null)


# ---------------------------------------------------------------------------
# Exactness, by enumeration
# ---------------------------------------------------------------------------
def _brute[op: Int](a: Int, b: Int, c: Int, d: Int) -> Bool:
    """Does some pair `(x, y)` with `a <= x <= b` and `c <= y <= d` satisfy the
    operator? Decidable by enumeration because integer intervals are dense."""
    for x in range(a, b + 1):
        for y in range(c, d + 1):
            comptime if op == 0:
                if x < y:
                    return True
            elif op == 1:
                if x <= y:
                    return True
            elif op == 2:
                if x > y:
                    return True
            elif op == 3:
                if x >= y:
                    return True
            elif op == 4:
                if x == y:
                    return True
            else:
                if x != y:
                    return True
    return False


def test_bounds_readings_are_exact_over_small_integers() raises:
    """Every reading, against brute force, over every interval pair in
    `[-3, 3]`.

    Asserted as **equality**, not one-sidedness. One-sidedness alone is
    satisfied by a kernel that always answers `True`, so it cannot fail; this
    pins each reading to the exact set of prunable configurations, which is
    what makes a sign error in `maybe_lt`/`maybe_le` visible.
    """
    var checked = 0
    for a in range(-3, 4):
        for b in range(a, 4):
            for c in range(-3, 4):
                for d in range(c, 4):
                    var l = _r(a, b)
                    var r = _r(c, d)
                    assert_equal(
                        LtBounds.maybe[I64](l, r), _brute[0](a, b, c, d)
                    )
                    assert_equal(
                        LeBounds.maybe[I64](l, r), _brute[1](a, b, c, d)
                    )
                    assert_equal(
                        GtBounds.maybe[I64](l, r), _brute[2](a, b, c, d)
                    )
                    assert_equal(
                        GeBounds.maybe[I64](l, r), _brute[3](a, b, c, d)
                    )
                    assert_equal(
                        EqBounds.maybe[I64](l, r), _brute[4](a, b, c, d)
                    )
                    assert_equal(
                        NeBounds.maybe[I64](l, r), _brute[5](a, b, c, d)
                    )
                    checked += 1
    assert_true(checked == 28 * 28)

"""Bounds kernels — marrow's comparisons read over `[lo, hi]` intervals instead
of over values.

Every comparison has two readings. `LtKernel.core` answers `a < b` for a SIMD
lane of concrete values; `LtBounds` answers *could* `a < b` be true for **some**
pair drawn from two intervals. Same operator, a different domain — so it is a
different kernel, not a method on the SIMD one.

That separation has a precedent here. `NumericCompareKernel` used to carry a
`comptime StringKernel` naming its string counterpart, and it was removed
because "which family `a < b` means is a question about the operands, and it
belongs to whoever is interpreting the operator, not to the SIMD kernel". An
interval reading is the same kind of claim, so it lives beside the SIMD kernels
rather than inside them, **and the expression node pairs the two**.

**The error is one-sided, and every line here exists to keep it that way.** A
caller may only skip data it has *proven* cannot match: a wrong "maybe" costs
time, a wrong "no" costs correctness. So the only question a kernel answers is
*could this be TRUE here?*, and every unknown resolves to `True`.

"""

from .core import Kernel


# ---------------------------------------------------------------------------
# Ord — a three-way comparison whose answer may be unknown
# ---------------------------------------------------------------------------
struct Ord(Copyable, Equatable, ImplicitlyCopyable, Movable, Writable):
    """The result of comparing two *bounds*: less, equal, greater, or unknown.

    A value type rather than an `Optional[Int]`, for the reason `Shape` is one:
    the four cases have names and the "unknown" case has a rule (it is
    permissive for every ordering question and never equal), which a bare
    integer would leave to a comment at each of six call sites.

    **This type is what lets one algebra serve both expression lanes.** The
    comptime lane compares two `Scalar[dt]` with no dispatch; the runtime lane
    compares two `DynScalar` through *one* dtype ladder. Both produce an `Ord`,
    and the six readings below are then the same six lines for both. Writing
    the readings twice — once typed, once erased — is how the two lanes drift
    into disagreeing about which row groups to skip, which is the failure mode
    `2026-08-24-expr2-pruning-pushdown-design.md` §8 stage 6 calls "the
    dangerous case".
    """

    var _v: Int8

    comptime lt = Ord(-1)
    comptime eq = Ord(0)
    comptime gt = Ord(1)
    comptime unknown = Ord(2)
    """Not comparable — a NaN, or two erased scalars of different dtypes.

    Distinct from `eq` on purpose: `x != y` is prunable only when the two sides
    are provably the *same* point, and an unknown comparison proves nothing.
    """

    def __init__(out self, v: Int8):
        self._v = v

    @staticmethod
    def of[dt: DType](a: Scalar[dt], b: Scalar[dt]) -> Self:
        """Compare two concrete bounds. IEEE-unordered pairs answer `unknown`.

        The trailing `else` is not dead: for a floating `dt` with `a` or `b`
        NaN, all three of `<`, `>` and `==` are `False`. Reporting that as
        `eq` — which a `-1 if a < b else (1 if a > b else 0)` three-way does —
        is exactly the defect recorded against `Interval._three_way`.
        """
        if a < b:
            return Ord.lt
        elif a > b:
            return Ord.gt
        elif a == b:
            return Ord.eq
        else:
            return Ord.unknown

    def __eq__(self, other: Self) -> Bool:
        return self._v == other._v

    def __ne__(self, other: Self) -> Bool:
        return self._v != other._v

    # -- the one-sided questions --------------------------------------------
    #
    # Each answers "could this ordering hold?", so `unknown` is `True` for all
    # four of them and `False` for `is_eq`, which asks whether it is *proven*.

    def maybe_lt(self) -> Bool:
        return self != Ord.eq and self != Ord.gt

    def maybe_le(self) -> Bool:
        return self != Ord.gt

    def maybe_gt(self) -> Bool:
        return self != Ord.eq and self != Ord.lt

    def maybe_ge(self) -> Bool:
        return self != Ord.lt

    def is_eq(self) -> Bool:
        """Proven equal. `unknown` answers `False` — the one place where the
        conservative direction is *away* from equality."""
        return self == Ord.eq

    def write_to[W: Writer](self, mut writer: W):
        if self == Ord.lt:
            writer.write("lt")
        elif self == Ord.eq:
            writer.write("eq")
        elif self == Ord.gt:
            writer.write("gt")
        else:
            writer.write("unknown")


# ---------------------------------------------------------------------------
# Bounds — what one typed sub-expression's value can be, over one granule
# ---------------------------------------------------------------------------
struct Bounds[dt: DType](Copyable, ImplicitlyCopyable, Movable, Writable):
    """What one sub-expression's value can be over one granule (a row group, a
    page, a partition — this type does not know which).

    Four register-sized fields and nothing else: no `Optional[DynScalar]`, no
    heap, no `ArcPointer`, no `Variant`. That is the whole reason a fused
    predicate can be pruned without dispatch, and it makes `Bounds[dt]` the
    pruning-domain twin of the `SIMD[dt, W]` a lane produces.

    **`lo`/`hi` bracket the granule's *non-null* values only**, which is what
    Parquet's `min`/`max` mean (`parquet/statistics.h:135-176`: computed over
    non-null values, and `HasMinMax()` false means no usable bound). Nulls are
    carried separately by `all_null`; a granule that is *partly* null needs no
    flag, because a null row makes a comparison `NULL` and a filter keeps only
    valid `TRUE`, so such a row can never be the reason to keep a granule.
    """

    var lo: Scalar[Self.dt]
    var hi: Scalar[Self.dt]

    var known: Bool
    """False means `lo`/`hi` are meaningless and this bounds nothing.

    Set by `range` when `lo <= hi` fails, which folds three separate causes into
    one flag: an absent statistic, a NaN bound (no IEEE comparison against NaN
    is true, so `lo <= hi` is `False`), and a cast that inverted the interval
    (see `cast`).
    """

    var all_null: Bool
    """Every value in the granule is NULL.

    Worth its own flag rather than being folded into `known`: it is the one
    fact that proves a comparison `never`, where `not known` proves nothing.
    Detectable exactly from `null_count == num_rows`, and from
    `ColumnIndex.null_pages` at page granularity.
    """

    def __init__(
        out self,
        lo: Scalar[Self.dt],
        hi: Scalar[Self.dt],
        known: Bool,
        all_null: Bool,
    ):
        self.lo = lo
        self.hi = hi
        self.known = known
        self.all_null = all_null

    @staticmethod
    def unknown() -> Self:
        """No usable bound — the conservative answer, and the default every
        node that cannot do better returns."""
        return Self(Scalar[Self.dt](0), Scalar[Self.dt](0), False, False)

    @staticmethod
    def null() -> Self:
        """Every value is NULL. Proves `never` for any comparison."""
        return Self(Scalar[Self.dt](0), Scalar[Self.dt](0), False, True)

    @staticmethod
    def point(v: Scalar[Self.dt]) -> Self:
        """A single known value — a literal, or a bound parameter."""
        return Self(v, v, True, False)

    @staticmethod
    def range(lo: Scalar[Self.dt], hi: Scalar[Self.dt]) -> Self:
        """A value known to lie in `[lo, hi]`, or `unknown` if that is not a
        well-formed interval.

        The `lo <= hi` test is the module's single NaN guard: for a floating
        `dt`, any NaN endpoint makes it `False` and the bound reads as unknown
        rather than being compared. Marrow's writer already skips NaN when it
        *computes* bounds (`parquet/statistics.mojo:101-119`); this is the
        reader-side mirror, and it also covers files other writers produced.
        """
        return Self(lo, hi, lo <= hi, False)

    def cast[to: DType](self) -> Bounds[to]:
        """These bounds in another representation, as the lane's operand
        promotion would produce them.

        **Sound only because the result is re-checked with `range`.** A
        comparison node casts both operands to a common type before comparing,
        and the interval reading has to make the identical cast or it is
        answering a different question. That is safe when the cast is monotone
        non-decreasing — every widening and every int->float rounding is — and
        *unsafe* when it is not.

        The one non-monotone cast marrow can produce is real and is not
        hypothetical: `promote[Int64Type, UInt64Type]` resolves to `Int64Type`
        (`comptime/rules.mojo`'s `_outranks` falls through to
        `bit_width_of[int64]() >= bit_width_of[uint64]()`), so a `uint64`
        operand is *reinterpreted*, not widened. A column holding `{1, 2**63}`
        then has bounds `[1, 2**63]` that cast to `[1, -2**63]` — an inverted
        interval. Reading `x > 0` off it would answer `hi > 0` = `False` and
        skip a group whose first row matches: a false negative, the one error
        class pruning may never produce.

        Routing the result through `range` catches exactly that case, because
        the u64->i64 rotation inverts the interval **iff** it straddles `2**63`,
        which is **iff** `cast(lo) > cast(hi)`. So the check is not merely
        conservative here, it is exact.

        (The underlying `promote` behaviour is a pre-existing defect of the
        comparison itself and is not fixed here; pruning simply refuses to
        speak where it applies.)
        """
        if self.all_null:
            return Bounds[to].null()
        elif not self.known:
            return Bounds[to].unknown()
        else:
            return Bounds[to].range(self.lo.cast[to](), self.hi.cast[to]())

    def write_to[W: Writer](self, mut writer: W):
        if self.all_null:
            writer.write("bounds(all-null)")
        elif not self.known:
            writer.write("bounds(unknown)")
        else:
            writer.write("bounds[", self.lo, ", ", self.hi, "]")


# ---------------------------------------------------------------------------
# BoundsKernel — one operator, read over bounds
# ---------------------------------------------------------------------------
trait BoundsKernel(Kernel):
    """One comparison, read over intervals: *could* it be true for some pair?

    A conformer writes **one** method, `decide`, over two `Ord`s. Everything
    else — the null rule, the unknown rule, and the extraction of the two
    orderings from a typed pair of bounds — is the `maybe` default below, so
    the six conformers cannot disagree about them.

    `maybe` is parameterised exactly as `NumericCompareKernel.core[dt, W]` is,
    for the same reason: the operand type is known at the node, so the reading
    should be too.
    """

    @staticmethod
    def decide(lo_hi: Ord, hi_lo: Ord) -> Bool:
        """Could this operator hold for some pair, given
        `lo_hi = cmp(l.lo, r.hi)` and `hi_lo = cmp(l.hi, r.lo)`?

        Those two orderings are the only information any of the six needs, and
        that is not a coincidence: a comparison is monotone in each operand, so
        whether it *can* hold over two intervals is decided at the extreme pair
        — the smallest left against the largest right, and the largest left
        against the smallest right.

        Called only after `maybe` has established that both sides are known and
        neither is all-null, so a conformer never repeats those checks.
        """
        ...

    @staticmethod
    def maybe[dt: DType](l: Bounds[dt], r: Bounds[dt]) -> Bool:
        """Could `l <op> r` be TRUE for some pair drawn from the two granules?

        A default, not an abstract member: the two guards below are where a
        hand-written conformer would silently go wrong, and there is exactly
        one correct spelling of each.

        - **All-null on either side proves `False`.** Every comparison with a
          NULL operand evaluates to NULL, and `FilterOperator` keeps only rows
          whose mask bit is valid *and* true, so no row of such a granule can
          survive. This is the one exactly-provable prune that needs no bound
          at all.
        - **An unknown bound on either side proves nothing**, so the answer is
          `True`. Absence of a statistic must never be read as a zero, an
          empty range, or a sentinel.
        """
        if l.all_null or r.all_null:
            return False
        elif not l.known or not r.known:
            return True
        else:
            return Self.decide(Ord.of(l.lo, r.hi), Ord.of(l.hi, r.lo))


struct LtBounds(BoundsKernel):
    """`l < r` — possible iff `min(l) < max(r)`.

    Every non-null `x` in `l` satisfies `l.lo <= x`, and every `y` in `r`
    satisfies `y <= r.hi`; so if `l.lo >= r.hi` then `x >= y` for every pair
    and no row can satisfy `x < y`. Conversely `l.lo < r.hi` is witnessed by
    the extreme pair, which the bounds assert exists.
    """

    comptime name = "less"

    @staticmethod
    def decide(lo_hi: Ord, hi_lo: Ord) -> Bool:
        return lo_hi.maybe_lt()


struct LeBounds(BoundsKernel):
    """`l <= r` — possible iff `min(l) <= max(r)`. Same argument as `LtBounds`
    with the non-strict ordering."""

    comptime name = "less_equal"

    @staticmethod
    def decide(lo_hi: Ord, hi_lo: Ord) -> Bool:
        return lo_hi.maybe_le()


struct GtBounds(BoundsKernel):
    """`l > r` — possible iff `max(l) > min(r)`.

    Mirror of `LtBounds`: if `l.hi <= r.lo` then `x <= y` for every pair, so no
    row satisfies `x > y`; and `l.hi > r.lo` is witnessed by the extreme pair.
    """

    comptime name = "greater"

    @staticmethod
    def decide(lo_hi: Ord, hi_lo: Ord) -> Bool:
        return hi_lo.maybe_gt()


struct GeBounds(BoundsKernel):
    """`l >= r` — possible iff `max(l) >= min(r)`. Same argument as `GtBounds`
    with the non-strict ordering."""

    comptime name = "greater_equal"

    @staticmethod
    def decide(lo_hi: Ord, hi_lo: Ord) -> Bool:
        return hi_lo.maybe_ge()


struct EqBounds(BoundsKernel):
    """`l == r` — possible iff the two intervals overlap.

    Two intervals are disjoint iff one lies entirely below the other, i.e. iff
    `l.hi < r.lo` or `l.lo > r.hi`; overlap is the negation, which is exactly
    `l.lo <= r.hi and l.hi >= r.lo`. Overlap does not prove a *shared value*
    exists, which is why this is one-sided and why it is `maybe`, not "yes".
    """

    comptime name = "equal"

    @staticmethod
    def decide(lo_hi: Ord, hi_lo: Ord) -> Bool:
        return lo_hi.maybe_le() and hi_lo.maybe_ge()


struct NeBounds(BoundsKernel):
    """`l != r` — possible unless both sides are provably the same single value.

    `l.lo == r.hi` and `l.hi == r.lo`, given `l.lo <= l.hi` and `r.lo <= r.hi`,
    forces `l.lo == l.hi == r.lo == r.hi`: every non-null pair is equal, so no
    row can differ. Any other configuration admits a differing pair.

    `interval.mojo` answered `True` unconditionally here, calling the point case
    "not worth a special case". It is two `Ord` reads, it is exactly sound, and
    it is the difference between `!=` being a real reading and a placeholder —
    which matters for a generative soundness test, where a kernel that always
    says "maybe" proves nothing.
    """

    comptime name = "not_equal"

    @staticmethod
    def decide(lo_hi: Ord, hi_lo: Ord) -> Bool:
        return not (lo_hi.is_eq() and hi_lo.is_eq())

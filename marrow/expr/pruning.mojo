"""Statistics-based pruning: evaluating a predicate over a *granule* instead of
over rows.

Given only what a row group, a page or a partition can say about itself — each
column's `[min, max]` and its null count — could this predicate be `TRUE` for
**some** row it covers? A proven "no" lets a source skip the granule without
decoding it.

**Pruning is `bind`/`lane` in a second domain.** A node answers for itself
through `prune`/`bounds`, typed by the same comptime parameters that type
`lane`, folded by the same promotion, over `Bounds[dt]` instead of
`SIMD[dt, W]`. That is the argument for this shape over every alternative: it
is not a new mechanism, it is the existing one read differently.

| fusion | pruning |
|---|---|
| `bind(batch, bindings) -> Self.Bound` | `bounds(stats, bindings) -> Bounds[Self.Type.native]` |
| `lane[W](bound, i) -> SIMD[Self.Type.native, W]` | folded inside `bounds` |
| `.cast[Self.ArgType.native]()` at a compare | `.cast[Self.ArgType.native]()` at a compare |
| `validity(bound)` | `Bounds.all_null` |

# The error is one-sided, and that is the whole design

Pruning may only produce **false positives**, never false negatives. Reading a
granule that turns out to hold no matching row costs time; skipping one that
held a match changes the answer.

So **the only question a node ever answers is "could this be TRUE here?"**.
There is no "could this be FALSE" and no "is this definitely TRUE". That is not
a simplification for v1 — it is what keeps the algebra sound with two lines of
proof per node, and it is why `NOT` and `XOR` prune nothing (negating "maybe
true" is not "maybe false", and inventing that answer is the classic way a
pruner goes silently wrong).

It also dissolves two things a two-sided lattice would have to carry.
ClickHouse's `BoolMask` needs `can_be_false` to make `NOT` composable, and it
needs a per-atom `relaxed` bit forced on by any *query-side* widening — an
inverted transform, a tokenizer that drops boundary tokens — because such a
widening is sound for `can_be_true` and unsound for `can_be_false`
(`KeyCondition.cpp:5581-5589`). With one side there is no false half to be
unsound about, so neither bit exists here. Both become necessary the day
`Truth` grows a second field, and not before.

# What the statistics actually mean

Consulted, and each rule below is a line of code somewhere in this file:

- **Parquet** (`cpp/src/parquet/statistics.h:135-176`): `null_count`,
  `has_min_max` and `has_null_count` are **independent**. `min`/`max` are
  computed over **non-null** values only, and absent min/max means no usable
  bound — not an empty range.
- **arrow-rs** (`parquet/src/arrow/arrow_reader/statistics.rs:1439-1471`)
  carries `missing_null_counts_as_zero` as an explicit *option*, because
  treating an absent null count as zero is a **soundness choice**, not a
  default. Here an absent null count is `-1` and reads as *unknown*, so
  `all_null` is never inferred from it.
- **NaN bounds bound nothing.** Marrow's writer already computes float bounds
  with `skip_nan=True` and normalises signed zero
  (`parquet/statistics.mojo:101-119`); `Bounds.range` is the reader-side
  mirror, and it also covers files other writers produced.
- **Truncated bounds are fine.** arrow-rs exposes `is_min_value_exact` /
  `is_max_value_exact`; range pruning does not need them, because a truncated
  bound only ever *widens* the interval. (A bloom-filter equality reading would
  need them; there is none here — see below.)

# Scope, and what is deliberately absent

`Truth` has two states; there is no `Facts`/`Index` trait, no bloom reading,
and no page-level or partition-level machinery. Each was designed and cut on
measurements rather than taste — scalar blooms add +0.0% over min/max on the
compound ClickBench queries, and an index trait would have zero methods with
one implementation. The measurements, and the honest ceiling on the data
marrow has (1.04x, against 3.6x for projection pushdown), are in
`docs/superpowers/specs/2026-08-27-index-and-pruning-plan.md` §0 and §5.
"""

from std.memory import ArcPointer

from ..dtypes import BoolType, DynType, NumericType, PrimitiveType
from ..kernels.bounds import Bounds, BoundsKernel, Ord
from ..scalars import DynScalar
from .bindings import Bindings


# ---------------------------------------------------------------------------
# Truth — the one-sided answer
# ---------------------------------------------------------------------------
struct Truth(Copyable, Equatable, ImplicitlyCopyable, Movable, Writable):
    """Could this predicate be `TRUE` for some row of this granule?

    Two states, and the asymmetry is the point: `never` is a *proof*, `maybe`
    is the absence of one. A value type rather than a bare `Bool` for the
    reason `Shape` is one — `True` and `False` are interchangeable to a reader,
    and getting the polarity backwards here silently drops rows.
    """

    var _maybe: Bool

    comptime never = Truth(False)
    """Proven: no row covered by these statistics can satisfy the predicate."""

    comptime maybe = Truth(True)
    """Not proven otherwise. The answer every node that cannot do better
    gives, and the only answer that is always correct."""

    def __init__(out self, maybe: Bool):
        self._maybe = maybe

    def __eq__(self, other: Self) -> Bool:
        return self._maybe == other._maybe

    def __ne__(self, other: Self) -> Bool:
        return self._maybe != other._maybe

    def __bool__(self) -> Bool:
        """`True` when the granule must be read. Reads as "could be true"."""
        return self._maybe

    def __and__(self, other: Self) -> Self:
        """`a AND b` — provably false as soon as either conjunct is.

        Sound because `AND` is Kleene here and a filter keeps only valid
        `TRUE`: if no row can make `a` true, no row can make `a AND b` true.
        """
        return Truth(self._maybe and other._maybe)

    def __or__(self, other: Self) -> Self:
        """`a OR b` — provably false only when *both* disjuncts are.

        This is what structurally prevents the classic sub-term-index defect:
        an index that disproves one disjunct while the other is unknown yields
        `never | maybe = maybe`, so it cannot skip a granule full of matches.
        """
        return Truth(self._maybe or other._maybe)

    def write_to[W: Writer](self, mut writer: W):
        writer.write("maybe" if self._maybe else "never")


# ---------------------------------------------------------------------------
# PruneStats — what one granule says about itself
# ---------------------------------------------------------------------------
struct ColumnBounds(Copyable, Movable):
    """One column's summary over one granule.

    Three fields mirroring Parquet's own independence rules: either bound may
    be absent on its own, and `null_count` is `-1` when the file did not store
    one. Deliberately *not* `parquet.ColumnStatistics`, so this module — and
    everything that prunes — stays free of a Parquet dependency and can be fed
    by an in-memory source, an Iceberg manifest, or a test.
    """

    var lo: Optional[DynScalar]
    var hi: Optional[DynScalar]
    var null_count: Int
    """Nulls in this column over this granule, or `-1` for *unknown*.

    Never silently zero. arrow-rs makes "treat missing as zero" an explicit
    option because it is a soundness choice; here the choice is made once, in
    favour of unknown, and `PruneStats.all_null` is the only reader of it.
    """

    def __init__(
        out self,
        var lo: Optional[DynScalar],
        var hi: Optional[DynScalar],
        null_count: Int,
    ):
        self.lo = lo^
        self.hi = hi^
        self.null_count = null_count


struct PruneStats(Copyable, Movable):
    """One granule's per-column bounds and null counts, keyed by column name.

    **A granule is whatever built this decided it was** — a row group, a page,
    a partition, a whole file. This type never says which, because whoever
    constructed it already knows which rows it covers. That is why there is no
    `parts()` and no container-identity problem: two `PruneStats` are never
    combined here.

    Keyed by *name*, not by position, because the expression that reads it
    knows a name and nothing else. Lookup is a linear scan, which is free at
    one call per leaf per row group; it would want an index only at page
    granularity over a wide schema.

    Both lists are capacity-reserved at construction. That is not a
    micro-optimisation: `List` reallocation is the trigger for the variant
    layout miscompile CLAUDE.md records, and `ColumnBounds` holds two
    `Optional[DynScalar]`. `test_pruning.mojo` asserts the alignment that makes
    it moot on the current tree; reserving keeps it moot if that changes.
    """

    var _names: List[String]
    var _cols: List[ColumnBounds]
    var _num_rows: Int
    """Rows in this granule, or `-1` when unknown. Read only to decide
    `all_null`; an unknown row count makes that unprovable, never false."""

    def __init__(out self, num_rows: Int = -1, capacity: Int = 0):
        self._names = List[String](capacity=capacity)
        self._cols = List[ColumnBounds](capacity=capacity)
        self._num_rows = num_rows

    def add(
        mut self,
        var name: String,
        var lo: Optional[DynScalar],
        var hi: Optional[DynScalar],
        null_count: Int = -1,
    ):
        """Record one column's summary. A column never added simply has no
        statistic, which prunes nothing."""
        self._names.append(name^)
        self._cols.append(ColumnBounds(lo^, hi^, null_count))

    def num_rows(self) -> Int:
        return self._num_rows

    def num_columns(self) -> Int:
        return len(self._cols)

    def index_of(self, name: String) -> Int:
        """This column's position, or `-1`. Non-raising: an unknown column is
        a missing statistic, not an error — the filter above the source still
        resolves the name and still raises there if it is genuinely wrong."""
        for i in range(len(self._names)):
            if self._names[i] == name:
                return i
        return -1

    def null_count(self, i: Int) -> Int:
        return self._cols[i].null_count

    def all_null(self, i: Int) -> Bool:
        """Every value in this column is NULL, provably.

        Needs both a stored null count and a known row count. `null_count == -1`
        is *unknown*, so it answers `False` — reading an absent count as zero
        would be a soundness choice this module declines to make.
        """
        return (
            self._num_rows >= 0
            and self._cols[i].null_count >= 0
            and self._cols[i].null_count == self._num_rows
        )

    # -- the typed reading: what the comptime lane uses ----------------------

    def bounds[T: NumericType](self, name: String) -> Bounds[T.native]:
        """One column's bounds in the type the reading node expects.

        **The dtype equality is a safety check, not a nicety.**
        `DynScalar.as_primitive[T]()` is a `debug_assert`, so a mismatch aborts
        the process rather than raising — the failure mode
        `comptime/boolean.mojo` documents at length ("the abort took down the
        whole test runner, failing all seven cases in the file rather than the
        one that was wrong"). Comparing `DynType(T())` against the stored
        scalar's own type is what makes the unwrap provably safe: a
        `PrimitiveScalar[T]` is the only variant member that reports
        `DynType(T())`, and it is what the file's own leaf type produced.

        Bound on `NumericType` rather than `PrimitiveType` for exactly that
        reason: `T()` needs `Defaultable`, which `NumericType` has and
        `TemporalType`/`DecimalType` do not. A temporal column therefore
        inherits `unknown` today. That is a real gap and it is cheap to close
        — the guard becomes a comparison against the granule's own field dtype,
        which `PruneStats` would have to start carrying — but it buys nothing
        measurable first: on `hits_0.parquet` the whole file is a single day,
        so `EventDate` bounds prune 0 of 2 row groups.
        """
        var i = self.index_of(name)
        if i < 0:
            return Bounds[T.native].unknown()
        elif self.all_null(i):
            return Bounds[T.native].null()
        else:
            return _typed_range[T](self._cols[i].lo, self._cols[i].hi)

    def bool_truth(self, name: String) -> Truth:
        """Could a bare `bool` column be `TRUE` for some row?

        Two provable cases, and both are cheap: an all-null column has no
        `TRUE` row, and a column whose `max` is `False` has none either. Bool
        bounds are `BoolScalar`, not `PrimitiveScalar[BoolType]` — bool is
        bit-packed and `BoolType` deliberately does not conform to
        `PrimitiveType` — so this cannot go through `bounds[T]`.
        """
        var i = self.index_of(name)
        if i < 0:
            return Truth.maybe
        elif self.all_null(i):
            return Truth.never
        else:
            ref hi = self._cols[i].hi
            if not hi:
                return Truth.maybe
            elif hi.value().type() != DynType(BoolType()):
                return Truth.maybe
            elif not hi.value().is_valid():
                return Truth.maybe
            else:
                return Truth(hi.value().as_bool().value())

    # -- the erased reading: what the runtime lane uses ----------------------

    def dyn_bounds(self, name: String) -> DynBounds:
        """One column's bounds, left erased.

        The runtime lane has no comptime type to unwrap into, so it carries
        `DynScalar` endpoints and pays one dtype ladder per comparison — see
        `compare_dyn`. That is the same asymmetry the lane already carries
        everywhere else, and it is confined to binaries that build expressions
        at run time.
        """
        var i = self.index_of(name)
        if i < 0:
            return DynBounds.unknown()
        elif self.all_null(i):
            return DynBounds.null()
        else:
            return DynBounds(
                self._cols[i].lo.copy(), self._cols[i].hi.copy(), False
            )


def _typed_range[
    T: NumericType
](lo: Optional[DynScalar], hi: Optional[DynScalar]) -> Bounds[T.native]:
    """Unwrap two erased bounds into `Bounds[T.native]`, or `unknown`.

    Every guard here removes one way of turning an absent or foreign statistic
    into a confident wrong answer: a half-present bound is no bound (Parquet's
    flags are independent), a null-valued bound carries no value, and a dtype
    that is not `T` would make the unwrap a process abort.
    """
    if not lo or not hi:
        return Bounds[T.native].unknown()
    var want = DynType(T())
    if lo.value().type() != want or hi.value().type() != want:
        return Bounds[T.native].unknown()
    elif not lo.value().is_valid() or not hi.value().is_valid():
        return Bounds[T.native].unknown()
    else:
        return Bounds[T.native].range(
            lo.value().as_primitive[T]().value(),
            hi.value().as_primitive[T]().value(),
        )


def scalar_bounds[T: NumericType](s: DynScalar) -> Bounds[T.native]:
    """A single known value as a point interval, or `unknown`.

    What a bound `Param` reads. A *null* binding answers `unknown` rather than
    `null()`: `Bounds.null()` would prove `never` and skip the whole file,
    which is only correct if `bind` agrees the parameter is NULL — and what
    `bind` does with a null binding is a separate question this module must not
    presume the answer to.
    """
    if s.type() != DynType(T()) or not s.is_valid():
        return Bounds[T.native].unknown()
    else:
        return Bounds[T.native].point(s.as_primitive[T]().value())


def param_bounds[
    T: NumericType
](
    bindings: Bindings, name: String, default: Optional[Scalar[T.native]]
) -> Bounds[T.native]:
    """A late-bound parameter's bounds for *this* execution.

    **A late-bound predicate prunes exactly as well as a literal one**, and
    that falls out of pruning happening at execution time rather than plan
    time: `prune(stats, bindings)` mirrors `bind(batch, bindings)` and sees the
    same `Bindings` the filter will. the previous expression package had to
    reach a process-global
    parameter registry from inside `prune` to get this, and a parameterised
    date filter still read `unknown` and decoded every row group.

    The default is used when nothing binds the name, and that is correct *here*
    for the same reason: `bind` will use the same default in the same
    execution. (`2026-08-27-index-and-pruning-plan.md` §4.4 says a `Param`
    must fold to unknown rather than to its default — that rule is about
    pruning at *plan* time, where the value used to skip a file can differ from
    the value used to filter. It does not bind here.)

    Unbound and undefaulted answers `unknown`: the scan reads everything, and
    `Param.bind` then raises **naming the parameter**. Pruning degrades;
    binding raises.
    """
    var got = bindings.get(name)
    if got:
        return scalar_bounds[T](got.value())
    elif default:
        return Bounds[T.native].point(default.value())
    else:
        return Bounds[T.native].unknown()


# ---------------------------------------------------------------------------
# DynBounds — the erased carrier, and the runtime lane's single dtype ladder
# ---------------------------------------------------------------------------
struct DynBounds(Copyable, Movable):
    """`Bounds` with its endpoints still erased — the runtime lane's carrier.

    Not `Bounds[dt]`, because a runtime node learns its dtype from the data and
    has no comptime type to project. Not
    the previous expression package's `Interval` either: this holds no
    `maybe_true`, so the boolean
    algebra stays on `Truth` and this type is only ever about intervals.
    """

    var lo: Optional[DynScalar]
    var hi: Optional[DynScalar]
    var all_null: Bool

    def __init__(
        out self,
        var lo: Optional[DynScalar],
        var hi: Optional[DynScalar],
        all_null: Bool,
    ):
        self.lo = lo^
        self.hi = hi^
        self.all_null = all_null

    @staticmethod
    def unknown() -> Self:
        return Self(None, None, False)

    @staticmethod
    def null() -> Self:
        return Self(None, None, True)

    @staticmethod
    def point(var v: DynScalar) -> Self:
        """A single known value. A null scalar carries none, so it degrades to
        `unknown` — see `scalar_bounds` for why not `null()`."""
        if not v.is_valid():
            return Self.unknown()
        return Self(Optional(v.copy()), Optional(v^), False)


def compare_dyn[P: BoundsKernel](l: DynBounds, r: DynBounds) -> Truth:
    """One comparison, read over erased bounds — **the runtime lane's only
    dtype ladder**.

    The two `_ord` calls are where the ladder lives, and the reading itself is
    `P.decide`, the same one line the fused lane runs. That factoring is
    deliberate: dispatching *inside* each of the six kernels would emit six
    twenty-arm ladders instead of one, and — much worse — it would let the two
    lanes' readings drift apart, which shows up as the two disagreeing about
    which row groups to skip.

    **Mixed dtypes refuse rather than promote.** `RuntimeValue._compare` casts
    the narrower operand up at *evaluation* time, so the bounds arriving here
    are un-promoted. Promoting them here would mean re-deriving `cast_array`'s
    rules in a second place; refusing is sound (it can only answer `maybe`) and
    it is one line. `_ord` answers `unknown` for a dtype mismatch and every
    reading then reads permissively.
    """
    if l.all_null or r.all_null:
        return Truth.never
    elif not l.lo or not l.hi or not r.lo or not r.hi:
        return Truth.maybe
    else:
        var lo_hi = _ord(l.lo.value(), r.hi.value())
        var hi_lo = _ord(l.hi.value(), r.lo.value())
        if lo_hi == Ord.unknown or hi_lo == Ord.unknown:
            return Truth.maybe
        else:
            return Truth(P.decide(lo_hi, hi_lo))


def _ord(a: DynScalar, b: DynScalar) -> Ord:
    """Three-way compare two erased bounds; `unknown` whenever it cannot.

    `is_primitive()` is exactly the set that conforms to `PrimitiveType`
    (`dtypes.mojo:1120-1132`), and every one of those scalars is stored as a
    `PrimitiveScalar[T]`, so the `as_primitive[T]` inside the arm cannot hit
    the `debug_assert`. `bool` is not in that set, by design — it is
    bit-packed — which is why a bare bool column goes through
    `PruneStats.bool_truth` instead.

    The `try` is the module's one containment point: `dispatch_primitive`
    raises for a non-primitive dtype, and a pruner that raises would turn one
    bad footer into a failed query. Failure degrades to `unknown`, which
    degrades to "read the granule".
    """
    if not a.is_valid() or not b.is_valid():
        return Ord.unknown
    var t = a.type()
    if t != b.type() or not t.is_primitive():
        return Ord.unknown

    def arm[T: PrimitiveType](witness: T) raises {imm} -> Ord:
        return Ord.of(a.as_primitive[T]().value(), b.as_primitive[T]().value())

    try:
        return t.dispatch_primitive(arm)
    except:
        return Ord.unknown


# ---------------------------------------------------------------------------
# Prunable — "I can answer, conservatively, whether I could be true here"
# ---------------------------------------------------------------------------
trait Prunable(Copyable, Deinitable):
    """A value that can be read over a granule's statistics.

    **One method, with a total default.** That default is the only soundness
    rule in the system: a node added tomorrow cannot be forgotten by a pruner
    written today, because forgetting it means answering `maybe`, which is
    always correct. Nothing is forced to name a kernel it does not use, which
    is also what keeps the closed-erasure property — a predicate that never
    writes `<` never links `LtBounds`.

    **Non-raising, deliberately.** "I don't know" is always a legal answer, so
    a pruner that can fail is a pruner with a bug: the fallback is correct by
    construction. It also means a pruning failure can never fail a query — a
    missing column, a dtype the file disagrees about, an unbound parameter each
    degrade to "read the granule" rather than aborting a scan. `_ord` is the
    one place a raising primitive is reached, and it contains it there.

    Refined into `ComptimeValue` and implemented on `RuntimeValue`, so **both
    lanes prune through the same trait** and a `PrunePredicate` erases either.
    """

    def prune(self, stats: PruneStats, bindings: Bindings) -> Truth:
        """Could this be `TRUE` for some row covered by `stats`?"""
        return Truth.maybe


struct PrunePredicate(Copyable, Movable):
    """A `Prunable` erased behind one function pointer.

    **The one indirect call in the subsystem, placed at the coarsest possible
    granularity.** A source calls this once per row group — order 10^6 rows —
    and everything below it is monomorphic: for
    `Gt[Column[Int64Type], Literal[Int64Type]]` the compiler sees two
    `Scalar[int64]` loads, two comparisons and a return, with every operand
    type, the promotion and the kernel fixed at compile time.

    **Built where the concrete type is still visible**, i.e. at the plan-building
    verb, and *not* as a sixth slot on `DynValue`. That is the single most
    important size decision here: a trampoline instantiated at
    `DynValue.__init__` is paid for every projection value, every sort key and
    every aggregate input in the program, where this one is paid per *filter
    predicate*. The recorded priors are `+3.2 MB / +24%` for one extra slot on
    the old aggregate box and `+662,740 bytes` for a shared generic dispatch
    adapter.

    The price is stated rather than hidden: a predicate that arrives already
    boxed as a `DynValue` gets no pruner and reads the whole file. The concrete
    type has to be visible somewhere, and `filter()` is where it still is.
    """

    var _boxed: ArcPointer[NoneType]
    var _prune: def(ArcPointer[NoneType], PruneStats, Bindings) thin -> Truth
    var _drop: def(var ArcPointer[NoneType]) thin
    """Erasure forgets the pointee's destructor; this carries it — the same
    shape `DynValue._drop` and `DynRelation._drop` use, and for the same
    reason."""

    @staticmethod
    def _prune_tramp[
        P: Prunable
    ](
        ptr: ArcPointer[NoneType], stats: PruneStats, bindings: Bindings
    ) -> Truth:
        return rebind[ArcPointer[P]](ptr)[].prune(stats, bindings)

    @staticmethod
    def _drop_tramp[P: Prunable](var ptr: ArcPointer[NoneType]):
        var typed = rebind[ArcPointer[P]](ptr)
        _ = ptr^
        _ = typed^

    def __init__[P: Prunable](out self, value: P):
        var ptr = ArcPointer[P](value.copy())
        self._boxed = rebind[ArcPointer[NoneType]](ptr^)
        self._prune = Self._prune_tramp[P]
        self._drop = Self._drop_tramp[P]

    def __deinit__(deinit self):
        self._drop(self._boxed^)

    def prune(self, stats: PruneStats, bindings: Bindings) -> Truth:
        return self._prune(self._boxed, stats, bindings)

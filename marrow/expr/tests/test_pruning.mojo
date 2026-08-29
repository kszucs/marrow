"""Statistics-based pruning: the vocabulary, the per-node readings, and the
one-sided property.

# Why this file defines its own node types

`Prunable` is a supertrait of `ComptimeValue` and a conformance on
`RuntimeValue` — both of those edits live in files another change owns, so the
nodes below stand in for them. **Each body is one line over `pruning.mojo`**,
and that is deliberate rather than incidental: the production override is the
same one line, so what is exercised here is the shared reading and not a
parallel implementation of it. `_PComparison.prune` is character-for-character
what `NumericCompare.prune` becomes.

# The property that matters

`test_pruning_never_excludes_a_matching_group` is the file's reason to exist.
It generates random data, builds the **real** comptime predicate and runs it
through the **real** engine, then asserts that a granule the pruner rejected
returned zero rows. That is the one-sided-error rule stated as an experiment
against the production evaluator rather than against a second model of it — a
model would only prove the two models agree.

It asserts one direction only, on purpose. `never` must imply "no rows", and
"no rows" need not imply `never`: a granule whose bounds admit a match that the
data happens not to contain is a false positive, which costs time and not
correctness.
"""

from std.sys import align_of
from std.testing import assert_equal, assert_false, assert_true

from ...builders import array
from ...dtypes import Int32Type, Int64Type, NumericType, int64
from ...kernels.bounds import (
    Bounds,
    EqBounds,
    GeBounds,
    GtBounds,
    LeBounds,
    LtBounds,
    NeBounds,
)
from ...kernels.numeric import (
    EqKernel,
    GeKernel,
    GtKernel,
    LeKernel,
    LtKernel,
    NeKernel,
    NumericCompareKernel,
)
from ...scalars import BoolScalar, DynScalar, Int32Scalar, Int64Scalar
from ...tabular import record_batch
from ...schema import Schema
from ...dtypes import BoolType, DynType
from ..builders import col, lit, table
from ..logical import DynRelation, DynValue, Shape, Value
from ..bindings import Bindings
from ..physical import DynOperator
from ..`comptime`.rules import promote
from ..pruning import (
    PrunePredicate,
    PruneStats,
    Prunable,
    Truth,
    param_bounds,
    scalar_bounds,
)


# ---------------------------------------------------------------------------
# Stand-ins for the node overrides — one line of body each
# ---------------------------------------------------------------------------
trait _PValue(Prunable):
    """What `PrimitiveValue` gains: a typed bounds reading.

    The exact shape the patch adds — `Bounds[Self.Type.native]` is the same
    projection `lane`'s `SIMD[Self.Type.native, W]` already makes and compiles
    at, which is why this return type is known to reduce.
    """

    comptime Type: NumericType

    def bounds(
        self, stats: PruneStats, bindings: Bindings
    ) -> Bounds[Self.Type.native]:
        """What this sub-expression's value can be. Default: unknown.

        **A trait *default* at this return type, and it reduces.** That was
        open question R-2 in `2026-08-27-index-and-pruning-plan.md`: an
        *abstract* declaration returning `SIMD[Self.Type.native, W]` is proven
        by `PrimitiveValue.lane`, but a defaulted *body* at a projected return
        type was untested, and the fallback was to declare it abstract and
        write a four-line unknown body per node. `_PUnprunable` below inherits
        this and compiles, so the fallback is not needed and a node that cannot
        do better costs one inherited `return`.
        """
        return Bounds[Self.Type.native].unknown()


struct _PColumn[T: NumericType](Copyable, Movable, _PValue):
    """Stands in for `Column[T]`."""

    comptime Type = Self.T
    var _name: String

    def __init__(out self, var name: String):
        self._name = name^

    def bounds(
        self, stats: PruneStats, bindings: Bindings
    ) -> Bounds[Self.Type.native]:
        return stats.bounds[Self.T](self._name)


struct _PLiteral[T: NumericType](Copyable, Movable, _PValue):
    """Stands in for `Literal[T]`."""

    comptime Type = Self.T
    var _value: Scalar[Self.T.native]

    def __init__(out self, value: Scalar[Self.T.native]):
        self._value = value

    def bounds(
        self, stats: PruneStats, bindings: Bindings
    ) -> Bounds[Self.Type.native]:
        return Bounds[Self.Type.native].point(self._value)


struct _PParam[T: NumericType](Copyable, Movable, _PValue):
    """Stands in for `Param[T]`."""

    comptime Type = Self.T
    var _name: String
    var _default: Optional[Scalar[Self.T.native]]

    def __init__(
        out self,
        var name: String,
        var default: Optional[Scalar[Self.T.native]] = None,
    ):
        self._name = name^
        self._default = default^

    def bounds(
        self, stats: PruneStats, bindings: Bindings
    ) -> Bounds[Self.Type.native]:
        return param_bounds[Self.T](bindings, self._name, self._default)


struct _PComparison[K: NumericCompareKernel, L: _PValue, R: _PValue](
    Copyable, Movable, Prunable
):
    """Stands in for `NumericCompare[K, L, R]`.

    **Keyed on the existing `K`, with no new struct parameter.** The 08-24
    design paired the two kernels by adding a `P: BoundsKernel` parameter
    (`Gt = NumericCompare[GtKernel, GtBounds, _, _]`); this pairs them in a
    `comptime if` ladder over `Self.K.name` instead, which is purely additive —
    no arity change, no alias change, no call site touched — and whose failure
    mode is conservative: a seventh comparison kernel that nobody adds an arm
    for falls through to `maybe`, which is always correct.

    `Self.K.name` resolving off a kernel parameter is recorded as working in
    CLAUDE.md; what this case additionally settles is that a **`comptime if`
    over a `String` equality** folds, so only the taken arm is emitted and the
    closed-erasure property holds — a `Gt` node links `GtBounds.decide` and
    nothing else.

    `ArgType` is a member rather than a projection written inline for the
    reason `NumericCompare` has one: `Self.L.Type.native` used directly "does
    not reduce — the compiler reports a type 'cannot be converted' to *itself*"
    (`comptime/numeric.mojo:283-291`).
    """

    comptime ArgType = promote[Self.L.Type, Self.R.Type]

    var l: Self.L
    var r: Self.R

    def __init__(out self, var l: Self.L, var r: Self.R):
        self.l = l^
        self.r = r^

    def prune(self, stats: PruneStats, bindings: Bindings) -> Truth:
        comptime dt = Self.ArgType.native
        var lb = self.l.bounds(stats, bindings).cast[dt]()
        var rb = self.r.bounds(stats, bindings).cast[dt]()
        comptime if Self.K.name == LtKernel.name:
            return Truth(LtBounds.maybe[dt](lb, rb))
        elif Self.K.name == LeKernel.name:
            return Truth(LeBounds.maybe[dt](lb, rb))
        elif Self.K.name == GtKernel.name:
            return Truth(GtBounds.maybe[dt](lb, rb))
        elif Self.K.name == GeKernel.name:
            return Truth(GeBounds.maybe[dt](lb, rb))
        elif Self.K.name == EqKernel.name:
            return Truth(EqBounds.maybe[dt](lb, rb))
        elif Self.K.name == NeKernel.name:
            return Truth(NeBounds.maybe[dt](lb, rb))
        else:
            return Truth.maybe


struct _PUnprunable[T: NumericType](Copyable, Movable, _PValue):
    """A node that can do no better than "unknown" — it defines **neither**
    `prune` nor `bounds` and inherits both totals.

    Stands in for `NumericBinary`, `CaseWhen`, `StringColumn` and every other
    node that this change does not teach to prune. Its existence is the proof
    that the two defaults reduce, which is what lets the production patch touch
    only the eight nodes that can do better.
    """

    comptime Type = Self.T

    def __init__(out self):
        pass


struct _PAnd[L: Prunable, R: Prunable](Copyable, Movable, Prunable):
    var l: Self.L
    var r: Self.R

    def __init__(out self, var l: Self.L, var r: Self.R):
        self.l = l^
        self.r = r^

    def prune(self, stats: PruneStats, bindings: Bindings) -> Truth:
        return self.l.prune(stats, bindings) & self.r.prune(stats, bindings)


struct _POr[L: Prunable, R: Prunable](Copyable, Movable, Prunable):
    var l: Self.L
    var r: Self.R

    def __init__(out self, var l: Self.L, var r: Self.R):
        self.l = l^
        self.r = r^

    def prune(self, stats: PruneStats, bindings: Bindings) -> Truth:
        return self.l.prune(stats, bindings) | self.r.prune(stats, bindings)


struct _PNot[A: Prunable](Copyable, Movable, Prunable):
    """Stands in for `Not[A]` — and it defines **no** `prune`.

    That is the whole point: it inherits the total `Truth.maybe` default, so a
    negation cannot prune, and nobody has to remember to make it not.
    """

    var a: Self.A

    def __init__(out self, var a: Self.A):
        self.a = a^


struct _PBoolColumn(Copyable, Movable, Prunable):
    """Stands in for `BoolColumn` read as a bare predicate."""

    var _name: String

    def __init__(out self, var name: String):
        self._name = name^

    def prune(self, stats: PruneStats, bindings: Bindings) -> Truth:
        return stats.bool_truth(self._name)


struct _MinimalValue(Copyable, Movable, Prunable, Value):
    """A node that is both a `Value` and a `Prunable`, and nothing more.

    It exists to settle open question 1 of
    `2026-08-24-expr2-pruning-pushdown-design.md`: `DynRelation.filter` must
    gain a `filter[V: Value & Prunable](V)` overload beside the existing
    `filter(DynValue)` so that a predicate reaches a scan with its concrete
    type intact — and `DynValue` carries an `@implicit __init__[V: Value]`, so
    it is not obvious the two candidate sets are disjoint. If they are not, the
    delivery half of this design is not applicable as written.

    `to_operator` raises rather than working: this probes overload resolution
    and the spellability of the `Value & Prunable` bound, not execution.
    """

    comptime shape = Shape.columnar

    def __init__(out self):
        pass

    def columns(self) -> List[String]:
        return List[String]()

    def name(self) -> String:
        return String("probe")

    def dtype(self, schema: Schema) raises -> DynType:
        return DynType(BoolType())

    def to_operator(
        self, schema: Schema, grouped: Bool, bindings: Bindings = Bindings()
    ) raises -> DynOperator:
        raise Error("probe: not executable")

    def prune(self, stats: PruneStats, bindings: Bindings) -> Truth:
        return Truth.never

    def write_to[W: Writer](self, mut writer: W):
        writer.write("probe")


def _pick[V: Value & Prunable](value: V) -> Bool:
    """The typed candidate — what the new `filter` overload would be."""
    return True


def _pick(value: DynValue) -> Bool:
    """The erased candidate — what `filter` already is."""
    return False


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _i64(v: Int) -> Optional[DynScalar]:
    return Optional(Int64Scalar(Scalar[int64.native](v)).to_dyn())


def _stats_i64(
    name: String, lo: Int, hi: Int, rows: Int = 100, nulls: Int = 0
) -> PruneStats:
    var s = PruneStats(rows, capacity=1)
    s.add(name.copy(), _i64(lo), _i64(hi), nulls)
    return s^


def _gt_lit(var name: String, v: Int) -> _PComparison[
    GtKernel, _PColumn[Int64Type], _PLiteral[Int64Type]
]:
    return _PComparison[GtKernel, _PColumn[Int64Type], _PLiteral[Int64Type]](
        _PColumn[Int64Type](name^), _PLiteral[Int64Type](Scalar[int64.native](v))
    )


# ---------------------------------------------------------------------------
# The readings
# ---------------------------------------------------------------------------
def test_pruning_a_range_predicate_prunes_and_keeps() raises:
    """The shape the whole subsystem exists for: `a > 5`."""
    var p = _gt_lit("a", 5)
    assert_true(p.prune(_stats_i64("a", 0, 3), Bindings()) == Truth.never)
    assert_true(p.prune(_stats_i64("a", 0, 9), Bindings()) == Truth.maybe)


def test_pruning_an_all_null_column_proves_never() raises:
    """`null_count == num_rows` is exactly provable, and it needs no bound —
    a comparison against NULL is NULL and a filter keeps only valid TRUE."""
    var s = PruneStats(100, capacity=1)
    s.add("a", None, None, 100)
    assert_true(_gt_lit("a", 5).prune(s, Bindings()) == Truth.never)


def test_pruning_a_missing_statistic_prunes_nothing() raises:
    """Three separate ways a statistic can be absent, and all three must read
    as *unknown* rather than as an empty range or a zero."""
    var no_column = PruneStats(100)
    assert_true(_gt_lit("a", 5).prune(no_column, Bindings()) == Truth.maybe)

    var no_bounds = PruneStats(100, capacity=1)
    no_bounds.add("a", None, None, 0)
    assert_true(_gt_lit("a", 5).prune(no_bounds, Bindings()) == Truth.maybe)

    var half_a_bound = PruneStats(100, capacity=1)
    half_a_bound.add("a", _i64(0), None, 0)
    assert_true(_gt_lit("a", 5).prune(half_a_bound, Bindings()) == Truth.maybe)


def test_pruning_an_unknown_null_count_is_not_zero() raises:
    """An absent null count is unknown, never zero.

    arrow-rs makes "missing null counts as zero" an explicit option because it
    is a soundness choice. Here `-1` means unknown, so `all_null` cannot be
    inferred and the granule is read.
    """
    var s = PruneStats(100, capacity=1)
    s.add("a", None, None, -1)
    assert_true(_gt_lit("a", 5).prune(s, Bindings()) == Truth.maybe)


def test_pruning_a_foreign_dtype_answers_maybe_without_aborting() raises:
    """The single most important case in this file.

    `DynScalar.as_primitive[T]()` is a `debug_assert`, so unwrapping a bound
    whose type is not `T` **aborts the process** rather than raising — and an
    abort inside a test runner fails every case in the file, not the one that
    was wrong. The dtype equality in `_typed_range` is what makes the unwrap
    provably safe. This case reaching its assertion at all is the result.
    """
    var s = PruneStats(100, capacity=1)
    s.add("a", Optional(Int32Scalar(0).to_dyn()), Optional(Int32Scalar(3).to_dyn()), 0)
    assert_true(_gt_lit("a", 5).prune(s, Bindings()) == Truth.maybe)


def test_pruning_a_null_valued_bound_answers_maybe() raises:
    var s = PruneStats(100, capacity=1)
    var null_i64 = Int64Scalar(Optional[Scalar[int64.native]](None)).to_dyn()
    s.add("a", Optional(null_i64.copy()), Optional(null_i64^), 0)
    assert_true(_gt_lit("a", 5).prune(s, Bindings()) == Truth.maybe)


def test_pruning_conjunction_follows_the_prunable_conjunct() raises:
    """`AND` of a prunable and an unprunable conjunct prunes."""
    var s = _stats_i64("a", 0, 3)
    var both = _PAnd(_gt_lit("a", 5), _gt_lit("b", 0))
    assert_true(both.prune(s, Bindings()) == Truth.never)

    var keeps = _PAnd(_gt_lit("a", 1), _gt_lit("b", 0))
    assert_true(keeps.prune(s, Bindings()) == Truth.maybe)


def test_pruning_disjunction_needs_both_disjuncts_disproved() raises:
    """The structural guard against the sub-term-index defect: one disproved
    disjunct beside an unknown one must not skip a granule full of matches."""
    var s = _stats_i64("a", 0, 3)
    var one_open = _POr(_gt_lit("a", 5), _gt_lit("b", 0))
    assert_true(one_open.prune(s, Bindings()) == Truth.maybe)

    var both_shut = _POr(_gt_lit("a", 5), _gt_lit("a", 9))
    assert_true(both_shut.prune(s, Bindings()) == Truth.never)


def test_pruning_negation_prunes_nothing() raises:
    """A one-sided domain has no negation: `p.prune()` answers "could `p` be
    TRUE", and `NOT p` needs "could `p` be FALSE", which is not derivable from
    it. Deriving it anyway produces false negatives — the one error class the
    subsystem forbids."""
    var s = _stats_i64("a", 0, 3)
    assert_true(_gt_lit("a", 5).prune(s, Bindings()) == Truth.never)
    assert_true(_PNot(_gt_lit("a", 5)).prune(s, Bindings()) == Truth.maybe)


def test_pruning_every_comparison_picks_its_own_reading() raises:
    """All six operators against one granule, `[0, 3]`, compared to `3`.

    This is what proves the `comptime if Self.K.name == ...` ladder *resolves*
    rather than merely compiles: six kernels differing only in that parameter
    must give six independently-correct verdicts. A ladder that fell through to
    `Truth.maybe` for every operator would pass every other case in this file.
    """
    var s = _stats_i64("a", 0, 3)
    var three = _PLiteral[Int64Type](Scalar[int64.native](3))

    # a < 3 : min 0 < 3            -> maybe
    # a <= 3: min 0 <= 3           -> maybe
    # a > 3 : max 3 > 3 is false   -> never
    # a >= 3: max 3 >= 3           -> maybe
    # a == 3: [0,3] overlaps [3,3] -> maybe
    # a != 3: not a shared point   -> maybe
    var lt = _PComparison[LtKernel, _PColumn[Int64Type], _PLiteral[Int64Type]](
        _PColumn[Int64Type]("a"), three.copy()
    )
    var le = _PComparison[LeKernel, _PColumn[Int64Type], _PLiteral[Int64Type]](
        _PColumn[Int64Type]("a"), three.copy()
    )
    var gt = _PComparison[GtKernel, _PColumn[Int64Type], _PLiteral[Int64Type]](
        _PColumn[Int64Type]("a"), three.copy()
    )
    var ge = _PComparison[GeKernel, _PColumn[Int64Type], _PLiteral[Int64Type]](
        _PColumn[Int64Type]("a"), three.copy()
    )
    var eq = _PComparison[EqKernel, _PColumn[Int64Type], _PLiteral[Int64Type]](
        _PColumn[Int64Type]("a"), three.copy()
    )
    var ne = _PComparison[NeKernel, _PColumn[Int64Type], _PLiteral[Int64Type]](
        _PColumn[Int64Type]("a"), three.copy()
    )
    assert_true(lt.prune(s, Bindings()) == Truth.maybe)
    assert_true(le.prune(s, Bindings()) == Truth.maybe)
    assert_true(gt.prune(s, Bindings()) == Truth.never)
    assert_true(ge.prune(s, Bindings()) == Truth.maybe)
    assert_true(eq.prune(s, Bindings()) == Truth.maybe)
    assert_true(ne.prune(s, Bindings()) == Truth.maybe)

    # and the mirrored granule, so each operator is exercised in both verdicts
    var high = _stats_i64("a", 7, 9)
    assert_true(lt.prune(high, Bindings()) == Truth.never)
    assert_true(le.prune(high, Bindings()) == Truth.never)
    assert_true(gt.prune(high, Bindings()) == Truth.maybe)
    assert_true(ge.prune(high, Bindings()) == Truth.maybe)
    assert_true(eq.prune(high, Bindings()) == Truth.never)
    assert_true(ne.prune(high, Bindings()) == Truth.maybe)

    var point = _stats_i64("a", 3, 3)
    assert_true(ne.prune(point, Bindings()) == Truth.never)


def test_pruning_a_node_with_no_override_answers_unknown() raises:
    """`_PUnprunable` defines neither `prune` nor `bounds`.

    Both totals reduce at a projected return type, which settles open question
    R-2 of `2026-08-27-index-and-pruning-plan.md` — the fallback there was to
    declare `bounds` abstract and write a four-line body on every node. It is
    also the property that makes the whole design safe to extend: a node added
    tomorrow cannot be forgotten by a pruner written today.
    """
    var u = _PUnprunable[Int64Type]()
    assert_true(u.prune(_stats_i64("a", 0, 3), Bindings()) == Truth.maybe)
    assert_false(u.bounds(_stats_i64("a", 0, 3), Bindings()).known)

    var cmp = _PComparison[
        GtKernel, _PUnprunable[Int64Type], _PLiteral[Int64Type]
    ](_PUnprunable[Int64Type](), _PLiteral[Int64Type](Scalar[int64.native](5)))
    assert_true(cmp.prune(_stats_i64("a", 0, 3), Bindings()) == Truth.maybe)


def test_pruning_a_bool_column_reads_its_maximum() raises:
    var all_false = PruneStats(100, capacity=1)
    all_false.add(
        "b", Optional(BoolScalar(False).to_dyn()), Optional(BoolScalar(False).to_dyn()), 0
    )
    assert_true(_PBoolColumn("b").prune(all_false, Bindings()) == Truth.never)

    var mixed = PruneStats(100, capacity=1)
    mixed.add(
        "b", Optional(BoolScalar(False).to_dyn()), Optional(BoolScalar(True).to_dyn()), 0
    )
    assert_true(_PBoolColumn("b").prune(mixed, Bindings()) == Truth.maybe)

    var nulls = PruneStats(100, capacity=1)
    nulls.add("b", None, None, 100)
    assert_true(_PBoolColumn("b").prune(nulls, Bindings()) == Truth.never)


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
def test_pruning_a_bound_parameter_prunes_like_a_literal() raises:
    """Pruning runs at execution time with the same `Bindings` the filter
    sees, so a late-bound predicate prunes exactly as well as a literal one.
    `exprold` had to read a process-global registry from inside `prune` for
    this, and still regressed a parameterised date filter to reading every row
    group."""
    var p = _PComparison[GtKernel, _PColumn[Int64Type], _PParam[Int64Type]](
        _PColumn[Int64Type]("a"), _PParam[Int64Type]("min-a")
    )
    var s = _stats_i64("a", 0, 3)
    assert_true(
        p.prune(s, {"min-a": Int64Scalar(5).to_dyn()}) == Truth.never
    )
    assert_true(
        p.prune(s, {"min-a": Int64Scalar(1).to_dyn()}) == Truth.maybe
    )


def test_pruning_an_unbound_parameter_answers_maybe_without_raising() raises:
    """Pruning degrades; binding raises. The scan reads everything and
    `Param.bind` then raises naming the parameter."""
    var p = _PComparison[GtKernel, _PColumn[Int64Type], _PParam[Int64Type]](
        _PColumn[Int64Type]("a"), _PParam[Int64Type]("min-a")
    )
    assert_true(p.prune(_stats_i64("a", 0, 3), Bindings()) == Truth.maybe)


def test_pruning_a_defaulted_parameter_uses_its_default() raises:
    """Correct *here* because `bind` will use the same default in the same
    execution. It would be wrong for plan-time pruning, where the value used
    to skip a file can differ from the value used to filter."""
    var p = _PComparison[GtKernel, _PColumn[Int64Type], _PParam[Int64Type]](
        _PColumn[Int64Type]("a"),
        _PParam[Int64Type]("min-a", Optional(Scalar[int64.native](5))),
    )
    assert_true(p.prune(_stats_i64("a", 0, 3), Bindings()) == Truth.never)


def test_pruning_scalar_bounds_reject_a_foreign_dtype() raises:
    assert_false(scalar_bounds[Int64Type](Int32Scalar(5).to_dyn()).known)
    assert_true(scalar_bounds[Int64Type](Int64Scalar(5).to_dyn()).known)


# ---------------------------------------------------------------------------
# Promotion
# ---------------------------------------------------------------------------
def test_pruning_promotes_operands_exactly_as_the_lane_does() raises:
    """`int32 > int64` compares in `int64`; the bounds take the identical
    cast, so the interval reading agrees with the lane's reading row for row.
    """
    var p = _PComparison[GtKernel, _PColumn[Int32Type], _PLiteral[Int64Type]](
        _PColumn[Int32Type]("a"), _PLiteral[Int64Type](Scalar[int64.native](5))
    )
    var s = PruneStats(100, capacity=1)
    s.add("a", Optional(Int32Scalar(0).to_dyn()), Optional(Int32Scalar(3).to_dyn()), 0)
    assert_true(p.prune(s, Bindings()) == Truth.never)

    var wide = PruneStats(100, capacity=1)
    wide.add("a", Optional(Int32Scalar(0).to_dyn()), Optional(Int32Scalar(9).to_dyn()), 0)
    assert_true(p.prune(wide, Bindings()) == Truth.maybe)


# ---------------------------------------------------------------------------
# The erasure box
# ---------------------------------------------------------------------------
def test_pruning_predicate_erases_a_node_and_keeps_its_answer() raises:
    var boxed = PrunePredicate(_gt_lit("a", 5))
    assert_true(boxed.prune(_stats_i64("a", 0, 3), Bindings()) == Truth.never)
    assert_true(boxed.prune(_stats_i64("a", 0, 9), Bindings()) == Truth.maybe)

    var copied = boxed.copy()
    assert_true(copied.prune(_stats_i64("a", 0, 3), Bindings()) == Truth.never)


def test_pruning_predicate_erases_a_composite() raises:
    var boxed = PrunePredicate(_PAnd(_gt_lit("a", 5), _PNot(_gt_lit("a", 1))))
    assert_true(boxed.prune(_stats_i64("a", 0, 3), Bindings()) == Truth.never)
    assert_true(boxed.prune(_stats_i64("a", 0, 9), Bindings()) == Truth.maybe)


def test_pruning_a_typed_overload_wins_over_the_erased_one() raises:
    """The delivery half depends on this, and it was an open question.

    A concrete node must select `filter[V: Value & Prunable]` and a
    `DynValue` must select `filter(DynValue)`, even though `DynValue` has an
    `@implicit __init__[V: Value]` that would let the erased overload accept
    the concrete node too. The candidate sets are disjoint because `DynValue`
    deliberately does not conform to the traits it erases — this pins that.

    The consequence when it goes the other way is the whole point: a predicate
    that arrives already boxed gets no pruner and the file is read in full,
    which is correct but slow. Silently taking that path for a *typed*
    predicate would make the fused pruner unreachable and unmeasurable.
    """
    assert_true(_pick(_MinimalValue()))
    assert_false(_pick(DynValue(_MinimalValue())))


def test_pruning_stats_hold_their_alignment() raises:
    """`ColumnBounds` holds two `Optional[DynScalar]` and lives in a `List`,
    which is the shape of the variant layout miscompile CLAUDE.md records: a
    `List` whose element is a variant whose largest member is not its
    most-aligned one drops every other element when it grows.

    `PrimitiveScalar` stores its payload as bytes precisely to keep
    `align_of[DynScalar]()` at 8, so the shape is currently benign. This asserts
    that, and then grows a list past several reallocations to catch it if it
    ever stops being true.
    """
    assert_equal(align_of[DynScalar](), 8)

    var s = PruneStats(1000)
    for i in range(64):
        s.add(String("c", i), _i64(i), _i64(i + 1), 0)
    assert_equal(s.num_columns(), 64)
    for i in range(64):
        var b = s.bounds[Int64Type](String("c", i))
        assert_true(b.known)
        assert_equal(Int(b.lo), i)
        assert_equal(Int(b.hi), i + 1)


# ---------------------------------------------------------------------------
# The soundness property, against the real engine
# ---------------------------------------------------------------------------
def _xorshift(mut s: UInt64) -> UInt64:
    s ^= s << 13
    s ^= s >> 7
    s ^= s << 17
    return s


def _group(mut seed: UInt64, n: Int) raises -> List[Optional[Int]]:
    """`n` values in `[-20, 20)`, roughly one in eight null."""
    var out = List[Optional[Int]](capacity=n)
    for _ in range(n):
        var r = _xorshift(seed)
        if r % 8 == 0:
            out.append(None)
        else:
            out.append(Int(r % 40) - 20)
    return out^


def _stats_of(values: List[Optional[Int]]) -> PruneStats:
    """The statistics a writer would emit: min/max over **non-null** values
    only, plus the null count — exactly Parquet's rule, so the pruner is fed
    what a real footer would give it."""
    var nulls = 0
    var lo = 0
    var hi = 0
    var seen = False
    for ref v in values:
        if not v:
            nulls += 1
        else:
            if not seen:
                lo = v.value()
                hi = v.value()
                seen = True
            else:
                lo = min(lo, v.value())
                hi = max(hi, v.value())
    var s = PruneStats(len(values), capacity=1)
    if seen:
        s.add("a", _i64(lo), _i64(hi), nulls)
    else:
        s.add("a", None, None, nulls)
    return s^


def _batch(values: List[Optional[Int]]) raises -> DynRelation:
    return table(record_batch([array(values, int64).copy()], names=["a"]))


def test_pruning_never_excludes_a_matching_group() raises:
    """The one-sided-error rule, as an experiment.

    For 240 randomly generated granules and four predicate shapes: build the
    real comptime predicate, run it through the real engine, and require that
    every granule the pruner rejected returned zero rows.

    One direction only. `never` must imply "no rows"; "no rows" need not imply
    `never`, because a granule whose bounds admit a match the data happens not
    to contain is a false positive — time, not correctness.

    The counters at the end are what stops this passing vacuously: a pruner
    that never pruned would satisfy the implication trivially, so the test
    also requires that pruning actually happened and that granules survived.
    """
    var seed: UInt64 = 0x9E3779B97F4A7C15
    var pruned = 0
    var kept = 0

    for _ in range(60):
        var values = _group(seed, 24)
        var stats = _stats_of(values)
        var rel = _batch(values)

        # 1) a > k
        var k = Int(_xorshift(seed) % 40) - 20
        var t1 = _gt_lit("a", k).prune(stats, Bindings())
        var n1 = rel.filter(col("a", int64) > lit(k, int64)).execute().num_rows()
        if t1 == Truth.never:
            pruned += 1
            assert_equal(n1, 0)
        else:
            kept += 1

        # 2) a == k
        var eq = _PComparison[
            EqKernel, _PColumn[Int64Type], _PLiteral[Int64Type]
        ](
            _PColumn[Int64Type]("a"),
            _PLiteral[Int64Type](Scalar[int64.native](k)),
        )
        var t2 = eq.prune(stats, Bindings())
        var n2 = rel.filter(col("a", int64) == lit(k, int64)).execute().num_rows()
        if t2 == Truth.never:
            pruned += 1
            assert_equal(n2, 0)
        else:
            kept += 1

        # 3) a > k AND a < k + 3  (an empty-ish window, so pruning bites)
        var band = _PAnd(
            _gt_lit("a", k),
            _PComparison[LtKernel, _PColumn[Int64Type], _PLiteral[Int64Type]](
                _PColumn[Int64Type]("a"),
                _PLiteral[Int64Type](Scalar[int64.native](k + 3)),
            ),
        )
        var t3 = band.prune(stats, Bindings())
        var n3 = (
            rel.filter(
                (col("a", int64) > lit(k, int64))
                & (col("a", int64) < lit(k + 3, int64))
            )
            .execute()
            .num_rows()
        )
        if t3 == Truth.never:
            pruned += 1
            assert_equal(n3, 0)
        else:
            kept += 1

        # 4) a != k  (the point reading; prunes only for a constant column)
        var ne = _PComparison[
            NeKernel, _PColumn[Int64Type], _PLiteral[Int64Type]
        ](
            _PColumn[Int64Type]("a"),
            _PLiteral[Int64Type](Scalar[int64.native](k)),
        )
        var t4 = ne.prune(stats, Bindings())
        var n4 = rel.filter(col("a", int64) != lit(k, int64)).execute().num_rows()
        if t4 == Truth.never:
            pruned += 1
            assert_equal(n4, 0)
        else:
            kept += 1

    assert_true(pruned > 0)
    assert_true(kept > 0)


def test_pruning_never_excludes_a_matching_group_when_all_null() raises:
    """The all-null granule, separately: it is the one case where pruning is
    driven by the null count rather than by a bound, and a generator that
    rarely produces one would not exercise it."""
    var values = List[Optional[Int]](capacity=8)
    for _ in range(8):
        values.append(None)
    var stats = _stats_of(values)
    var rel = _batch(values)

    for k in range(-3, 4):
        assert_true(_gt_lit("a", k).prune(stats, Bindings()) == Truth.never)
        assert_equal(
            rel.filter(col("a", int64) > lit(k, int64)).execute().num_rows(), 0
        )

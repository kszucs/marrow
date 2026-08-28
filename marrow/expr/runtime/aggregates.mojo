"""Aggregates named at run time — the runtime lane's half of aggregation.

Its counterpart is `comptime/aggregates.mojo`, which holds the same aggregates
as **types**: `Aggregate[Agg, A]`, which fuses or materialises according to
`Agg` and `A`. Both lanes share one
vocabulary — `AggKernel` in `kernels/aggregate.mojo` — and differ only
in when the aggregate is chosen.

This module is what a caller uses when the aggregate is a **string**: a plan
parsed from Python, a `("count_distinct", "x")` pair, anything built after the
program started. Both the aggregate *and* its operands are erased here, which
is the honest encoding — a frontend that names its aggregate at run time named
its column expression at run time too.

**One resolution, not two.** `RuntimeAggregate.resolve` answers the output
dtype and the implementation *together*, from one name x dtype ladder, and
takes both off the same `AggKernel`. Two free functions with the same
parameters answering different aspects of one thing were methods on a missing
type — and, worse, two independently hand-written ladders that could silently
disagree. `min` over `timestamp[us]` declaring `timestamp[s]` is a `Variant`
misaccess at emit, not a raise, so agreement cannot be left to review; taking
both answers from one type makes disagreement unrepresentable.

Resolution still happens **twice, at two different times**, and that part is
mandatory:

- `Aggregate._output_schema` calls `a.dtype(schema)` before any batch exists,
  so it resolves from schema-derived dtypes and keeps only its `dtype`.
- `BufferedAccumulator` holds the node and resolves again on first push,
  from the morsel's real dtypes, keeping only `fold`. It cannot resolve
  earlier: a `RuntimeValue` operand has no dtype until a schema is in hand.
  Holding the node is what removes the third type an earlier draft had — an
  operator in this tree already holds its node (`FusedAccumulator._input`,
  `EvalOperator._value`), so `resolve` can simply be a method on it.

Discarding a `AggregateFn` at plan time is one function-pointer assignment, and
paying it is what buys a single ladder.

**Non-generic on purpose.** A `resolve[Job: def[A: ...]()]` form instantiates
once per closure type — the `_arith[K]` shape, measured at +115,600 bytes. This
resolver has no type parameter, so it is one instantiation for the whole tree,
and the comptime lane's DCE property holds because nothing there names it: a
program built from `col("a", int64)` reaches `AggKernel` conformers
directly and never this ladder.
"""

from ...arrays import DynArray, StructArray
from ...dtypes import DynType, int64
from ...kernels.aggregate import (
    AggKernel,
    AggregateFn,
    Dispersion,
    DistinctCount,
    MaxKernel,
    MaxOp,
    MeanKernel,
    MinKernel,
    MinOp,
    Fold,
    ProductKernel,
    StringExtremum,
    SumKernel,
    ValidCount,
)
from ...schema import Schema
from ..logical import DynValue, Shape, Value, merged
from ..params import Bindings
from ...execution import ExecContext
from ...kernels.concat import concat
from ...kernels.core import Groups
from ..physical import (
    Accumulable,
    AggregateOperator,
    Datum,
    DynOperator,
    Morsel,
)


comptime SUM = "sum"
comptime PRODUCT = "product"
comptime MEAN = "mean"
comptime COUNT = "count"
comptime COUNT_DISTINCT = "count_distinct"
comptime APPROX_COUNT_DISTINCT = "approx_count_distinct"
comptime VARIANCE = "variance"
comptime STDDEV = "stddev"
comptime MIN = "min"
comptime MAX = "max"


struct ResolvedAggregate(Copyable, Movable):
    """What a named aggregate becomes once its input types are known.

    A named pair rather than a `Tuple`, for the reason `Groups` and `JoinIndex`
    are named types: two fields that must agree with each other should not be
    positional. Here the agreement is load-bearing — `dtype` reaches the
    output schema and `fold` produces the column, and a mismatch is a `Variant`
    misaccess at emit rather than a raise.
    """

    var dtype: DynType
    """What the produced column will be typed as.

    Bare `dtype`, because there is no *input* dtype in this struct for a prefix
    to disambiguate against — the inputs were consumed by `resolve`.
    """

    var grouped: AggregateFn
    """The implementation. Carries no identity — a `thin` fn cannot be asked
    which aggregate it is — which is why `of` is the only thing that builds
    one."""

    def __init__(out self, var dtype: DynType, grouped: AggregateFn):
        self.dtype = dtype^
        self.grouped = grouped

    @staticmethod
    def of[Agg: AggKernel](in_dtypes: List[DynType]) raises -> Self:
        """Both halves of a resolution, off **one** `AggKernel`.

        This is what makes the dtype and the implementation unable to
        disagree: naming `Agg` once produces both, so there is no second table
        to keep in step. A static constructor rather than a free
        `_resolved[Agg]` helper — same instantiation count, but it reads as
        construction instead of as a loose generic beside the type it builds.
        """
        return Self(Agg.dtype(in_dtypes), Agg.grouped)


# ---------------------------------------------------------------------------
# RuntimeAggregate — the node
# ---------------------------------------------------------------------------
struct RuntimeAggregate(Value):
    """An aggregate resolved by name, over erased operands.

    The runtime lane's aggregate node, and the counterpart of
    `Aggregate[Agg, A]`. That one keeps its operand **typed**, so
    `count_distinct(upper(region))` still fuses `upper(region)` into one loop;
    this one cannot, because a caller who names its aggregate with a string
    built its operand at run time too. Reach for it from a frontend, not from
    Mojo source: `col("region", string).count_distinct()` gives the fused-operand
    node instead.

    **No `_dtype` and no `_grouped` field.** Neither is available where the node
    is built: `col("ts", timestamp(us)).min()` is written where no schema
    exists, and a temporal dtype is not constructible from its type. The node
    stores the *function* and resolves twice — once for the schema, once for
    the columns.
    """

    comptime aggregates = True
    """It answers from `AggregateOperator.drain`, never per batch — the same
    answer `Aggregate` gives, for the same reason."""

    var _inputs: List[DynValue]
    """The operands, erased. A `List` because `AggregateFn` takes one."""

    var _name: String
    """The aggregate's own name, and the resolver key. **Validated in
    `__init__`**, so `col("s").count_distnct()` — or any frontend handing over
    a typo — raises where it was written rather than on the first morsel of a
    long scan.

    Never changed by `alias`: one combined name field would make `write_to`
    print `n(region)` after `.alias("n")` and send the resolver looking for an
    aggregate called `n`."""

    var _alias: String
    """What `Value.name()` answers, and what `Aggregate._output_schema` reads.
    Defaults to the aggregate's own name, the same way
    `col("a", int64).sum()` is named `"sum"`."""

    def __init__(out self, var input: DynValue, var name: String) raises:
        self._inputs = [input^]
        self._alias = name.copy()
        self._name = Self._checked(name^)

    def __init__(
        out self,
        var inputs: List[DynValue],
        var name: String,
        var display: String,
    ) raises:
        self._inputs = inputs^
        self._name = Self._checked(name^)
        self._alias = display^

    @staticmethod
    def vocabulary() -> List[String]:
        """Every aggregate name this lane serves.

        Public because it is the *only* thing that can keep `_checked` and
        `resolve` in step. They are two tables over one vocabulary, and the
        hazard is real rather than theoretical: `variance` and `stddev` were
        added to `resolve` first, and every query naming them raised "unknown
        aggregate" from `__init__` before reaching it.

        `resolve` cannot be derived from this list — its arms bind types, not
        strings — so instead `test_named_aggregate_vocabulary_all_resolves`
        walks this list and resolves each entry, which fails loudly the next
        time one side gains a name the other lacks.
        """
        return [
            SUM,
            PRODUCT,
            MEAN,
            VARIANCE,
            STDDEV,
            COUNT,
            COUNT_DISTINCT,
            APPROX_COUNT_DISTINCT,
            MIN,
            MAX,
        ]

    @staticmethod
    def _checked(var name: String) raises -> String:
        """`name`, or a raise naming it. The only gate on the vocabulary."""
        for ref known in Self.vocabulary():
            if name == known:
                return name^
        raise Error("unknown aggregate '", name, "'")

    def resolve(self, in_dtypes: List[DynType]) raises -> ResolvedAggregate:
        """The one name x dtype ladder in the system.

        Public because the **operator holds this node** and calls it on first
        push, the same way `FusedAccumulator` reaches into the `A` it holds. That
        is what lets one type carry both the plan-time and the execution-time
        answer instead of a separate function object beside it.

        Each arm names a single `AggKernel` and takes **both** answers
        from it, so the output dtype and the implementation cannot disagree —
        they come from the same type on the same branch. Two hand-written
        tables could, and `min` over `timestamp[us]` declaring `timestamp[s]`
        is a `Variant` misaccess at emit rather than a raise.

        `List[DynType]` rather than one dtype because `string_agg(x, sep)` has
        two different input dtypes; every aggregate resolved today takes one,
        which each conformer's `dtype` states for itself.
        """
        if self._name == COUNT_DISTINCT:
            return ResolvedAggregate.of[DistinctCount[True]](in_dtypes)
        elif self._name == APPROX_COUNT_DISTINCT:
            return ResolvedAggregate.of[DistinctCount[False]](in_dtypes)
        elif self._name == COUNT:
            # A validity scan, defined for every dtype — which is why it is
            # not a `Fold[CountKernel]` here. The comptime lane still
            # fuses numeric `count`; see `ValidCount`.
            return ResolvedAggregate.of[ValidCount](in_dtypes)
        elif self._name == SUM:
            return ResolvedAggregate.of[Fold[SumKernel]](in_dtypes)
        elif self._name == PRODUCT:
            return ResolvedAggregate.of[Fold[ProductKernel]](in_dtypes)
        elif self._name == MEAN:
            return ResolvedAggregate.of[Fold[MeanKernel]](in_dtypes)
        elif self._name == VARIANCE:
            # `ddof=0` — the population form, Arrow's default. A sample
            # variance is `Dispersion[1, False]`, which needs two more names
            # here (`var_samp`, `stddev_samp`) and nothing else.
            return ResolvedAggregate.of[Dispersion[0, False]](in_dtypes)
        elif self._name == STDDEV:
            return ResolvedAggregate.of[Dispersion[0, True]](in_dtypes)
        elif self._name == MIN or self._name == MAX:
            # The only place a *dtype* selects an implementation. A string
            # extremum is a bytewise scan and a fixed-width one is a grouped, and
            # they are two `AggKernel`s rather than one with a runtime
            # branch, so neither compiles the other's body.
            if len(in_dtypes) != 1:
                raise Error(
                    "aggregate '",
                    self._name,
                    "' takes exactly one input, got ",
                    len(in_dtypes),
                )
            var stringly = (
                in_dtypes[0].is_string() or in_dtypes[0].is_large_string()
            )
            if self._name == MIN:
                if stringly:
                    return ResolvedAggregate.of[StringExtremum[MinOp]](
                        in_dtypes
                    )
                return ResolvedAggregate.of[Fold[MinKernel]](in_dtypes)
            else:
                if stringly:
                    return ResolvedAggregate.of[StringExtremum[MaxOp]](
                        in_dtypes
                    )
                return ResolvedAggregate.of[Fold[MaxKernel]](in_dtypes)
        else:
            raise Error("unknown aggregate '", self._name, "'")

    def empty(self) raises -> Optional[DynArray]:
        """This aggregate's answer over an input that produced no column at
        all, or `None` when it has none.

        Delegates to the same conformers `resolve` names, and needs no dtypes —
        which is the whole point. A filter that keeps nothing answers with no
        batch, so the aggregate above it never learns its input's type.
        `COUNT(x)` and `COUNT(DISTINCT x)` of nothing are 0 regardless; `sum`,
        `mean` and the extrema are NULL, and their *dtype* is known only to the
        plan's schema, so they decline here and `GroupByOperator` fills the
        slot.
        """
        if self._name == COUNT_DISTINCT or self._name == APPROX_COUNT_DISTINCT:
            return DistinctCount[True].empty()
        elif self._name == COUNT:
            return ValidCount.empty()
        return Fold[SumKernel].empty()

    # -- Value --------------------------------------------------------------

    def columns(self) -> List[String]:
        var out = List[String]()
        for ref i in self._inputs:
            out = merged(out^, i.columns())
        return out^

    def name(self) -> String:
        return self._alias.copy()

    def dtype(self, schema: Schema) raises -> DynType:
        """Resolve from schema-derived dtypes and keep only the output type.

        The `AggregateFn` this also produces is discarded, deliberately: it is
        one function-pointer assignment, and paying it is what makes the dtype
        and the implementation come from the same branch.
        """
        var in_dtypes = List[DynType](capacity=len(self._inputs))
        for ref i in self._inputs:
            in_dtypes.append(i.dtype(schema))
        return self.resolve(in_dtypes).dtype.copy()

    comptime shape = Shape.scalar
    """One value per group, so scalar-shaped in the same sense a literal is."""

    # -- to_operator --------------------------------------------------------

    def to_operator(
        self, grouped: Bool, bindings: Bindings = Bindings()
    ) raises -> DynOperator:
        """`grouped` stays a runtime `Bool` here rather than becoming a
        comptime placement: there is no fused loop for the choice to
        specialise, and the operator only needs to know whether to keep the
        group ids."""
        var inputs = List[DynOperator](capacity=len(self._inputs))
        for ref i in self._inputs:
            inputs.append(i.to_operator(False, bindings))
        return AggregateOperator(
            RuntimeAccumulator(inputs^, self.copy(), grouped)
        )

    def alias(self, var name: String) raises -> Self:
        """Rename this aggregate. Changes **only** `_alias`, so the resolver
        still sees `_name` and `.alias("n")` cannot change which kernel
        runs."""
        return Self(self._inputs.copy(), self._name.copy(), name^)

    def write_to[W: Writer](self, mut writer: W):
        writer.write(self._name, "(")
        for i in range(len(self._inputs)):
            if i > 0:
                writer.write(", ")
            writer.write(self._inputs[i])
        writer.write(")")


struct RuntimeAccumulator(Accumulable):
    """The runtime lane's aggregate operator: a **name**, resolved on the first
    morsel, over erased inputs.

    Erased in three places, all forced by the lane rather than chosen: the
    operands are `DynOperator` because a `RuntimeValue` subtree has no type to
    keep, the implementation is an `AggregateFn` pointer because the name is
    not known until run time, and the inputs' dtypes are not known until a
    morsel arrives. `BufferedAccumulator` in `comptime/` is the same
    machine with all three known statically, and it shares no code with this on
    purpose: erasure is the *only* difference, and paying for it in a lane that
    does not need it is what the size gates exist to catch.

    `_scatters` stays a runtime `Bool` where `BufferedAccumulator` makes it a
    `G: Grouping` parameter. That asymmetry is deliberate: parameterising here
    would double an instantiation on the lane that has already accepted
    interpretation, to remove one predictable branch per morsel.

    It also buffers — the accumulated columns and ids are **O(rows)** — but
    that is a property of `AggregateFn` being one-shot, not of erasure. Its
    typed counterpart buffers identically.

    **Zero type parameters, shared by both lanes.** The comptime lane hands it
    `Agg.grouped` — a statically known method taken as a thin pointer — and the
    runtime lane hands it the `RuntimeAggregate` node to resolve on first push.
    The
    operand keeps its fusion either way, because this operator does not
    evaluate the operand: `_inputs` holds the operand's *own* operator, which
    for a comptime subtree is a fully monomorphised `EvalOperator[A]`.

    **It buffers, and that is inherent.** A `AggregateFn` is one-shot over the
    whole input, so each morsel's evaluated columns and group ids are kept and
    `concat`ed at `drain`. Concatenating ids across morsels is sound because
    `HashGrouping` assigns dense ids that are stable across batches — batch
    N+1 extends the numbering rather than restarting it.

    Buffering the ids per aggregate is a known, bounded redundancy: three
    column-aggregates in one query keep three copies of the same `Int32` per
    row. `GroupByOperator` sees every morsel and could keep one, but only by
    special-casing this operator among the values it holds, which is the
    uniformity that made "every value answers `to_operator`" work at all.

    **Exactly one of `_grouped` and `_node` is set at construction, and which
    one says which lane built this.** The comptime lane knows its `AggKernel`
    at compile time and supplies `_grouped` directly — which is what keeps a
    binary that only ever aggregates a comptime expression from linking the
    name ladder at all, and that DCE property is what the whole two-node design
    turns on. The runtime lane knows only a name, so it supplies `_node` and
    `_grouped` is filled on **first push**, from the morsel's real dtypes; it
    cannot be filled earlier, because a `RuntimeValue` operand has no dtype
    until a schema is in hand. `_grouped` is therefore `Some` from the first
    push onward in both cases, which is the invariant `drain` reads.

    Two nullable fields rather than a `Variant`: the alignment hazard
    CLAUDE.md records for `Variant` is not worth paying for two small members,
    so the invariant is stated here instead of encoded.
    """

    var _inputs: List[DynOperator]
    var _grouped: Optional[AggregateFn]
    """The implementation. `Some` from the first push onward, in both lanes."""

    var _node: Optional[RuntimeAggregate]
    """The node itself, for the lane that has a name instead of a type — an
    operator here already holds its node (`FusedAccumulator._input`,
    `EvalOperator._value`), so the ladder needs no object of its own."""

    var _empty: Optional[DynArray]
    """This aggregate's answer over an input that produced no column at all.
    Known without any dtype, so it is computed at construction rather than
    being a third code path at `drain`."""

    var _scatters: Bool
    """Whether the query has `GROUP BY` keys.

    A runtime `Bool` rather than a per-morsel `Groups.is_single()` read: a
    grouped query whose first batch happens to hold exactly one group would
    otherwise be mistaken for an ungrouped one, and that batch's ids dropped.
    """

    var _chunks: List[List[DynArray]]
    var _ids: List[DynArray]
    var _num_groups: Int
    var _rows: Int

    def __init__(
        out self,
        var inputs: List[DynOperator],
        var node: RuntimeAggregate,
        scatters: Bool,
    ) raises:
        """The runtime lane's constructor: the aggregate is a name, so the
        implementation waits for the first morsel's dtypes."""
        self._chunks = List[List[DynArray]]()
        for _ in range(len(inputs)):
            self._chunks.append(List[DynArray]())
        self._inputs = inputs^
        self._grouped = None
        self._empty = node.empty()
        self._node = node^
        self._scatters = scatters
        self._ids = List[DynArray]()
        self._num_groups = 0 if scatters else 1
        self._rows = 0

    def absorb(mut self, morsel: Morsel) raises:
        """Evaluate this aggregate's inputs against the morsel and keep them.

        Empty batches are kept too rather than skipped: an empty column still
        carries its dtype, which is the one thing a name-resolved fold needs
        and the one thing a later batch cannot supply retroactively.
        """
        var n = len(morsel.batch)
        # Indexed: an operator is move-only, so a `List` of them cannot be
        # iterated by reference.
        for i in range(len(self._inputs)):
            var d = self._inputs[i].push(morsel)
            self._chunks[i].append(d.value().to_array(n))
        if not self._grouped:
            var in_dtypes = List[DynType](capacity=len(self._chunks))
            for i in range(len(self._chunks)):
                in_dtypes.append(self._chunks[i][0].dtype())
            self._grouped = self._node.value().resolve(in_dtypes).grouped
        self._rows += n
        if self._scatters:
            self._ids.append(morsel.groups.ids.copy().to_dyn())
            self._num_groups = max(self._num_groups, morsel.groups.num_groups)

    def finish(mut self) raises -> Optional[DynArray]:
        if len(self._chunks) == 0 or len(self._chunks[0]) == 0:
            # No morsel ever arrived, so there is no dtype to resolve against.
            # A distinct count still has an answer; an extremum does not, and
            # `GroupByOperator` fills its slot from the output schema.
            if self._empty:
                return self._empty.value().copy()
            return None
        var columns = List[DynArray](capacity=len(self._chunks))
        for i in range(len(self._chunks)):
            columns.append(concat(self._chunks[i], ExecContext.serial()))
        var groups: Groups
        if self._scatters:
            groups = Groups(
                concat(self._ids, ExecContext.serial()).as_int32().copy(),
                self._num_groups,
            )
        else:
            groups = Groups.single(self._rows)
        return self._grouped.value()(groups, columns)

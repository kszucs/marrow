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
- `BufferedAggregateOperator` holds the node and resolves again on first push,
  from the morsel's real dtypes, keeping only `fold`. It cannot resolve
  earlier: a `RuntimeValue` operand has no dtype until a schema is in hand.
  Holding the node is what removes the third type an earlier draft had — an
  operator in this tree already holds its node (`FusedAggregateOperator._input`,
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
from ..physical import (
    BufferedAggregateOperator,
    Datum,
    DynOperator,
    Evaluable,
)


comptime SUM = "sum"
comptime PRODUCT = "product"
comptime MEAN = "mean"
comptime COUNT = "count"
comptime COUNT_DISTINCT = "count_distinct"
comptime APPROX_COUNT_DISTINCT = "approx_count_distinct"
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
struct RuntimeAggregate(Evaluable, Value):
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
    def _checked(var name: String) raises -> String:
        """`name`, or a raise naming it. The only gate on the vocabulary."""
        if not (
            name == SUM
            or name == PRODUCT
            or name == MEAN
            or name == COUNT
            or name == COUNT_DISTINCT
            or name == APPROX_COUNT_DISTINCT
            or name == MIN
            or name == MAX
        ):
            raise Error("unknown aggregate '", name, "'")
        return name^

    def resolve(self, in_dtypes: List[DynType]) raises -> ResolvedAggregate:
        """The one name x dtype ladder in the system.

        Public because the **operator holds this node** and calls it on first
        push, the same way `FusedAggregateOperator` reaches into the `A` it holds. That
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

    # -- Evaluable ----------------------------------------------------------

    def evaluate(self, batch: StructArray, bindings: Bindings) raises -> Datum:
        """An aggregate has no per-batch value, and saying so is the point.

        The same message `Aggregate.evaluate` raises, for the same
        reason: an aggregate reached through an elementwise path is a mistake
        in the plan, and naming it beats half-computing it.
        """
        raise Error(
            "aggregate '",
            self._alias,
            (
                "' cannot be evaluated per batch; use .aggregate() rather than"
                " projecting or filtering on it"
            ),
        )

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
        return BufferedAggregateOperator(inputs^, self.copy(), grouped)

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

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
from ...dtypes import DynType, NumericType, TemporalType, int64
from ...kernels.aggregate import (
    ArithmeticAgg,
    FoldKernel,
    AggKernel,
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
from std.memory import ArcPointer

from ...kernels.core import Groups
from ..physical import (
    Datum,
    DynOperator,
    Morsel,
    Operator,
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
    """What a named aggregate becomes once its input types are known: the
    output dtype **and** the opened, type-erased state machine that produces it.

    **One type, because the two halves must agree.** `of[Agg]` names `Agg`
    exactly once and takes both answers from it, so the declared dtype and the
    implementation cannot disagree — a mismatch is a `Variant` misaccess at
    emit, not a raise. A separate `dtype` struct beside a separate kernel box
    would be two tables to keep in step.

    **Glue, not compute, which is why the erasure lives here.** The runtime
    lane resolves a *name* to a kernel, so it cannot name the type it got;
    nothing in `marrow.kernels` has that problem. The mechanism is `DynValue`'s
    — an `ArcPointer[NoneType]` plus thin trampolines that `rebind` it back —
    except that these trampolines *mutate* the boxed value, because an
    `AggKernel` is a state machine (`open` / `update` / `finish`) and a bare
    function pointer cannot carry state.

    Mojo has no dynamic dispatch, so the indirection is not avoidable here —
    but it is paid once per morsel, not once per row, and the comptime lane
    never constructs one.
    """

    var dtype: DynType
    """What the produced column will be typed as.

    Bare `dtype`, because there is no *input* dtype in this struct for a prefix
    to disambiguate against — the inputs were consumed by `of`.
    """

    var _boxed: ArcPointer[NoneType]
    var _update: def(ArcPointer[NoneType], Groups, List[DynArray]) thin raises
    var _finish: def(ArcPointer[NoneType]) thin raises -> DynArray

    @staticmethod
    def _update_tramp[
        Agg: AggKernel
    ](ptr: ArcPointer[NoneType], groups: Groups, inputs: List[DynArray]) raises:
        rebind[ArcPointer[Agg]](ptr)[].update(groups, inputs)

    @staticmethod
    def _finish_tramp[
        Agg: AggKernel
    ](ptr: ArcPointer[NoneType]) raises -> DynArray:
        return rebind[ArcPointer[Agg]](ptr)[].finish()

    @staticmethod
    def of[Agg: AggKernel](in_dtypes: List[DynType]) raises -> Self:
        """Both halves of a resolution, off **one** `AggKernel`. The only thing
        that builds one, so neither the box/trampoline pairing nor the
        dtype/implementation pairing can be wrong."""
        var ptr = ArcPointer[Agg](Agg.open(in_dtypes))
        return Self(
            Agg.dtype(in_dtypes),
            rebind[ArcPointer[NoneType]](ptr^),
            Self._update_tramp[Agg],
            Self._finish_tramp[Agg],
        )

    def __init__(
        out self,
        var dtype: DynType,
        var boxed: ArcPointer[NoneType],
        update: def(ArcPointer[NoneType], Groups, List[DynArray]) thin raises,
        finish: def(ArcPointer[NoneType]) thin raises -> DynArray,
    ):
        self.dtype = dtype^
        self._boxed = boxed^
        self._update = update
        self._finish = finish

    def update(mut self, groups: Groups, inputs: List[DynArray]) raises:
        self._update(self._boxed, groups, inputs)

    def finish(mut self) raises -> DynArray:
        return self._finish(self._boxed)


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
    """It answers from `RuntimeAggregateOperator.drain`, never per batch — the same
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

    @staticmethod
    def _fold[
        K: FoldKernel
    ](in_dtypes: List[DynType]) raises -> ResolvedAggregate:
        """Resolve a lane algebra against a runtime dtype.

        **The dispatch lives here, not in `Fold`.** `Fold[K, V]` is typed on
        its input like every other kernel, so something has to map a runtime
        dtype onto `V` — and that is the expression layer's job, the same one
        it does for `filter`, `take` and `cast`. `Fold` used to do it itself
        and paid for it: once it had to hold state across morsels, the state
        could not be a typed field and went behind an `ArcPointer`.

        The numeric and temporal arms are separate rather than one
        `dispatch_primitive`, so a kernel whose domain excludes temporal
        columns is never instantiated over one — `AggState`'s compile-time
        domain assertion would fail the build rather than raise.
        """
        if len(in_dtypes) != 1:
            raise Error(
                "aggregate '",
                K.name,
                "' takes exactly one input, got ",
                len(in_dtypes),
            )
        ref d = in_dtypes[0]
        var box = List[ResolvedAggregate]()
        box.reserve(1)

        def numeric[V: NumericType](x: V) raises {mut box, imm}:
            box.append(ResolvedAggregate.of[Fold[K, V]](in_dtypes))

        comptime if conforms_to(K, ArithmeticAgg):
            if not d.is_numeric():
                raise Error(
                    "aggregate '",
                    K.name,
                    "' needs arithmetic, so it is not defined for ",
                    d,
                    " columns",
                )
            d.dispatch_numeric(numeric)
        else:
            if d.is_numeric():
                d.dispatch_numeric(numeric)
            elif d.is_temporal():

                def temporal[V: TemporalType](x: V) raises {mut box, imm}:
                    box.append(ResolvedAggregate.of[Fold[K, V]](in_dtypes))

                d.dispatch_temporal(temporal)
            else:
                raise Error(
                    "aggregate '",
                    K.name,
                    "' is not defined for ",
                    d,
                    " columns",
                )
        return box.pop()

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
            return Self._fold[SumKernel](in_dtypes)
        elif self._name == PRODUCT:
            return Self._fold[ProductKernel](in_dtypes)
        elif self._name == MEAN:
            return Self._fold[MeanKernel](in_dtypes)
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
                return Self._fold[MinKernel](in_dtypes)
            else:
                if stringly:
                    return ResolvedAggregate.of[StringExtremum[MaxOp]](
                        in_dtypes
                    )
                return Self._fold[MaxKernel](in_dtypes)
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
        # Every remaining aggregate declines: `sum`, `mean`, the extrema and
        # the dispersions all answer NULL, and their *dtype* is known only to
        # the plan's schema, so `GroupByOperator` fills the slot.
        return None

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
        return RuntimeAggregateOperator(inputs^, self.copy(), grouped)

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


struct RuntimeAggregateOperator(Operator):
    """The runtime lane's aggregate operator: a **name**, resolved on the first
    morsel, over erased inputs — and streaming, like its comptime sibling.

    Erased in three places, all forced by the lane rather than chosen: the
    operands are `DynOperator` because a `RuntimeValue` subtree has no type to
    keep, the kernel is a `ResolvedAggregate` because the name is not known until
    run time, and the inputs' dtypes are not known until a morsel arrives.
    `ColumnAggregateOperator` in `comptime/` is the same machine with all three
    known statically, and shares no code with this on purpose: erasure is the
    only difference, and paying for it in a lane that does not need it is what
    the size gates exist to catch.

    It holds O(groups), not O(rows). Both operators used to buffer every
    morsel's columns and concatenate at `drain`, because `AggKernel.grouped`
    was one-shot; the kernel is a state machine now and absorbs each morsel as
    it arrives.
    """

    var _inputs: List[DynOperator]
    var _kernel: Optional[ResolvedAggregate]
    """`None` until the first morsel: resolving the name needs the inputs'
    real dtypes, and a `RuntimeValue` operand has none until a schema is in
    hand."""

    var _node: RuntimeAggregate
    """The node itself, kept to resolve against those first dtypes."""

    var _scatters: Bool
    var _emitted: Bool

    def __init__(
        out self,
        var inputs: List[DynOperator],
        var node: RuntimeAggregate,
        scatters: Bool,
    ) raises:
        self._inputs = inputs^
        self._kernel = None
        self._node = node^
        self._scatters = scatters
        self._emitted = False

    def push(mut self, morsel: Morsel) raises -> Optional[Datum]:
        """Evaluate this aggregate's inputs and let the kernel absorb them.

        Empty batches are absorbed too rather than skipped: an empty column
        still carries its dtype, which is the one thing a name-resolved
        aggregate needs and the one thing a later batch cannot supply
        retroactively.
        """
        var n = len(morsel.batch)
        var columns = List[DynArray](capacity=len(self._inputs))
        # Indexed: an operator is move-only, so a `List` of them cannot be
        # iterated by reference.
        for i in range(len(self._inputs)):
            var d = self._inputs[i].push(morsel)
            columns.append(d.value().to_array(n))
        if not self._kernel:
            var in_dtypes = List[DynType](capacity=len(columns))
            for i in range(len(columns)):
                in_dtypes.append(columns[i].dtype())
            self._kernel = self._node.resolve(in_dtypes)
        var groups = morsel.groups.copy() if self._scatters else Groups.single(
            n
        )
        self._kernel.value().update(groups, columns)
        return None

    def drain(mut self) raises -> Optional[Datum]:
        if self._emitted:
            return None
        self._emitted = True
        if not self._kernel:
            # No morsel ever arrived, so there was no dtype to resolve against.
            # A distinct count still has an answer; an extremum does not, and
            # `GroupByOperator` fills its slot from the output schema.
            var answer = self._node.empty()
            if answer:
                return Datum(answer.value().copy())
            return None
        return Datum(self._kernel.value().finish())

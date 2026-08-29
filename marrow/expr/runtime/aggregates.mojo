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
- `to_operator` resolves again, against the same schema, and keeps only the
  `AggKernel` it binds — which is what lets it construct a **fully typed**
  `BufferedAggregateOperator[Agg, RuntimeValue]` with no erased aggregate
  state anywhere. It cannot resolve earlier: a `RuntimeValue` operand has no
  dtype until a schema is in hand.

Both resolutions take their answer off the same branch of the same ladder, so
two-ladder drift is unrepresentable.

**Non-generic on purpose.** A `resolve[Job: def[A: ...]()]` form instantiates
once per closure type — the `_arith[K]` shape, measured at +115,600 bytes. This
resolver has no type parameter, so it is one instantiation for the whole tree,
and the comptime lane's DCE property holds because nothing there names it: a
program built from `col("a", int64)` reaches `AggKernel` conformers
directly and never this ladder.
"""

from ...arrays import Array
from ...dtypes import (
    DynType,
    NumericType,
    PrimitiveType,
    StringLikeType,
    TemporalType,
    int64,
)
from ...kernels.aggregate import (
    dispatch_agg_array,
    agg_vocabulary,
    APPROX_COUNT_DISTINCT,
    COUNT,
    COUNT_DISTINCT,
    MAX,
    MEAN,
    MIN,
    PRODUCT,
    STDDEV,
    SUM,
    VARIANCE,
    ArithmeticAgg,
    MinMaxOp,
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
from ..`comptime`.aggregates import BufferedAggregateOperator
from .values import RuntimeValue
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


def _fold_agg[
    R: Movable, //, K: FoldKernel, Func: def[Agg: AggKernel]() raises -> R
](in_dtype: DynType, func: Func) raises -> R:
    """A lane algebra against a runtime dtype.

    The dispatch lives here, not in `Fold`: `Fold[K, V]` is typed on its input
    like every other kernel, so something has to map a runtime dtype onto `V`,
    and that is the expression layer's job — the same one it does for `filter`,
    `take` and `cast`.

    Numeric and temporal are separate arms rather than one
    `dispatch_primitive`, so a kernel whose domain excludes temporal columns is
    never instantiated over one: `AggState`'s compile-time domain assertion
    would fail the *build* rather than raise.
    """

    def numeric[V: NumericType](d: V) raises {imm func} -> R:
        return func[Fold[K, V]]()

    def temporal[V: TemporalType](d: V) raises {imm func} -> R:
        return func[Fold[K, V]]()

    comptime if conforms_to(K, ArithmeticAgg):
        if not in_dtype.is_numeric():
            raise Error(
                "aggregate '",
                K.name,
                "' needs arithmetic, so it is not defined for ",
                in_dtype,
                " columns",
            )
        return in_dtype.dispatch_numeric(numeric)
    else:
        if in_dtype.is_numeric():
            return in_dtype.dispatch_numeric(numeric)
        elif in_dtype.is_temporal():
            return in_dtype.dispatch_temporal(temporal)
        else:
            raise Error(
                "aggregate '",
                K.name,
                "' is not defined for ",
                in_dtype,
                " columns",
            )


def dispatch_agg[
    R: Movable, //, Func: def[Agg: AggKernel]() raises -> R
](name: StringSlice, in_dtype: DynType, func: Func) raises -> R:
    """**The** name x dtype ladder — one, not two.

    Every arm names a single `AggKernel` and hands it to `func`, so whatever
    the caller wants — an output dtype, an operator — comes off the same type
    on the same branch. Two ladders could disagree, and `min` over
    `timestamp[us]` declaring `timestamp[s]` is a `Variant` misaccess at emit
    rather than a raise.

    The job is a parametric *value*, the same mechanism `DynType.dispatch_*`
    uses. That is what lets one ladder serve two callers: a closure cannot be
    generic over its own trait bound, but a job whose signature names
    `AggKernel` directly can be.
    """
    if name == COUNT or name == COUNT_DISTINCT or name == APPROX_COUNT_DISTINCT:

        def counted[A: Array]() raises {imm func, imm name} -> R:
            if name == COUNT:
                return func[ValidCount[A]]()
            elif name == COUNT_DISTINCT:
                return func[DistinctCount[True, A]]()
            else:
                return func[DistinctCount[False, A]]()

        return dispatch_agg_array(in_dtype, counted)
    elif name == VARIANCE or name == STDDEV:
        # `ddof=0` — the population form, Arrow's default. A sample variance is
        # `Dispersion[1, root, V]`: two more names here and nothing else.
        def dispersed[V: NumericType](d: V) raises {imm func, imm name} -> R:
            if name == VARIANCE:
                return func[Dispersion[0, False, V]]()
            else:
                return func[Dispersion[0, True, V]]()

        if not in_dtype.is_numeric():
            raise Error(
                "aggregate '",
                name,
                "' needs arithmetic, so it is not defined for ",
                in_dtype,
                " columns",
            )
        return in_dtype.dispatch_numeric(dispersed)
    elif name == MIN or name == MAX:
        # The only place a *dtype* selects an implementation. A string extremum
        # is a bytewise scan and a fixed-width one is a fold — two kernels
        # rather than one with a runtime branch, so neither compiles the
        # other's body.
        def extremum[T: StringLikeType](d: T) raises {imm func, imm name} -> R:
            if name == MIN:
                return func[StringExtremum[MinOp, T]]()
            else:
                return func[StringExtremum[MaxOp, T]]()

        if in_dtype.is_string() or in_dtype.is_large_string():
            return in_dtype.dispatch_stringlike(extremum)
        elif name == MIN:
            return _fold_agg[K=MinKernel](in_dtype, func)
        else:
            return _fold_agg[K=MaxKernel](in_dtype, func)
    elif name == SUM:
        return _fold_agg[K=SumKernel](in_dtype, func)
    elif name == PRODUCT:
        return _fold_agg[K=ProductKernel](in_dtype, func)
    elif name == MEAN:
        return _fold_agg[K=MeanKernel](in_dtype, func)
    else:
        raise Error("unknown aggregate '", name, "'")


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
    """It answers from `BufferedAggregateOperator.drain`, never per batch — the
    same answer `Aggregate` gives, for the same reason, through the same
    operator."""

    var _input: RuntimeValue
    """The operand.

    Typed as `RuntimeValue` rather than boxed in a `DynValue`, and singular
    rather than a list. Every construction site already passes exactly one
    `RuntimeValue`; the box cost an erasure hop per morsel and the list cost
    five kernels an arity check apiece. A genuinely two-operand aggregate
    (`corr`, `covar`) is a different family with a different trait, not a
    longer list."""

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

    def __init__(out self, var input: RuntimeValue, var name: String) raises:
        self._input = input^
        self._alias = name.copy()
        self._name = Self._checked(name^)

    def __init__(
        out self,
        var input: RuntimeValue,
        var name: String,
        var display: String,
    ) raises:
        self._input = input^
        self._name = Self._checked(name^)
        self._alias = display^

    @staticmethod
    def vocabulary() -> List[String]:
        """Every name this node accepts.

        Forwards to `agg_vocabulary()` in `kernels/aggregate.mojo`, which is
        where the catalog lives: the names are the kernels' own
        `Kernel.name` constants, and keeping a second list here meant nothing
        enforced that the two agreed. Kept as a method because it is part of
        this node's surface — a frontend asks *this* what it accepts.
        """
        return agg_vocabulary()

    @staticmethod
    def _checked(var name: String) raises -> String:
        """`name`, or a raise naming it. The only gate on the vocabulary, and
        it runs in `__init__` so `col("s").count_distnct()` raises where it was
        written rather than on the first morsel of a long scan."""
        for ref known in agg_vocabulary():
            if name == known:
                return name^
        raise Error("unknown aggregate '", name, "'")

    def columns(self) -> List[String]:
        return self._input.columns()

    def name(self) -> String:
        return self._alias.copy()

    comptime shape = Shape.scalar
    """One value per group, so scalar-shaped in the same sense a literal is."""

    # -- plan time ----------------------------------------------------------

    def dtype(self, schema: Schema) raises -> DynType:
        """The output dtype, at plan time, before any data exists."""
        var d = self._input.dtype(schema)

        def job[Agg: AggKernel]() raises {imm} -> DynType:
            return Agg.dtype(d)

        return dispatch_agg(self._name, d, job)

    # `empty()` — the one-row answer over an input that produced no column at
    # all — used to live here, a four-arm switch on `_name` that only the
    # cardinalities answered. It had **no production caller**: `GroupByOperator`
    # holds its folds as `DynOperator`s and cannot reach a node, so there was
    # nowhere for it to be called from, and its null-fill branch was unreachable
    # besides — the aggregate operators always answer `Some` from their first
    # `drain`. `AggKernel.reserve` now covers the case properly and covers all
    # ten names rather than three: an operator with no keys seeds one slot at
    # `drain`, so `count_distinct` of nothing is one `0` and `min` of nothing is
    # one NULL, computed by the kernel that owns the answer instead of by a
    # second switch that could disagree with it.

    # -- to_operator --------------------------------------------------------

    def to_operator(
        self, schema: Schema, grouped: Bool, bindings: Bindings = Bindings()
    ) raises -> DynOperator:
        """Resolve the name against the input dtype and hand back a **fully
        typed** operator, in the box every operator already pays for.

        This is why `to_operator` takes a schema. `dispatch_agg` binds a
        concrete `Agg`, so the job below constructs
        `BufferedAggregateOperator[Fold[K, V], RuntimeValue]` outright: no
        erased aggregate state, no per-morsel narrowing, no second box.

        **Always the buffered operator, never the fused one.** Fusion needs an
        operand that answers `lane[W]`, and a `RuntimeValue` materialises a
        `DynArray` per node by construction — so `grouped` stays a plain `Bool`
        field here rather than becoming a comptime parameter. Reusing the
        comptime lane's operator is what deleted this lane's own copy of it.
        """
        var d = self._input.dtype(schema)

        def job[Agg: AggKernel]() raises {imm} -> DynOperator:
            return BufferedAggregateOperator[Agg, RuntimeValue](
                self._input.copy(), bindings.copy(), grouped, d
            )

        return dispatch_agg(self._name, d, job)

    def alias(self, var name: String) raises -> Self:
        """Rename this aggregate. Changes **only** `_alias`, so the resolver
        still sees `_name` and `.alias("n")` cannot change which kernel runs."""
        return Self(self._input.copy(), self._name.copy(), name^)

    def write_to[W: Writer](self, mut writer: W):
        writer.write(self._name, "(", self._input, ")")

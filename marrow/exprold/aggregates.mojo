"""Aggregate expressions — the aggregate side of the expression system.

`values.mojo` describes what a *column* expression is; this module describes
what an *aggregate* is, in the same spirit: a type, resolved as far as it can
be before anything runs.

Three layers, each one narrower than the last:

- **`AggFunction`** (``Sum``, ``Product``, ``Mean``, ``Min``, ``Max``,
  ``Count``, ``CountDistinct``, ``ApproxCountDistinct``) — an aggregate before
  its input type is known. It states which dtypes it is defined for and, for
  each, which implementation runs.
- **`Aggregation`** (``NumericAgg[K, V]``,
  ``StringMinMax[Op, T]``, ``CountAgg``, ``DistinctAgg[exact]``) — that function
  bound to one input type. These are *kernels* and live in
  ``marrow.kernels.aggregate``: each names its own ``InArray``/``OutArray`` and
  carries the whole per-column implementation, so the routing that used to be a
  name comparison (the bytewise string scan, the temporal fold, the
  validity-only count, the distinct sketches) *is* which type was resolved.
- **`AggFunc` / `FoldedAggregates`** — the erasure a *plan* needs, because a
  query's aggregate list is heterogeneous and only known at run time.
  ``AggFunc`` is one aggregate behind a single function pointer into its
  ``Aggregation``, and a plan holds a plain ``List`` of them — there is nothing
  a wrapper would add. ``FoldedAggregates`` is what an eager ``GroupBy`` driver
  holds instead, adding the partial/merge fold that lets a runtime aggregate set
  take the thread-local path. Both boxes for a member come out of *one*
  ``resolve_agg``, and nothing dispatches on a name afterwards.

Two ways in, one destination:

- **fused / AOT** — ``aggs.append[NumericAgg[SumKernel, Int64Type]](int64)``.
  Nothing is interpreted: no name, no dtype resolution, a direct pointer to
  ``AggState[SumKernel, Int64Type]``, everything else dead code.
- **dynamic** — ``aggs.append("sum", value_dtype)``, which goes through
  ``marrow.exprold.aggregates.resolve_agg``: one string comparison per aggregate, once,
  when the plan is built. The Python ``group_by(...).aggregate([...])`` binding
  and ``DynRelation.aggregate(...)`` start here.

The fused path is the dynamic one with the resolution removed — never a second
implementation.
"""

from ..arrays import (
    DynArray,
    StructArray,
    Int32Array,
    Int64Array,
)
from ..dtypes import DynType, Field
from ..execution import ExecContext
from ..schema import Schema
from ..tabular import RecordBatch
from ..dtypes import NumericType, StringLikeType, TemporalType
from ..kernels.core import Groups, Kernel
from ..kernels.aggregate import (
    AggKernel,
    FoldKernel,
    Fold,
    StringExtremum,
    ValidCount,
    DistinctCount as DistinctCountKernel,
    MinMax,
    MinMaxOp,
    MinOp,
    MaxOp,
    SumKernel,
    ProductKernel,
    MeanKernel,
    CountKernel,
)
from ..kernels.groupby import AggregateSet, GroupBy, GroupedColumns


# ---------------------------------------------------------------------------
# The aggregate functions — the frontend's vocabulary.
#
# An `AggFunction` is an aggregate before its input type is known: it states
# which dtypes it is defined for and, for each, which `Aggregation` implements
# it. That is the only dispatch left, and `resolve_agg` below is the only place
# a runtime *name* is compared — once per aggregate, when the plan is built.
# ---------------------------------------------------------------------------


trait AggFunction(Kernel):
    """An aggregate *function*: a name plus the input dtypes it supports.

    Moved out of ``marrow.kernels.aggregate`` when the aggregate vocabulary was
    unified: no kernel in that package turns a name into behaviour, and this is
    a name. It stays here only as long as this package does.
    """

    @staticmethod
    def resolve[
        Job: def[A: AggKernel]() raises -> None
    ](value_dtype: DynType, job: Job) raises:
        """Run ``job[A]`` with the ``AggKernel`` implementing this function over
        a ``value_dtype`` column. Raises if it is not defined for it."""
        ...


struct NumericFold[K: FoldKernel](AggFunction):
    """`sum` / `product` / `mean` — folds defined over numeric columns only."""

    comptime name = Self.K.name

    @staticmethod
    def resolve[
        Job: def[A: AggKernel]() raises -> None
    ](value_dtype: DynType, job: Job) raises:
        # No dtype dispatch left: `Fold[K]` takes an erased column and gates
        # the domain itself.
        _ = Fold[Self.K].dtype([value_dtype.copy()])
        job[Fold[Self.K]]()


struct OrderPreserving[Op: MinMaxOp](AggFunction):
    """`min` / `max` — defined wherever a total order is: numeric and temporal
    columns fold through the same typed `AggState`, string columns through the
    bytewise scan. Both keep the input dtype."""

    comptime name = Self.Op.name

    @staticmethod
    def resolve[
        Job: def[A: AggKernel]() raises -> None
    ](value_dtype: DynType, job: Job) raises:
        if value_dtype.is_numeric() or value_dtype.is_temporal():
            job[Fold[MinMax[Self.Op]]]()
        elif value_dtype.is_string() or value_dtype.is_large_string():
            job[StringExtremum[Self.Op]]()
        else:
            raise Error(
                "aggregate '",
                Self.name,
                "' is not defined for ",
                value_dtype,
                " columns",
            )


struct CountValid(AggFunction):
    """`count` — defined for every dtype. Numeric columns take the mergeable
    `AggState` fold; everything else the validity-only scan."""

    comptime name = CountKernel.name

    @staticmethod
    def resolve[
        Job: def[A: AggKernel]() raises -> None
    ](value_dtype: DynType, job: Job) raises:
        if value_dtype.is_numeric():
            job[Fold[CountKernel]]()
        else:
            job[ValidCount]()


struct DistinctCount[exact: Bool](AggFunction):
    """`count_distinct` / `approx_count_distinct` — defined for every dtype."""

    comptime name = DistinctCountKernel[Self.exact].name

    @staticmethod
    def resolve[
        Job: def[A: AggKernel]() raises -> None
    ](value_dtype: DynType, job: Job) raises:
        job[DistinctCountKernel[Self.exact]]()


comptime Sum = NumericFold[SumKernel]
comptime Product = NumericFold[ProductKernel]
comptime Mean = NumericFold[MeanKernel]
comptime Min = OrderPreserving[MinOp]
comptime Max = OrderPreserving[MaxOp]
comptime Count = CountValid
comptime CountDistinct = DistinctCount[True]
comptime ApproxCountDistinct = DistinctCount[False]


def resolve_agg[
    Job: def[A: AggKernel]() raises -> None
](name: String, value_dtype: DynType, job: Job) raises:
    """Resolve an aggregate function *name* over a ``value_dtype`` column to the
    ``Aggregation`` that implements it, and run ``job[A]``.

    Keyed on the functions' own ``name``, with no tag in between. Each branch
    hands the dtype straight to that function's own ``resolve``, so which input
    types an aggregate supports — and what it does for each — is stated by the
    aggregate itself, never re-listed here."""
    if name == Sum.name:
        Sum.resolve(value_dtype, job)
    elif name == Product.name:
        Product.resolve(value_dtype, job)
    elif name == Mean.name:
        Mean.resolve(value_dtype, job)
    elif name == Min.name:
        Min.resolve(value_dtype, job)
    elif name == Max.name:
        Max.resolve(value_dtype, job)
    elif name == Count.name:
        Count.resolve(value_dtype, job)
    elif name == CountDistinct.name:
        CountDistinct.resolve(value_dtype, job)
    elif name == ApproxCountDistinct.name:
        ApproxCountDistinct.resolve(value_dtype, job)
    else:
        raise Error("unknown aggregate function: ", name)


# ---------------------------------------------------------------------------
# AggFunc — the erased aggregate a plan node holds
# ---------------------------------------------------------------------------


struct AggFunc(Copyable, Movable, Writable):
    """One aggregate, erased to a single thin function pointer over a
    **comptime** ``Aggregation``.

    *Why a function pointer and not a type parameter?* Because a plan holds a
    **runtime-length list of different aggregates** — ``sum(a), min(b),
    count(c)`` is three unrelated ``Aggregation`` types in one ``List``, and
    Mojo has no dynamic dispatch. A type parameter carries the aggregation only
    while there is a single statically known one, which is exactly the AOT path:
    ``GroupBy.aggregate[A]`` takes the type, and no box exists at all. This box
    is what the *heterogeneous* case costs, and it is deliberately one pointer
    wide.

    ``_grouped_fn`` is a comptime instantiation of ``_grouped``, whose body is
    two O(1) handle conversions around ``A.grouped``, so nothing is interpreted
    at run time: the plan points straight at the monomorphized aggregation, and
    the input dtype was resolved when the plan was built, not per batch.

    ``out_dtype`` and ``is_mergeable`` are *values* read off the aggregation at
    construction, not calls: the output dtype of ``min`` over a timestamp column
    carries that column's unit and timezone, and mergeability is a property of
    the resolved (kernel, input type) pair.

    **Kept deliberately small.** Every field of an erased box is live code for
    every aggregation the name switch can produce, so a capability nobody on this
    path calls is still paid for in full. Folding the eager drivers' whole-table
    reduce and partial/merge fold in here (see ``AggFold``) measured **+3.2 MB
    (+24 %)** on the aggregate binary-size gate."""

    var name: String
    """The function's own name — the default output column name. Nothing
    dispatches on it: the one place a name becomes a type is ``resolve_agg``,
    and everything an aggregate can do comes out of that single resolution."""

    var out_dtype: DynType
    """This aggregate's output dtype over the column it was resolved against."""

    var is_mergeable: Bool
    """Whether this aggregate can run as thread-local partials plus a merge (see
    ``AggFold``). The sketch aggregates carry no mergeable scalar accumulator,
    and the non-fold scans (string ``min``/``max``, ``count`` of a non-numeric
    column) are dedicated per-group passes rather than the ``AggState`` fold the
    merge understands."""

    var _grouped_fn: def(Groups, DynArray) thin raises -> DynArray

    @staticmethod
    def _grouped[
        A: AggKernel
    ](groups: Groups, value: DynArray) raises -> DynArray:
        return A.grouped(groups, [value.copy()])

    def __init__(
        out self,
        *,
        var func_name: String,
        var out_dtype: DynType,
        is_mergeable: Bool,
        grouped_fn: def(Groups, DynArray) thin raises -> DynArray,
    ):
        self.name = func_name^
        self.out_dtype = out_dtype^
        self.is_mergeable = is_mergeable
        self._grouped_fn = grouped_fn

    @staticmethod
    def of[A: AggKernel](value_dtype: DynType) raises -> AggFunc:
        """The fused (AOT) form: the aggregation is named directly, so nothing
        is resolved at run time and every other instantiation is dead code."""
        return AggFunc(
            func_name=String(A.name),
            out_dtype=A.dtype([value_dtype.copy()]),
            is_mergeable=A.mergeable,
            grouped_fn=Self._grouped[A],
        )

    def __init__(out self, name: String, value_dtype: DynType) raises:
        """The dynamic form: resolve a function *name* over a column dtype
        (``marrow.exprold.aggregates.resolve_agg`` — the one string comparison).
        """
        var box = List[AggFunc]()

        def make[A: AggKernel]() raises {mut box, imm}:
            box.append(Self.of[A](value_dtype))

        resolve_agg(name, value_dtype, make)
        self = box[0].copy()

    def grouped(self, groups: Groups, value: DynArray) raises -> DynArray:
        """One aggregate column over precomputed group ids."""
        return self._grouped_fn(groups, value)

    def write_to[W: Writer](self, mut writer: W):
        writer.write(self.name)


struct AggFold(Copyable, Movable):
    """The thread-local partial/merge fold, erased over the same comptime
    ``Aggregation`` as ``AggFunc``.

    Split out of ``AggFunc`` on purpose: a relational plan never merges partials
    or reduces a whole table, and an erased box pays for every field it declares
    (see ``AggFunc``'s note), so the fused/AOT path links none of this.

    Constructible **only** from a comptime ``AggKernel`` (``of[A]``). There is
    no name-keyed constructor, deliberately: a second dispatch on a string is
    exactly what the aggregate layer exists not to do, and
    ``FoldedAggregates.append`` gets this and its ``AggFunc`` out of the same
    single resolution."""

    var _in_dtype: DynType
    """The dtype of the column these fold. Held because ``merge`` only ever
    sees accumulators, and a widening fold loses the input type on the way out:
    ``sum(int32)`` hands back an int64 column that cannot say what it came
    from."""

    var _whole_fn: def(DynType, DynArray) thin raises -> DynArray
    var _partials_fn: def(DynType, Groups, DynArray) thin raises -> Tuple[
        DynArray, Int64Array
    ]
    var _merge_fn: def(
        DynType, List[Int32Array], List[DynArray], List[Int64Array], Int
    ) thin raises -> DynArray

    @staticmethod
    def _whole[
        A: AggKernel
    ](in_dtype: DynType, value: DynArray) raises -> DynArray:
        return A.grouped(Groups.single(len(value)), [value.copy()])

    @staticmethod
    def _partials[
        A: AggKernel
    ](in_dtype: DynType, groups: Groups, value: DynArray) raises -> Tuple[
        DynArray, Int64Array
    ]:
        return A.partials(in_dtype, groups, [value.copy()])

    @staticmethod
    def _merge[
        A: AggKernel
    ](
        in_dtype: DynType,
        remap: List[Int32Array],
        accs: List[DynArray],
        cnts: List[Int64Array],
        num_groups: Int,
    ) raises -> DynArray:
        return A.merge(in_dtype, remap, accs, cnts, num_groups)

    def __init__(
        out self,
        *,
        var in_dtype: DynType,
        whole_fn: def(DynType, DynArray) thin raises -> DynArray,
        partials_fn: def(DynType, Groups, DynArray) thin raises -> Tuple[
            DynArray, Int64Array
        ],
        merge_fn: def(
            DynType, List[Int32Array], List[DynArray], List[Int64Array], Int
        ) thin raises -> DynArray,
    ):
        self._in_dtype = in_dtype^
        self._whole_fn = whole_fn
        self._partials_fn = partials_fn
        self._merge_fn = merge_fn

    @staticmethod
    def of[A: AggKernel](var in_dtype: DynType) -> AggFold:
        return AggFold(
            in_dtype=in_dtype^,
            whole_fn=Self._whole[A],
            partials_fn=Self._partials[A],
            merge_fn=Self._merge[A],
        )

    def whole(self, value: DynArray) raises -> DynArray:
        """The whole-table aggregate as a one-row column.

        Takes no `ExecContext`: `AggregateFn` carries none, so every aggregate
        decides its own parallelism from `ExecContext.auto()` internally. That
        is a real narrowing — `count_distinct` used to receive the caller's
        context and go radix-parallel against *its* worker budget, and a GPU
        device could be handed down. Both now come from `auto()`."""
        return self._whole_fn(self._in_dtype, value)

    def partials(
        self, groups: Groups, value: DynArray
    ) raises -> Tuple[DynArray, Int64Array]:
        """A thread-local partial fold: raw per-group accumulator + valid counts.
        """
        return self._partials_fn(self._in_dtype, groups, value)

    def merge(
        self,
        remap: List[Int32Array],
        accs: List[DynArray],
        cnts: List[Int64Array],
        num_groups: Int,
    ) raises -> DynArray:
        """Fold every thread's partials at remapped group ids and finalize."""
        return self._merge_fn(self._in_dtype, remap, accs, cnts, num_groups)


struct FoldedAggregates(AggregateSet, Copyable, Movable, Sized):
    """N aggregates resolved *once* into everything the eager group-by drivers
    need: the per-column `grouped` entry point and the partial/merge fold.

    Both boxes come out of a single `resolve_agg` — one name comparison per
    aggregate, at construction, and none afterwards. That is the whole point:
    resolving the fold separately, later, by name would be a second dispatch on
    a string, and doing it eagerly on the plan-side `Aggregates` instead would
    link the fold code into every plan (measured **+1.2x** on the runtime-named
    binary-size gate, for capabilities a plan never calls).

    So the split is by *who needs what*, not by convenience: a plan holds
    `Aggregates` and aggregates column by column; an eager `GroupBy` driver
    holds this and can also fold thread-local partials."""

    var _funcs: List[AggFunc]
    var _folds: List[AggFold]

    def __init__(out self):
        self._funcs = List[AggFunc]()
        self._folds = List[AggFold]()

    def __len__(self) -> Int:
        return len(self._funcs)

    def append[A: AggKernel](mut self, value_dtype: DynType) raises:
        """Add aggregation ``A`` over a ``value_dtype`` column (AOT — the
        aggregation is named, so there is nothing to resolve)."""
        self._funcs.append(AggFunc.of[A](value_dtype))
        self._folds.append(AggFold.of[A](value_dtype.copy()))

    def append(mut self, name: String, value_dtype: DynType) raises:
        """Add the aggregate ``name`` over a ``value_dtype`` column — the one
        name→type step, producing both boxes from the same ``AggKernel``."""
        var funcs = List[AggFunc]()
        var folds = List[AggFold]()

        def make[A: AggKernel]() raises {mut funcs, mut folds, imm}:
            funcs.append(AggFunc.of[A](value_dtype))
            folds.append(AggFold.of[A](value_dtype.copy()))

        resolve_agg(name, value_dtype, make)
        self._funcs.append(funcs[0].copy())
        self._folds.append(folds[0].copy())

    def names(self) -> List[String]:
        var out = List[String]()
        for ref f in self._funcs:
            out.append(f.name)
        return out^

    # -- AggregateSet: what the grouper asks of an aggregate set --------

    def num_columns(self) -> Int:
        return len(self._funcs)

    def mergeable(self) -> Bool:
        """Whether *every* member can run as thread-local partials + a merge.
        One that cannot disqualifies the set, because the whole set shares a
        single grouping."""
        for ref f in self._funcs:
            if not f.is_mergeable:
                return False
        return True

    def grouped(
        self, column: Int, groups: Groups, values: DynArray
    ) raises -> DynArray:
        return self._funcs[column].grouped(groups, values)

    def partials(
        self, column: Int, groups: Groups, values: DynArray
    ) raises -> Tuple[DynArray, Int64Array]:
        return self._folds[column].partials(groups, values)

    def merge(
        self,
        column: Int,
        remap: List[Int32Array],
        accs: List[DynArray],
        cnts: List[Int64Array],
        num_groups: Int,
    ) raises -> DynArray:
        return self._folds[column].merge(remap, accs, cnts, num_groups)

    # -- the drivers --------------------------------------------------------

    def grouped(
        self, gb: GroupBy, values: List[DynArray]
    ) raises -> RecordBatch:
        """Apply every aggregate over one grouping of ``gb``'s keys.

        ``values[j]`` is aggregated with member ``j``. Returns the unique key
        columns followed by one column per aggregate, so a multi-aggregate query
        hashes and probes the keys once instead of once per aggregate. Which
        execution strategy runs is the grouper's decision: it is handed this set
        and asks it whether the columns are mergeable."""
        return self._named(gb.aggregate_all(self, values), gb.keys())

    def whole(
        self, values: List[DynArray], ctx: ExecContext = ExecContext.auto()
    ) raises -> RecordBatch:
        """Whole-table aggregation — ``SELECT agg(x), ...`` with no GROUP BY.

        A single implicit group, computed with each aggregation's own
        whole-column path (the vectorized SIMD reduce, ``O(1)`` count, direct
        ``count_distinct``) rather than the grouped scatter. Returns a one-row
        batch. Each aggregate decides its own parallelism internally — the fold
        reductions force one worker, the distinct sketches self-gate on size —
        so ``ctx`` no longer reaches them; see ``AggFold.whole``."""
        if len(values) != len(self._funcs):
            raise Error("aggregate: one value column per aggregate is required")
        var out_fields = List[Field]()
        var out_cols = List[DynArray]()
        for j in range(len(self._funcs)):
            var col = self._folds[j].whole(values[j])
            out_fields.append(Field(self._funcs[j].name, col.dtype().copy()))
            out_cols.append(col^)
        return RecordBatch(schema=Schema(fields=out_fields^), columns=out_cols^)

    def _named(
        self, var grouped: GroupedColumns, keys: StructArray
    ) raises -> RecordBatch:
        """Label a grouping's columns. The kernel layer returns key and
        aggregate *columns* and knows none of their names; the key names come
        from the keys struct and the aggregate names from this set."""
        ref kstruct = keys.dtype.as_struct()
        var fields = List[Field]()
        var cols = List[DynArray]()
        for k in range(len(grouped.keys)):
            fields.append(kstruct.fields[k].copy())
            cols.append(grouped.keys[k].copy())
        for j in range(len(grouped.aggregates)):
            fields.append(
                Field(self._funcs[j].name, grouped.aggregates[j].dtype().copy())
            )
            cols.append(grouped.aggregates[j].copy())
        return RecordBatch(schema=Schema(fields=fields^), columns=cols^)

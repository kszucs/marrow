"""Aggregate expressions — the aggregate side of the expression system.

`values.mojo` describes what a *column* expression is; this module describes
what an *aggregate* is, in the same spirit: a type, resolved as far as it can
be before anything runs.

Three layers, each one narrower than the last:

- **`AggFunction`** (``Sum``, ``Product``, ``Mean``, ``Min``, ``Max``,
  ``Count``, ``CountDistinct``, ``ApproxCountDistinct``) — an aggregate before
  its input type is known. It states which dtypes it is defined for and, for
  each, which implementation runs.
- **`Aggregation`** (``NumericAgg[K, V]``, ``TemporalMinMax[Op, T]``,
  ``StringMinMax[Op, T]``, ``CountAgg``, ``DistinctAgg[exact]``) — that function
  bound to one input type. These are *kernels* and live in
  ``marrow.kernels.aggregate``: each names its own ``InArray``/``OutArray`` and
  carries the whole per-column implementation, so the routing that used to be a
  name comparison (the bytewise string scan, the temporal fold, the
  validity-only count, the distinct sketches) *is* which type was resolved.
- **`AggFunc` / `Aggregates` / `FoldedAggregates`** — the erasure a *plan*
  needs, because a query's aggregate list is heterogeneous and only known at run
  time. ``AggFunc`` is one aggregate behind a single function pointer into its
  ``Aggregation`` and ``Aggregates`` is the list of them, which is all a plan
  holds; ``FoldedAggregates`` is what an eager ``GroupBy`` driver holds, adding
  the partial/merge fold that lets a runtime aggregate set take the
  thread-local path. Both boxes for a member come out of *one* ``resolve_agg``,
  and nothing dispatches on a name afterwards.

Two ways in, one destination:

- **fused / AOT** — ``aggs.append[NumericAgg[SumKernel, Int64Type]](int64)``.
  Nothing is interpreted: no name, no dtype resolution, a direct pointer to
  ``AggState[SumKernel, Int64Type]``, everything else dead code.
- **dynamic** — ``aggs.append("sum", value_dtype)``, which goes through
  ``marrow.expr.dynamic.resolve_agg``: one string comparison per aggregate, once,
  when the plan is built. The Python ``group_by(...).aggregate([...])`` binding
  and ``AnyRelation.aggregate(...)`` start here.

The fused path is the dynamic one with the resolution removed — never a second
implementation.
"""

from ..arrays import (
    AnyArray,
    StructArray,
    Int32Array,
    Int64Array,
)
from ..dtypes import AnyDataType, Field
from ..schema import Schema
from ..tabular import RecordBatch
from ..dtypes import NumericType, StringLikeType, TemporalType
from ..kernels.aggregate import (
    Aggregation,
    AggFunction,
    AggKernel,
    NumericAgg,
    TemporalMinMax,
    StringMinMax,
    CountAgg,
    DistinctAgg,
    MinMax,
    MinMaxOp,
    MinOp,
    MaxOp,
    SumKernel,
    ProductKernel,
    MeanKernel,
    CountKernel,
)
from ..kernels.groupby import ColumnAggregator, GroupBy, GroupedColumns


# ---------------------------------------------------------------------------
# The aggregate functions — the frontend's vocabulary.
#
# An `AggFunction` is an aggregate before its input type is known: it states
# which dtypes it is defined for and, for each, which `Aggregation` implements
# it. That is the only dispatch left, and `resolve_agg` below is the only place
# a runtime *name* is compared — once per aggregate, when the plan is built.
# ---------------------------------------------------------------------------


struct NumericFold[K: AggKernel](AggFunction):
    """`sum` / `product` / `mean` — folds defined over numeric columns only."""

    comptime name = Self.K.name

    @staticmethod
    def resolve[
        job: def[A: Aggregation]() raises capturing[_] -> None
    ](value_dtype: AnyDataType) raises:
        if not value_dtype.is_numeric():
            raise Error(
                "aggregate '",
                Self.name,
                "' is not defined for ",
                value_dtype,
                " columns",
            )

        @parameter
        def numeric[V: NumericType](d: V) raises:
            job[Self.K.Grouped[V]]()

        value_dtype.dispatch_numeric[numeric]()


struct OrderPreserving[Op: MinMaxOp](AggFunction):
    """`min` / `max` — defined wherever a total order is: numeric columns fold
    through `AggState`, temporal columns through their integer backing, and
    string columns through the bytewise scan. All three keep the input dtype."""

    comptime name = Self.Op.name

    @staticmethod
    def resolve[
        job: def[A: Aggregation]() raises capturing[_] -> None
    ](value_dtype: AnyDataType) raises:
        if value_dtype.is_numeric():

            @parameter
            def numeric[V: NumericType](d: V) raises:
                job[MinMax[Self.Op].Grouped[V]]()

            value_dtype.dispatch_numeric[numeric]()
        elif value_dtype.is_temporal():

            @parameter
            def temporal[T: TemporalType](d: T) raises:
                job[TemporalMinMax[Self.Op, T]]()

            value_dtype.dispatch_temporal[temporal]()
        elif value_dtype.is_string() or value_dtype.is_large_string():

            @parameter
            def stringly[T: StringLikeType](d: T) raises:
                job[StringMinMax[Self.Op, T]]()

            value_dtype.dispatch_stringlike[stringly]()
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
        job: def[A: Aggregation]() raises capturing[_] -> None
    ](value_dtype: AnyDataType) raises:
        if value_dtype.is_numeric():

            @parameter
            def numeric[V: NumericType](d: V) raises:
                job[NumericAgg[CountKernel, V]]()

            value_dtype.dispatch_numeric[numeric]()
        else:
            job[CountAgg]()


struct DistinctCount[exact: Bool](AggFunction):
    """`count_distinct` / `approx_count_distinct` — defined for every dtype."""

    comptime name = DistinctAgg[Self.exact].name

    @staticmethod
    def resolve[
        job: def[A: Aggregation]() raises capturing[_] -> None
    ](value_dtype: AnyDataType) raises:
        job[DistinctAgg[Self.exact]]()


comptime Sum = NumericFold[SumKernel]
comptime Product = NumericFold[ProductKernel]
comptime Mean = NumericFold[MeanKernel]
comptime Min = OrderPreserving[MinOp]
comptime Max = OrderPreserving[MaxOp]
comptime Count = CountValid
comptime CountDistinct = DistinctCount[True]
comptime ApproxCountDistinct = DistinctCount[False]


def resolve_agg[
    job: def[A: Aggregation]() raises capturing[_] -> None
](name: String, value_dtype: AnyDataType) raises:
    """Resolve an aggregate function *name* over a ``value_dtype`` column to the
    ``Aggregation`` that implements it, and run ``job[A]``.

    Keyed on the functions' own ``name``, with no tag in between. Each branch
    hands the dtype straight to that function's own ``resolve``, so which input
    types an aggregate supports — and what it does for each — is stated by the
    aggregate itself, never re-listed here."""
    if name == Sum.name:
        Sum.resolve[job](value_dtype)
    elif name == Product.name:
        Product.resolve[job](value_dtype)
    elif name == Mean.name:
        Mean.resolve[job](value_dtype)
    elif name == Min.name:
        Min.resolve[job](value_dtype)
    elif name == Max.name:
        Max.resolve[job](value_dtype)
    elif name == Count.name:
        Count.resolve[job](value_dtype)
    elif name == CountDistinct.name:
        CountDistinct.resolve[job](value_dtype)
    elif name == ApproxCountDistinct.name:
        ApproxCountDistinct.resolve[job](value_dtype)
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

    var out_dtype: AnyDataType
    """This aggregate's output dtype over the column it was resolved against."""

    var is_mergeable: Bool
    """Whether this aggregate can run as thread-local partials plus a merge (see
    ``AggFold``). The sketch aggregates carry no mergeable scalar accumulator,
    and the non-fold scans (string ``min``/``max``, ``count`` of a non-numeric
    column) are dedicated per-group passes rather than the ``AggState`` fold the
    merge understands."""

    var _grouped_fn: def(Int32Array, AnyArray, Int) thin raises -> AnyArray

    @staticmethod
    def _grouped[
        A: Aggregation
    ](gids: Int32Array, value: AnyArray, num_groups: Int) raises -> AnyArray:
        """The erasure boundary: a typed column in, a typed column out, widened
        back to ``AnyArray`` only for the caller's heterogeneous column list."""
        return A.grouped(gids, A.from_any(value), num_groups).to_any()

    def __init__(
        out self,
        *,
        var func_name: String,
        var out_dtype: AnyDataType,
        is_mergeable: Bool,
        grouped_fn: def(Int32Array, AnyArray, Int) thin raises -> AnyArray,
    ):
        self.name = func_name^
        self.out_dtype = out_dtype^
        self.is_mergeable = is_mergeable
        self._grouped_fn = grouped_fn

    @staticmethod
    def of[A: Aggregation](value_dtype: AnyDataType) raises -> AggFunc:
        """The fused (AOT) form: the aggregation is named directly, so nothing
        is resolved at run time and every other instantiation is dead code."""
        return AggFunc(
            func_name=String(A.name),
            out_dtype=A.out_dtype(value_dtype),
            is_mergeable=A.is_mergeable,
            grouped_fn=Self._grouped[A],
        )

    def __init__(out self, name: String, value_dtype: AnyDataType) raises:
        """The dynamic form: resolve a function *name* over a column dtype
        (``marrow.expr.dynamic.resolve_agg`` — the one string comparison)."""
        var box = List[AggFunc]()

        @parameter
        def make[A: Aggregation]() raises:
            box.append(Self.of[A](value_dtype))

        resolve_agg[make](name, value_dtype)
        self = box[0].copy()

    def grouped(
        self, gids: Int32Array, value: AnyArray, num_groups: Int
    ) raises -> AnyArray:
        """One aggregate column over precomputed group ids."""
        return self._grouped_fn(gids, value, num_groups)

    def write_to[W: Writer](self, mut writer: W):
        writer.write(self.name)


struct AggFold(Copyable, Movable):
    """The thread-local partial/merge fold, erased over the same comptime
    ``Aggregation`` as ``AggFunc``.

    Split out of ``AggFunc`` on purpose: a relational plan never merges partials
    or reduces a whole table, and an erased box pays for every field it declares
    (see ``AggFunc``'s note), so the fused/AOT path links none of this.

    Constructible **only** from a comptime ``Aggregation`` (``of[A]``). There is
    no name-keyed constructor, deliberately: a second dispatch on a string is
    exactly what the aggregate layer exists not to do, and
    ``FoldedAggregates.append`` gets this and its ``AggFunc`` out of the same
    single resolution."""

    var _whole_fn: def(AnyArray, Int) thin raises -> AnyArray
    var _partials_fn: def(Int32Array, AnyArray, Int) thin raises -> Tuple[
        AnyArray, Int64Array
    ]
    var _merge_fn: def(
        List[Int32Array], List[AnyArray], List[Int64Array], Int
    ) thin raises -> AnyArray

    @staticmethod
    def _whole[
        A: Aggregation
    ](value: AnyArray, num_threads: Int) raises -> AnyArray:
        return A.whole(A.from_any(value), num_threads).to_any()

    @staticmethod
    def _partials[
        A: Aggregation
    ](gids: Int32Array, value: AnyArray, num_groups: Int) raises -> Tuple[
        AnyArray, Int64Array
    ]:
        var parts = A.partials(gids, A.from_any(value), num_groups)
        return (parts[0].copy().to_any(), parts[1].copy())

    @staticmethod
    def _merge[
        A: Aggregation
    ](
        remap: List[Int32Array],
        accs: List[AnyArray],
        cnts: List[Int64Array],
        num_groups: Int,
    ) raises -> AnyArray:
        var typed = List[A.OutArray]()
        for t in range(len(accs)):
            typed.append(A.OutArray(accs[t].to_data()))
        return A.merge(remap, typed, cnts, num_groups).to_any()

    def __init__(
        out self,
        *,
        whole_fn: def(AnyArray, Int) thin raises -> AnyArray,
        partials_fn: def(Int32Array, AnyArray, Int) thin raises -> Tuple[
            AnyArray, Int64Array
        ],
        merge_fn: def(
            List[Int32Array], List[AnyArray], List[Int64Array], Int
        ) thin raises -> AnyArray,
    ):
        self._whole_fn = whole_fn
        self._partials_fn = partials_fn
        self._merge_fn = merge_fn

    @staticmethod
    def of[A: Aggregation]() -> AggFold:
        return AggFold(
            whole_fn=Self._whole[A],
            partials_fn=Self._partials[A],
            merge_fn=Self._merge[A],
        )

    def whole(self, value: AnyArray, num_threads: Int = 0) raises -> AnyArray:
        """The whole-table aggregate as a one-row column."""
        return self._whole_fn(value, num_threads)

    def partials(
        self, gids: Int32Array, value: AnyArray, num_groups: Int
    ) raises -> Tuple[AnyArray, Int64Array]:
        """A thread-local partial fold: raw per-group accumulator + valid counts.
        """
        return self._partials_fn(gids, value, num_groups)

    def merge(
        self,
        remap: List[Int32Array],
        accs: List[AnyArray],
        cnts: List[Int64Array],
        num_groups: Int,
    ) raises -> AnyArray:
        """Fold every thread's partials at remapped group ids and finalize."""
        return self._merge_fn(remap, accs, cnts, num_groups)


struct FoldedAggregates(ColumnAggregator, Copyable, Movable, Sized):
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

    def append[A: Aggregation](mut self, value_dtype: AnyDataType) raises:
        """Add aggregation ``A`` over a ``value_dtype`` column (AOT — the
        aggregation is named, so there is nothing to resolve)."""
        self._funcs.append(AggFunc.of[A](value_dtype))
        self._folds.append(AggFold.of[A]())

    def append(mut self, name: String, value_dtype: AnyDataType) raises:
        """Add the aggregate ``name`` over a ``value_dtype`` column — the one
        name→type step, producing both boxes from the same ``Aggregation``."""
        var funcs = List[AggFunc]()
        var folds = List[AggFold]()

        @parameter
        def make[A: Aggregation]() raises:
            funcs.append(AggFunc.of[A](value_dtype))
            folds.append(AggFold.of[A]())

        resolve_agg[make](name, value_dtype)
        self._funcs.append(funcs[0].copy())
        self._folds.append(folds[0].copy())

    def names(self) -> List[String]:
        var out = List[String]()
        for ref f in self._funcs:
            out.append(f.name)
        return out^

    # -- ColumnAggregator: what the grouper asks of an aggregate set --------

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
        self, column: Int, gids: Int32Array, values: AnyArray, num_groups: Int
    ) raises -> AnyArray:
        return self._funcs[column].grouped(gids, values, num_groups)

    def partials(
        self, column: Int, gids: Int32Array, values: AnyArray, num_groups: Int
    ) raises -> Tuple[AnyArray, Int64Array]:
        return self._folds[column].partials(gids, values, num_groups)

    def merge(
        self,
        column: Int,
        remap: List[Int32Array],
        accs: List[AnyArray],
        cnts: List[Int64Array],
        num_groups: Int,
    ) raises -> AnyArray:
        return self._folds[column].merge(remap, accs, cnts, num_groups)

    # -- the drivers --------------------------------------------------------

    def grouped(
        self, gb: GroupBy, values: List[AnyArray]
    ) raises -> RecordBatch:
        """Apply every aggregate over one grouping of ``gb``'s keys.

        ``values[j]`` is aggregated with member ``j``. Returns the unique key
        columns followed by one column per aggregate, so a multi-aggregate query
        hashes and probes the keys once instead of once per aggregate. Which
        execution strategy runs is the grouper's decision: it is handed this set
        and asks it whether the columns are mergeable."""
        return self._named(gb.aggregate_all(self, values), gb.keys())

    def whole(
        self, values: List[AnyArray], num_threads: Int = 0
    ) raises -> RecordBatch:
        """Whole-table aggregation — ``SELECT agg(x), ...`` with no GROUP BY.

        A single implicit group, computed with each aggregation's own
        whole-column path (the vectorized SIMD reduce, ``O(1)`` count, direct
        ``count_distinct``) rather than the grouped scatter. Returns a one-row
        batch. Each aggregation decides what to do with the worker budget: the
        fold reductions stay serial, the distinct sketches self-gate on size."""
        if len(values) != len(self._funcs):
            raise Error("aggregate: one value column per aggregate is required")
        var out_fields = List[Field]()
        var out_cols = List[AnyArray]()
        for j in range(len(self._funcs)):
            var col = self._folds[j].whole(values[j], num_threads)
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
        var cols = List[AnyArray]()
        for k in range(len(grouped.keys)):
            fields.append(kstruct.fields[k].copy())
            cols.append(grouped.keys[k].copy())
        for j in range(len(grouped.aggregates)):
            fields.append(
                Field(self._funcs[j].name, grouped.aggregates[j].dtype().copy())
            )
            cols.append(grouped.aggregates[j].copy())
        return RecordBatch(schema=Schema(fields=fields^), columns=cols^)


# ---------------------------------------------------------------------------
# Aggregates — the aggregate *set* a query applies in one pass
# ---------------------------------------------------------------------------


struct Aggregates(Copyable, Movable, Sized, Writable):
    """The N aggregates a query computes, and how to compute them together.

    What a *plan* holds: one `AggFunc` per output column, each already bound to
    the dtype its input turned out to have. A plan aggregates column by column
    (`AggregateProcessor` groups incrementally as morsels arrive), so this list
    deliberately carries nothing else — reaching the thread-local fold from here
    would link that code into every plan.

    An eager `GroupBy` driver wants more, and gets its own type:
    `FoldedAggregates`. Members are added either by naming the `Aggregation`
    (`append[A]`, the AOT form) or by resolving a function name against the
    column's dtype (`append(name, dtype)`, the dynamic form)."""

    var _funcs: List[AggFunc]

    def __init__(out self):
        self._funcs = List[AggFunc]()

    def __init__(out self, var funcs: List[AggFunc]):
        self._funcs = funcs^

    def __len__(self) -> Int:
        return len(self._funcs)

    def __getitem__(ref self, index: Int) -> ref[self._funcs[index]] AggFunc:
        return self._funcs[index]

    def add(mut self, var func: AggFunc):
        """Add an already-resolved aggregate — what an ``AggExpr`` hands over
        once the plan knows its input's dtype."""
        self._funcs.append(func^)

    def append[A: Aggregation](mut self, value_dtype: AnyDataType) raises:
        """Add aggregation ``A`` over a ``value_dtype`` column (AOT)."""
        self._funcs.append(AggFunc.of[A](value_dtype))

    def append(mut self, name: String, value_dtype: AnyDataType) raises:
        """Add the aggregate ``name`` over a ``value_dtype`` column (dynamic).
        """
        self.add(AggFunc(name, value_dtype))

    def names(self) -> List[String]:
        """Each aggregate's default output column name."""
        var out = List[String]()
        for ref f in self._funcs:
            out.append(f.name)
        return out^

    def write_to[W: Writer](self, mut writer: W):
        for i in range(len(self._funcs)):
            if i > 0:
                writer.write(", ")
            self._funcs[i].write_to(writer)

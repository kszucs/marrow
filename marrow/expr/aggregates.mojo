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
- **`AggFunc` / `Aggregates`** — the erasure a *plan* needs, because a query's
  aggregate list is heterogeneous and only known at run time. ``AggFunc`` is one
  aggregate behind a single function pointer into its ``Aggregation``;
  ``Aggregates`` is the set, and it owns the drivers that make N aggregates
  share one grouping pass.

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

from std.algorithm.functional import sync_parallelize

from ..arrays import (
    AnyArray,
    StructArray,
    Int32Array,
    Int64Array,
)
from ..dtypes import AnyDataType, Field
from ..schema import Schema
from ..tabular import RecordBatch
from ..kernels.aggregate import Aggregation
from ..kernels.groupby import GroupBy, HashGrouper, GROUP_THREAD_LOCAL
from .dynamic import resolve_agg


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
    """The function's own name — the default output column name, never dispatch.
    """

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

    Two pointers rather than one box per capability: ``partials`` is called once
    per *thread* and ``merge`` once per aggregate, so resolving them together
    and calling them many times is what the box buys. Anything called exactly
    once (the whole-table reduce) goes straight through ``resolve_agg`` instead.

    Split out of ``AggFunc`` on purpose: a relational plan never merges partials,
    and an erased box pays for every field it declares (see ``AggFunc``'s note),
    so the fused/AOT path links none of this."""

    var _partials_fn: def(Int32Array, AnyArray, Int) thin raises -> Tuple[
        AnyArray, Int64Array
    ]
    var _merge_fn: def(
        List[Int32Array], List[AnyArray], List[Int64Array], Int
    ) thin raises -> AnyArray

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
        partials_fn: def(Int32Array, AnyArray, Int) thin raises -> Tuple[
            AnyArray, Int64Array
        ],
        merge_fn: def(
            List[Int32Array], List[AnyArray], List[Int64Array], Int
        ) thin raises -> AnyArray,
    ):
        self._partials_fn = partials_fn
        self._merge_fn = merge_fn

    @staticmethod
    def of[A: Aggregation]() -> AggFold:
        return AggFold(
            partials_fn=Self._partials[A],
            merge_fn=Self._merge[A],
        )

    def __init__(out self, name: String, value_dtype: AnyDataType) raises:
        var box = List[AggFold]()

        @parameter
        def make[A: Aggregation]() raises:
            box.append(Self.of[A]())

        resolve_agg[make](name, value_dtype)
        self = box[0].copy()

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


struct ThreadPartials(Copyable, Movable):
    """One worker's contribution to a thread-local aggregation: the unique keys
    it saw, and the raw (non-finalized) accumulator + valid counts for each
    aggregate over those keys.

    The typed per-group state itself is ``AggState[K, V]``, inside the
    aggregation; what a worker hands back is that state *frozen* and erased, one
    entry per aggregate, which is what the merge re-keys and folds."""

    var keys: StructArray
    var accs: List[AnyArray]
    var cnts: List[Int64Array]

    def __init__(
        out self,
        var keys: StructArray,
        var accs: List[AnyArray],
        var cnts: List[Int64Array],
    ):
        self.keys = keys^
        self.accs = accs^
        self.cnts = cnts^


# ---------------------------------------------------------------------------
# Aggregates — the aggregate *set* a query applies in one pass
# ---------------------------------------------------------------------------


struct Aggregates(Copyable, Movable, Sized, Writable):
    """The N aggregates a query computes, and how to compute them together.

    A query aggregates several columns at once (``sum(a), min(b), count(c)``),
    and what makes that fast is doing the *grouping* once and every aggregate in
    that one pass. That is a property of the set, not of any single aggregate,
    so the set is the type that owns the drivers:

    - ``grouped(gb, values)`` — the whole GROUP BY, including the choice between
      thread-local partial folds (only when every member is mergeable) and the
      key-partitioned path;
    - ``whole(values)`` — no GROUP BY: one implicit group, each aggregate's own
      whole-column path.

    Members are added either by naming the ``Aggregation`` (``append[A]``, the
    AOT form) or by resolving a function name against the column's dtype
    (``append(name, dtype)``, the dynamic form)."""

    var _funcs: List[AggFunc]

    def __init__(out self):
        self._funcs = List[AggFunc]()

    def __init__(out self, var funcs: List[AggFunc]):
        self._funcs = funcs^

    def __len__(self) -> Int:
        return len(self._funcs)

    def __getitem__(ref self, index: Int) -> ref[self._funcs[index]] AggFunc:
        return self._funcs[index]

    def append[A: Aggregation](mut self, value_dtype: AnyDataType) raises:
        """Add aggregation ``A`` over a ``value_dtype`` column (AOT)."""
        self._funcs.append(AggFunc.of[A](value_dtype))

    def append(mut self, name: String, value_dtype: AnyDataType) raises:
        """Add the aggregate ``name`` over a ``value_dtype`` column (dynamic).
        """
        self._funcs.append(AggFunc(name, value_dtype))

    def names(self) -> List[String]:
        """Each aggregate's default output column name."""
        var out = List[String]()
        for ref f in self._funcs:
            out.append(f.name)
        return out^

    def is_mergeable(self) -> Bool:
        """Whether *every* member can run as thread-local partials + a merge —
        the gate for the thread-local path, since the whole set shares one
        grouping."""
        for ref f in self._funcs:
            if not f.is_mergeable:
                return False
        return True

    def grouped(
        self, gb: GroupBy, values: List[AnyArray]
    ) raises -> RecordBatch:
        """Apply every aggregate over one grouping of ``gb``'s keys.

        ``values[j]`` is aggregated with member ``j``. Returns the unique key
        columns followed by one column per aggregate, so a multi-aggregate query
        hashes and probes the keys once instead of once per aggregate."""
        if len(values) != len(self._funcs):
            raise Error("aggregate: one value column per aggregate is required")
        if gb.strategy() == GROUP_THREAD_LOCAL and self.is_mergeable():
            return self._thread_local(gb.keys(), values, gb.num_threads())
        else:
            # `aggregate_columns` groups once and picks its partitioner itself;
            # the per-column aggregate is supplied as the comptime aggregator.
            @parameter
            def by_func(
                j: Int, gids: Int32Array, value: AnyArray, ng: Int
            ) raises -> AnyArray:
                return self._funcs[j].grouped(gids, value, ng)

            return gb.aggregate_columns[by_func](values, self.names())

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
            var col = List[AnyArray]()

            @parameter
            def run[A: Aggregation]() raises:
                col.append(A.whole(A.from_any(values[j]), num_threads).to_any())

            resolve_agg[run](self._funcs[j].name, values[j].dtype())
            out_fields.append(Field(self._funcs[j].name, col[0].dtype().copy()))
            out_cols.append(col[0].copy())
        return RecordBatch(schema=Schema(fields=out_fields^), columns=out_cols^)

    def _thread_local(
        self, keys: StructArray, values: List[AnyArray], num_threads: Int
    ) raises -> RecordBatch:
        """Thread-local partial aggregation for N *fold* aggregates.

        Every worker groups an equal contiguous chunk once and folds each
        aggregate into its own `AggState`; a serial merge then re-keys the chunks
        into a global grouper and folds the partials. Only valid when every
        aggregate is mergeable — `grouped` gates on `is_mergeable`."""
        var n = len(keys)
        var na = len(self._funcs)
        var chunk = (n + num_threads - 1) // num_threads
        var folds = List[AggFold]()
        for j in range(na):
            folds.append(AggFold(self._funcs[j].name, values[j].dtype()))

        # Pre-sized per-thread slots — no races on list growth.
        var partials = List[Optional[ThreadPartials]](
            length=num_threads, fill=None
        )

        @parameter
        def worker(t: Int) raises:
            var start = t * chunk
            if start >= n:
                return
            var length = min(chunk, n - start)
            var kchunk = keys.slice(start, length)

            var grouper = HashGrouper()
            var gids = grouper.consume_keys(kchunk)  # group this chunk ONCE
            var ng = grouper.num_groups()

            var kfields = grouper.key_fields(kchunk)
            var kcols = grouper.key_columns(kfields)
            var accs = List[AnyArray]()
            var cnts = List[Int64Array]()
            for j in range(na):
                var parts = folds[j].partials(
                    gids, values[j].slice(start, length), ng
                )
                accs.append(parts[0].copy())
                cnts.append(parts[1].copy())

            partials[t] = ThreadPartials(
                StructArray(
                    dtype=keys.dtype.copy(),
                    length=ng,
                    nulls=0,
                    offset=0,
                    bitmap=None,
                    children=kcols^,
                ),
                accs^,
                cnts^,
            )

        sync_parallelize[worker](num_threads)

        # Merge — re-key every chunk into the global grouper ONCE (shared across
        # aggregates), then fold each aggregate's partials at the global ids.
        var gg = HashGrouper()
        var live = List[Int]()
        var remap = List[Int32Array]()
        for t in range(num_threads):
            if partials[t]:
                live.append(t)
                remap.append(gg.consume_keys(partials[t].value().keys))
        var ngg = gg.num_groups()

        var kfields = gg.key_fields(keys)
        var out_fields = List[Field]()
        for k in range(len(kfields)):
            out_fields.append(kfields[k].copy())
        var out_cols = gg.key_columns(kfields)

        for j in range(na):
            var accs = List[AnyArray]()
            var cnts = List[Int64Array]()
            for i in range(len(live)):
                ref part = partials[live[i]].value()
                accs.append(part.accs[j].copy())
                cnts.append(part.cnts[j].copy())
            var col = folds[j].merge(remap, accs, cnts, ngg)
            out_fields.append(Field(self._funcs[j].name, col.dtype().copy()))
            out_cols.append(col^)

        return RecordBatch(schema=Schema(fields=out_fields^), columns=out_cols^)

    def write_to[W: Writer](self, mut writer: W):
        for i in range(len(self._funcs)):
            if i > 0:
                writer.write(", ")
            self._funcs[i].write_to(writer)

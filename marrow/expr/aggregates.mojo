"""Aggregate *functions* for the relational layer — one representation, two
resolutions.

The kernel layer (``marrow.kernels.aggregate`` / ``marrow.kernels.groupby``)
knows aggregates only as types: ``SumKernel``, ``AggState[K, V]``,
``GroupBy.aggregate[K]``. It has no vocabulary for ``"sum"``. This module is the
*only* place an aggregate identity becomes concrete, and it offers exactly two
ways to do it — both landing on the same comptime ``AggKernel``:

- **fused / AOT (F2)** — ``AggFunc.of[SumKernel]()`` or, when the input dtype is
  also statically known, ``AggFunc.typed[SumKernel, Int64Type]()``. Nothing is
  interpreted: no function-name string exists, and the fully typed form carries
  no dtype dispatch either, so a fused plan monomorphises down to
  ``AggState[K, V]`` and the rest is dead code.
- **dynamic / interpreted (F1)** — ``AggFunc("sum")``. The single runtime→comptime
  boundary in the whole system, keyed on the kernel's own ``name`` with no tag in
  between. The Python ``group_by(...).aggregate([...])`` binding and
  ``AnyRelation.aggregate(...)`` start here.

``AggFunc`` is a closed erasure over that choice: two thin function pointers,
each a comptime instantiation of the *same* generic bodies (``agg_grouped`` /
``agg_out_dtype``). The two paths therefore bottom out in identical kernels — the
fused one is the dynamic one with the dispatch removed, never a second
implementation.

``AggFold`` carries the capabilities only the eager ``GroupBy`` drivers need (the
whole-array reduce, and the thread-local partial/merge fold). It is a *separate*
box because every field of an erased box is live code for every kernel the name
switch can produce: folding these three into ``AggFunc`` cost the aggregate
binary-size gate 3.2 MB (+24 %) for capabilities a relational plan never calls.

Also here: the runtime multi-aggregate drivers (``aggregate_grouped``,
``aggregate_whole``) — group once (or not at all, for ``SELECT agg(x)`` with no
``GROUP BY``) and apply N aggregates in the same pass.
"""

from std.algorithm.functional import sync_parallelize

from ..arrays import (
    AnyArray,
    StructArray,
    Int32Array,
    Int64Array,
)
from ..builders import Int32Builder
from ..dtypes import (
    AnyDataType,
    Field,
    NumericType,
    int32,
    int64,
    float64,
)
from ..schema import Schema
from ..tabular import RecordBatch
from ..kernels.aggregate import (
    AggKernel,
    AggState,
    SumKernel,
    ProductKernel,
    MinKernel,
    MaxKernel,
    CountKernel,
    MeanKernel,
    min_max_string_grouped,
    count_valid_grouped,
    reinterpret_array,
    temporal_backing_dtype,
)
from ..kernels.distinct import (
    count_distinct,
    approx_count_distinct,
    count_distinct_grouped,
    approx_count_distinct_grouped,
)
from ..kernels.execution import ExecutionContext
from ..kernels.groupby import (
    GroupBy,
    HashGrouper,
    slice_struct,
    GROUP_THREAD_LOCAL,
)


# ---------------------------------------------------------------------------
# The generic bodies. Every one is parameterized on a comptime `AggKernel`, so
# each is the *whole* implementation of that aggregate — the erased `AggFunc`
# below only chooses which instantiation to point at.
# ---------------------------------------------------------------------------


def agg_out_dtype[K: AggKernel](value_dtype: AnyDataType) raises -> AnyDataType:
    """The output dtype of aggregate ``K`` over a ``value_dtype`` column.

    For a numeric input the rule *is* the kernel's own accumulator algebra
    (``AggKernel.AccType``): ``sum``/``product`` widen so narrow integers don't
    overflow, ``min``/``max`` keep the input type, ``count`` is int64 and
    ``mean`` float64. Only non-numeric inputs need a rule of their own —
    order-preserving ``min``/``max`` keep the (string / temporal) input dtype and
    ``count`` still counts."""
    if value_dtype.is_numeric():
        return _acc_dtype[K](value_dtype)
    else:
        comptime if K.name == "min" or K.name == "max":
            return value_dtype.copy()
        elif K.name == "count":
            return AnyDataType(int64)
        elif K.name == "mean":
            return AnyDataType(float64)
        else:
            # `sum` / `product` of a non-numeric column: raises, in the one place
            # that knows the supported set.
            return _acc_dtype[K](value_dtype)


def _acc_dtype[K: AggKernel](value_dtype: AnyDataType) raises -> AnyDataType:
    @parameter
    def by_value[V: NumericType](d: V) raises -> AnyDataType:
        return AnyDataType(K.AccType[V]())

    return value_dtype.dispatch_numeric[by_value]()


def agg_grouped[
    K: AggKernel
](gids: Int32Array, value: AnyArray, num_groups: Int) raises -> AnyArray:
    """One aggregate column over precomputed group ids.

    The shared per-column entry point for every aggregate — used by the
    multi-aggregate drivers below and by the expression layer's
    ``AggregateProcessor``:

    - string ``min``/``max`` → the dedicated bytewise scan;
    - temporal ``min``/``max`` → its (order-preserving) signed-integer backing,
      reinterpreted here and relabelled to the temporal dtype on the way out;
    - non-numeric ``count`` → the validity-only per-group scan;
    - everything else → the typed ``AggState`` fold.

    ``aggregate_grouped`` reinterprets temporal columns *before* partitioning
    (``take``/``concat`` don't handle temporal dtypes) and relabels the output
    columns itself, so the temporal branch here is inert on that path."""
    comptime if K.name == "min" or K.name == "max":
        var vdt = value.dtype()
        if vdt.is_string() or vdt.is_large_string():
            return min_max_string_grouped(
                gids, value, num_groups, K.name == "min"
            )
        elif vdt.is_temporal():
            var backing = temporal_backing_dtype(vdt)
            var out = _fold_grouped[K](
                gids, reinterpret_array(value, backing), num_groups
            )
            return reinterpret_array(out, vdt)
        else:
            return _fold_grouped[K](gids, value, num_groups)
    elif K.name == "count":
        if value.dtype().is_numeric():
            return _fold_grouped[K](gids, value, num_groups)
        else:
            return count_valid_grouped(gids, value, num_groups).to_any()
    else:
        return _fold_grouped[K](gids, value, num_groups)


def _fold_grouped[
    K: AggKernel
](gids: Int32Array, value: AnyArray, num_groups: Int) raises -> AnyArray:
    """The `AggState` fold with the input dtype resolved at runtime."""

    @parameter
    def by_value[V: NumericType](d: V) raises -> AnyArray:
        return _fold_grouped_typed[K, V](gids, value, num_groups)

    return value.dtype().dispatch_numeric[by_value]()


def _fold_grouped_typed[
    K: AggKernel, V: NumericType
](gids: Int32Array, value: AnyArray, num_groups: Int) raises -> AnyArray:
    """The `AggState` fold with *nothing* left to resolve — the fused leaf."""
    var state = AggState[K, V]()
    state.update(gids, value.as_primitive[V](), num_groups)
    return state.finish(num_groups).to_any()


def agg_whole[
    K: AggKernel
](value: AnyArray, ctx: ExecutionContext) raises -> AnyArray:
    """One whole-table aggregate as a 1-row column — the SIMD/``O(1)`` scalar
    reduction broadcast to length 1 (``AnyScalar.repeat``)."""
    comptime if K.name == "min" or K.name == "max":
        var vdt = value.dtype()
        if vdt.is_string() or vdt.is_large_string():
            # String min/max doesn't broadcast through `AnyScalar.repeat`;
            # compute the single-group result by scanning every row into
            # group 0.
            var gb = Int32Builder(len(value))
            for _ in range(len(value)):
                gb.append(Scalar[int32.native](0))
            return min_max_string_grouped(
                gb.finish(), value, 1, K.name == "min"
            )
        elif vdt.is_temporal():
            # Reduce over the integer backing, then relabel to the temporal
            # dtype (the whole-array reduce already handles all-null → null).
            var iv = reinterpret_array(value, temporal_backing_dtype(vdt))
            return reinterpret_array(K.reduce(iv, ctx).repeat(1), vdt)
        else:
            return K.reduce(value, ctx).repeat(1)
    else:
        return K.reduce(value, ctx).repeat(1)


def _partials[
    K: AggKernel
](gids: Int32Array, value: AnyArray, num_groups: Int) raises -> Tuple[
    AnyArray, Int64Array
]:
    """A thread-local partial fold: the raw per-group accumulator + valid counts,
    for a later `merge`."""

    @parameter
    def by_value[V: NumericType](d: V) raises -> Tuple[AnyArray, Int64Array]:
        return _partials_typed[K, V](gids, value, num_groups)

    return value.dtype().dispatch_numeric[by_value]()


def _partials_typed[
    K: AggKernel, V: NumericType
](gids: Int32Array, value: AnyArray, num_groups: Int) raises -> Tuple[
    AnyArray, Int64Array
]:
    var state = AggState[K, V]()
    state.update(gids, value.as_primitive[V](), num_groups)
    var parts = state.into_partials()
    return (parts[0].copy().to_any(), parts[1].copy())


def _merge_partials[
    K: AggKernel
](
    value_dtype: AnyDataType,
    remap: List[Int32Array],
    accs: List[AnyArray],
    cnts: List[Int64Array],
    num_groups: Int,
) raises -> AnyArray:
    """Fold every thread's partial into one global `AggState` and finalize."""

    @parameter
    def by_value[V: NumericType](d: V) raises -> AnyArray:
        return _merge_partials_typed[K, V](
            value_dtype, remap, accs, cnts, num_groups
        )

    return value_dtype.dispatch_numeric[by_value]()


def _merge_partials_typed[
    K: AggKernel, V: NumericType
](
    value_dtype: AnyDataType,
    remap: List[Int32Array],
    accs: List[AnyArray],
    cnts: List[Int64Array],
    num_groups: Int,
) raises -> AnyArray:
    var state = AggState[K, V]()
    for t in range(len(remap)):
        state.merge(
            remap[t], accs[t].as_primitive[K.AccType[V]](), cnts[t], num_groups
        )
    return state.finish(num_groups).to_any()


# ---------------------------------------------------------------------------
# dispatch_agg — the one runtime -> comptime boundary
# ---------------------------------------------------------------------------


def dispatch_agg[
    job: def[K: AggKernel]() raises capturing[_] -> None
](name: String) raises:
    """Resolve an aggregate *fold* function name to its kernel and run ``job[K]``.

    The single place in the system that maps a runtime aggregate identity onto a
    comptime one, keyed on the kernels' own ``name`` with no tag in between. The
    two distinct aggregates are not folds (their per-group state is a hash set /
    HLL sketch, not a scalar accumulator) and so have no kernel — callers handle
    them before reaching here."""
    if name == SumKernel.name:
        job[SumKernel]()
    elif name == MinKernel.name:
        job[MinKernel]()
    elif name == MaxKernel.name:
        job[MaxKernel]()
    elif name == CountKernel.name:
        job[CountKernel]()
    elif name == MeanKernel.name:
        job[MeanKernel]()
    elif name == ProductKernel.name:
        job[ProductKernel]()
    else:
        raise Error("unknown aggregate function: ", name)


comptime COUNT_DISTINCT = "count_distinct"
comptime APPROX_COUNT_DISTINCT = "approx_count_distinct"


# ---------------------------------------------------------------------------
# AggFunc — the erased aggregate function a plan node holds
# ---------------------------------------------------------------------------


struct AggFunc(Copyable, Movable, Writable):
    """An aggregate function, erased to thin function pointers over a **comptime**
    ``AggKernel``.

    Both fields are a comptime instantiation of the generic bodies above, so an
    ``AggFunc`` built with ``of``/``typed`` carries no interpretation whatsoever:
    the plan holds a direct pointer to the monomorphised kernel. Only
    ``AggFunc(name)`` — the dynamic frontend's entry — ever consults a name, and
    it does so exactly once, at plan-build time.

    ``name`` is the kernel's own comptime name, used as the default output column
    name; ``is_distinct`` marks the two sketch aggregates, whose per-group state
    is a hash set / HLL rather than a mergeable scalar accumulator.

    **Kept deliberately small.** Every field of an erased box is live code for
    every kernel the name switch can produce, so a capability nobody on this path
    calls is still paid for in full. Folding the eager drivers' whole-array
    reduce and partial/merge fold in here (see ``AggFold``) measured **+3.2 MB
    (+24 %)** on the aggregate binary-size gate."""

    var name: String
    var is_distinct: Bool
    var _out_dtype_fn: def(AnyDataType) thin raises -> AnyDataType
    var _grouped_fn: def(Int32Array, AnyArray, Int) thin raises -> AnyArray

    @staticmethod
    def _distinct_out_dtype(value_dtype: AnyDataType) raises -> AnyDataType:
        return AnyDataType(int64)

    @staticmethod
    def _distinct_grouped[
        exact: Bool
    ](gids: Int32Array, value: AnyArray, num_groups: Int) raises -> AnyArray:
        comptime if exact:
            return count_distinct_grouped(gids, value, num_groups)
        else:
            return approx_count_distinct_grouped(gids, value, num_groups)

    def __init__(
        out self,
        *,
        var func_name: String,
        is_distinct: Bool,
        out_dtype_fn: def(AnyDataType) thin raises -> AnyDataType,
        grouped_fn: def(Int32Array, AnyArray, Int) thin raises -> AnyArray,
    ):
        self.name = func_name^
        self.is_distinct = is_distinct
        self._out_dtype_fn = out_dtype_fn
        self._grouped_fn = grouped_fn

    @staticmethod
    def of[K: AggKernel]() -> AggFunc:
        """Aggregate ``K`` with the input dtype resolved per column at runtime —
        the general form, and what the dynamic frontend resolves a name to."""
        return AggFunc(
            func_name=String(K.name),
            is_distinct=False,
            out_dtype_fn=agg_out_dtype[K],
            grouped_fn=agg_grouped[K],
        )

    @staticmethod
    def typed[K: AggKernel, V: NumericType]() -> AggFunc:
        """Aggregate ``K`` over a statically known numeric input dtype ``V`` —
        the fused (AOT) form. Nothing is resolved at runtime: no name, no tag and
        no dtype dispatch, so the plan reaches ``AggState[K, V]`` directly and
        every other instantiation is dead code."""
        return AggFunc(
            func_name=String(K.name),
            is_distinct=False,
            out_dtype_fn=_typed_out_dtype[K, V],
            grouped_fn=_fold_grouped_typed[K, V],
        )

    @staticmethod
    def distinct[exact: Bool]() -> AggFunc:
        """``count_distinct`` (exact) / ``approx_count_distinct`` (HLL)."""
        return AggFunc(
            func_name=String(COUNT_DISTINCT) if exact else String(
                APPROX_COUNT_DISTINCT
            ),
            is_distinct=True,
            out_dtype_fn=Self._distinct_out_dtype,
            grouped_fn=Self._distinct_grouped[exact],
        )

    def __init__(out self, name: String) raises:
        """Resolve an aggregate function *name* to its kernel."""
        if name == COUNT_DISTINCT:
            self = Self.distinct[True]()
        elif name == APPROX_COUNT_DISTINCT:
            self = Self.distinct[False]()
        else:
            var box = List[AggFunc]()

            @parameter
            def make[K: AggKernel]() raises:
                box.append(Self.of[K]())

            dispatch_agg[make](name)
            self = box[0].copy()

    # -- the aggregate surface ----------------------------------------------

    def out_dtype(self, value_dtype: AnyDataType) raises -> AnyDataType:
        """This aggregate's output dtype over a ``value_dtype`` column."""
        return self._out_dtype_fn(value_dtype)

    def grouped(
        self, gids: Int32Array, value: AnyArray, num_groups: Int
    ) raises -> AnyArray:
        """One aggregate column over precomputed group ids."""
        return self._grouped_fn(gids, value, num_groups)

    def is_mergeable(self, value_dtype: AnyDataType) -> Bool:
        """Whether this aggregate over ``value_dtype`` can run as thread-local
        partials + a merge. The sketch aggregates carry no mergeable scalar
        accumulator, and the non-numeric scans (string ``min``/``max``,
        ``count`` of a non-numeric column) are dedicated per-group passes rather
        than the ``AggState`` fold the merge understands."""
        return not self.is_distinct and value_dtype.is_numeric()

    def write_to[W: Writer](self, mut writer: W):
        writer.write(self.name)


def _typed_out_dtype[
    K: AggKernel, V: NumericType
](value_dtype: AnyDataType) raises -> AnyDataType:
    """`agg_out_dtype` with the input dtype already known — a constant."""
    return AnyDataType(K.AccType[V]())


# ---------------------------------------------------------------------------
# AggFold — the extra capabilities only the eager drivers below need
# ---------------------------------------------------------------------------


struct AggFold(Copyable, Movable):
    """The whole-array reduce and the partial/merge fold, erased over the same
    comptime ``AggKernel`` as ``AggFunc``.

    Split out of ``AggFunc`` on purpose: a relational plan never calls these, and
    an erased box pays for every field it declares (see ``AggFunc``'s note). Only
    ``aggregate_whole`` / ``aggregate_grouped`` — the eager ``GroupBy`` drivers
    behind the Python bindings — build one, so the fused/AOT path links none of
    it. Resolution goes through the same ``dispatch_agg`` switch, so there is
    still exactly one list of kernels."""

    var _whole_fn: def(AnyArray, ExecutionContext) thin raises -> AnyArray
    var _partials_fn: def(Int32Array, AnyArray, Int) thin raises -> Tuple[
        AnyArray, Int64Array
    ]
    var _merge_fn: def(
        AnyDataType, List[Int32Array], List[AnyArray], List[Int64Array], Int
    ) thin raises -> AnyArray

    @staticmethod
    def _distinct_whole[
        exact: Bool
    ](value: AnyArray, ctx: ExecutionContext) raises -> AnyArray:
        comptime if exact:
            return count_distinct(value, ctx).to_any().repeat(1)
        else:
            return approx_count_distinct(value, ctx).to_any().repeat(1)

    @staticmethod
    def _no_partials(
        gids: Int32Array, value: AnyArray, num_groups: Int
    ) raises -> Tuple[AnyArray, Int64Array]:
        raise Error("aggregate has no mergeable partial state")

    @staticmethod
    def _no_merge(
        value_dtype: AnyDataType,
        remap: List[Int32Array],
        accs: List[AnyArray],
        cnts: List[Int64Array],
        num_groups: Int,
    ) raises -> AnyArray:
        raise Error("aggregate has no mergeable partial state")

    def __init__(
        out self,
        *,
        whole_fn: def(AnyArray, ExecutionContext) thin raises -> AnyArray,
        partials_fn: def(Int32Array, AnyArray, Int) thin raises -> Tuple[
            AnyArray, Int64Array
        ],
        merge_fn: def(
            AnyDataType, List[Int32Array], List[AnyArray], List[Int64Array], Int
        ) thin raises -> AnyArray,
    ):
        self._whole_fn = whole_fn
        self._partials_fn = partials_fn
        self._merge_fn = merge_fn

    @staticmethod
    def of[K: AggKernel]() -> AggFold:
        return AggFold(
            whole_fn=agg_whole[K],
            partials_fn=_partials[K],
            merge_fn=_merge_partials[K],
        )

    @staticmethod
    def distinct[exact: Bool]() -> AggFold:
        return AggFold(
            whole_fn=Self._distinct_whole[exact],
            partials_fn=Self._no_partials,
            merge_fn=Self._no_merge,
        )

    def __init__(out self, name: String) raises:
        if name == COUNT_DISTINCT:
            self = Self.distinct[True]()
        elif name == APPROX_COUNT_DISTINCT:
            self = Self.distinct[False]()
        else:
            var box = List[AggFold]()

            @parameter
            def make[K: AggKernel]() raises:
                box.append(Self.of[K]())

            dispatch_agg[make](name)
            self = box[0].copy()

    def whole(self, value: AnyArray, ctx: ExecutionContext) raises -> AnyArray:
        """The whole-table aggregate as a one-row column."""
        return self._whole_fn(value, ctx)

    def partials(
        self, gids: Int32Array, value: AnyArray, num_groups: Int
    ) raises -> Tuple[AnyArray, Int64Array]:
        """A thread-local partial fold: raw per-group accumulator + valid counts.
        """
        return self._partials_fn(gids, value, num_groups)

    def merge(
        self,
        value_dtype: AnyDataType,
        remap: List[Int32Array],
        accs: List[AnyArray],
        cnts: List[Int64Array],
        num_groups: Int,
    ) raises -> AnyArray:
        """Fold every thread's partials at remapped global ids and finalize."""
        return self._merge_fn(value_dtype, remap, accs, cnts, num_groups)


# ---------------------------------------------------------------------------
# Runtime multi-aggregate drivers — group ONCE, apply N aggregates in the same
# pass. The aggregate columns are named by kernel (callers rename as needed).
# ---------------------------------------------------------------------------


def aggregate_grouped(
    gb: GroupBy, values: List[AnyArray], funcs: List[AggFunc]
) raises -> RecordBatch:
    """Apply several aggregates over one grouping of ``gb``'s keys.

    ``values[j]`` is aggregated with ``funcs[j]``. Returns the unique key columns
    followed by one column per aggregate. This is the path the Python
    ``group_by(...).aggregate([...])`` binding uses, so a multi-agg query
    hashes/probes the keys once instead of once per aggregate."""
    # Temporal min/max are order-preserving on the signed-integer backing, so
    # reinterpret the value column to int up front. Then every downstream
    # path — serial / thread-local / radix, `take`, `concat`, `AggState` — is
    # fully numeric (`take` doesn't handle temporal dtypes), and the output
    # column is relabelled back to the temporal dtype at the end. An aggregate
    # keeps a temporal input dtype exactly when it is order-preserving, so the
    # rule is *derived* from `out_dtype` rather than re-listed here.
    var work = List[AnyArray]()
    var relabel = List[Optional[AnyDataType]]()  # per aggregate column
    var names = List[String]()
    for j in range(len(funcs)):
        var vdt = values[j].dtype()
        if vdt.is_temporal() and funcs[j].out_dtype(vdt).is_temporal():
            work.append(
                reinterpret_array(values[j], temporal_backing_dtype(vdt))
            )
            relabel.append(vdt.copy())
        else:
            work.append(values[j].copy())
            relabel.append(None)
        names.append(funcs[j].name)

    # Some aggregates can't be combined by the thread-local partial + merge
    # path, so they must take the radix (or serial) route instead. The radix
    # path partitions by key hash, so a group lands wholly in one partition and
    # its result is final without any cross-partition merge.
    var mergeable = True
    for j in range(len(funcs)):
        if not funcs[j].is_mergeable(work[j].dtype()):
            mergeable = False
            break

    var result: RecordBatch
    if gb.strategy() == GROUP_THREAD_LOCAL and mergeable:
        result = _thread_local_multi(
            gb.keys(), work, funcs, names, gb.num_threads()
        )
    else:
        # `aggregate_columns` groups once and picks serial vs. radix itself;
        # the per-column aggregate is supplied as the comptime aggregator.
        @parameter
        def by_func(
            j: Int, gids: Int32Array, value: AnyArray, ng: Int
        ) raises -> AnyArray:
            return funcs[j].grouped(gids, value, ng)

        result = gb.aggregate_columns[by_func](work, names)

    return _relabel_temporal(result^, len(funcs), relabel)


def _relabel_temporal(
    var result: RecordBatch,
    num_aggs: Int,
    relabel: List[Optional[AnyDataType]],
) raises -> RecordBatch:
    """Relabel the temporal min/max aggregate columns (computed over their
    integer backing) back to their temporal dtype. A no-op when no aggregate
    was reinterpreted."""
    var any_relabel = False
    for j in range(len(relabel)):
        if relabel[j]:
            any_relabel = True
            break
    if not any_relabel:
        return result^

    var num_keys = result.num_columns() - num_aggs
    var out_fields = List[Field]()
    var out_cols = List[AnyArray]()
    for c in range(result.num_columns()):
        var agg_idx = c - num_keys
        if agg_idx >= 0 and relabel[agg_idx]:
            var dt = relabel[agg_idx].value().copy()
            out_cols.append(reinterpret_array(result.column(c), dt))
            out_fields.append(Field(result.schema.fields[c].name, dt.copy()))
        else:
            out_fields.append(result.schema.fields[c].copy())
            out_cols.append(result.column(c).copy())
    return RecordBatch(schema=Schema(fields=out_fields^), columns=out_cols^)


def _thread_local_multi(
    keys: StructArray,
    values: List[AnyArray],
    funcs: List[AggFunc],
    names: List[String],
    num_threads: Int,
) raises -> RecordBatch:
    """Thread-local partial aggregation for N *fold* aggregates.

    Every worker groups an equal contiguous chunk once and folds each aggregate
    into its own `AggState`; a serial merge then re-keys the chunks into a global
    grouper and folds the partials. Only valid when every aggregate is
    mergeable — the caller gates on `AggFunc.is_mergeable`."""
    var n = len(keys)
    var na = len(funcs)
    var chunk = (n + num_threads - 1) // num_threads
    var folds = List[AggFold]()
    for j in range(na):
        folds.append(AggFold(funcs[j].name))

    var part_keys = List[Optional[StructArray]](length=num_threads, fill=None)
    # Per (thread, aggregate) partial state, flattened as [t * na + j].
    var part_acc = List[Optional[AnyArray]](length=num_threads * na, fill=None)
    var part_cnt = List[Optional[Int64Array]](
        length=num_threads * na, fill=None
    )

    @parameter
    def worker(t: Int) raises:
        var start = t * chunk
        if start >= n:
            return
        var length = min(chunk, n - start)
        var kchunk = slice_struct(keys, start, length)

        var grouper = HashGrouper()
        var gids = grouper.consume_keys(kchunk)  # group this chunk ONCE
        var ng = grouper.num_groups()

        var kfields = grouper.key_fields(kchunk)
        var kcols = grouper.key_columns(kfields)
        part_keys[t] = StructArray(
            dtype=keys.dtype.copy(),
            length=ng,
            nulls=0,
            offset=0,
            bitmap=None,
            children=kcols^,
        )

        for j in range(na):
            var parts = folds[j].partials(
                gids, values[j].slice(start, length), ng
            )
            part_acc[t * na + j] = parts[0].copy()
            part_cnt[t * na + j] = parts[1].copy()

    sync_parallelize[worker](num_threads)

    # Merge — re-key every chunk into the global grouper ONCE (shared across
    # aggregates), then fold each aggregate's partials at the global ids.
    var gg = HashGrouper()
    var live = List[Int]()
    var remap = List[Int32Array]()
    for t in range(num_threads):
        if part_keys[t]:
            live.append(t)
            remap.append(gg.consume_keys(part_keys[t].value()))
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
            accs.append(part_acc[live[i] * na + j].value().copy())
            cnts.append(part_cnt[live[i] * na + j].value().copy())
        var col = folds[j].merge(values[j].dtype(), remap, accs, cnts, ngg)
        out_fields.append(Field(names[j], col.dtype().copy()))
        out_cols.append(col^)

    return RecordBatch(schema=Schema(fields=out_fields^), columns=out_cols^)


# ---------------------------------------------------------------------------
# Whole-table aggregation (no GROUP BY)
# ---------------------------------------------------------------------------


def aggregate_whole(
    values: List[AnyArray], funcs: List[AggFunc], num_threads: Int = 0
) raises -> RecordBatch:
    """Whole-table aggregation — ``SELECT agg(x), ...`` with no GROUP BY.

    A single implicit group, computed with the vectorized whole-array
    reductions (SIMD ``AggKernel.reduce``, ``O(1)`` count, direct scalar
    ``count_distinct``) rather than the grouped scatter. Returns a one-row
    batch of the aggregate columns (named by kernel; callers rename).

    Distinct aggregates get a parallel ctx (``count_distinct`` self-gates on
    size, going radix-partition-parallel at scale); fold reductions stay
    serial — the whole-array SIMD reduce only benefits from threads well above
    these sizes, and that gating belongs in the reduce primitive itself."""
    var par = ExecutionContext.parallel(num_threads)
    var ser = ExecutionContext.serial()
    var out_fields = List[Field]()
    var out_cols = List[AnyArray]()
    for j in range(len(funcs)):
        var col = AggFold(funcs[j].name).whole(
            values[j], par if funcs[j].is_distinct else ser
        )
        out_fields.append(Field(funcs[j].name, col.dtype().copy()))
        out_cols.append(col^)
    return RecordBatch(schema=Schema(fields=out_fields^), columns=out_cols^)

# Refactoring & simplification notes

Findings from a read of the `parquet-native` diff (merge-base `ac4b8a0` →
`parquet-native`, ~202 commits, +37k/−8k). Grouped by subsystem and ranked by
payoff within each group. Line anchors are approximate (they reflect the state
at time of writing). Nothing here is applied — this is an inventory to work
from.

Many `refactor(...)` commits already landed on this branch (shared radix
partitioner, `codecs.Dictionary`, the `statistics` module, `_FixedWidthAcc`,
`_walk_slots`, `_LeafTypeRow`). The items below are the remaining holdouts that
follow the *same* sharing logic those commits started.

## Highest-leverage themes

1. **A runtime-dtype → comptime-type dispatch cascade is hand-copied ~20 times**
   across kernels, Parquet value-codecs, and Python bindings. `cast._over_numeric`
   / `utils.variant_dispatch_raises` / `pruning.PruneBound._cmp_scalar` already
   show the right pattern; almost nothing else uses it. This is the single
   biggest duplication surface in the codebase.
2. **The Parquet leaf `(arrow dtype) → (store, phys, width)` mapping lives in 6
   value-codec functions** even though `schema._LeafTypeRow` is declared as "the
   single source of truth." Drift-prone.
3. **Three hand-rolled type-erasure boxes** (`AnyValue`, `AnyRelation`,
   `AnyProcessor`) reimplement identical trampoline machinery.
4. **The Python `RecordBatch.join` wrapper is an outright bug** (below) — worth
   fixing regardless of the cleanup work.

---

## 0. Bugs found while reading (fix independently of refactors)

### 0.1 `RecordBatch.join` Python wrapper forwarded the wrong arguments — FIXED in this branch
`python/marrow/__init__.py` `RecordBatch.join` called
`self._binding.join(right.unwrap(), keys, right_keys)` — only 3 of the binding's
6 required parameters (`_record_batch_join` needs `right, keys, right_keys,
join_type, num_threads`), and defaulted `join_type="left semi"` while the
binding documents `"inner"`. Any `rb.join(...)` from Python raised `TypeError`,
and no Python test exercised it (only `str.join` matches in the test suite). The
documented join behaviour (6 join types, `num_threads`) was unreachable.

**Applied fix** (needed for the join docs to run): forward `join_type` and
`num_threads`, default `join_type="inner"`, and accept a bare string for
`keys`/`right_keys`. A Python-level `test_join` covering the six join types
should be added — the gap is why this went unnoticed. (Verified: all six join
types now run from Python; collisions suffix `_right` as documented.)

### 0.2 `take` aborts the interpreter from Python (NOT fixed — needs a Mojo fix)
`ma.take(array, indices)` / `marrow.compute.take` crash the process with
`ABORT: marrow/arrays.mojo:2166: get: wrong variant type` for *any* element
type (reproduced with int64 and string arrays). The abort is in
`AnyArray._as[T]` (`arrays.mojo:2164-2166`) — the take binding/kernel downcasts a
result (or the indices) to the wrong typed variant. No Python test exercises
`take` (only `str.join`-style matches in the suite), so it went unnoticed.
Because it aborts rather than raising, it cannot be caught. `take` was
consequently **left out of the docs**. Fix the variant downcast in the take
path and add a Python `test_take`.

### 0.3 Aggregate functions reject non-numeric value columns
`RecordBatch.aggregate` / `group_by(...).aggregate` raise
`aggregate: unsupported input dtype string` for a string value column — including
`count`. So `("name", "count")` fails; only numeric value columns work. This
contradicts the `RecordBatch.aggregate` docstring ("`count` of a non-null column
gives `COUNT(*)`"), which implies any column. Either (a) special-case `count`
(and `count_distinct`) to accept any dtype — the natural `COUNT(*)` / distinct
semantics — or (b) narrow the docstring. The docs now count a numeric column and
state the numeric-only limitation, but the underlying gap remains.

---

## 1. Kernels (`marrow/kernels/`)

### 1.1 Duplicated 11-way numeric dtype dispatch cascade — highest payoff
The `if dtype()==int8: apply(as_int8()) … elif float64` chain (~13 lines each,
~130 lines total) is copy-pasted across at least 10 dispatch sites:
`arithmetic.mojo:107-135, 152-167, 219-`; `compare.mojo:131-155`;
`boolean.mojo:78-, 290-296, 355-365`; `filter.mojo` filter/`drop_null`/`take`
(~490, ~585, ~808); `hashing.mojo:~405`; `sort.mojo:572-625`;
`aggregate.mojo:132-155`. Meanwhile `cast.mojo:73-94` already wraps exactly this
as `_over_numeric[func](dt)` over `utils.variant_dispatch_raises`.
**Refactor:** promote a shared `for_numeric_array[func]` / `for_float_array`
helper into `kernels/helpers.mojo`; each `dispatch` collapses to one line.

### 1.2 Two parallel runtime→comptime numeric mechanisms
`aggregate.for_value_dtype` (`aggregate.mojo:559-587`, `job[V]()` returning None
via a capture box) vs `cast._over_numeric` (value-returning). The box-and-append
idiom (`var box=List[...]; for_value_dtype[job](dt); return box[0]`) recurs in
`groupby.mojo` at `343-351, 539-548, 577-583, 690-703` purely because
`for_value_dtype` can't return a value.
**Refactor:** give `for_value_dtype` a value-returning form (or reuse
`variant_dispatch_raises` directly), deleting ~5 `List[...]` round-trips.

### 1.3 GroupBy: the typed single-aggregate path duplicates the runtime multi path
`groupby.mojo` `_serial/_thread_local/_radix` (333-586) vs
`_serial_multi/_thread_local_multi/_radix_multi` (705-906) are the same three
strategies with an outer `for j in range(na)` loop — chunking, `HashGrouper`
build, per-worker `_slice_struct`, partition/merge, key-column emit all repeated.
**Refactor:** implement `aggregate[K]` in terms of the runtime multi path
(single-element tags/values, rename output to `K.name`), or extract the shared
per-strategy skeleton behind a "produce agg columns over gids" closure. ~200
lines. *Risk: medium* — verify the closure indirection doesn't defeat AOT
monomorphization of the typed entry.

### 1.4 GroupBy result-batch assembly repeated 6×
`groupby.mojo:354-364, 475-483, 568-586, 713-725, 803-838, 889-906` all end with
the same "copy key fields+columns, then append one `Field`+column per aggregate,
build `RecordBatch`" block. Extract `_emit_batch(key_fields, key_cols,
agg_names, agg_cols)`. Low risk, independent of 1.3.

### 1.5 First-occurrence-row scan duplicated 3×
`groupby.mojo:113-122, 528-536, 859-866` — identical "walk rows, append index when
`bids[i]==next_new`, stop at `ng`". Extract `_first_occurrence_rows(gids, ng,
rows=None)`.

### 1.6 `cast.mojo` builder per-element map loops (6-7 near-identical)
`cast.mojo:480-495, 522-538, 565-571, 592-600, 658-664, 738-746` all share
`for i: if is_valid: b.append(transform(x)) else: b.append_null()` (9
`append_null()` sites). Extract `map_valid[op](array, builder)`; keep the
per-kernel `safe`/formatting logic in `op`.

### 1.7 sort.mojo valid/null partition + merge repeated across overloads
The "scan into valid+null lists, sort valid, place nulls via
`null_off`/`valid_off`" logic exists 3× (`sort.mojo:394-419` + the `array()`
helper at `349-374`; bool at `433-481`; string at `499-544` which *re-inlines*
the `array()` merge instead of calling it). Have the string path reuse
`array()`; extract `_partition_valid_null(array)`.
**Naming nit:** `sort.mojo:349` defines a free function literally named `array()`
while the file also imports `from ..builders import array as _primitive_array` —
the local `array` shadows the Arrow constructor concept. Rename to
`_merge_sorted_indices`.

### 1.8 distinct.mojo HLL register-update loop duplicated
`distinct.mojo:118-127` (whole-array) and `199-207` (grouped) share the
skip-null / `idx = h >> (64-p)` (grouped adds `gid*m`) / `rho` / max-update loop.
Extract `_hll_add(registers, base, h)`. Small but the two must stay
bit-identical for correctness.

### 1.9 Redundant per-kernel `reduce` override in aggregate.mojo
`aggregate.mojo` Sum/Product/Min/Max (186-266) each define an identical
`reduce` that just calls `Self.dispatch(...)`. Replace with a
`comptime has_simd_reduce` flag (or a `SameTypeAggKernel` marker sub-trait).
Also `AggKernel.reduce` builds zero gids with `n` appends (`90-94`) while
`aggregate_whole` uses the `zeros.set_length(n)` fast path (`groupby.mojo:640-643`)
— share `_all_zero_gids(n)`.

### Kernel convention scan — clean
`grep` over `marrow/kernels/*` found no `fn`, `alias`, `unsafe_ptr`, or
`AnyOrigin`/`unsafe_origin_cast`. Typed shorthands and `BoolArray`/`BoolBuilder`
are used correctly; `.as_primitive[V]()` appears only where `V` is a generic
parameter (sanctioned).

---

## 2. Parquet (`marrow/parquet/`)

### 2.1 Arrow-dtype value-codec cascade duplicated across 6 functions — highest payoff
The `(arrow dtype) → (store DType, phys DType, width)` dispatch (including the
"int8/16 + uint8/16 stored as physical INT32" widening rule) is hand-written in
`reader.mojo:1614-1718` (`ColumnReader._dispatch`), `writer.mojo:93-154`
(`_encode_values`) and `211-277` (`_bloom_hashes`), `codecs.mojo:643-708`
(`Dictionary.encode`), `statistics.mojo:181-289` (`min_max`) and `291-388`
(`decode`). `schema._LeafTypeRow` (`schema.mojo:420-443`) is documented as the
single source of truth but none of these consult it.
**Refactor:** extend `_LeafTypeRow`/`LeafColumn` to carry `store/phys/width`, and
drive each operation through one `dispatch_leaf[Op]()` visitor whose per-dtype
arm calls `Op.apply[store, phys]`. `_encode_values`, `_bloom_hashes`, `min_max`,
`Dictionary.encode` become four tiny `Op` structs over one cascade.

### 2.2 FLBA / INT96 decode duplicated between flat and leveled paths (per type)
The "decode present values: `is_dictionary` → RLE indices; `is_plain` → in place;
else → `decode_flba`" block exists twice for each of decimal
(`reader.mojo:812-845` vs `1358-1380`), fixed_size_binary (`931-965` vs
`1413-1432`), and INT96 (`879-899` vs `1555-1583`). `DecimalLeafBuilder._place`
and `FixedSizeBinaryLeafBuilder._place` are themselves structurally identical.
**Refactor:** a shared `decode_flba_present[native](page, width, dict, mut out)`
that both the flat `_place` and leveled `decode_present` call.

### 2.3 `PrimitiveLeafBuilder` reimplements `_FixedWidthAcc`
`_FixedWidthAcc[native]` (`reader.mojo:739-782`) was factored out to share
values-buffer + lazy-bitmap + `wpos` + `null_count` bookkeeping, but
`PrimitiveLeafBuilder` (`486-518`) redeclares the same five fields and
re-implements `_ensure_bitmap` (`523-529`) verbatim against
`_FixedWidthAcc.ensure_bitmap` (`761-765`). Have it hold a
`_FixedWidthAcc[store_dt]` and keep only the memcpy/gather fast paths.

### 2.4 Present-value "physical bytes" extraction loop copied 4×
`for i: if is_valid: b = arr[i].value().cast[phys]().as_bytes[...]()` in
`codecs.mojo:602-613` (`Dictionary._encode_prim`), `823-827`
(`ByteStreamSplit.encode`), `500-509` (`Plain.encode_primitive`), and
`writer.mojo:170-180` (`_hash_prim`). Extract
`for_each_present_phys_bytes[store, phys](arr, fn)`.

### 2.5 Leaf-builder `place` closures repeat the null/present skeleton
Seven `@parameter def place(present_here, selected, vi)` closures fed to
`_walk_slots` (`reader.mojo:552-561, 659-665, 681-691, 828-844, 947-964,
1003-1012, 1024-1031`) share the `if selected: if present_here: <append> else:
<append_null>` shape. A companion `_scatter_present[append_present,
append_null](page, max_def, mask)` collapses each to two one-liners.

### 2.6 Thrift `read`/`write` field boilerplate across 14 structs
Every metadata struct in `format.mojo` hand-writes mirror `read`/`write`
methods; the `list<struct>` read loop alone appears ~8×. Full codegen is out of
scope, but a `read_struct_list[T](r, mut out)` (mirroring the existing
`write_struct_list` at `format.mojo:274-283`) removes the most-copied half.

### 2.7 `z_stream` manual layout duplicated in utils.mojo
`gzip_decompress` (`utils.mojo:154-181`) and `gzip_compress` (`261-296`) both
hand-lay-out the 112-byte `z_stream` with copy-pasted offset arithmetic. Extract
a small `_ZStream` helper struct.

### 2.8 `Statistics.min_max` re-inlines the byte-wise min/max it already extracted
`_update_minmax`/`_bytes_stats` (`statistics.mojo:136-179`) exist to share the
unsigned-lexicographic seen/lo/hi fold, but the `fixed_size_binary` arm
(`259-271`) re-implements it inline. Route FSB through the shared helper.

### 2.9 Three encodings of the `PAR1` magic
`PARQUET_MAGIC` (`format.mojo:303`) is defined but unused; `write_magic`
(`1359-1364`) and the footer check (`1373-1378`) each hardcode the four bytes.
Fold both onto the constant.

### Parquet convention violations (new code)
- **`unsafe_ptr()` outside the three permitted files** (CLAUDE.md restricts it to
  `buffers.mojo`/`views.mojo`/`c_data.mojo`): `codecs.mojo:180-181` (`Rle.gather`,
  perf-justified in its docstring) and `reader.mojo:534, 543-545, 572-573, 586,
  606, 617` (`PrimitiveLeafBuilder`). Either carve out an explicit exception or
  rewrite through `BufferView`.
- **Typed builder alias required:** `reader.mojo:1560`
  `PrimitiveBuilder[dt.Int64Type]` → `Int64Builder`; `schema.mojo:179`
  `PrimitiveBuilder[dt.Int32Type]` → `Int32Builder`. (The generic-`T`
  `PrimitiveBuilder[T]` uses in the drivers are correctly exempt.)

---

## 3. Expr layer (`marrow/exprold/`)

### 3.1 Three hand-rolled erasure boxes duplicate the same trampoline machinery
`AnyValue` (`values.mojo:736-802`), `AnyRelation` (`relations.mojo:120-207`),
`AnyProcessor` (`execution.mojo:93-132`) each implement `var
_data: ArcPointer[NoneType]`, a set of `_virt_*`/`_tramp_*` statics that
`rebind[ArcPointer[T]](ptr)[].method()`, an `@implicit __init__[T]`, and
`_tramp_drop`/`__del__`. `AnyRelation` and `AnyProcessor` even share
`_tramp_schema`/`_tramp_drop` verbatim.
**Refactor:** a shared `ErasedBox[Trait]` helper (or at minimum shared
`_tramp_drop` + `rebind` boilerplate). Collapses ~60 lines × 3.
*Keep the closed-erasure / small-binary property (CLAUDE.md) — the helper must
stay comptime-parameterized, not an open dispatcher.*

### 3.2 `DynValue` carries three parallel tag-switch dispatchers
`eval()` (`dynamic.mojo:196-263`), `prune()` (`272-315`), `_op_name()`
(`317-358`) each ladder over the same tag universe. In `eval`, the 12 binary-op
arms (`210-243`) are byte-identical modulo the kernel called; in `prune`, the 5
comparison arms (`284-303`) modulo `.maybe_eq/lt/le/gt/ge`.
**Refactor:** a tag→metadata table (op name + comparison-method selector)
driving one loop; at minimum extract `_eval_binary[kernel]`.

### 3.3 Multi-batch concat duplicated verbatim
`AnyProcessor.collect()` (`execution.mojo:144-154`) and
`ParquetScanProcessor.pull()` (`326-338`) contain the identical "0→empty; 1→copy;
else concat each column across batches" block. Extract
`_concat_batches(List[RecordBatch])` in `execution.mojo`.

### 3.4 `NumericValue.execute` / `BoolValue.execute` share the vectorize skeleton
`values.mojo:148-179` and `436-462` both do comptime native/width → alloc →
`fill` closure → `_vectorize_dispatch` → wrap, differing only in output
container (`Buffer`→`PrimitiveArray` vs `Bitmap`→`BoolArray`). Lift behind a
`_fused_run[native, Sink]` helper. Lower priority (2 occurrences).

---

## 4. Python bindings (`python/bindings/`)

> Note: the pervasive `UnsafePointer[T, MutAnyOrigin]` / `ImmutAnyOrigin` here is
> **mandated by the CPython `def_method` FFI signature**, not a violation of the
> kernel/array `AnyOrigin` ban. Correctly confined to the Python boundary.

### 4.1 `pyfunction`/`pymethod` overload explosion — 26 near-identical wrappers
`helpers.mojo` (whole file): the generic block (24-385) and the concrete
`Int`/`Bool` block (399-518) differ only in arity (0-4) and return shape. The
`Int`/`Bool` block re-duplicates the generic one solely because `Int`/`Bool`
don't nominally conform to `ConvertibleFromPython`.
**Refactor:** a single `PyArg` wrapper conforming to `ConvertibleFromPython` that
delegates to `Int(py=)`/`Bool(py=)` eliminates the 399-518 block.

### 4.2 Per-dtype tables re-enumerated in 4+ binding sites
`PyAnyConverter.__init__` 17-way switch (`arrays.mojo:536-583`), `_as_py` scalar
switch (`scalars.mojo:31-74`), the 12 nullary dtype factories
(`dtypes.mojo:31-83` + registration `222-234`). `pruning.PruneBound._cmp_scalar`
(`pruning.mojo:116-126`) already shows the one-call
`variant_dispatch_raises[dt.NumericType, …]` pattern to adopt for the numeric
arms; the 12 factories could be generated from one comptime helper in a loop.

### 4.3 `RecordBatch` and `Table` binding methods are near-verbatim twins
`tabular.mojo:126-225` (RecordBatch schema/columns/shape/column/column_names/
to_pydict/to_pylist/arrow_c_schema) vs `272-354` (Table). Each pair is identical
except `Table` first does `combine_chunks()`; `_table_column_names` (289-296) and
`_record_batch_column_names` (142-149) are byte-identical. Parameterize on "how
to get schema+columns."

### 4.4 `PyConverter` implementations share scaffolding + null/non-null loop
`PyPrimitiveConverter`/`PyBoolConverter`/`PyStringConverter`/`PyBinaryConverter`
(`arrays.mojo:605-802`) each declare the same `_builder`/`_has_nulls`/`py`
fields, a 3-line `__init__`, a `builder()` accessor, and an `extend()` with
identical null/non-null branch shape; only the element converter differs.
`PyStringConverter`/`PyBinaryConverter` also share `_count_bytes` verbatim.
A generic `PyLeafConverter[conv]` replaces four structs.

### 4.5 Column-name→index resolution loop repeated ~8×
`tabular.mojo:167-169, 433-435, 443-452, 515-518, 527-529, 577-578, 619-641` —
`idx = schema.get_field_index(name); if idx == -1: raise …`. Extract
`_resolve_index(schema, name)` / `_resolve_indices(schema, names)` and unify the
(currently inconsistent) error messages.

### 4.6 String-parsing / sugar in binding files (CLAUDE.md: bindings stay minimal)
Move to the pure-Python wrapper layer: join-kind parsing
(`tabular.mojo:455-466`), `sort_by` spec + direction + null_placement parsing
(`609-641`), compression-name `_codec` (`parquet.mojo:19-29`),
`_parse_time_unit` (`dtypes.mojo:151-162`), and the `record_batch()`/`table()`
try/except-isinstance dispatch (`tabular.mojo:242-264, 370-391`). The Mojo entry
points should take already-resolved enums/ints.

### 4.7 Smaller binding dups
- `_visit_list` vs `_visit_tuple` (`arrays.mojo:355-362` / `364-371`) differ by
  one call (`list_getitem` vs `tuple_getitem`) — parameterize on the accessor.
- `_build_from_dict` / `_build_from_arrays` (`tabular.mojo:82-118`) share the
  field/column append loop differing only in name source.

---

## Suggested sequencing

1. Land **0.1** (join wrapper — already applied here) + add a Python join test.
2. Kernels **1.1** (shared numeric dispatch) — unblocks the most sites, low risk.
3. Parquet **2.1** (leaf-type visitor) — highest structural payoff, drift risk.
4. Expr **3.1** (erasure box) and bindings **4.1/4.2** — largest line reductions.
5. The remaining per-file dups (1.4-1.8, 2.2-2.9, 3.2-3.4, 4.3-4.7) as
   opportunistic cleanups; each is independent and low risk.

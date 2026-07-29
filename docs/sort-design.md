# Sort Kernel Design for Marrow

## Context

Marrow needs a sort kernel — a prerequisite for Sort-Merge Join, ASOF Join, inequality joins, and ORDER BY in query plans. The joins-design.md sketches the public API; this document fills in the algorithmic details gathered from ClickHouse, DuckDB, Polars, and DataFusion/Arrow-RS, cross-checked against Marrow's existing kernel patterns and the Mojo standard library.

**Key findings from prior art research:**
- Mojo stdlib has no radix sort — must be implemented. It *does* have `sort[cmp_fn](span, stable)` (quicksort + merge sort) and `partition[cmp_fn](span, k)` (quickselect for top-K), which save significant implementation work.
- The fastest systems (ClickHouse, DuckDB) use a *permutation-first* approach: always sort an index array, never reorder column data until the final materialisation step.
- Row-based key normalisation (DuckDB, Polars row encoding, Arrow RowConverter) is faster for multi-column comparison-based sorts but conflicts with Marrow's columnar architecture. ClickHouse's permutation refinement avoids row conversion entirely at the cost of one sequential scan per additional sort key.
- `Buffer[mut=True].to_immutable()` is a zero-cost type rebind — radix sort can allocate uninitialised buffers and freeze results with no copy.
- The existing `take()` / `_gather()` in `filter.mojo` already takes an optimal fast path for full permutations (no `-1` indices, no source nulls) — no separate `permute()` function is needed.

---

## Prior Art

### ClickHouse — Permutation-First, Columnar, Hybrid Radix+PDQ

**Algorithm selection** (`src/Columns/ColumnVector.cpp`, `src/Common/RadixSort.h`, `base/base/sort.h`):

| Input type | N ≥ 64 | N < 64 | Top-K (limit set) |
|---|---|---|---|
| Fixed-width numeric | LSB Radix (stable, 8-bit passes) | Insertion sort | MSB Radix (partial) or FloydRivest |
| String / complex | PDQSort (`::pdqsort_try_sort`) | PDQSort | PDQSort + FloydRivest |

**Permutation approach** (key insight): Always sort an index array; never reorder column data. `iota(0..N)` initialises the permutation; all algorithms sort the index array; `column->permute(perm, limit)` materialises at the very end. Multi-column: `getPermutation()` on col[0] → `updatePermutation()` on col[1..N-1] within equal ranges tracked by `EqualRanges`. Column data is never touched until the final permute call.

**Null handling**: Two-pointer O(N) scan moves nulls to the requested end *after* sorting valid indices. For stable sort: extra `::sort()` call on the null range preserves insertion order.

**NaN / float**: Bit-flip transform (positive: XOR `1<<31`; negative: XOR `0xFFFFFFFF`) maps IEEE 754 bit patterns to uint32/uint64 that sort correctly as unsigned integers — NaN maps to uint max → sorts last for ascending.

**Parallelism**: Pipeline-level (PartialSortingTransform → MergeSortingTransform → MergingSortedTransform). No intra-block parallel sort. Priority-queue K-way merge across blocks. Radix sort itself is sequential per block.

**SIMD**: Radix histogram loop unrolled with `UNROLL_DISTANCE`; `__builtin_prefetch` inside scatter loop; SSE2 identity-permutation fast-exit check.

**External sort**: MergeSortingTransform spills blocks to disk (NativeFormat) when memory exceeds threshold; remerge strategy when savings justify it.

---

### DuckDB — Row-Normalised Keys + Merge-Path Parallelism

**Row normalisation** (`src/include/duckdb/common/sorting/sort_key.hpp`, `src/function/scalar/create_sort_key.cpp`):
- All sort columns encoded into a single byte-comparable key via `create_sort_key` expression. 9 `SortKeyType` variants by total width (`NO_PAYLOAD_FIXED_{8,16,24,32}`, `PAYLOAD_*`, `*_VARIABLE_32`).
- Null bytes: `0x01` = NULLS FIRST, `0x02` = NULLS LAST. Signed int: big-endian XOR 0x80 on MSB. Strings: each byte +1, null-terminated. `sort_skippable_bytes` lets the radix pass skip all-zero byte positions.

**Algorithm selection** (`src/common/sort/sorted_run.cpp`):
- **Vergesort** (primary): detects natural sorted/reverse-sorted runs; quicksort for small (< 80 elements) or non-run data.
- **SKA sort** (byte-at-a-time radix, fallback): extracts uint64 from sort key, skips skippable bytes.
- **PDQSort branchless**: used during merge operations.

**Parallel K-way merge** (`src/common/sort/sorted_run_merger.cpp`): Thread-local sort in sink phase → lock-free merge-path boundary computation; each merge partition processed by one thread.

**External sort**: `TupleDataPinProperties::UNPIN_AFTER_DONE` during reorder; `TemporaryMemoryManager` doubles reservation until it fits; `external = true` flag switches to spill mode.

---

### Polars — Permutation Refinement + Rayon Parallel Sort

**Algorithms** (`crates/polars-core/src/chunked_array/ops/sort/`):
- `sort_unstable_by()` (Rust std introsort) for single column.
- `select_nth_unstable_by()` (quickselect) for top-K — partitions at K-th element then sorts the lower K.
- `par_sort_unstable_by()` / `par_sort_by()` via Rayon for parallel path.

**Multi-column routing** (`mod.rs:823-863`): Row-encoded path (polars-row) for nested/complex types or `all(nulls_last)`; columnar comparison cascade otherwise. Polars prefers row encoding; Marrow will use permutation refinement instead.

**Null handling**: Validity bitmap scan partitions nulls *before* sorting; nulls reinserted at correct end afterward (Strategy A).

**Fast paths**: Already-sorted → clone; already-reverse-sorted → `reverse_stable_no_nulls()` O(N) reverse; `bottom_k_impl` for `df.sort().slice(0, K)` (quickselect).

**Row encoding** (polars-row): Null = `0x00`; valid unsigned: `0x01` + big-endian bytes; valid signed: `0x01` + big-endian with MSB XOR'd; valid float: sortable int representation. 32-byte block scheme + `0xFF` sentinel for strings enables unescaped `memcmp`. Designed for AVX-256.

---

### DataFusion / Arrow-RS — RowConverter + Loser Tree Merge

**Arrow sort** (`arrow-rs/arrow-ord/src/sort.rs`):
- `sort_to_indices(array, options, limit)`: single column, type-dispatched; `partial_sort` when `limit < N`.
- `lexsort_to_indices(columns)`: multi-column; specialised `LexicographicalComparator` with compile-time specialisations for N=2..5.
- All paths unstable.

**Row format** (`arrow-rs/arrow-row/`): `RowConverter::new(SortField[])` encodes columns to normalised byte sequences; `memcmp` gives correct total order. Fixed encoding: sign-bit toggle + big-endian; complement for DESC.

**TopK** (`datafusion/physical-plan/src/topk/mod.rs`): `TopKHeap` — max-heap of K rows; dynamic filter pushdown prunes upstream rows once K items are collected. Partial sort operator for inputs already partially sorted by a prefix.

**Merge**: Loser tree (tournament tree) — O(log K) per element across K sorted streams. Round-robin tiebreaker for stability.

**External sort**: Buffer batches → Arrow IPC spill when memory exhausted → streaming merge. Configurable compression via `SpillCompression`.

**String prefix optimisation**: Extract first 4 bytes as `UInt32`; compare as integer before full string comparison — skips most full string compares on typical data.

---

### Mojo Standard Library (`std/builtin/sort.mojo`)

| Primitive | Signature | Notes |
|---|---|---|
| `sort[T, cmp_fn]` | `(span, stable=False) -> None` | Unstable: iterative quicksort, median-of-3, insertion sort fallback at 32 |
| `sort[T, cmp_fn]` | `(span, stable=True) -> None` | Stable: merge sort, allocates temp buffer internally |
| `partition[cmp_fn]` | `(span, k) -> None` | Quickselect — O(N) average; elements [0,k) are k smallest |
| `_small_sort[n, T, cmp_fn]` | compile-time n=2..5 | Optimal sort networks, minimal comparator count |
| `vectorize[func, W]` | SIMD loop with scalar remainder | Fundamental vectorization building block |
| `prefetch[dtype]` | `(addr, options)` | READ/WRITE, locality 0-3 |
| `Span.unsafe_swap_elements` | `(a, b) -> None` | Branchless element swap |

Mojo stdlib has **no radix sort** — must implement.

---

### Parallel Approach Comparison

| Aspect | ClickHouse | DuckDB | Polars | DataFusion |
|---|---|---|---|---|
| **Sort algorithm** | LSB Radix + PDQSort | Vergesort + SKA radix | std introsort | std unstable |
| **Multi-column** | Permutation refinement | Row-normalised key | Row encoding | Row converter |
| **Top-K** | FloydRivest + MSB radix | MergeSortTree | quickselect | partial_sort + TopKHeap |
| **Parallelism** | Pipeline (block-level) | Merge-path K-way | Rayon par_sort | Loser tree merge |
| **Null placement** | Two-pointer post-sort | Encoded in key byte | Partition pre-sort | Partition pre-sort |
| **External sort** | MergeSorting + spill | TupleData unpin | — | Arrow IPC spill |

**Recommendation for Marrow**: Adopt ClickHouse's permutation-first approach (stays columnar, zero row conversion) combined with Polars' pre-sort null partitioning (cleaner comparators, faster sort body). Parallelism from Phase 1 using the same `sync_parallelize` / `wants_parallel` pattern as `join.mojo`.

---

## Marrow Design

### Allocation Model

From `marrow/buffers.mojo`:
- `Buffer.alloc_uninit[T](n)` — O(1), no memset. Use whenever every element will be written before being read.
- `Buffer.alloc_zeroed[T](n)` — O(n), calls `memset_zero`. Only needed for sparse output.
- `buffer.to_immutable()` — **zero-cost type rebind**. `ArcPointer[Allocation]` already created at allocation time; conversion is purely a compile-time origin change, no copy.
- `Buffer[mut=True]` / `BufferView[mut=True]` support in-place mutation via `unsafe_set()` and SIMD `store[W]()`.

**Radix sort allocation budget** (two-buffer LSD scheme):
- Two `alloc_uninit` buffers S and D, each N × `sizeof((encoded_key, orig_index))`.
- Histogram: `InlineArray[Int32, 256]` on the stack — no heap allocation.
- After the final pass, `winner.to_immutable()` → `Int32Array`. Zero extra copies.
- **Total heap**: 2 × N × pair size. Nothing else.

**`take()` for full permutations**: The existing `take()` / `_gather()` in `filter.mojo` already hits the optimal fast path for full permutations (no `-1` null indices, no source nulls) — SIMD gather via `src.gather[W](offsets)` with no bitmap overhead. The sort kernel always produces full permutations, so it will always take this path.

---

### Public API

The API mirrors the join API: `StructArray` is the table/batch container; sort key columns are identified by positional index. Single-column sort is the degenerate case of a one-field StructArray with `key_indices=[0]`.

```mojo
# ---- Primary API (follows join pattern) ----

# Returns sorted permutation indices. Apply with take().
def argsort(
    array: StructArray,
    key_indices: List[Int],             # sort keys in priority order
    ascending: List[Bool],              # per-key direction
    nulls_first: Bool = True,
    stable: Bool = False,
    limit: Optional[Int] = None,        # top-K: return only first `limit` indices
    ctx: ExecutionContext = ExecutionContext.serial(),
) raises -> Int32Array

# Convenience: single-column argsort on DynArray.
def argsort(
    array: DynArray,
    ascending: Bool = True,
    nulls_first: Bool = True,
    stable: Bool = False,
    limit: Optional[Int] = None,
    ctx: ExecutionContext = ExecutionContext.serial(),
) raises -> Int32Array

# Apply a permutation to materialise sorted output. Reuses _gather() from join.mojo.
def take(array: DynArray, indices: Int32Array, ctx: ExecutionContext = ...) raises -> DynArray
def take(array: StructArray, indices: Int32Array, ctx: ExecutionContext = ...) raises -> StructArray

# Convenience: sort StructArray and return sorted copy.
def sort(
    array: StructArray,
    key_indices: List[Int],
    ascending: List[Bool],
    nulls_first: Bool = True,
    stable: Bool = False,
    limit: Optional[Int] = None,
    ctx: ExecutionContext = ExecutionContext.serial(),
) raises -> StructArray
```

**`SortOptions`** companion struct (for future per-column `nulls_first` extension):
```mojo
struct SortOptions:
    var ascending: Bool
    var nulls_first: Bool
```

---

### Sorter Catalog

#### 1. RadixSortPrimitive[T] — O(N) for fixed-width numerics

**Inspired by**: ClickHouse LSB radix, DuckDB SKA sort
**When**: Single primitive column (`int8`..`int64`, `uint8`..`uint64`, `float32`, `float64`), N ≥ 64.

**Sort key encoding** (in-place into pair buffer before radix passes):

| Source type | Encoding |
|---|---|
| Unsigned | Value as-is (already unsigned order) |
| Signed int | XOR `1 << (bits-1)` — flips sign bit → unsigned order |
| Float32 | Positive: XOR `0x80000000`; Negative: XOR `0xFFFFFFFF`. NaN → uint max → sorts last |
| Float64 | Same pattern with 64-bit masks |
| Descending | Complement all encoded bits (XOR `0xFF..FF`) |

Null indices: already removed by null partitioning (see below); radix never sees nulls.

**LSD radix (zero-copy)**:
1. Allocate pair buffers S and D: `alloc_uninit[(encoded_key, UInt32)](N)` each. Fill S.
2. For each byte pass (2 passes for 16-bit, 4 for 32-bit, 8 for 64-bit):
   - **Histogram**: `vectorize[hist_fn, W]` over S; accumulate `InlineArray[Int32, 256]` (stack).
   - **Prefix sum**: 256-element serial pass → output offsets.
   - **Scatter** with prefetch: read S[i], `prefetch(S_ptr + i + 16)`, write to D[offset[byte]++].
   - Swap S ↔ D pointers.
3. `winner.to_immutable()` → zero-cost freeze → extract `Int32Array` of `orig_index` values.

**Insertion sort fallback**: N < 64 → Mojo stdlib `sort[cmp_fn](span)` on an index span; no pair buffers needed.

**Stable**: LSB radix preserves relative input order for equal keys — inherently stable. No extra work for `stable=True`.

**CPU parallelism** (active from Phase 1, enabled when `ctx.wants_parallel(N)` and N ≥ 65_536):
1. **Parallel histogram**: `sync_parallelize[worker](nt)` — each worker fills a thread-local `InlineArray[Int32, 256]`; serial reduce → global histogram → prefix sum → per-thread output offsets.
2. **Parallel scatter**: each worker reads its stripe and writes to pre-computed disjoint output slots. No atomics, no contention.

Factor `_parallel_radix_histogram` as a shared utility callable from both `sort.mojo` and `join.mojo`'s `RadixPartitioner`.

**GPU path**:
```mojo
if ctx.is_gpu():
    var s = Buffer.alloc_device[pair](ctx.device.value(), n)
    var d = Buffer.alloc_device[pair](ctx.device.value(), n)
    # encode + histogram via elementwise[encode_hist_fn, W, target="gpu"](n, ctx.device.value())
    # scatter via elementwise[scatter_fn, W, target="gpu"] with GPU atomics on histogram
    # winner.to_immutable() → Int32Array (device)
```
GPU histogram uses per-warp local histograms or atomic adds on the 256-bucket array — standard GPU radix pattern. Phase 1 raises `Error("RadixSort: GPU not yet implemented")`.

---

#### 2. ComparisonSort — Mojo stdlib for strings, bools, small arrays

**Inspired by**: ClickHouse PDQSort, Mojo stdlib `sort[cmp_fn]`
**When**: `StringArray`, `BoolArray`, primitives with N < 64, or when `stable=True` for primitives.

**How**:
1. Allocate `alloc_uninit[int32](N)` index buffer; fill with `iota(0..N)` via `vectorize`.
2. Define `cmp_fn(a: Int32, b: Int32) -> Bool` closure: `array[a] < array[b]` with null semantics already handled by pre-partitioning.
3. Call Mojo stdlib `sort[cmp_fn](span, stable=stable)`:
   - `stable=False` → iterative quicksort + insertion sort (threshold=32). O(N log N).
   - `stable=True` → merge sort. Stdlib allocates its own temp buffer internally; freed on return.
4. `buf.to_immutable()` → `Int32Array`. Zero-copy.

**Allocation budget**: one `alloc_uninit[int32](N)`. Stdlib merge sort's internal buffer is temporary.

**Top-K**: `partition[cmp_fn](span, limit)` → quickselect O(N); then `sort[cmp_fn](span[:limit])` → O(K log K). Equivalent to ClickHouse's FloydRivest + partial_sort.

**String comparison**: `StringArray[a] < StringArray[b]` via `StringSlice.__lt__`. **4-byte prefix optimisation** (DataFusion approach): extract first 4 bytes as `UInt32`, compare as integer; fall through to full string compare only when equal. Avoids most full compares on real data.

**BoolArray fast path**: one `vectorize` pass counting `false/true/null` → directly construct index array in three memcpy-like fills. O(N), zero comparisons, no sort call.

**GPU**: ComparisonSort requires sequential pivoting — not amenable to GPU. Phase 1 strings: sort CPU-side, copy result to device.

---

#### 3. Permutation Refinement — Multi-column columnar sort

**Inspired by**: ClickHouse `getPermutation` + `updatePermutation` + EqualRanges
**When**: `argsort(StructArray, key_indices, ...)` with `len(key_indices) > 1`.

**How**:
1. Sort by `col[key_indices[0]]` → permutation P (using RadixSortPrimitive or ComparisonSort).
2. **Find equal ranges**: one O(N) scan under P — where `col[P[i]] == col[P[i-1]]`, extend the current equal range. Build `List[(start, end)]`.
3. For each subsequent key column `col[key_indices[k]]`:
   - For each equal range `(s, e)`: sort sub-span `P[s..e]` by `col[key_indices[k]]`.
   - Update equal ranges within each processed range.
   - Stop when no equal ranges remain or all key columns consumed.
4. P is the final multi-column sort permutation.

**Equal range scan — vectorizable**:
```mojo
@parameter
def detect_fn[W: Int](idx: Int):
    var a = col_view.load[W](idx)
    var b = col_view.load[W](idx + 1)   # shifted read
    var eq = a.eq(b)                     # SIMD bool mask
    # record boundary positions from eq mask
vectorize[detect_fn, W](n - 1)
```

**Within-range parallelism**: per-range sorts are typically tiny (real data has few exact ties across columns). For pathological data (many equal keys in col[0]), dispatch range list via `sync_parallelize` over ranges.

**Advantages over row encoding**: no temporary byte buffer; no encoding / decoding overhead; stays fully columnar; SIMD through comparison and gather paths unchanged; allocation budget is just the permutation array itself.

---

#### 4. Top-K / Partial Sort

**Inspired by**: ClickHouse FloydRivest, Polars quickselect, DataFusion TopKHeap
**When**: `limit` is set and `limit < N / 2`.

**Full-array case** (limit known upfront):
- `partition[cmp_fn](index_span, limit)` — Mojo stdlib quickselect, O(N) average.
- `sort[cmp_fn](index_span[:limit])` — sort the lower K elements, O(K log K).
- Total: O(N + K log K). For small K this dominates over O(N log N) full sort.

**Multi-column top-K**: same pattern applied to the permutation span in step 1 of Permutation Refinement.

**Streaming top-K** (future, ORDER BY LIMIT on unbounded input):
- `TopKHeap`: max-heap of K rows. Push each row; when full, pop if incoming row is smaller. O(N log K).
- Requires a row comparator over `StructArray`.

---

### Null Handling

**Strategy A — Partition pre-sort** (all paths):
```
1. Scan validity bitmap → valid_indices[], null_indices[]
2. Sort valid_indices[] (radix / comparison / refinement)
3. nulls_first: output = concat(null_indices, sorted_valid_indices)
   nulls_last:  output = concat(sorted_valid_indices, null_indices)
```

Pre-partitioning means the sort algorithm never sees nulls — no null checks in comparators or encoding transforms, no special-casing inside radix passes.

**Bitmap scan**: `BitmapView` from `views.mojo`. Future: vectorize with `vectorize[scan_fn, W]` loading W validity bits at a time via SIMD bitmasking.

**Multi-column**: null partitioning applied per-column per-range during refinement steps. Null placement is consistent: `nulls_first` applies uniformly to all key columns.

---

### NaN / Float Handling

Bit-flip transform applied in RadixSortPrimitive during pair construction:

```
positive float:  bit_cast<uint>(f) XOR 0x80000000   (flip sign bit only)
negative float:  bit_cast<uint>(f) XOR 0xFFFFFFFF   (flip all bits)
NaN:             bit_cast<uint>(NaN) XOR 0xFFFFFFFF → uint max → sorts last (ascending)
-0.0 == +0.0:   both map to same uint after transform ✓
+Inf > any:     bit pattern after transform preserves this ✓
-Inf < any:     bit pattern after transform preserves this ✓
```

For ComparisonSort: `Float.is_nan()` → treat as greater than any finite value (consistent with PyArrow default).

---

### Stable Sort

Default: **unstable** — matches NumPy/PyArrow `argsort` default; faster.

`stable=True`:
- **RadixSortPrimitive**: already stable (LSB radix preserves input order for equal keys). No code change.
- **ComparisonSort**: pass `stable=True` to Mojo stdlib → merge sort path.
- **PermutationRefinement**: each sub-sort within an equal range uses `stable=True`.

---

### Vectorization Strategy

All vectorization follows the existing `apply[]` / `vectorize[]` / `elementwise[]` patterns from `views.mojo` and `kernels/`.

**Null partitioning (bitmap scan)**:
```mojo
var valid_indices = List[Int32](capacity=n)
var null_indices  = List[Int32]()
if bitmap:
    for i in range(n):
        if bitmap.is_valid(i): valid_indices.append(i)
        else:                  null_indices.append(i)
```
Future: load W validity bits at a time via SIMD, extract set/unset positions.

**Radix histogram (SIMD load, scalar scatter)**:
```mojo
var hist = InlineArray[Int32, 256](fill=0)
@parameter
def hist_fn[W: Int](idx: Int):
    var keys = src_view.load[W](idx)
    for lane in range(W):                        # scalar scatter to 256 buckets
        hist[Int(keys[lane] & 0xFF)] += 1
vectorize[hist_fn, simd_width_of[T.native]()](n)
```
Inner scatter stays scalar — gather/scatter SIMD for 256 buckets requires careful offset handling that doesn't pay off at this bucket count.

**Radix scatter (with prefetch)**:
```mojo
for i in range(n):
    prefetch[DType.uint8](pairs_ptr + i + 16)    # ClickHouse's PREFETCH_DISTANCE
    var byte = Int(pairs_s[i].key >> (pass_idx * 8)) & 0xFF
    pairs_d[offsets[byte]] = pairs_s[i]
    offsets[byte] += 1
```

**Iota (index initialisation)**:
```mojo
@parameter
def iota_fn[W: Int](idx: Int):
    var v = SIMD[DType.int32, W]()
    for lane in range(W): v[lane] = idx + lane
    dst.store[W](idx, v)
vectorize[iota_fn, simd_width_of[DType.int32]()](n)
```

**Permutation application**: reuse `take()` from `join.mojo` — SIMD gather via `src.gather[W](offsets)` in the no-null fast path.

**Equal-range detection**:
```mojo
@parameter
def detect_fn[W: Int](idx: Int):
    var a = col_view.load[W](idx)
    var b = col_view.load[W](idx + 1)
    var eq = a.eq(b)                             # SIMD bool mask
    # extract boundary indices from eq mask
vectorize[detect_fn, simd_width_of[T.native]()](n - 1)
```

---

### Parallelisation

Parallelism is a first-class design axis, active from Phase 1. All parallel paths use `sync_parallelize` / `ctx.wants_parallel` as in `join.mojo` and `hashing.mojo`.

**Parallel RadixSortPrimitive** (N ≥ 65_536, `ctx.wants_parallel(N)`):
```
1. sync_parallelize[worker](nt):
       each worker fills thread-local InlineArray[Int32, 256] over its stripe
2. Serial reduce: sum thread histograms → global histogram
3. Prefix sum → per-thread output offsets
4. sync_parallelize[scatter_worker](nt):
       each worker scatters its stripe to pre-computed disjoint output slots
```
No atomics. No contention. Identical pattern to `build_parallel` in `join.mojo`. Shared utility `_parallel_radix_histogram` extracted from `RadixPartitioner`.

**Parallel ComparisonSort** (large N):
- Split index span into T halves; `sync_parallelize` sorts each independently.
- Serial merge-sort T sorted halves. O(N log T) merge cost amortised by parallel sort.

**Permutation Refinement parallelism**:
- First-column sort is already parallel via RadixSortPrimitive.
- Sub-range sorts (col[1..]) are typically tiny — serial.
- For pathological equal-key data: `sync_parallelize` over the range list.

**GPU parallelism**:
```mojo
if ctx.is_gpu():
    var s = Buffer.alloc_device[pair](ctx.device.value(), n)
    var d = Buffer.alloc_device[pair](ctx.device.value(), n)
    elementwise[encode_fn, gpu_width, target="gpu"](n, ctx.device.value())
    # ... GPU histogram + scatter passes ...
    return result.to_immutable()
```
`elementwise[target="gpu"]` handles thread/warp scheduling. No explicit warp management at the Mojo level.

**Batch-level parallelism** (ORDER BY with streaming batches):
- Each batch sorted independently via `sync_parallelize` over batch list.
- Final K-way merge via priority queue (loser tree, Phase 3).

---

### Algorithm Selection

```
argsort(DynArray, ...)
  │
  ├── BoolArray                → BoolSorter (count pass, O(N), zero comparisons)
  │
  ├── PrimitiveArray[T]
  │     ├── N < 64             → ComparisonSort (stdlib sort, no pair buffers)
  │     └── N ≥ 64             → RadixSortPrimitive[T] (LSD, parallel when N ≥ 65k)
  │
  └── StringArray              → ComparisonSort (stdlib sort + 4-byte prefix opt)

argsort(StructArray, key_indices, ...)
  │
  ├── len(key_indices) == 1    → argsort(DynArray) on that column
  │
  └── len(key_indices) > 1     → PermutationRefinement
          Sort by key[0] (dispatch to above)
          → find equal ranges
          → refine by key[1], key[2], ...

limit set and limit < N / 2   → Top-K: partition[cmp_fn](span, limit) + sort lower K
```

---

### Python Bindings

Following PyArrow API naming:

```python
class Array:
    def sort(self, ascending=True, null_placement="at_start") -> Array
    def argsort(self, ascending=True, null_placement="at_start") -> Array  # → Int32Array

class RecordBatch:
    def sort_by(
        self,
        by: list[str | tuple[str, str]],   # name or (name, "ascending"/"descending")
        null_placement: str = "at_start",
    ) -> RecordBatch
```

Matches PyArrow's `Array.sort()`, `Array.argsort()`, `RecordBatch.sort_by()`.

---

### Relation Node

```mojo
comptime SORT_NODE: UInt8 = 7

struct Sort(Relation):
    var input: DynRelation
    var keys: List[AnyValue]           # column exprs resolved to input schema
    var ascending: List[Bool]
    var nulls_first: Bool
    var limit: Optional[Int]
    var schema_: Schema                # same schema as input
```

`SortProcessor.pull()`: collect all input batches → concat → `argsort()` → `take()` → emit one batch.

---

## Implementation Phases

Each phase ships with tests **and** benchmarks.

### Phase 1 — Single-column argsort: radix (CPU parallel) + comparison

**New files**:
- `marrow/kernels/sort.mojo`
- `marrow/tests/test_sort.mojo`
- `marrow/kernels/tests/bench_sort.mojo`

1. **Null partitioning helper** (`_partition_nulls`): bitmap scan → `valid_indices[]`, `null_indices[]`.
2. **`_argsort_primitive[T](values, ascending, nulls_first, stable, limit, ctx)`**: two `alloc_uninit` pair-buffers; encoding transform; LSD radix with serial + parallel branches (`ctx.wants_parallel`); insertion sort fallback for N < 64.
3. **`_argsort_string(array, ascending, nulls_first, stable, limit, ctx)`**: `alloc_uninit` index buffer + iota; stdlib `sort[cmp_fn]` with 4-byte prefix optimisation. CPU only (Phase 1).
4. **`_argsort_bool(array, ascending, nulls_first)`**: count pass, direct index construction. O(N).
5. **`argsort(DynArray, ...)`**: type-dispatch via `ArrayVisitor`.
6. **`take(DynArray, Int32Array, ctx)`**: thin wrapper over `_gather()` from `join.mojo`. Full permutations always hit the no-null fast path.
7. **`sort(StructArray, key_indices, ascending, ...)`**: `argsort` → `take` per child column → new StructArray (following join's `_assemble` pattern).

**Tests** (`marrow/tests/test_sort.mojo`): correctness vs PyArrow `sort_to_indices` / `RecordBatch.sort_by` as oracle. Cases: ascending/descending, nulls_first/last, stable/unstable, floats with NaN/±Inf/−0.0, BoolArray, empty array, single element, already-sorted input (fast-path validation).

**Benchmarks** (`marrow/kernels/tests/bench_sort.mojo`):
```mojo
def bench_sort_int64_1m(mut b: Benchmark) raises:    _bench_sort(b, int64,   1_000_000)
def bench_sort_float64_1m(mut b: Benchmark) raises:  _bench_sort(b, float64, 1_000_000)
def bench_sort_string_100k(mut b: Benchmark) raises: _bench_sort_string(b,   100_000)
```
Use `--competition` flag to compare vs Polars, PyArrow, DuckDB.

---

### Phase 2 — Multi-column argsort via Permutation Refinement

1. **`_equal_ranges(col: DynArray, perm: Int32Array)`**: vectorized adjacent-element comparison → `List[(Int, Int)]` equal-value ranges.
2. **`_refine_ranges(perm, ranges, col, ascending, cmp_fn)`**: per-range sub-span sort using same `_argsort_primitive` / `_argsort_string` dispatch.
3. **`argsort(StructArray, key_indices, ascending, nulls_first, stable, limit, ctx)`**: permutation refinement loop.
4. Per-column null partitioning within ranges.
5. Multi-column `take` in `sort(StructArray, ...)`: `take()` per child column → new StructArray.

**Tests**: multi-column sorts against PyArrow `lexsort_to_indices` and `RecordBatch.sort_by`. Include: mixed types, large equal-key groups, descending sub-keys, 3-column sorts.

**Benchmarks**: `bench_sort_multi_2col_1m`, `bench_sort_multi_3col_1m`. Compare vs Polars `df.sort(by=[...])`.

---

### Phase 3 — Top-K, GPU, Relation Node, Python Bindings

1. **`_nth_element(perm_span, k, cmp_fn)`**: Mojo stdlib `partition[cmp_fn](span, k)` wrapper.
2. **`argsort(..., limit=K)`**: nth_element + sort lower K slice; wired into both radix and comparison paths.
3. **GPU radix sort**: `elementwise[encode_fn, W, target="gpu"]` for pair encoding + per-warp histogram; GPU scatter; `Buffer.alloc_device` for S and D pair buffers.
4. **`Sort` relation node** + **`SortProcessor`** in `marrow/expr/relations.mojo` + executor.
5. **`SortMergeJoinProcessor`**: `argsort` + two-pointer merge algorithm (joins-design.md Phase 3).
6. **Python bindings**: `Array.sort()`, `Array.argsort()`, `RecordBatch.sort_by()`.
7. **ASAN tests**: `pixi run -e asan test_mojo_asan` on sort tests — catches use-after-free in pair buffers.

**Benchmarks**: `bench_sort_topk_100_of_1m`, `bench_sort_gpu_float32_1m` (GPU baseline), `bench_sort_gpu_preloaded_float32_1m` (data already on device).

---

## Key Files

| File | Role |
|---|---|
| `marrow/kernels/sort.mojo` | New — all sort primitives |
| `marrow/kernels/join.mojo` | `_gather()` reused for `take()` |
| `marrow/kernels/hashing.mojo` | Parallel histogram pattern reference |
| `marrow/views.mojo` | `apply[]`, `vectorize[]`, `BitmapView` for null scan |
| `marrow/arrays.mojo` | `StructArray.select()`, child iteration |
| `marrow/expr/relations.mojo` | `Sort` relation node (Phase 3) |
| `marrow/tests/test_sort.mojo` | Correctness tests vs PyArrow |
| `marrow/kernels/tests/bench_sort.mojo` | Benchmarks and GPU tests |
| `docs/sort-design.md` | This document |

### Reused Utilities

| Utility | Source | Purpose |
|---|---|---|
| `_gather(array, indices)` | `join.mojo` | Implements `take()` for free; full-permutation fast path |
| `BitmapView` | `views.mojo` | Null partitioning (validity bitmap scan) |
| `ExecutionContext` | `execution.mojo` | Thread count + GPU device dispatch |
| `sync_parallelize` | Mojo stdlib | Parallel radix histogram + scatter |
| `sort[cmp_fn](span, stable)` | Mojo stdlib | ComparisonSort for strings / small arrays |
| `partition[cmp_fn](span, k)` | Mojo stdlib | Top-K quickselect |
| `vectorize[fn, W]` | Mojo stdlib | SIMD loops (iota, histogram, equal-range scan) |
| `prefetch[dtype](addr)` | Mojo stdlib | Radix scatter prefetch |
| `StructArray.select(indices)` | `arrays.mojo` | Zero-copy key column extraction |
| `RadixPartitioner` histogram | `join.mojo` | Factor into shared `_parallel_radix_histogram` |

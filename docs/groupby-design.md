# Plan: Group-By Kernel for Marrow

## Context

Marrow needs a group-by kernel — the fundamental primitive that assigns each row a group index. The goal is state-of-the-art performance with multi-key support, good parallelization, and memory-constrained operation. This means multiple specialized grouper implementations, like ClickHouse (40+), DuckDB (4), and Polars (3) all have.

---

## Prior Art

### ClickHouse — Specialization King

**Single-key specializations by type:**
- `key8/key16`: UInt8/16 → `FixedHashMap` (direct array indexing, 256/65536 slots, O(1), zero hash)
- `key32/key64`: UInt32/64 → `HashMap` with CRC32 hash
- `key_string`: → `StringHashMap` with inline short-string storage
- Low-cardinality variants: work with dictionary indices instead of values

**Multi-key packing:**
- `keys16/32/64/128/256`: Pack N fixed-width keys into 2-32 byte blob → single hash lookup
- Decision: total key bytes ≤ 32 and all fixed-width → pack; else serialize
- `serialized`/`prealloc_serialized`: Binary concatenation fallback for mixed types

**Nullable keys:** Every method has `nullable_*` variant with separate `null_key_data`.

**Two-level hash tables** (for parallelism + spilling):
- 256 fixed sub-tables, bucket = `(hash >> 24) & 0xFF`
- Enables lock-free parallel merge AND partition-aligned spilling

**Parallel merge** (Aggregator.cpp:3722-3871):
- `std::atomic<UInt32> next_bucket_to_merge` distributes 256 buckets across threads
- Each thread grabs bucket via `fetch_add(1)`, merges all chunks for that bucket
- No locks: each bucket is a disjoint hash table, threads write to independent targets
- **Scales linearly up to 256 threads**

**External aggregation** (spilling, Aggregator.cpp:1875-2117):
- Triggered when `current_memory_usage > max_bytes_before_external_group_by`
- Writes each of 256 buckets independently to temp files
- Later merges: read spilled buckets back, merge with in-memory buckets
- Two-level structure is critical: bucket-aligned writes = efficient sequential I/O

**Aggregation-in-order** (pre-sorted input):
- No hash table, O(n) scan: accumulate while key matches, emit on change

### DuckDB — Vectorized & Adaptive

**4 strategies:** Ungrouped, PerfectHash, Partitioned, HashAggregate.

**Hash Aggregate (main workhorse):**
- Linear probing with salt-based collision detection (top hash bits as fingerprint)
- Vectorized probing: branchless hash loop, multi-pass with selection vectors
- Radix partitioning for cache locality

**Parallel aggregation:**
- Thread-local hash tables during sink (no synchronization)
- **Sequential combine** after sink (each thread's table merged one-at-a-time into global state)
- Weakest parallel merge of the three systems

**Memory management** (radix_partitioned_hashtable.cpp:220-236):
- Reservation-based: `min_reservation = num_threads × num_partitions × blocks_per_partition × block_size`
- Adaptive radix bits: starts at 4, increases to 8 under memory pressure → more, smaller partitions
- Repartitioning on OOM: `SetRadixBitsToExternal()` increases partition count

### Polars — Streaming & Cache-Efficient

**3 groupers:** SingleKey, Binview (strings), RowEncoded (multi-key).

**Hot/cold streaming** (fixed_index_table.rs:19-44):
- `FixedIndexTable`: 4096 slots, cuckoo hashing with 2 probes (`h1`, `h2 = hash * H2_MULT`)
- LRU eviction: `last_access_tag` tracks recency; on miss, evict oldest entry
- Evicted keys stored as pre-aggregates in per-partition cold storage
- Hot table fits in L2 cache → excellent locality

**Parallel group-by** (group_by.rs:251-334, 4 phases):
1. Streaming ingest: each thread has own hot table, processes morsels independently
2. Partition evictions: evicted keys hashed to partition P, pre-aggs stored per-partition
3. Per-partition grouper: sized via cardinality estimation, pre-aggs merged in
4. Output: partitions converted to DataFrames in parallel

**Cardinality estimation** (Chao-Srichand):
- Sample-based: `u + (ui / m) * (s - m)` where u = unique in sample, ui = singletons
- Pre-sizes output grouper at estimate × 1.25

---

## Parallel Approach Comparison

| Aspect | ClickHouse | DuckDB | Polars |
|--------|-----------|--------|--------|
| **Merge parallelism** | **Best**: atomic counter distributes 256 buckets | Worst: sequential combine | Good: parallel per-partition |
| **Scalability** | Linear to 256 threads | Sublinear (merge bottleneck) | Linear with partition count |
| **Cache behavior** | Good (bucket-isolated) | Good (thread-local sink) | **Best** (4KB hot table in L2) |
| **Memory control** | Threshold-based spill | Adaptive radix bits | Fixed hot table + pre-agg buffer |
| **Spill mechanism** | Bucket-aligned file writes | Repartition + block writes | Pre-agg overflow to cold storage |
| **Complexity** | High (256 × method variants) | Medium | Medium-high (4 phases) |

**Recommendation for Marrow**: Adopt **ClickHouse's two-level approach** for the parallel grouper (best merge scalability, proven at massive scale), combined with **DuckDB's adaptive memory management** (radix bits increase under pressure). Polars' hot/cold approach is excellent for streaming but adds complexity we don't need initially.

---

## Mojo Stdlib Building Blocks

| Component | What it provides |
|---|---|
| `Dict[K, V]` | Swiss Table, 16-way SIMD probing, 7/8 load factor, insertion-order |
| `hash()` / `AHasher` | 64-bit SIMD-aware hash, works on any `Hashable` |
| `Scalar[T]` | Satisfies `KeyElement` → `Dict[Scalar[T], UInt32]` works directly |
| `String` | Satisfies `KeyElement` → `Dict[String, UInt32]` for string keys |
| `elementwise()` | SIMD vectorized element-wise loop (CPU + GPU) |
| `vectorize()` | Lower-level SIMD loop with width + unroll |
| `Tuple[*Ts]` / `InlineArray[T, N]` | Hashable composite keys / fixed-size packing |

---

## Grouper Catalog

### 1. DirectMapGrouper — Zero-hash array-indexed

**Inspired by**: ClickHouse `key8`/`key16`, DuckDB PerfectHash

**When**: Single key with tiny value range — `bool_`, `uint8`, `int8` (and optionally `uint16`/`int16`).

**How**: Flat array of `Int32` slots indexed by key value. No hashing.
- `bool_`: 3 slots (false, true, null)
- `uint8`: 256 slots, `index = value`
- `int8`: 256 slots, `index = value + 128`
- `uint16`/`int16`: 65536 slots (128KB, fits in L2)

**Performance**: O(1), zero hash overhead, perfectly cache-friendly.

### 2. PrimitiveHashGrouper[T] — Swiss Table for single numeric keys

**Inspired by**: ClickHouse `key32`/`key64`, Polars SingleKeyHashGrouper

**When**: Single numeric key not handled by DirectMap (int32, int64, uint32, uint64, float32, float64).

**How**: `Dict[Scalar[T.native], UInt32]` — stdlib Swiss Table with 16-way SIMD probing, AHasher.

### 3. StringHashGrouper — Swiss Table for string keys

**Inspired by**: ClickHouse `key_string`, Polars BinviewHashGrouper

**When**: Single string key.

**How**: `Dict[String, UInt32]`. StringSlice from array → String for Dict key.

### 4. PackedKeyGrouper — Multi-key fixed-width packing

**Inspired by**: ClickHouse `keys16`/`keys32`/`keys64`/`keys128`/`keys256`

**When**: Multiple key columns, all fixed-width, total bytes ≤ 16.

**How**: Pack all key bytes into a single blob per row → single hash lookup.
- Total ≤ 8 bytes → `Dict[UInt64, UInt32]`
- Total ≤ 16 bytes → `Dict[InlineArray[UInt8, 16], UInt32]`

Packing: concatenate column bytes + 1-byte null flag per nullable column.

### 5. RowEncodedGrouper — Multi-key general fallback

**Inspired by**: ClickHouse `serialized`/`prealloc_serialized` (preferred), Polars RowEncodedHashGrouper

**When**: Multi-key with variable-width types or total bytes > 16.

**How**: Serialize each row's keys into a contiguous byte sequence, then hash/compare the bytes. Following **ClickHouse's approach** (simpler and faster than Polars for hash-based grouping — no byte normalization needed since we only hash, not sort):

**Encoding format** (per row, columns concatenated):
- **Fixed-width types**: Raw `memcpy`, native byte order. No sign-flipping or big-endian conversion.
  - int32 → 4 bytes, int64 → 8 bytes, float64 → 8 bytes, etc.
- **Strings**: 8-byte UInt64 length prefix + raw bytes.
  ```
  [length: UInt64 (8B)] [payload: N bytes]
  ```
- **Nullable columns**: 1-byte prefix per column (0x00 = valid, 0x01 = null). Value omitted if null.
  ```
  [null_flag: 1B] [value: N bytes if not null]
  ```

**Example** — row with keys `(int32=42, string="hello", nullable int64=None)`:
```
[42 as 4 raw bytes] [0x00][5 as UInt64][h][e][l][l][o] [0x01]
     int32               valid  strlen      payload       null
```

**Two variants** (following ClickHouse):
1. **On-demand serialization**: For each row, serialize into arena, hash, insert into Dict. If key already exists, rollback arena allocation. Simple, good for moderate key counts.
2. **Prealloc batch serialization** (ClickHouse's `prealloc_serialized`): Pre-calculate all row sizes in one pass, batch-allocate the full buffer, then serialize all rows. Better cache behavior, chosen when all keys are numeric or string.

**Hash table**: `Dict[String, UInt32]` where the String holds the serialized bytes. Or a custom approach using arena-allocated `StringSlice` keys to avoid copying.

**Why ClickHouse over Polars**: ClickHouse skips byte normalization (no sign-flipping, no big-endian) because hash equality doesn't require it. This saves ~2 instructions per numeric key per row. Polars normalizes bytes because its RowEncoded format doubles as a sort key — we don't need that for group-by.

### 6. TwoLevelGrouper — Parallel wrapper (ClickHouse approach)

**Inspired by**: ClickHouse two-level hash tables

**When**: Large datasets where single-threaded grouping is the bottleneck.

**How**: 256 sub-groupers (one per bucket). Three phases:

1. **Hash + bucket assignment**: For each key, compute hash, extract bucket = `(hash >> 56) & 0xFF`. Each thread accumulates keys into per-bucket sub-groupers.
2. **Parallel merge**: Atomic counter distributes 256 buckets across threads. Each thread merges all partial results for its assigned buckets. No locks — bucket targets are disjoint.
3. **Global ID assignment**: Walk buckets 0..255, assign global IDs by offsetting each bucket's local IDs.

**SIMD pre-hashing with `elementwise()`**: Hash all keys in a single SIMD pass, then scatter to buckets. This is the natural fit for `elementwise()`.

```
Thread 0: aggregate into local sub-groupers[0..255]
Thread 1: aggregate into local sub-groupers[0..255]
...
Barrier
Thread 0: merge bucket 0, 4, 8, ... (atomic counter)
Thread 1: merge bucket 1, 5, 9, ...
...
```

### 7. SpillingGrouper — Memory-constrained operation

**Inspired by**: ClickHouse external aggregation, DuckDB adaptive radix

**When**: Memory budget exceeded during grouping.

**How**: Wraps TwoLevelGrouper. When memory threshold crossed:
1. Flush each of 256 buckets independently to temp files (sequential I/O per bucket)
2. Clear in-memory hash tables, reclaim memory
3. Continue aggregating into fresh tables
4. At end: for each bucket, merge in-memory results with spilled data (read back per-bucket)

**Why bucket-aligned spilling works**: Each of the 256 buckets is a self-contained hash table — no cross-bucket dependencies. Flushing bucket N only requires sequential writes. Merging bucket N only requires reading bucket N's spill files + in-memory bucket N. This is exactly how ClickHouse achieves efficient external aggregation.

**Memory tracking**: Track total hash table memory. When exceeding threshold, trigger spill. DuckDB's adaptive approach (increase radix bits under pressure to create more, smaller partitions) is also worth considering as an alternative — it avoids disk I/O by reducing per-partition memory footprint.

---

## Selection Logic

```
group_id(keys: Array)             → single-key dispatch
group_id(keys: List[Array])       → multi-key dispatch

Single-key:
  bool/uint8/int8       → DirectMapGrouper (256 slots)
  uint16/int16          → DirectMapGrouper (65536 slots)
  other numeric         → PrimitiveHashGrouper[T]
  string                → StringHashGrouper

Multi-key:
  all fixed, total ≤ 8B     → PackedKeyGrouper (UInt64)
  all fixed, total ≤ 16B    → PackedKeyGrouper (InlineArray)
  mixed/variable             → RowEncodedGrouper

Parallel (orthogonal, wraps any above):
  len(keys) > parallel_threshold → TwoLevelGrouper(inner)

Memory-constrained (orthogonal):
  memory > budget → SpillingGrouper(TwoLevelGrouper(inner))
```

---

## Public API

```mojo
# Single-key (typed)
def group_id[T: DataType](keys: PrimitiveArray[T]) raises -> PrimitiveArray[uint32]
def group_id(keys: StringArray) raises -> PrimitiveArray[uint32]

# Single-key (type-erased)
def group_id(keys: Array) raises -> PrimitiveArray[uint32]

# Multi-key
def group_id(keys: List[Array]) raises -> PrimitiveArray[uint32]

# Unique (companion)
def unique[T: DataType](keys: PrimitiveArray[T]) raises -> PrimitiveArray[T]
def unique(keys: StringArray) raises -> StringArray
def unique(keys: Array) raises -> Array
```

---

## Implementation Order

### Step 1: PrimitiveHashGrouper + StringHashGrouper + type-erased dispatch

Core Dict-based groupers covering all numeric and string types. Minimum viable `group_id`.

**Files:**
- Create `marrow/kernels/groupby.mojo`
- Create `marrow/kernels/tests/test_groupby.mojo`
- Update `marrow/kernels/__init__.mojo` docstring
- Update `CHANGELOG.md`

### Step 2: DirectMapGrouper

Zero-hash fast path for bool/uint8/int8. Small self-contained optimization.

### Step 3: PackedKeyGrouper (multi-key, fixed-width)

Multi-key support for the common case (e.g., `group_by(year, month)`). Packs columns into UInt64 or InlineArray.

### Step 4: RowEncodedGrouper (multi-key, general)

General multi-key fallback. Requires implementing row encoding format.

**File:** `marrow/kernels/row.mojo` (reusable for sorting, joins).

### Step 5: Vectorized hash pre-computation

`elementwise()` to SIMD-parallelize hash computation. Foundation for TwoLevelGrouper.

### Step 6: TwoLevelGrouper (parallel)

256 sub-groupers + atomic bucket distribution for parallel merge. Depends on Mojo threading primitives.

### Step 7: SpillingGrouper (memory-constrained)

Bucket-aligned spill to temp files when memory budget exceeded. Wraps TwoLevelGrouper.

### Step 8: Fused grouped aggregation

`group_sum`, `group_min`, `group_max`, `group_count` — accumulate aggregate state during grouping pass.

### Step 9: Sorted input optimization

Pre-sorted keys → run-detection, no hash table, O(n).

---

## Tests

**File: `marrow/kernels/tests/test_groupby.mojo`**

- `test_group_id_int32_basic` — `[1, 2, 1, 3, 2]` → `[0, 1, 0, 2, 1]`
- `test_group_id_int32_all_same` — `[5, 5, 5]` → `[0, 0, 0]`
- `test_group_id_int32_all_unique` — `[1, 2, 3]` → `[0, 1, 2]`
- `test_group_id_int32_empty` — `[]` → `[]`
- `test_group_id_int32_with_nulls` — `[1, null, 2, null, 1]` → nulls get own group
- `test_group_id_uint8_directmap` — exercises DirectMapGrouper path
- `test_group_id_bool_directmap` — `[true, false, true, null]`
- `test_group_id_string_basic` — `["foo", "bar", "foo"]` → `[0, 1, 0]`
- `test_group_id_array_dispatch` — type-erased dispatch
- `test_group_id_large` — 200+ elements for hash table growth
- `test_group_id_multikey_packed` — two int32 columns
- `test_group_id_multikey_mixed` — int32 + string columns
- `test_unique_int32` — encounter-order uniqueness
- `test_unique_string` — string uniqueness

## Files to Create/Modify

| Step | File | Action |
|---|---|---|
| 1 | `marrow/kernels/groupby.mojo` | Create — core groupers + public API |
| 1 | `marrow/kernels/tests/test_groupby.mojo` | Create — tests |
| 1 | `marrow/kernels/__init__.mojo` | Update docstring |
| 1 | `CHANGELOG.md` | Add feature entry |
| 4 | `marrow/kernels/row.mojo` | Create — row encoding |

## Verification

```bash
mojo test marrow/kernels/tests/test_groupby.mojo -I .
pixi run test
```

# Duplication and placement audit

Repeated code, misplaced code, single-use helpers and over-abstraction across the
whole library. Verified against the working tree at **`5435f59` + uncommitted
changes (2026-08-16)**, with `mojo precompile marrow` clean at the time of
writing. **Nothing here has been changed** — this is a findings list, not a plan.

Method: a clone detector (maximal repeated blocks, normalized lines, overlapping
windows merged) over all 134 `.mojo` files — 43,272 source lines and 32,526 test
lines — plus `python/bindings/` and `python/marrow/`; then every hit read in
place and confirmed.

---

# 1. Repeated code patterns worth factoring out

Ranked by redundant-line count and by risk of the copies drifting apart.

### 1.1 `expr/values.mojo` — validity/state delegation, ~200 lines across 37 nodes

The same three-to-four method bodies are re-typed on every node family:

| Shape | Sites | Lines each |
|---|---|---|
| Unary passthrough (`validity` + `state_validity` + `referenced_columns` + `state` → `self.a.*`) | 8 — [800](../marrow/expr/values.mojo#L800), [837](../marrow/expr/values.mojo#L837), [927](../marrow/expr/values.mojo#L927), [1196](../marrow/expr/values.mojo#L1196), [1280](../marrow/expr/values.mojo#L1280), [1354](../marrow/expr/values.mojo#L1354), [1390](../marrow/expr/values.mojo#L1390), [1740](../marrow/expr/values.mojo#L1740) | ~10 |
| Binary intersect (`Bitmap.intersect(l, r)` + `Pair` state + `_union_columns`) | 4 — [756](../marrow/expr/values.mojo#L756), [884](../marrow/expr/values.mojo#L884), [1074](../marrow/expr/values.mojo#L1074), [1695](../marrow/expr/values.mojo#L1695) | ~16 |
| Breaker (`_result(batch)` → `owned_validity`) | 2 — [2290](../marrow/expr/values.mojo#L2290), [2355](../marrow/expr/values.mojo#L2355) | ~21 |

Including the multi-line docstring, byte-for-byte. `NumericUnary` and
`NumericCast` at [729-776](../marrow/expr/values.mojo#L729-L776) /
[858-904](../marrow/expr/values.mojo#L858-L904) share a 38-line identical run.

⚠️ **This is the one place where the obvious fix is measured-wrong.**
`docs/backlog.md` §0 records that folding twelve promote-then-dispatch sites into
one `_arith[K]` helper cost **+115,600 bytes**. Any consolidation here must go
through `pixi run binary_size`, and `query_streaming_agg_fused` is already at
+0.449% of a 0.5% ceiling.

### 1.2 The four `*Column` structs are one struct written four times

[`NumericColumn`](../marrow/expr/values.mojo#L634),
[`StringColumn`](../marrow/expr/values.mojo#L1592),
[`TemporalColumn`](../marrow/expr/values.mojo#L2436),
[`ListColumn`](../marrow/expr/values.mojo#L2549) — same `_name: String` field,
same `__init__`, `referenced_columns`, `materialize`, `validity`, `name`.

**This one has already cost a bug and will again.** `bound_column`/`prune` are
overridden on Numeric ([647](../marrow/expr/values.mojo#L647)) and String
([1604](../marrow/expr/values.mojo#L1604)) only. `StringColumn.prune`'s own
docstring says these "were on `NumericColumn` alone, so a string column could not
be a join key and a string predicate pruned nothing". **`TemporalColumn` and
`ListColumn` still have neither** — they inherit the conservative `-1` /
`Interval.unknown()` defaults at [439](../marrow/expr/values.mojo#L439)/[450](../marrow/expr/values.mojo#L450).
A date predicate prunes nothing in the fused lane today. Sound, but silently
no-op.

### 1.3 Operator↔interval-kernel pairing is hand-maintained in 20 places, twice

The AOT lane pairs each kernel with its interval kernel by hand —
[values.mojo:1094-1099](../marrow/expr/values.mojo#L1094-L1099),
[1209-1211](../marrow/expr/values.mojo#L1209-L1211),
[1936-1953](../marrow/expr/values.mojo#L1936-L1953) — while the runtime lane
re-encodes the *same table* as a name-keyed ladder at
[dynamic.mojo:610-626](../marrow/expr/dynamic.mojo#L610-L626). Two independent
copies of one mapping; a mismatched pair yields silently wrong pruning, not an
error.

An associated `comptime Interval: IntervalKernel` on
`NumericCompareKernel`/`BoolBinaryKernel`/`StringPredicateKernel` would let both
lanes derive it. Cheap to state, needs a size measurement.

Relatedly, the nine `*Interval` structs at
[interval.mojo:171-230](../marrow/kernels/interval.mojo#L171-L230) are 5-line
wrappers each forwarding to `Interval.maybe_*`.

### 1.4 `arrays.mojo` — `slice()` and `__eq__`

**Superseded by `docs/duplication-audit-1.4.md`.** Working it through established
that the fix filed here — adding `unsafe_get` to the `Array` trait so a generic
helper could express the element loop — does not work, and neither does any
variant of it. Read that document, not this entry.

`slice()` is still written 7 times ([422](../marrow/arrays.mojo#L422),
[504](../marrow/arrays.mojo#L504), [745](../marrow/arrays.mojo#L745),
[996](../marrow/arrays.mojo#L996), [1247](../marrow/arrays.mojo#L1247),
[1542](../marrow/arrays.mojo#L1542), [1715](../marrow/arrays.mojo#L1715),
[1969](../marrow/arrays.mojo#L1969)) and `__eq__` 5 times
([862](../marrow/arrays.mojo#L862), [1073](../marrow/arrays.mojo#L1073),
[1322](../marrow/arrays.mojo#L1322), [1596](../marrow/arrays.mojo#L1596),
[1780](../marrow/arrays.mojo#L1780)), but the bodies are now ~10 lines rather
than 13 — the three-line comment repeated above every `_sliced_null_count` call
was most of the apparent duplication measured here, and has been deleted.

What remains is structurally unfactorable (see §1.11) plus one real change: the
`_validity_equal` rewrite and two `DictionaryArray` bugs, both detailed in the
superseding document.

### 1.5 The GPU-or-host allocation preamble, 10 sites

```mojo
var buf: Buffer[mut=True]
comptime if GPU_ENABLED:
    if ctx.is_gpu(): buf = Buffer.alloc_device[native](ctx.device.value(), length)
    else:            buf = Buffer.alloc_zeroed[native](length)
else:                buf = Buffer.alloc_zeroed[native](length)
```

[numeric.mojo:99](../marrow/kernels/numeric.mojo#L99),
[178](../marrow/kernels/numeric.mojo#L178),
[500](../marrow/kernels/numeric.mojo#L500);
[cast.mojo:127](../marrow/kernels/cast.mojo#L127),
[208](../marrow/kernels/cast.mojo#L208),
[258](../marrow/kernels/cast.mojo#L258);
[hashing.mojo:383](../marrow/kernels/hashing.mojo#L383),
[446](../marrow/kernels/hashing.mojo#L446),
[545](../marrow/kernels/hashing.mojo#L545);
[boolean.mojo:360](../marrow/kernels/boolean.mojo#L360). A
`Buffer.alloc_for[T](ctx, n)` / `Bitmap.alloc_for(ctx, n)` in
[buffers.mojo](../marrow/buffers.mojo) collapses all ten and is the natural home
for the `GPU_ENABLED` gate.

### 1.6 `kernels/filter.mojo` — the set-bit iteration loop, 5 sites

The word-at-a-time CTZ loop is hand-written at
[252](../marrow/kernels/filter.mojo#L252),
[274](../marrow/kernels/filter.mojo#L274),
[364](../marrow/kernels/filter.mojo#L364),
[425](../marrow/kernels/filter.mojo#L425),
[480](../marrow/kernels/filter.mojo#L480) — including the `rem < 64` tail mask,
which is exactly the kind of detail that gets fixed in one copy. Two of them
differ *only* by an `if src_bm.test(idx)` inside the body. The validity-filter
preamble repeats 4× more at [379](../marrow/kernels/filter.mojo#L379)/[444](../marrow/kernels/filter.mojo#L444)/[493](../marrow/kernels/filter.mojo#L493)/[549](../marrow/kernels/filter.mojo#L549).

An `@always_inline` unified-closure `_for_each_set_bit` is expressible today, but
this is the hottest loop in the library — needs `bench_filter` before/after with
the drift-normalization rule from §0.

### 1.7 `builders.mojo` — validity extend, 5 sites

The 11-line "reserve, then propagate nulls or set-range" block at
[696](../marrow/builders.mojo#L696), [827](../marrow/builders.mojo#L827),
[1022](../marrow/builders.mojo#L1022), [1229](../marrow/builders.mojo#L1229),
[1370](../marrow/builders.mojo#L1370). Pure bitmap bookkeeping, no type
dependency — a `_extend_validity(arr, n)` helper on the builders.

### 1.8 `c_data.mojo` — the release handshake, 3 sites

`is_released` / `mark_released` / `__deinit__` are verbatim (docstrings included)
on [`CArrowSchema`:304-323](../marrow/c_data.mojo#L304-L323),
[`CArrowArray`:948-967](../marrow/c_data.mojo#L948-L967), and partially on
[`CArrowArrayStream`:1563](../marrow/c_data.mojo#L1563). This is the double-free
guard — the one place where three copies drifting is a memory-safety bug.

### 1.9 `marrow/testing/` — 40 lines duplicated verbatim between two files

[`CLIFlags`](../marrow/testing/test.mojo#L25) and
[`_print_json_array`](../marrow/testing/test.mojo#L46) are byte-identical in
[bench.mojo:34](../marrow/testing/bench.mojo#L34)/[55](../marrow/testing/bench.mojo#L55).
Zero risk to extract into `marrow/testing/_cli.mojo` — no generics, no size-gate
exposure. **Lowest-cost item on this list.**

### 1.10 `parquet/format.mojo` — the Thrift field loop, 12 sites

`@staticmethod def read[o](mut r) -> Self` + `var f = FieldHeader()` + `while
r.next_field(f)` + `else: r.skip(f.type)` at
[432](../marrow/parquet/format.mojo#L432), [564](../marrow/parquet/format.mojo#L564),
[614](../marrow/parquet/format.mojo#L614), [665](../marrow/parquet/format.mojo#L665),
[707](../marrow/parquet/format.mojo#L707), [1028](../marrow/parquet/format.mojo#L1028),
[1065](../marrow/parquet/format.mojo#L1065), [1104](../marrow/parquet/format.mojo#L1104),
[1237](../marrow/parquet/format.mojo#L1237), [1275](../marrow/parquet/format.mojo#L1275),
[1315](../marrow/parquet/format.mojo#L1315), plus
[bloom.mojo:260](../marrow/parquet/bloom.mojo#L260).

I'd rate this **leave alone** — it's a wire-format decoder where per-field
explicitness is the point, and Arrow C++/arrow-rs generate the equivalent. Worth
noting instead is the *inconsistency*: `read` is not a trait requirement while
`write` is ([`ThriftWritable`](../marrow/parquet/format.mojo#L286)), and 5 of the
12 structs don't conform to `ThriftWritable` at all.

### 1.11 The four `_dispatch` narrowing adapters

[`DynType`](../marrow/dtypes.mojo#L834), [`DynArray`](../marrow/arrays.mojo#L2412),
[`DynScalar`](../marrow/scalars.mojo#L614),
[`DynBuilder`](../marrow/builders.mojo#L318) each carry an identical
raising/non-raising pair, differing only by the trait name. Per §0 and CLAUDE.md
this is **structurally forced** — a closure type cannot be generic over its own
trait bound. Listing it so it isn't rediscovered as an opportunity; the nine
`dispatch_*` family adapters below it are the same story.

### 1.12 Tests (32.5k lines, no shared fixtures)

- The `List[Int]` + 6× `.append` + `_batch(...)` join-fixture block appears **9
  times** across [expr/tests/test_join.mojo](../marrow/expr/tests/test_join.mojo#L54)
  and [kernels/tests/test_join.mojo](../marrow/kernels/tests/test_join.mojo#L211).
- `_batch` defined 4×, `_to_marrow` 3×, `_make_struct`/`_int32_struct`/`_keys`
  variants 5× — each private to its file.
- The MapBuilder setup block is copied into
  [test_cast.mojo:542](../marrow/kernels/tests/test_cast.mojo#L542),
  [test_nested.mojo:80](../marrow/kernels/tests/test_nested.mojo#L80),
  [test_ipc.mojo:845](../marrow/tests/test_ipc.mojo#L845).
- **Hardcoded `/tmp/` paths, ~15 sites, suite-wide.**
  [test_codecs.mojo](../marrow/parquet/tests/test_codecs.mojo#L96),
  [test_bloom.mojo](../marrow/parquet/tests/test_bloom.mojo#L150),
  [bench_parquet.mojo](../marrow/parquet/tests/bench_parquet.mojo#L46) and
  [test_parquet.mojo](../marrow/parquet/tests/test_parquet.mojo#L29) all write
  fixed paths like `/tmp/marrow_codec_snappy.parquet`, while
  [test_ipc.mojo:52](../marrow/tests/test_ipc.mojo#L52) uses `tempfile.mkstemp`.
  Two concurrent `pytest` invocations — which the harness explicitly supports —
  collide on the fixed ones. This is the *prevailing* convention in
  `parquet/tests/`, so it has to be fixed suite-wide or not at all.

`marrow/testing/` holds only the runner. A `marrow/testing/fixtures.mojo` is the
missing piece — and per the "one selection = one compilation unit" note, shared
fixtures cost nothing extra to compile.

### 1.13 Python bindings

- The builder `extend` loop repeats 3× in
  [bindings/arrays.mojo:647](../python/bindings/arrays.mojo#L647)/[692](../python/bindings/arrays.mojo#L692)/[862](../python/bindings/arrays.mojo#L862).
- The `pymethod` arity overloads in
  [helpers.mojo](../python/bindings/helpers.mojo#L108) repeat their wrapper body
  per arity — probably irreducible.

---

# 2. Code that belongs somewhere else

1. **`Crc32`** ([utils.mojo:202](../marrow/utils.mojo#L202)) is used *only* by
   `parquet/{reader,writer,bloom}.mojo`. It's a Parquet page-checksum concern
   sitting in a core module whose docstring says it is about "Generic Variant
   dispatch utilities".
2. **`LittleEndian`** ([utils.mojo:82](../marrow/utils.mojo#L82)) is used by
   `ipc.mojo` and 6 parquet files — a genuine shared byte-order utility, but it
   doesn't belong in a file documented as variant dispatch. `utils.mojo` is
   currently four unrelated things: variant dispatch, byte order, CRC32, and GPU
   capability.
3. **`GPU_ENABLED` / `has_accelerator_support`**
   ([utils.mojo:67](../marrow/utils.mojo#L67), [235](../marrow/utils.mojo#L235))
   are device-policy, and [`marrow/execution.mojo`](../marrow/execution.mojo)
   already owns device policy (`ExecContext`).
4. **`Grouping`** ([kernels/core.mojo:47](../marrow/kernels/core.mojo#L47)) is a
   data type living in a file whose stated job is "the root of the kernel trait
   hierarchy". Used by `groupby`, `aggregate`, `distinct`, `expr/aggregates` —
   it belongs in `groupby.mojo` or its own module.
5. **`marrow/kernels/tests/test_execution.mojo` + `test_execution_gpu.mojo`**
   test `marrow/execution.mojo` (core) — they import `...execution` from inside
   `kernels/tests/`. Left behind by the `ExecContext`-out-of-kernels refactor;
   they belong in `marrow/tests/`.
6. **Two files named `execution.mojo`** with unrelated responsibilities:
   [`marrow/execution.mojo`](../marrow/execution.mojo) (`ExecContext` —
   threads/device) and [`marrow/expr/execution.mojo`](../marrow/expr/execution.mojo)
   (`Processor` — the pull engine). And two `struct Filter` —
   [kernels/filter.mojo:49](../marrow/kernels/filter.mojo#L49) and
   [expr/relations.mojo:867](../marrow/expr/relations.mojo#L867).

---

# 3. Single-use and unreferenced functions

**Zero call sites in Mojo** — all C-ABI callbacks, referenced only as function
pointers, so this is expected, not dead code: `_release_*` ×6, `_stream_*` ×4 in
[c_data.mojo](../marrow/c_data.mojo); `_rapidhash_bool_masked`
([hashing.mojo:206](../marrow/kernels/hashing.mojo#L206)) is the one that
deserves a look — it has no pointer-taking site either.

**Exactly one call site** (20 functions). Most are deliberate named steps and
should stay. The ones where the split earns nothing:

| Function | Defined | Used |
|---|---|---|
| `_fold_kind` / `_ascii_lower` / `_match_tokens` | [string.mojo:523](../marrow/kernels/string.mojo#L523)/[536](../marrow/kernels/string.mojo#L536)/[546](../marrow/kernels/string.mojo#L546) | 681/685/705 — three helpers, one caller each, ~150 lines apart |
| `_rapid_mix_wide`, `_rapidhash_bool`, `_indices_as_int32`, `_rapidhash_primitive_masked` | [hashing.mojo](../marrow/kernels/hashing.mojo#L60) | 4 of the file's helpers are single-use |
| `_format_ns` | [bench.mojo:457](../marrow/testing/bench.mojo#L457) | 344 — defined *after* its only caller |
| `read_metadata` | [parquet/reader.mojo:2384](../marrow/parquet/reader.mojo#L2384) | **only from a test** ([test_metadata.mojo:39](../marrow/parquet/tests/test_metadata.mojo#L39)) — public API with no production caller |

---

# 4. Over-abstraction

- **`trait Join`** ([join.mojo:384](../marrow/kernels/join.mojo#L384)) — 3
  abstract methods, **one conformer** (`HashJoin`), one commented-out
  (`SortMergeJoin`, [:829](../marrow/kernels/join.mojo#L829)). Its own docstring
  says "operators use concrete types directly". A trait carrying no dispatch and
  no second implementation.
- **`trait ByteSource`** ([parquet/source.mojo:20](../marrow/parquet/source.mojo#L20))
  — one conformer (`MappedFile`).
- **`trait WindowKernel`** ([values.mojo:2199](../marrow/expr/values.mojo#L2199))
  — one conformer (`RowNumberKernel`); windows are AOT-only, which §0 already
  flags as violating the two-lane invariant (M2.3).
- **`trait ListValue`** ([values.mojo:2539](../marrow/expr/values.mojo#L2539)) —
  one conformer; `TemporalValue` has two, one of which is the column.
- **The `AggFunction` split** — trait in
  [kernels/aggregate.mojo:854](../marrow/kernels/aggregate.mojo#L854), all four
  conformers in [expr/aggregates.mojo](../marrow/expr/aggregates.mojo#L86).
  Defensible (kernels owns the protocol, expr populates it), but it means the
  aggregate vocabulary is documented in two files and the section banner
  explaining it is stranded at the other end of a 1,193-line file.

---

# 5. What I deliberately did *not* re-file

`docs/backlog.md` §8 already records, with more measurement behind it than I
have: the `values.mojo` split (**L2**, attempted and reverted — the blocker is
the shared `col`/`lit` name, not the file), the three `marrow.expr` import cycles
(**Q-NEW**, with the exact one-move fix), `ipc.mojo` → package (**Q4.4**), the
11-array `ArrayData`-codec duplication, `RecordBatch`/`Table` as
container-plus-query-surface, `SwissHashTable`'s four responsibilities,
`SortIndices`' five, and `Value` as the union of four consumers' protocols.

---

## Suggested order, if you want one

**Free / no size exposure:** 1.9 (testing CLI), §2 items 1–8 (moves and stale
banners), 1.12–1.13 (test + binding fixtures).

**Needs a benchmark:** 1.5 (`alloc_for`), 1.6 (`_for_each_set_bit`), 1.7 (builder
validity).

**Needs `pixi run binary_size` first, and may come back "no":** 1.1, 1.2, 1.3 —
all inside the size-gated fused lane.

**Correctness-adjacent, independent of any refactor:** 1.2's missing
`prune`/`bound_column` on `TemporalColumn`/`ListColumn`, and 1.8's three copies
of the double-free guard.

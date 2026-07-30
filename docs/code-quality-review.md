# Code quality review — leaky abstractions, encapsulation, and pattern inventory

**Date:** 2026-07-24 · **Base commit:** `80ebc10` (branch `complete`) · **Status:** discovery only — nothing fixed yet.

Scope: six parallel read-only reviews covering the whole non-test tree (~39k lines) —
core types, memory/views, elementwise kernels, relational kernels, expr engine +
Python bindings, parquet + IPC. This document records **what to fix and why**, and
distils the **patterns worth keeping vs avoiding**. It is a planning input, not a
task list; see §5 for sequencing.

Findings are grouped by **root cause**, not by file, because ~70 individual findings
collapse into nine systemic causes. Fixing a cause removes a class of future bugs;
fixing findings one by one does not.

---

## 1. Verified defects

These five were confirmed by reading the code and, where marked, by running a probe.
They are not style opinions. Each is a symptom of a root cause in §2.

| # | Defect | Evidence | Cause |
|---|--------|----------|-------|
| D1 | **`slice()` reports the parent's null count.** Every array type copies `nulls=self.nulls` into the slice. | **Probe-confirmed**: slicing `[1,null,null,4,5]` → `[4,5]` gives `null_count() == 2` with every element valid. `arrays.mojo:267,493,720,966,1257,1427,1676,1901` | RC3 |
| D2 | **Indexing a sliced `BoolArray` aborts.** `values()` returns an offset-applied view; `__getitem__` adds the offset again. | **Probe-confirmed**: `arr.slice(2,3)[0]` aborts on the bounds assert. Silently reads the wrong bit in release builds. `arrays.mojo:306` + `views.mojo:655` | RC3 |
| D3 | **The expr layer binds the null-incorrect `Any`/`All`.** Two same-named kernel pairs exist; `kernels/__init__.mojo:48` re-exports the correct one, `expr/values.mojo:138` imports the broken one. | `boolean.mojo:288` (`count_set_bits() > 0`, no validity mask — its sibling `AllKernel` even says *"nulls ignored; full null handling is a follow-up"*) vs `aggregate.mojo:696` (masks against validity). Bound at `values.mojo:1007-1008`. | RC2 |
| D4 | **Fused mixed-width compare truncates.** `NumericCompare` takes the *left* operand's native type and casts the right operand down into it. | `values.mojo:833,849-851`. `col("a",int32) > col("b",int64)` narrows int64→int32. `NumericBinary` (`values.mojo:583`) correctly uses `promote[L,R]` — so `a+b` and `a>b` disagree. No test covers a mixed-width compare. | RC2 |
| D5 | **`GROUP BY`/`ORDER BY` on temporal columns raises.** Both dispatchers are hand-written ladders that omit whole type families. | `rapidhash` (`hashing.mojo:456-498`) and `sort_indices` (`sort.mojo:547-625`) cover numerics + string, then `raise`. Neither handles **temporal, large_string, decimal, dictionary**. | RC4 |

**D5 blocks M1.** `GROUP BY date_trunc(...)` (ClickBench Q19/35/36/40/43) produces a
temporal key that `HashGrouper.consume_keys` cannot hash, and sorting by a timestamp
(Q8/24–27) raises. The ClickBench `hits` table is keyed on `EventDate`/`EventTime`.
This is a roadmap gap, not just a quality issue — it is not currently tracked in
`execution-engine-tasks.md`. (FU-1 tracks only the `large_string` half.)

---

## 2. Root causes

### RC1 — Missing public accessors, so types reach into each other's `_private` fields

The single most pervasive issue; present in every subsystem.

- `PrimitiveArray.__init__` reads `data.dtype._v[Self.T]` (`arrays.mojo:447`) — and unlike
  its siblings performs **no** variant check, so a mismatched `ArrayData` aborts instead of raising.
- `dispatch_over_*` read `DynType._v` from another module (`utils.mojo:287-316`).
- `Buffer.is_cpu/is_device/is_host/resize/to_cpu` probe `Allocation._host`/`._device`
  (`buffers.mojo:563-573,600,732`). **Consequence:** `Buffer.resize` branches on `_host` only,
  so a DEVICE buffer falls through to a CPU `alloc_zeroed` + memcpy from a device pointer.
- `Bitmap` pokes `self._buffer._ptr` throughout (`buffers.mojo:950-1165`).
- A free function assigns `b._null_count = size` (`builders.mojo:1827`).
- `Context.get[A]` does `self._slots[i][DynArray]._v[A]` (`values.mojo:208`), bypassing the
  `debug_assert` in `DynArray._as[T]`.

Each is a missing accessor, and each disables the owning type's ability to enforce its
own invariants. **Rule: add the accessor; never reach through.**

### RC2 — One concept with two homes, selected by import path

- `AnyKernel`/`AllKernel`/`BoolReduceKernel` in both `boolean.mojo` and `aggregate.mojo`,
  **with different null semantics** → **D3**.
- `MinKernel`/`MaxKernel` mean *element-wise* in `arithmetic.mojo:262,271` and *reducing* in
  `aggregate.mojo:306,345`; `kernels/__init__.mojo` can only re-export one, so the
  element-wise pair is unreachable through the package namespace.
- String comparison has **three** implementations: `compare.mojo`'s `apply_string`/`str_predicate`
  (a line-for-line clone of `string.mojo`'s `StringPredicateKernel.apply`), the `StringEq/Ne`
  structs, and the legacy `equal(StringArray)` free function. *(Tracked as FU-3.)*
- Legacy free functions shadow the kernel structs that replaced them — and **downstream latched
  onto the old, narrower ones**: `dynamic.mojo:381` calls the numeric-only `is_null` (so
  `col.is_null()` fails on a string column while `not_null()` succeeds), and `:389` calls
  `select`, which silently drops validity, instead of `conditional._multiplex`.
- `Bitmap` re-implements `BitmapView`'s entire bit API against the raw pointer, and has already
  drifted: `Bitmap.__eq__` is a bit-at-a-time loop where `BitmapView.__eq__` does 64-bit words
  (~64× slower on the owner type than its own view).
- Two `sort_indices` in the Python layer with **opposite** null defaults
  (`__init__.py:568` `"at_start"` vs `compute.py:225` `"at_end"`; PyArrow's is `"at_end"`).
- `Bitmap.test(raw_index)` and `BitmapView.test(logical_index)` share a name but use
  **different index bases** → **D2**.

**Rule: one concept, one owner, one name.** A duplicate that "looks equivalent" will drift,
and the drift will be invisible because both compile.

### RC3 — Derived state cached in mutable public fields, and offsets applied at call sites

- `nulls` survives `slice()` unchanged → **D1**. It is read by kernels (`cast.mojo:155,221,…`)
  and gates three builder fast paths (`builders.mojo:606,735,1657`).
- `ChunkedArray.length` is cached beside a public `chunks` list, so `ca.chunks.append(...)`
  desynchronises it.
- Offset arithmetic at call sites → **D2**; the same shape recurs in three builders
  (`builders.mojo:748`).
- The incantation `nulls = length - bm.count_set_bits() if bm else 0` is inlined into
  constructor argument lists **eight times** (`arithmetic.mojo:76,162`, `compare.mojo:89,159,332`,
  `string.mojo:294`, `boolean.mojo:213,335`).
- `bitmap_and` takes the raw `.bitmap` field, not the offset-adjusted `.validity()`
  (`helpers.mojo:16`), while the data is read through offset-applied accessors — so a sliced
  nullable input yields validity shifted by `offset` bits relative to its data.

**Rule: derived values are computed or lazily cached behind a method; a type that already
carries an offset is the only thing allowed to know about it.**

### RC4 — Hand-written dtype ladders where a dispatcher exists

~20 copies of the Arrow type ladder across the tree: 5 in core
(`arrays.mojo:2392`, `builders.mojo:174`, `scalars.mojo:576`, two dictionary-index cascades that
are literal duplicates), 7 in elementwise kernels, 2 in relational (`rapidhash`, `sort_indices`),
6 full + 2 partial in parquet (`reader.mojo:1582`, `writer.mojo:93,211`, `statistics.mojo:180,290`,
`schema.mojo:578,907`).

The cost is not line count — it is **silent coverage gaps**, because a ladder that forgets a
type compiles fine and raises at runtime: **D5**; `DynScalar.repeat` covers 11 numeric types and
raises for bool/string/temporal/decimal, invisible from its signature; parquet's `_dispatch`
handles INT96 that no sibling ladder knows about, while `min_max` silently returns `False` for
`binary`/`large_binary`.

`Filter.dispatch`/`Take.dispatch` (`filter.mojo:121,632`) and `cast.mojo:436,603,667` show the
closed form. Missing from the family: `dispatch_over_integer`, `dispatch_over_primitive`,
`dispatch_over_listlike`, `dispatch_over_temporal` — their absence is *why* the cascades were
hand-written.

**Rule: more than two `elif dtype ==` arms whose bodies differ only by type ⇒ use or add a dispatcher.**

### RC5 — Escape hatches baked into abstraction signatures

When the safe API has a gap, an escape hatch gets added instead of closing the gap — and then
the hatch becomes the path of least resistance.

- **`ByteSource.read_at` returns `Span[UInt8, ImmUntrackedOrigin]`** (`parquet/source.mojo:36`).
  The lifetime opt-out is in the *signature*, so it propagates: `MappedFile` stores an
  `UnsafePointer` (`:56`), `reader.mojo:157` has a `_untracked()` helper that `rebind`s any span
  into the untracked origin, and page/decoder types store such spans as fields. **This contract is
  satisfiable only by a whole-file mmap** — a streaming source must own recycled buffers, and the
  untracked origin removes the compiler's ability to catch the dangle. Directly blocks T2.4.
- **`ParquetFile._span()`** (`reader.mojo:1732`) reads the whole file and is the *only* way any
  decode path gets bytes — so the `ByteSource` seam currently carries zero load.
- **`views.as_span()` / `BufferView.unsafe_ptr()`** exist because `Span` subscripting is ~3× slower
  on the decode path (documented at `codecs.mojo:174-177`); the result is **20 `unsafe_ptr()` sites
  outside the three sanctioned modules** (`parquet/utils.mojo` ×10, `reader.mojo` ×6, `codecs.mojo` ×4).
  Several exist only because decode functions are *typed* on `Span`/`UnsafePointer`
  (`reader.mojo:500`) so no view can reach them.
- **`ArrayData`**, an explicitly-scoped interop DTO (`arrays.mojo:119`), is now used by six kernels
  that index `data.buffers[0]` and do manual offset arithmetic — exactly what `.values()`/`.validity()`
  exist to prevent.

**Rule: if a safe API is too slow or too narrow, fix the API. An abstraction with a documented
bypass is not in force.** (Convention hygiene is otherwise excellent: 0 uses of `fn`, 0 real uses
of `alias` tree-wide.)

### RC6 — Policy duplicated at call sites instead of owned by the type that has the information

- **Allocation kind**: `if ctx.is_gpu(): alloc_device else alloc_uninit` appears at **10 kernel
  sites**, each also unwrapping `ctx.device.value()`.
- **Parallel striping**: the `chunk = (n+nt-1)//nt; start = t*chunk; …` loop is hand-written **7 times**
  (`filter.mojo:724`, `sort.mojo:279,322`, `partition.mojo:46,205,296`, `groupby.mojo:437,885`) plus
  once in `views.mojo` — despite `ExecutionContext`'s docstring saying its purpose is to remove exactly this.
- **Gating** is spelled three incompatible ways (`ctx.wants_parallel(...)`, `if nt <= 1 or n < T`,
  `if n < T: nt = 1`) against **7 unrelated thresholds**. Net effect: `ExecutionContext.serial()`
  cannot force `RadixPartitioner` serial.
- **`ExecutionContext` is destructured to a bare `Int` at every layer boundary** and re-created
  (`HashJoin` stores `_num_threads`, then rebuilds `ExecutionContext.parallel(...)` at five sites,
  **silently dropping any GPU device**). `hash_join` takes both `ctx` and `num_threads`; **`ctx` is dead**.
- **Writer options**: 7 loose parameters repeated across 3 layers — already causing `row_group_size`
  to be unreachable from `write_table`.

**Rule: the type that owns the information owns the decision.**

### RC7 — Missing value types for data that always travels together

- **`(gids: Int32Array, num_groups: Int)`** as two parameters across 8+ signatures
  (`distinct.mojo:167,214`, `groupby.mojo:812,834`, `aggregate.mojo:823,888`). Nothing checks
  `num_groups > max(gids)`; callers must read `num_groups` *after* `consume_keys`. `HashGrouper`
  owns both and returns one. → a `Grouping` type.
- **Three lockstep-indexed `List`s** in `HashJoin` (`join.mojo:355-361`) that must share length and
  ordering, with the tables smuggled past `map_partitions` by out-of-band mutation. → one
  `BuildPartition` struct.
- **`JOIN_*` as bare `UInt8`** in one flat namespace (kind, strictness and algorithm are mutually
  assignable), with "does this kind emit right columns" re-derived **three times with different
  membership** (`join.mojo:645,681,582`); `JOIN_CROSS`/`MARK`/`SINGLE` are declared but unhandled,
  and `CROSS` silently falls into the LEFT/RIGHT/FULL branch. → a `JoinKind` struct with predicates.
- **`SwissHashTable` has two mutually exclusive lifecycles in one struct** — the CSR index is empty
  until `build_hashes`, and `probe` requires the caller to hand the build keys back in. → split the
  probe-side state into a `JoinIndex` that owns it.
- **`TagValue._name` is triple-overloaded** (column name / LIKE pattern / `date_trunc` unit), and
  `name()` returns it with no tag check — so a LIKE node reports `"%foo%"` as its output column name.
  `_value_set` and `_cast_to` likewise sit on all 42 tags. *(Extends FU-7.)*

**Rule: if two values always travel together and one constrains the other, they are one type.**

### RC8 — Whole-file / whole-result buffering

- Parquet `read()` decodes **every** selected row group before assembling any
  (`reader.mojo:1798-1866`), so peak memory is the whole decoded table regardless of source.
  No `iter_batches`.
- Parquet `FileWriter` accumulates the entire output in a `List[UInt8]` (`writer.mojo:827`) — no sink seam.
- IPC reads the whole file into an owned `List[UInt8]` and copies again per message; `write_array`
  copies buffers byte-at-a-time and re-packs bitmaps bit-by-bit (`ipc.mojo:1671-1685,1930`).

Directly blocks T2.4. Note the read-side *projection* work is essentially done already —
`SchemaMapping.project` and `Projection.decode_order` are real; only the bytes are over-fetched.

### RC9 — The F1/F2 wiring tax (expr-specific)

Op *semantics* live once, in the kernel structs — that factoring is right. The *wiring* is
duplicated, and the two sides paid it differently: **F2 solved it** (15 parameterized shells; a new
op is one `comptime X = Shell[XKernel, _]` line), **F1 did not** (every op is hand-written arms in
**three** parallel `elif` chains over the same 42 tags: `eval`, `_op_name`, `prune`).

Result: **F2 has ~30 operators F1 lacks** (`sqrt`, `exp`, `ln`, `pow`, all six reductions, all four
string ordering compares, `startswith`/`endswith`/`contains`, `row_number`, …). `test_parity.mojo`
covers only the intersection and nothing enforces coverage parity.

**Verdict: ~80% is eliminable and none of the eliminable part is load-bearing for DCE.** The
small-binary property comes from *which trampoline `DynValue.__init__` instantiates*, not from the
shape of the interpreter body. What *is* irreducible is having two **executors** — `vectorwise[W]`
cannot be reached through a runtime tag. A comptime op registry in `marrow/expr/ops.mojo` would
drive F1's tags/names/uniform arms and F2's aliases from one entry (~30 of 42 tags are uniform
unary/binary dispatch), and is the natural place to assert parity. **Must be gated on
`pixi run binary_size`.**

### Also: documentation drift on load-bearing docs

`CLAUDE.md` still lists `marrow/bitmap.mojo`, `marrow/visitor.mojo`, `BuilderData`,
`ArcPointer[BuilderData]`, `ArrayVisitor`/`DataTypeVisitor` and `BitmapBuilder` — **none of which
exist**. `buffers.mojo`'s own module docstring points readers to "`marrow.bitmap`" for types defined
700 lines below in itself. `benchmarks/binary_size/README.md` documents four binaries against module
names that no longer exist and quotes a ratio (30.9×) contradicting CLAUDE.md's (~12×) — i.e. **the
documentation of the project's central architectural gate does not describe the gate.**

Three docstrings assert capabilities the code lacks: `pruning.mojo:14` claims fused nodes implement
`prune` (they do not — `_prune_tramp` unconditionally returns `unknown()`, so the AOT frontend
cannot prune row groups at all), `values.mojo:515` says a schema lookup is "per pass" when it is per
SIMD chunk, and `dynamic.mojo:5` describes bindings that do not exist.

---

## 3. Patterns to codify

Distilled across all six reviews — these are what marrow does well and should keep doing.

1. **Type erasure via `Variant` + `variant_dispatch`** — never fn-pointer trampolines or `rebind`.
   `DynArray`/`DynScalar`/`DynBuilder`/`DynType` share one shape: `comptime VariantType`, one
   `_v`, `@implicit __init__[T: Trait]`, private `_as[T]()` + per-type `as_x()` borrows. Keeps
   dispatch closed and DCE-friendly. *(The `DynValue` trampolines are the deliberate exception —
   see §4.)*
2. **Physically-identical Arrow types are one parameterized struct + aliases** —
   `BinaryLikeArray[T]`, `ListLikeArray[T]`, `_IntegerType[DType]`. Retag rather than rebuild:
   `to_map()`/`to_list()` is the model.
3. **Three-tier kernels: `core[W]` → `apply` (typed) → `dispatch` (erased), as a struct.**
   A concrete op struct should be ~6 lines containing only `comptime name` and `core`
   (`arithmetic.mojo:207-278`). **Always expose `core[W]` even if `apply` doesn't use it** — without
   it the kernel is invisible to `marrow.expr` and cannot fuse (`ArrayLengthKernel` lacks one and
   therefore cannot).
4. **Close every dtype cascade with a `dispatch_over_*` helper** (RC4).
5. **Correctness-relevant flags are `comptime` parameters branched with `comptime if`** —
   `cast.mojo:117` (`safe`), `boolean.mojo:351` (`negate`), `reader.mojo:1427` (`leveled`). One
   monomorphized path, dead branch eliminated; this is what preserves the small-binary property.
6. **Compute validity with bitmap set algebra, not element loops.** `_kleene`
   (`boolean.mojo:165-216`) expresses the whole Arrow Kleene rule as
   `av.difference(a_data) | bv.difference(b_data) | (av & bv)`.
7. **Build type-agnostic ops by composing kernels.** `_multiplex` (`conditional.mojo:66`) reduces
   `case_when`/`coalesce`/`nullif`/`fill_null` to "selector + `concat` + `take`", working on strings,
   lists and structs for free. Document the N×L materialization cost.
8. **One `mut: Bool`-parameterized struct, not a Type/TypeBuilder pair** — `Buffer`, `Bitmap`,
   `BufferView`, `BitmapView`, with write methods constrained by `self: Buffer[mut=True]`.
9. **Views borrow via `origin_of(self)`; owners never hand out pointers.**
10. **One `Allocation` with exactly one active release mechanism**, shared through a single
    `ArcPointer`. New memory provenance = a fifth `Allocation` factory, not a new `Buffer` field.
11. **Let the kernel own its output-type algebra via an associated type** —
    `Reduction`'s `Self.K.AccType[Self.A.OutType]`. (Applying this to `BinaryKernel`/`UnaryKernel`
    deletes `FloatBinary`/`FloatUnary` outright.)
12. **One universal verb + a `Datum` wire format** (`execute(batch, ctx) -> Datum`), mirroring
    Arrow C++ `Datum` / DataFusion `ColumnarValue`; lets scalars stay lazy.
13. **Breakers are ordinary fused leaves** — `prepare` into a slot, then splat/load, so the stage
    above still fuses. The cleanest idea in the expr layer; preserve it.
14. **One partition-parallel skeleton, per-call-site body** (`map_partitions`), and **lock-free
    result collection via pre-sized `Optional[R]` slots + move-out**.
15. **Bidirectional format mappings defined once, as data** — `_LeafTypeRow` "so they cannot drift".
16. **Every tuning constant carries its benchmark in its docstring** (`sort.mojo:61-89`). *A
    constant without a recorded measurement is a guess nobody will dare change.*
17. **PyArrow-shaped static factories** (`from_arrays`, incl. PyArrow's `mask` convention).
18. **Composition in the Python layer** — `_Wrapper` + `._binding`.

## 4. Anti-patterns to avoid

1. **Don't cache a derived quantity in a public field other operations invalidate** (RC3).
2. **Don't expose layout fields publicly on a type documented as immutable.** `arrays.mojo:386`'s
   own TODO says so; `arr.length = 99` currently compiles.
3. **Don't do offset arithmetic at call sites** — if a view carries the offset, only the type may know it.
4. **Don't reach into another type's `_`-prefixed members** (RC1).
5. **Don't keep a free function beside the kernel struct that replaced it** — in all three cases
   found, downstream latched onto the *old*, narrower, null-buggier version (RC2).
6. **Don't reuse a struct/trait name across sibling modules** — meaning then depends on the import
   line, and `__init__` can only re-export one (RC2, D3).
7. **Don't let an interop DTO become the working representation** (`ArrayData` in six kernels).
8. **Don't add an escape hatch to make a caller fast — fix the view API** (RC5).
9. **Don't type non-memory-module parameters on `UnsafePointer`/`Span`** — it forces callers to
   produce raw pointers.
10. **Don't copy-paste a state machine across N implementations.** The builder `finish()`/`reset()`
    sequence exists 7× and the one that drifted (`BoolBuilder`, `builders.mojo:1667`) leaves a stale
    `_capacity` after freeing its buffer — a heap overflow on reuse.
11. **Don't add trait methods whose default implementation always raises** (`Array.to_device`/`to_cpu`:
    8 of 10 types fail at runtime, turning a static capability question into a dynamic failure).
12. **Don't drop type identity when erasing** — `ListScalar` backs list/fixed-size-list/map but stores
    no dtype, so `type()` lies for two of three.
13. **Don't take `var self` for a read-only operation** (`combine_chunks` forces three defensive clones).
14. **Don't return `-1` sentinels from raising-capable APIs** (`get_field_index` → the same
    `if idx == -1: raise` block three times).
15. **Don't let constructors accept structurally invalid values** — `RecordBatch(schema, columns)`
    validates nothing while `num_rows()` assumes the invariant holds.
16. **Don't pass parallel `List`s indexed in lockstep**, or **a value and its size as separate
    parameters** (RC7).
17. **Don't use `UInt8` constants as an enum with predicates re-derived per call site** (RC7).
18. **Don't hand-roll `sync_parallelize` stripe loops** — chunking is the context's job (RC6).
19. **Don't accept a `ctx`/kwarg you ignore.** `hash_join`'s `ctx` is never referenced;
    `compute.sum(skip_nulls=False)` returns the `skip_nulls=True` answer. *Raise `NotImplementedError`
    rather than lie.*
20. **Don't keep traits/impls "for the future" with zero users** (`Partitioner`, `NoPartition`,
    `original_row`, commented-out `SortMergeJoin`) — add the abstraction with the second implementation.
21. **Don't use `Optional` as a sentinel the only consumer unconditionally unwraps.**
22. **Don't name a module `utils`.** `marrow/utils.mojo` holds five unrelated things under a docstring
    describing one; `kernels/helpers.mojo` is a 2-item junk drawer; `parquet/utils.mojo` is a single
    FFI struct.
23. **Don't put a schema/name lookup inside a `vectorwise`/`elementwise` body** — anything invariant
    across the batch belongs in `prepare`.
24. **Don't re-execute a subtree to answer a question about a result you already materialized** —
    change the signature so it can read the slot.
25. **Don't invent an ad-hoc trick for a missing primitive** — "copy a `BitmapView` into an owned
    `Bitmap`" has three different encodings; add the method once.
26. **Don't put string parsing / `isinstance` dispatch / defaulting in a Mojo binding** — CLAUDE.md
    mandates the reverse split, and the code currently inverts it.
27. **Don't let a docstring assert a capability the code lacks** — a false invariant in a comment is
    how the next bug gets written.
28. **Don't leave the fix as a TODO in the type's own docstring.** ~20 sit in the memory layer alone,
    several describing the correct design; each is a decision made but not executed.

## 5. Forced by Mojo — do not "fix"

Distinguishing these matters; they look like smells and are not.

- **`DynValue`'s hand-rolled fn-pointer trampolines** — Mojo has no dynamic dispatch; this closed
  erasure *is* the DCE mechanism. The fused/dyn trampoline duplication is the price of the two-box design.
- **`comptime NativeType` on `BoolValue`** and the "wider of the two" rule — the bit-pack driver must
  pick a SIMD width no operand load can overflow. (D4 is a bug precisely because `NumericCompare`
  *doesn't* follow this.)
- **`comptime IsBreaker` + `comptime if` in `Value.execute`** — a runtime `if` would link both paths.
- **`TagValue.__del__(deinit self): pass`** — required for `ImplicitlyDeletable` on a self-referential
  `List[TagValue]`.
- **Associated-type rules** — single projections off a direct trait bound reduce; chained ones
  (`Self.OutType.ArrayType`) do not. Keep companion types as direct associated members. See CLAUDE.md's
  "Associated-type & trait gotchas".
- **`_view_to_owned`'s `difference`-against-zero trick** — `union(v, v)` trips the exclusivity checker.
  Only the missing offset-0 fast path is self-inflicted.
- **The two executors** (`vectorwise[W]` vs tag interpreter) — genuinely irreducible. Only the *wiring*
  around them is removable (RC9).

---

## 6. Suggested sequencing

Nothing here is started. Proposed order, by (risk removed ÷ effort):

**Tier 0 — correctness, small, do first.** D1 (`slice` null count), D2 (`BoolArray` offset), D3
(re-point the `Any`/`All` import), D4 (`NumericCompare` → `promote`), plus `TagValue.name()`
returning `""` unless `_tag == LOAD`. All are small and independently testable.

**Tier 1 — unblocks M1/T2.4.** D5 (temporal/large_string in `rapidhash` + `sort_indices` — fold into
RC4 by adding the missing `dispatch_over_*` helpers); `ByteSource.read_at → Buffer[mut=False]`
**before** any other parquet streaming work, then delete `_span()` and make `PageReader`
chunk-relative (RC5/RC8).

**Tier 2 — removes whole classes of future bugs.** RC1 (add the missing accessors), RC2 (de-duplicate
the doubled concepts, starting with the kernel-name collisions), RC3 (a shared validity/null-count
value type), RC6 (`ctx.stripe(...)` + context-aware allocation).

**Tier 3 — larger refactors, schedule deliberately.** RC7 (`Grouping`, `JoinKind`, `JoinIndex`),
RC9 (the op registry — gate on `binary_size`), the parquet visitor to collapse the eight ladders,
`ipc.mojo` → package.

**Continuous.** Fix the doc drift now (it is cheap and actively misleading), and fold the
un-tracked items above into `docs/execution-engine-tasks.md` as FU entries so they are dated
and closeable.

---

## 7. Free-function census (RC10)

Every module-level function in the tree was read and classified against a strict rubric:
**KEEP-API** (PyArrow/ibis exposes the same name as a free function — equivalent must be named),
**KEEP-FREE** (forced: no receiver is representable), **METHOD / FACTORY / PRIVATE-METHOD**
(belongs on a type), **KERNEL** (belongs in a `Kernel` struct), **MOVE**, **DELETE-DUP**,
**DELETE-DEAD**.

### Headline

**262 module-level functions in `marrow/` (175 public, 87 private). ~200 of them (≈76%) should
not be module-level functions.** Of the survivors, ~48 are PyArrow-parity names and only **~14
are genuinely forced**.

| subsystem | fns | should stop being free | genuinely KEEP-FREE |
|---|---:|---:|---:|
| kernels | 122 | **115 (94%)** | **0** |
| parquet + IPC | 27 | **24 (89%)** | **0** |
| core + memory | 89 | 41 (46%) | 13 |
| expr + testing | 26 | ~13 | 1 |
| *(python + bindings, separately)* | *236* | *65 delete, 19 relocate* | *97 (ABI-forced)* |

**Two subsystems returned zero justified free functions.** Kernels and parquet/IPC have no
free-function survivors that aren't verbatim PyArrow names — which is strong support for the
thesis: in this codebase a free function is almost always a workaround.

The 13 core KEEP-FREE are almost entirely **C-ABI callbacks in `c_data.mojo`** (whose address is
handed to CPython and which need `abi("C")` linkage) plus the three `variant_dispatch*` adapters,
which take `std.utils.Variant` — a **stdlib type marrow cannot extend**. That is the honest shape
of the exception: *no receiver exists, or the receiver is not ours.*

### The structural argument

`marrow/__init__.mojo` is **0 bytes** — there is no curated top-level surface, so "public" means
*not underscore-prefixed*, not *deliberately exported*. Of 99 distinct public free-function names,
only ~24 appear in any `__init__.mojo`. **The "it's public API" defence mostly does not apply.**

Worse, the curated surface itself is unreliable: `kernels/__init__.mojo` exports `equal` (a legacy
duplicate slated for deletion — the per-element `String`-allocating string path that also mishandles
`large_string`) and `bitmap_and`, an internal validity helper whose exported status is how it kept
the raw-`.bitmap`-instead-of-`.validity()` signature bug nobody questioned.

### What the categories reveal

- **DELETE-DUP is the largest kernel category (39).** These are typed overloads that only
  round-trip through `DynArray`, or free functions shadowing the `Kernel` struct that replaced
  them. The 18 typed `filter`/`take` overloads are byte-for-byte `return Filter.apply(...)`,
  forming a *third* layer beside `dispatch` and `apply` — which is why adding an array type is a
  six-site edit. Recommendation: **keep exactly 3 free functions here** (`filter`, `take`,
  `drop_null` — all `pc.*` names, all needed by the Python binding), delete the other 20.
- **Whole modules have no kernel struct at all.** `hashing.mojo` (17 functions) and `sort.mojo`
  (8) are 100% free functions despite being textbook three-tier kernels — and they are exactly
  the two files with the **D5** dtype-coverage gaps. The missing struct *is* the missing
  dispatcher.
- **9 temporal extractors** (`year`, `month`, …) are `return XKernel.dispatch(array)` delegators
  **called only by tests** — a redundant naming layer with zero production use.
- **Verified dead:** `hash_identity` ×3 (all references are self-recursive within its own file),
  `_drop_null_bool`, `scalar()` ×2, plus 8 in the Python layer (`*_array_from_arrays` ×3 and their
  3 bindings, `read_ipc_stream_schema` ×2) — **14 dead functions**, free deletions.
- **Free functions hide encapsulation breaks.** `nulls()` writes `b._null_count` from module scope;
  `dispatch_over_*` (64 call sites) exist *only* to read `DynType._v` from another module;
  `views.mojo`'s two bitmap↔bitmap `apply` overloads read `_data`/`_offset`/`_length` off
  `BitmapView`. Each is RC1 wearing a different hat — **making them methods fixes the reach-through
  for free.**
- **Free functions strand core logic in the wrong layer.** `_as_py` (58 lines of full type-system
  dispatch) lives in `python/bindings/scalars.mojo`; `_record_batch_join`/`group_by`/`aggregate`/
  `sort_by` (~265 lines of real semantics, incl. output-column renaming) exist **only for Python
  callers** and drag a function-body import of `marrow.expr.relations` into the bindings —
  the exact inversion of CLAUDE.md's mandated split.
- **`_view_to_owned`, `_apply_dispatch`, `_reduce_dispatch`** are textbook feature envy: the
  latter two read *only* `ExecutionContext` accessors, so CPU/GPU/parallel policy lives in
  `views.mojo` rather than on the type that owns it.

### Proposed rule (to codify)

A module-level function is justified **only** if one of:

1. **PyArrow/ibis parity** — the equivalent is a free function there (`pc.filter`, `pa.list_`,
   `pq.read_table`, `ibis.memtable`). CLAUDE.md already mandates PyArrow naming. *Must name it.*
2. **DSL entry point** — `col`, `lit`, `if_else`; no receiver by construction.
3. **No representable receiver** — C-ABI callbacks, comptime adapters over **stdlib** types
   (`Variant`, `Span`, `UnsafePointer`) that marrow cannot extend, or a cited Mojo limitation.

Everything else is a method, a static factory, a private method of its one owning type, or a
`Kernel` struct. Note even (3) is narrower than it looks: `dispatch_over_*` *appears* to qualify,
but `DynType` **is** marrow's own type — making them methods removes 64 private-field
reach-ins.

### Sequencing note

Most of this is mechanical and low-risk, but four items touch the **binary-size/DCE gate** and
must be measured (`pixi run binary_size`): promoting `Datum` to a real struct (highest risk —
recommend leaving `into_array` free), moving `_rank`/`promote` into `dtypes.mojo`, introducing a
`ColumnSet` type reachable from every node's `referenced_columns()`, and adding per-dtype methods
to `BitmapView`/`Bitmap`/`ArrayData`.

---

## 8. Test-suite baseline — RESOLVED

**Current state (`d0ecad7`): 1826 passed, 314 skipped, 0 failed.** The suite is green.

Previously measured at `80ebc10`: **59 failed, 1737 passed, 305 skipped**, confined to
`parquet/tests/test_reader.mojo` (35) and `expr/tests/test_streaming.mojo` (24).

**All 59 were one bug, now fixed upstream** by the Mojo `dev2026072217 → dev2026072406`
upgrade: `ArcPointer[TagValue]` wrote a `Variant` discriminant one byte past its allocation,
corrupting the heap cumulatively until it hit live allocator metadata. It was a toolchain
defect, not marrow logic. Both files now pass in full (43 and 35), and ASAN reports **0**
`heap-buffer-overflow` hits where it previously reported 86. See `code-quality-tasks.md` Q0.0
for the verification table and the methodology lessons (ASAN masked it; build failures look
identical to clean runs; minimal reproducers were useless).

Note the T2.3a Sort/Limit/TopK/Project tests were never broken — they were collateral, since
`TestSuite` runs a whole file in one process and one corrupting test fails all of them.

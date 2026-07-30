# Arrow type coverage — map gap & the view layouts

Companion to the type-gap analysis run on 2026-07-30. This file is the executable
half: discrete, worktree-ready tasks with explicit file ownership, following the
conventions of [`tasks-execution-engine.md`](tasks-execution-engine.md) (one task =
one worktree = one owned write-set; ownership rules and the quality gate in §1 of
that file apply verbatim here).

**Scope:** the **map serialization hole** (V0) and the **view layouts** (V1–V6), with
the string-like views (`Utf8View`/`BinaryView`) as the priority. Union and run-end
encoding are *out of scope* for this document — they are real gaps, but nothing on
the execution-engine roadmap needs them and they share no machinery with the views.

---

## 1. Why these two, and why in this order

The gap analysis found marrow implements 22 of the 26 core `Type` members in
`format/Schema.fbs`, including all four decimal widths and all three interval units.
Exactly four core families are absent: sparse/dense **union**, **run-end encoded**,
the **binary-view** layouts, and the **list-view** layouts. `CLAUDE.md`'s claim about
those three families is accurate — but it is *silent* on map, which is the one type
that is half-shipped.

### V0 — map is a partial, and the hole is serialization-shaped

| axis | status |
|---|---|
| dtype | ✅ `MapType` — [dtypes.mojo:621](../marrow/dtypes.mojo#L621), `map_()` factory at :1372 |
| array | ✅ `MapArray = ListLikeArray[MapType]` — [arrays.mojo:1213](../marrow/arrays.mojo#L1213) |
| builder | ✅ `MapBuilder` — [builders.mojo:1093](../marrow/builders.mojo#L1093) |
| scalar | ❌ no `MapScalar` |
| C-Data | ✅ emits and parses `+m` |
| IPC | ❌ **neither direction** |
| cast | ❌ no `is_map()` arm |

Verified by grep, not by trusting a doc: `_TYPE_MAP`, `is_map`, and `MapType` have
**zero occurrences** in [ipc.mojo](../marrow/ipc.mojo). The `_TYPE_*` block
(:43–65) runs 1–13, 15, 16, 18–21 and skips **17** with no comment, sitting right
next to the deliberately-skipped 14 (Union) and 22–26 (REE/views). `_type_code`
(:785) has no map arm, so a map column falls through to
`raise Error("_IpcEncoder: unsupported dtype")`; `_write_type_table` raises at
:1006; the decoder ladder raises `unsupported type_type` at :1527.

Net effect: **a map column round-trips through PyArrow via C-Data but cannot be
written to or read from an Arrow IPC file.** Everything expensive — the layout, the
buffers, the offsets — is already the list path. This is hours of work closing a
real hole in a type marrow already advertises, which is why it goes first.

### V1–V5 — string views are the layout the rest of the ecosystem computes on

`Utf8View`/`BinaryView` are format **1.4** (2024), and status.html shows them
implemented in C++, Java, Go, Rust, C#, and nanoarrow. More to the point for
marrow: **DuckDB, Polars, and Velox all use the German-string representation
internally.** Today a string column crossing that boundary costs a full
materializing conversion. That is an interop-performance argument, not a
correctness one — nothing in ClickBench/H2O/TPC-H requires views — but it is the
strongest reason on the list to build them.

### V6 — list views are the same spec version, a fraction of the value

`ListView`/`LargeListView` add a third `sizes` buffer alongside offsets and permit
**out-of-order and overlapping** offsets. That breaks the invariant every nested
kernel in the tree currently assumes (child order matches logical order), so the
blast radius is much larger than the buffer count suggests, and the payoff is much
smaller. Scheduled last, deliberately.

---

## 2. What the view layouts actually require

Spec references: `Columnar.rst:482` (variable-size binary view), the list-view
section following it, and `Schema.fbs` Type members 23–26.

### 2.1 The view buffer

A 16-byte fixed-width element, two forms discriminated by the leading length:

```
length <= 12   (fully inline)     length > 12   (spilled)
┌────────┬──────────────────┐     ┌────────┬────────┬─────────┬────────┐
│ len:i32│ data: byte[12]   │     │ len:i32│ pfx[4] │ buf_idx │ offset │
└────────┴──────────────────┘     └────────┴────────┴─────────┴────────┘
```

The 4-byte prefix in the spilled form is what makes comparison fast: two views
with different prefixes are unequal without touching the data buffers. Any
equality/compare/sort kernel that skips this fast path has thrown away the reason
the layout exists.

### 2.2 Variadic buffers — the one genuinely new concept

A view array carries `1 + N` buffers: the views buffer plus a **variable number**
of data buffers. arrow-rs models this as `DataTypeLayout.variadic: true`.

Two pieces of good news from reading the source:

- **`ArrayData.buffers` is already `List[Buffer[mut=False]]`**
  ([arrays.mojo:199](../marrow/arrays.mojo#L199)) — the interop DTO already
  accommodates a variable buffer count. Nothing to change there.
- The constraint lives in the **typed** structs, which are fixed-arity:
  `BinaryLikeArray[T]` ([arrays.mojo:728](../marrow/arrays.mojo#L728)) holds
  exactly `offsets: Buffer` + `values: Buffer`. A view array cannot reuse it and
  needs its own struct holding `views: Buffer` + `data: List[Buffer]`.

This materially softens the "large effort" estimate: it is a *new struct*, not a
*model change*. See the risk in §5 about `DynArray`'s variant, which is where the
real cost hides.

### 2.3 The IPC wrinkle

`Message.fbs`'s `RecordBatch` table carries **`variadicBufferCounts: [long]`** —
one entry per view-typed field, giving that field's data-buffer count. Marrow's
encoder and decoder currently do not read or write this field at all. It is not
optional: without it a reader cannot partition the flat buffer list back into
per-field groups. Budget for it explicitly; it is the single most likely thing to
be missed and to produce files that PyArrow rejects.

### 2.4 C-Data format strings

`vu` (utf8 view), `vz` (binary view), `+vl` (list view), `+vL` (large list view).
Emission lives in `CArrowSchema.from_dtype` ([c_data.mojo:304](../marrow/c_data.mojo#L304))
and parsing in the format-string ladder around :620–757. Note the exported
`ArrowArray.n_buffers` is variable for these types.

---

## 3. Tasks

Effort labels: **S** = hours, **M** = 1–2 days, **L** = several days. These are
engineering judgement, not measurements.

---

### V0 — Map through IPC, plus `MapScalar` and a cast arm · **S**

**Owns:** `marrow/ipc.mojo`, `marrow/scalars.mojo`, `marrow/kernels/cast.mojo`,
`marrow/tests/test_ipc.mojo`

1. Add `comptime _TYPE_MAP: UInt8 = 17` to the [:43–65](../marrow/ipc.mojo#L43) block.
2. `_type_code` (:785): add an `elif dtype.is_map(): return _TYPE_MAP` arm.
3. `_write_type_table` (:830): write a `Map` table carrying the **`keysSorted`**
   bool. Do not drop it — it is the only field the table has, and a reader that
   assumes `false` silently loses a sortedness guarantee.
4. Decoder ladder (~:1411–1527): add a `_TYPE_MAP` arm reconstructing
   `MapType(entries_field, sorted)` from the single non-nullable `entries` struct
   child. Buffers are the list path unchanged.
5. `MapScalar` in `scalars.mojo`, alongside the existing `ListScalar`/`StructScalar`.
6. A `is_map()` arm in `cast.mojo`'s top-level ladder (:1015–1051).

**DoD:** a map column round-trips through `ipc` file *and* stream, PyArrow reads
marrow's output and marrow reads PyArrow's, `keysSorted` survives both directions,
and a map scalar can be extracted from a `MapArray`.

**Open question, resolve before starting:** was 17 skipped deliberately (deferred
because C-Data interop already works) or forgotten? The absence of a comment next
to the deliberately-skipped 14/22–26 reads like an oversight, but confirm — if
there is a known blocker it will be in the map-array child-field handling.

---

### V1 — `Utf8View`/`BinaryView` dtype + array layout · **M**

**Owns:** `marrow/dtypes.mojo`, `marrow/arrays.mojo`, `marrow/views.mojo`,
`marrow/tests/test_arrays.mojo`

- `BinaryViewType` / `StringViewType` structs, `is_binary_view()`/`is_string_view()`
  predicates, and inclusion in the **`dispatch_binarylike` / `dispatch_stringlike`**
  families ([dtypes.mojo:826–913](../marrow/dtypes.mojo#L826)). Per the kernel
  pattern in `CLAUDE.md`: dispatch on the widest family the typed leaf accepts —
  do not invent a `dispatch_view` sibling.
- `BinaryViewArray[T]` (parameterized like `BinaryLikeArray` so utf8/binary share
  one struct): `length`, `nulls`, `offset`, `bitmap`, `views: Buffer[mut=False]`,
  `data: List[Buffer[mut=False]]`.
- A `StringViewElement` value type in `views.mojo` decoding the 16-byte element,
  exposing `length()`, `prefix()`, `is_inline()`, and a `bytes()` accessor that
  resolves against `data`. Element access goes through this — **no raw pointer
  arithmetic outside `buffers.mojo`/`views.mojo`** (`unsafe_ptr()` restriction).
- `to_data()` / `from_data()` mapping `data` onto `ArrayData.buffers[1:]`.

**DoD:** construct from hand-built buffers, index, slice (offset applies to the
views buffer only — data buffers are never sliced), equality, `Writable` output.

---

### V2 — View builders · **M** · depends on V1

**Owns:** `marrow/builders.mojo`, `marrow/tests/test_builders.mojo`

`BinaryViewBuilder[T]` with the inline/spill decision at `append`. Needs a stated
**block policy**: target data-buffer size (arrow-rs uses 8 KiB doubling up to 2
GiB), when to seal a block and start another. Write the policy down in the
docstring — it is the thing a reader will want to tune, and an undocumented magic
constant here is worse than a suboptimal documented one.

**DoD:** builder output byte-identical to PyArrow's for the same input sequence
across inline-only, spill-only, and mixed cases, including strings of exactly 12
and 13 bytes.

---

### V3 — C-Data import/export for `vu`/`vz` · **S–M** · depends on V1

**Owns:** `marrow/c_data.mojo`, `marrow/tests/test_c_data.mojo`

Emit `vu`/`vz` in `from_dtype`; parse them in the format ladder; export/import a
variable `n_buffers`. **Watch the 64-byte alignment assert** — the known flaky
C-Data import failure with numpy-backed PyArrow arrays will look like a V3 bug when
it is not. Re-run before diagnosing.

**DoD:** zero-copy round-trip with PyArrow in both directions, including an array
with ≥2 data buffers.

---

### V4 — IPC read/write for the view types · **M** · depends on V1, conflicts with V0

**Owns:** `marrow/ipc.mojo`, `marrow/tests/test_ipc.mojo`

Type codes 23/24, the type tables (both are empty tables — no fields), and
**`variadicBufferCounts` in the `RecordBatch` message** (§2.3). The decoder must
partition the flat buffer list per field using those counts.

**Must not run concurrently with V0** — same owned file. V0 lands first.

**DoD:** marrow→PyArrow and PyArrow→marrow file *and* stream round-trips, including
a batch mixing a view column with a non-view column (which is what exercises the
buffer partitioning).

---

### V5 — Kernel coverage for string views · **M–L** · depends on V1

**Owns:** `marrow/kernels/cast.mojo`, `marrow/kernels/compare.mojo`,
`marrow/kernels/string.mojo`, `marrow/kernels/filter.mojo`,
`marrow/kernels/tests/*`

Minimum viable set, in order: `cast` utf8 ↔ utf8_view (both directions),
`take`/`filter` (cheap — copy 16-byte views, share the data buffers, do not
materialize), equality and ordering **with the 4-byte prefix fast path**, then the
`string.mojo` predicates.

**DoD:** each kernel has a test with inline, spilled, and mixed inputs; `take` and
`filter` are verified to *share* rather than copy data buffers.

---

### V6 — `ListView`/`LargeListView` · **L** · schedule last

**Owns:** `marrow/dtypes.mojo`, `marrow/arrays.mojo`, `marrow/builders.mojo`,
`marrow/c_data.mojo`, `marrow/ipc.mojo` (serialized against V0/V4)

Layout is validity + offsets + **sizes** + one child. The hard part is not the
extra buffer, it is that offsets may be **out of order and overlapping**, so:

- `sum(sizes)` may exceed the child length, and the child may contain elements
  reachable from no list at all;
- any kernel that walks the child linearly assuming logical order is **wrong** for
  list-view, silently;
- slicing cannot be expressed as a child-range narrowing.

**Before writing code**, audit which nested kernels make the ordered-child
assumption and decide per kernel: support, or reject list-view explicitly with a
clear error. A silent wrong answer here is the failure mode.

---

## 4. Wave schedule

Waves are grouped so that within a wave every owned file has exactly one writer.

| wave | tasks | note |
|---|---|---|
| 1 | **V0** | owns `ipc.mojo`; unblocks nothing, ships value immediately |
| 2 | **V1** | foundation for everything after |
| 3 | **V2**, **V3** | disjoint (`builders.mojo` vs `c_data.mojo`) |
| 4 | **V4**, **V5** | disjoint (`ipc.mojo` vs `kernels/*`); V4 needs V0 merged |
| 5 | **V6** | serialized against V0/V4 on `ipc.mojo` |

`dtypes.mojo` and `arrays.mojo` are hotspots owned by V1 in wave 2 and by V6 in
wave 5 — never concurrently.

---

## 5. Risks and constraints

**`DynArray` is an inline `Variant`, and this is where the cost hides.**
[arrays.mojo:2103](../marrow/arrays.mojo#L2103) enumerates ~35 typed arrays in one
`Variant`; adding 2 (V1) or 4 (V1+V6) grows every `DynArray` by the delta of the
largest member. Two things this collides with:

1. **The binary-size gate.** `pixi run binary_size` and the closed-erasure/DCE
   property of `marrow.aot`/`marrow.expr` are hard constraints. Measure before and
   after V1; if a view variant is larger than the current widest member, the growth
   is paid by *every* `DynArray` in the tree, not just view columns.
2. **The open `ArcPointer[TagValue]` bug** (`expr/values.mojo:2299` — trailing
   `Variant` discriminant written one byte past the allocation, `size_of` 416 vs
   ≥417 needed). Anything that changes variant sizes near that boundary will
   perturb this. Verify **without ASAN** — ASAN masks it.

**The "no array/scalar/builder layout changes" constraint** from the code-quality
program does not block V1 as designed: the view array is a *new* struct and
`ArrayData.buffers` already accommodates a variable count. Confirm that reading is
shared before wave 2 opens; if the constraint is read more strictly than that, V1
needs an explicit exemption.

**Kernel coverage was never audited.** The gap analysis assessed per-type kernel
support from `cast.mojo`'s dispatch ladder only; the other ~20 kernel files were
not checked. The precedent in `CLAUDE.md` — `filter`/`take` carrying separate
numeric and temporal arms that silently left decimal and interval unsupported — is
exactly the failure mode to expect. **A separate audit task, independent of
everything above, is worth more than any single V-task here:** walk every
`dispatch_*` call site and record which dtypes each kernel actually accepts.

**Effort estimates are judgement.** Treat them as relative ordering.

---

## 6. Explicitly out of scope

- **Union** (sparse + dense) — two layouts, one `Schema.fbs` member, type code 14.
  Pre-1.0 core, so the highest interop expectation of the missing four, but nothing
  in the roadmap needs it.
- **Run-end encoding** — type code 22. Parent has no buffers and no validity;
  children named `run_ends`/`values`. Real cost is logical↔physical index
  translation (arrow-rs needs binary search plus a logical-length/offset-carrying
  run buffer for random access and slicing).
- **Canonical extension types** — `ARROW:extension:name`/`:metadata` has zero
  occurrences in the tree, so all 8 canonical extensions are unavailable. Cheap to
  close (metadata plumbing plus a registry, no new layout) and marrow already owns
  the storage layout for 7 of 8 — the `StringView`-backed variant of `arrow.json`
  is the exception, and V1 would unblock it.

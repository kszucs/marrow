# Duplication audit — solution proposals

Design options for every item in `docs/duplication-audit.md`, each from two or
three angles so the design can be chosen rather than accepted. **Status:
proposals, nothing implemented.**

§1.4 has its own document, `docs/duplication-audit-1.4.md`, because working it
through changed what the item was. Everything else is here.

Verified against the working tree at `5435f59` + uncommitted changes
(2026-08-16).

## How to read this

Each item states the problem in a line or two, then gives the angles. A verdict
line says what each angle costs and buys; a **What decides it** line names the
fact or measurement that should pick between them, so none of these turns into a
matter of taste.

Items are grouped by **risk class**, not by audit order, because the class is
what determines how much evidence a change needs:

| Part | Class | Gate |
|---|---|---|
| 1 | Size-gated (`marrow.expr`) | `pixi run binary_size`, and may come back "no" |
| 2 | Hot path | A benchmark, drift-normalised per §0 |
| 3 | Free | Tests only |
| 4 | Placement | Import-graph check |
| 5 | Deletion | Judgement |

A decision table is at the end.

---

# Part 1 — Size-gated

`docs/backlog.md` §0: folding twelve promote-then-dispatch sites into one
`_arith[K]` helper cost **+115,600 bytes**. `query_streaming_agg_fused` is at
+0.449% of a 0.5% ceiling. Measure one gate binary directly
(`mojo build -O3 -g0 -I . …query_dynvalue.mojo`, ~2.5 min), not the 10-minute
sweep.

**One argument runs through all three items and is worth stating once:** the
§0 trap is about a *generic wrapper instantiated per kernel*, each instantiation
carrying its own copy of what it touches. A **trait default method** is not that
shape — it is still instantiated once per conforming struct, exactly as the
hand-written copy was, so DCE sees the same reachability graph. If that holds,
Part 1 is source-level deduplication with no codegen change. **It is an
argument, not a measurement.** One spike settles it for all three items, and
should be run before any of them: pick the eight `NumericUnary`-family nodes,
move their four methods to a sub-trait default, and measure.

## §1.1 — `expr/values.mojo` validity/state delegation (~200 lines, 37 nodes)

Three method-body shapes re-typed per node family: unary passthrough (8 sites ×
~10 lines), binary intersect (4 × ~16), breaker (2 × ~21). Docstrings included,
byte for byte.

**Angle A — accept it.**
Leave all 200 lines. The nodes are the size-gated surface, the duplication is
mechanical and inert, and every alternative spends a spike to find out whether it
is even legal.
*Buys:* nothing. *Costs:* nothing. The honest baseline.

**Angle B — sub-trait defaults plus an operand accessor.**
Add `trait UnaryValue(Value)` (and `BinaryValue`, `BreakerValue`) carrying
default `validity` / `state_validity` / `referenced_columns` / `state` bodies.
The defaults need to reach the operand, which trait methods cannot do for a
field — so each node adds `def operand(self) -> ref[self.a] Self.A: return self.a`.
Net: ~10 lines per node becomes 1.
*Buys:* ~180 lines, and the `state_validity` docstring — which explains the FU-7a
double-execution it exists to prevent — stops being copied eight times.
*Costs:* three new traits; a spike to confirm legality; a size measurement.
*Known hazards:* CLAUDE.md records that re-defaulting a base trait's abstract
method in a sub-trait **recurses** when that method returns `Self.ArrayType` and
a conformer's `ArrayType` references another trait-member child. These methods
return `Optional[Bitmap[mut=False]]` and `Self.State` — the first is concrete and
safe; `state()` returning `Self.State` is the one to test first, since `State` is
an associated type and that is precisely the shape the note warns about.

**Angle C — Angle B for the unary cluster only.**
Eight sites, one new trait, one shape. Leaves binary and breaker alone.
*Buys:* ~80 lines and most of the risk reduction, since the unary shape has no
`Pair` state and the simplest `State` projection.
*Costs:* the tree ends with one deduplicated family and two duplicated ones,
which is a worse *story* than either A or B even though it is a strictly better
diff.

**What decides it:** the spike. If `state()` returning `Self.State` recurses as a
sub-trait default, B and C both collapse to "defaults for `validity` and
`referenced_columns` only" — about 90 lines rather than 180, which is likely no
longer worth three new traits, and A wins.

## §1.2 — the four `*Column` structs

`NumericColumn` / `StringColumn` / `TemporalColumn` / `ListColumn` share
`_name: String`, `__init__`, `referenced_columns`, `materialize`, `validity`,
`name`. Two of the four are also missing `bound_column` and `prune` — so a date
or list column is not join-key-eligible and prunes nothing in the fused lane.

**Angle A — fix the gap, keep the four structs.**
Add `bound_column` and `prune` to `TemporalColumn` and `ListColumn`, copying
`StringColumn`'s. `prune` needs a dtype-appropriate `Interval.bounds`; for a
list column "no information" may genuinely be right, in which case only
`bound_column` is added.
*Buys:* closes a real behavioural gap. *Costs:* +20 lines — it makes the
duplication *worse* while making the behaviour correct.
*This is the minimum, and it is independent of every other angle.*

**Angle B — extract a `_ColumnRef` payload.**
A small struct holding `_name` with `referenced_columns` / `materialize` /
`validity` / `name` / `bound_column` / `prune` as its methods. Each column
becomes `var _ref: _ColumnRef` plus six one-line forwards.
*Buys:* single-source logic; the gap in Angle A cannot recur, because there is
one implementation. *Costs:* ~24 forwarding lines remain; a field change to four
fused nodes, so it needs a size measurement.
*Note:* the standing "do not change array, scalar and builder layout" constraint
does **not** apply — these are expression nodes.

**Angle C — one `Column[T: DataType]` struct.**
*Ruled out.* It would have to satisfy `NumericValue.lane` (returns
`SIMD[T.native, W]`) and `StringValue.lane` (returns `String`) simultaneously.
Different return types, so no single `lane` and no `comptime if` rescue —
CLAUDE.md's conditional-type note covers exactly this. Recorded so it is not
re-attempted.

**What decides it:** whether B's field change is size-neutral, and whether a
list column has anything useful to say to `prune`. **A should land regardless**
— it is a correctness fix wearing a deduplication costume.

## §1.3 — operator↔interval-kernel pairing, 20 sites, encoded twice

The AOT lane pairs each operator with its interval kernel by hand
(`values.mojo:1094-1099`, `:1209-1211`, `:1936-1953`); the runtime lane
re-encodes the same table as a name-keyed ladder (`dynamic.mojo:610-626`). A
mismatched pair is silently wrong pruning, not an error.

**Angle A — associated type on the kernel trait.**
`comptime Interval: IntervalKernel` on `NumericCompareKernel`,
`BoolBinaryKernel`, `StringPredicateKernel`. `NumericCompare[K, KI, L, R]` loses
`KI` and reads `Self.K.Interval`; all 20 pairings become 20 kernel declarations
that already exist.
*Buys:* the mapping stops being a table anyone can get wrong. Both lanes derive
from one source.
*Costs:* touches three traits and every conforming kernel struct; size
measurement.
*Precedent is good:* CLAUDE.md warns that a `comptime name: T` trait requirement
"does not resolve reliably when read as `E.name` from a function generic over
`E: SomeTrait`" — but records that `Self.K.name` **on a kernel parameter** does
resolve, citing `NumericCompare.prune` (`values.mojo:965`), which is the same
position. Low risk, still a spike.

**Angle B — a parity test instead.**
Leave the pairings; add a case to `expr/tests/test_parity.mojo` asserting
`XKernel.name == XInterval.name` for every pairing, and that every operator the
runtime ladder handles has an AOT pairing and vice versa.
*Buys:* drift becomes a test failure. Zero risk, zero size exposure, ~30 lines of
test.
*Costs:* the table still exists twice; the test has to be extended when an
operator is added — though `test_parity.mojo` is already the file whose job that
is, per §0's "one engine, two drivers" invariant.

**Angle C — both.**
A closes the AOT half; the runtime ladder stays a ladder by design (§0: rewriting
the runtime lane as a tag switch cost **+1,807,168 bytes**, which is why
`EvalFn` exists). So even under A, the two lanes still encode the mapping
separately and B is what ties them.

**What decides it:** whether `Self.K.Interval` resolves. If yes, C. If no, B
alone — which is most of the safety for none of the risk.

---

# Part 2 — Hot path

## §1.5 — the GPU-or-host allocation preamble, 10 sites

Seven lines of `comptime if GPU_ENABLED: if ctx.is_gpu(): alloc_device else
alloc_zeroed`, at `numeric.mojo:99, 178, 500`, `cast.mojo:127, 208, 258`,
`hashing.mojo:383, 446, 545`, `boolean.mojo:360`. Three of the ten write the
same decision as a ternary rather than an if/else.

**Angle A — `Buffer.alloc_for[T](ctx, n)` / `Bitmap.alloc_for(ctx, n)` in `buffers.mojo`.**
*Costs:* `buffers.mojo` does not import `.execution` today (it takes
`DeviceContext` from `max.gpu.host` directly) and does not import `GPU_ENABLED`
from `.utils`. Both edges would be new, pointing from the lowest-level module at
policy. No cycle results — but it is the wrong direction.

**Angle B — `ExecContext.alloc_buffer[T](n)` / `.alloc_bitmap(n)` in `execution.mojo`. (Recommended.)**
*Buys:* the decision lives on the type that owns the device. "Where does this
memory go" is a device-policy question and `ExecContext` **is** the device-policy
type. Ten call sites become one line each.
*Costs:* a new `execution → buffers` edge. Checked: `views.mojo` already imports
both `.buffers` and `.execution`, and `buffers ⟷ views` is already a cycle, so
this adds no new cycle and no new layering violation.
*Converges with §2.3* — that item wants `GPU_ENABLED` and
`has_accelerator_support` moved out of `utils.mojo` to `execution.mojo`. Doing
both makes `ExecContext` the single answer to every "is there a device, and
what follows from that" question. **Take these two together or neither.**

**Angle C — normalise, don't extract.**
Rewrite the three ternary sites as if/else so all ten read identically, and stop.
*Buys:* consistency, zero risk. *Costs:* ten copies remain.

**What decides it:** whether you want `ExecContext` to grow an allocation
surface. If yes, B+§2.3 is the coherent move. If the preference is to keep
`ExecContext` a pure policy object, C — A is the worst of the three.

## §1.6 — `filter.mojo`'s set-bit iteration loop, 5 sites

The word-at-a-time CTZ loop, tail mask included, at `filter.mojo:252, 274, 364,
425, 480`. The pair at 252/274 differs *only* by an `if src_bm.test(idx)` in the
body. Validity-filter preamble repeats 4× more at `:379, 444, 493, 549`.

**Angle A — `@always_inline` helper taking a unified closure.**
`_for_each_set_bit(mask, n, off, body)` with `body` capturing the mutated state
(`{mut byte_pos, mut j, mut null_count, mut bm_builder, imm}`).
*Buys:* one copy of the tail-mask logic — the detail most likely to be fixed in
one place and not the others.
*Costs:* a benchmark gate. §0's sibling-mutable-capture limit does **not** bite
(one closure, not several), but the §0 warning about `sync_parallelize`
miscompiling implicitly-captured closures means any "assignment was never used"
warning on `byte_pos`/`j` invalidates the run.

**Angle B — a `SetBits` iterator struct.**
`for idx in SetBits(mask, n, off):` — no capture list, no closure, mutation stays
in the caller's frame where the compiler can see it.
*Buys:* sidesteps the capture-miscompile class entirely; reads better.
*Costs:* iterator overhead versus a fully inlined loop is unmeasured here, and
this is the hottest loop in the library.

**Angle C — collapse 5 copies to 3 without abstraction.**
Hoist the `if src_bm.test(idx)` out of the 252/274 pair by making it a
`Optional[BitmapView]` test inside one loop, and likewise for whichever of
364/425/480 differ only in the body.
*Buys:* two copies gone for zero new machinery and no benchmark risk.
*Costs:* three copies remain.

**What decides it:** `bench_filter`, five interleaved repeats, normalised against
untouched rows. If A and B both come out flat, prefer B — it is immune to the
capture-miscompile failure mode, which per §0 has already cost this project two
wrong conclusions. If either regresses, C is the fallback and is worth doing
regardless.

## §1.7 — `builders.mojo` validity extend, 5 sites

The 11-line "reserve, then propagate nulls or `set_range`" block at
`builders.mojo:696, 827, 1022, 1229, 1370`.

**Angle A — a free function with `mut` parameters.**
`_extend_validity(mut bitmap, mut null_count, mut length, arr)`.
*Costs:* three `mut` out-params; the ugliest option and the one most likely to be
misused.

**Angle B — a method on `Bitmap[mut=True]`. (Recommended.)**
`bitmap.extend_validity(src: Optional[BitmapView], src_offset, dest_offset, n) -> Int`
returning the null count added. Each builder's `extend` becomes ~3 lines.
*Buys:* the logic is *bitmap* logic — "extend these bits from an optional source,
or set a range true" — not builder logic. It belongs beside `Bitmap.extend` and
`Bitmap.set_range`, which it already calls. Every builder holds
`_bitmap: Bitmap[mut=True]` (confirmed at `builders.mojo:562+`), so there is no
field-access obstacle.
*Costs:* one new `Bitmap` method; a test.

**Angle C — a `trait ValidityBuilder` default.**
*Ruled out.* The default would need `self._bitmap`, `self._length` and
`self._null_count` — trait methods cannot reach fields. Same wall as
`validity()` in §1.4.

**What decides it:** nothing contentious. B unless there is a reason to keep
`Bitmap`'s surface narrow. Not on a hot path (builders are per-append, and this
is the bulk path), so no benchmark gate.

---

# Part 3 — Free

## §1.8 — `c_data.mojo`'s release handshake, 3 structs

`is_released` / `mark_released` verbatim on `CArrowSchema:304`,
`CArrowArray:948`, `CArrowArrayStream:1563`; `__deinit__` on the first two. Six
occurrences of
`Pointer(to=self.release).unsafe_bitcast[UInt64]()[unsafe_offset=0]`.

**Angle A — free helpers over the release slot. (Recommended.)**
`_release_slot_is_null(slot: Pointer[UInt64, …]) -> Bool` and
`_null_release_slot(slot)`, called as
`_release_slot_is_null(Pointer(to=self.release).unsafe_bitcast[UInt64]())`.
*Buys:* the raw pointer arithmetic exists once. Line count is the small part —
CLAUDE.md restricts `unsafe_ptr`-class code to a named set of files precisely
because it is the code that must not be reasoned about six times, and this is the
spec's double-free guard.
*Costs:* three one-line wrappers per struct remain; the `Pointer(to=self.release)`
half still repeats, because the field is what makes it work.

**Angle B — leave it.**
These are C-ABI structs whose layouts are fixed by the spec. Explicitness at an
ABI boundary is defensible, and the copies have not drifted.

**Angle C — a `CReleasable` trait with defaults.**
*Ruled out.* Names `self.release`.

**What decides it:** how much weight the "unsafe code in one place" rule carries
versus "ABI structs should read literally". A is a small, safe improvement; B is
a legitimate choice.

## §1.9 — `marrow/testing/` duplication, 40 lines verbatim

`CLIFlags` and `_print_json_array` byte-identical in `test.mojo:25, 46` and
`bench.mojo:34, 55`.

**Done 2026-08-16**, by a route neither angle anticipated: the whole
`marrow/testing/` package collapsed into a single `marrow/utils/testing.mojo`, so
there is one `CLIFlags` because there is one file. Both angles assumed the
two-file split was fixed.

## §1.12 — tests: 32,526 lines, no shared fixtures

The nine-times-repeated join fixture; `_batch` defined 4×, `_to_marrow` 3×; the
`MapBuilder` setup block in three files; a 35-line identical run in
`bench_bitmap.mojo`; two conventions for temp paths.

**Angle A — `marrow/testing/fixtures.mojo`.**
*Buys:* one home; per the "one selection = one compilation unit" property, shared
fixtures cost nothing extra to compile.
*Costs:* `marrow/testing/` ships in the package, so test-only builders would
ship too. Mitigated by DCE and by the fact that tests already live inside the
package — but it does put fixture code on the public import path.

**Angle B — `_fixtures.mojo` per tests directory.**
*Buys:* fixtures sit next to their users; nothing new on the public path.
*Costs:* `_batch`-shaped helpers get written up to four times still — though
once per directory instead of once per file, which is where 8 of the 9 join-fixture
copies go.

**Angle C — fix only the outliers.**
The `bench_bitmap.mojo` duplicate is **done (2026-08-16)** — the three
`_bench_pack_bools_w{8,32,64}` helpers are now one `_bench_pack_bools[W]`, −130
lines.

The temp-path half turned out **not** to be an outlier and was re-filed. Fixed
`/tmp/` paths are the *prevailing convention* across `parquet/tests/` — ~15 sites
in `test_codecs.mojo`, `test_bloom.mojo`, `bench_parquet.mojo` and
`test_parquet.mojo` — while only `tests/test_ipc.mojo:52` uses
`tempfile.mkstemp`. Converting one file would make it inconsistent with its
neighbours, so this is suite-wide or nothing. It is a real hazard: two concurrent
`pytest` invocations, which the harness explicitly supports, collide on the fixed
paths.

**What decides it:** whether test helpers on the shipped import path is
acceptable. If not, B.

## §1.13 — Python bindings

`RecordBatch:225` and `Table:356` in `python/marrow/__init__.py` duplicate ~34
lines of `self._binding` delegation plus `to_pydict`/`to_pylist`. The builder
`extend` loop repeats 3× in `python/bindings/arrays.mojo`.

The Python half is **done (2026-08-16)** — a `_Tabular(_Wrapper)` base now holds
the 15 shared methods. What is left is the Mojo half.

**Angle A — extract the builder `extend` loop.**
`python/bindings/arrays.mojo:647, 692, 862` share a null-checking append loop.
*Costs:* the loops differ slightly — one counts bytes first — so the helper needs
a parameter or two variants.

**Angle B — leave it.**
Three loops of eight lines, in binding glue that changes rarely.

**What decides it:** low stakes either way. The `pymethod` arity overloads in
`helpers.mojo` are irreducible without variadic forwarding, which §0 rules out —
leave them regardless.

---

# Part 4 — Placement

Nine items from audit §2. Most have one obvious answer; the two that do not are
`utils.mojo` and `Grouping`.

## `utils.mojo` is four unrelated things

Variant dispatch (its documented purpose), `LittleEndian`, `Crc32`, and
`GPU_ENABLED`/`has_accelerator_support`. Plus a 19-line orphaned banner at `:315`
documenting adapters that live in `dtypes.mojo`.

**Angle A — full split.**
`Crc32` → `parquet/` (its only users are `parquet/{reader,writer,bloom}.mojo`);
`LittleEndian` → a new `marrow/byteorder.mojo` (used by `ipc.mojo` and six
parquet files); `GPU_ENABLED` + `has_accelerator_support` → `execution.mojo`
(**converges with §1.5 Angle B**); delete the orphaned banner. `utils.mojo`
becomes what its docstring says it is, and could be renamed `dispatch.mojo`.
*Costs:* touches ~12 files' imports. Mechanical, but a wide diff.

**Angle B — move only `Crc32`.**
The single clearest case: one consumer package, zero ambiguity.
*Costs:* `utils.mojo` remains three things.

**Angle C — leave, and fix the docstring.**
Rewrite `utils.mojo`'s module docstring to describe all four, delete the orphaned
banner, stop.
*Buys:* honesty for ~5 lines of work.

**What decides it:** whether §1.5 Angle B is taken. If it is, the GPU half of A
happens anyway and the rest is cheap to finish. If not, B or C.

## `Grouping` in `kernels/core.mojo`

A data type in a file whose docstring says it is "the root of the kernel trait
hierarchy". Used by `groupby` (12), `aggregate` (11), `expr/aggregates` (11),
`distinct` (3), `expr/execution` (3).

**Angle A — to `groupby.mojo`.** Its densest user, and the type is a group-by
output. *Costs:* `aggregate.mojo` and `distinct.mojo` gain an import of
`groupby`, which may invert a dependency — check before committing.
**Angle B — its own `kernels/grouping.mojo`.** A leaf module five files import,
no direction question. *Costs:* one more small file.
**Angle C — leave, widen `core.mojo`'s docstring** to "kernel trait root and the
vocabulary types kernels share".

**What decides it:** the import direction under A. If `aggregate → groupby` is an
up-edge, B.

## The unambiguous ones

**Done 2026-08-16:** `marrow/tests/test_parquet.mojo` → `marrow/parquet/tests/`
(imports re-depthed `..` → `...`); `load_word_le` → `parquet/codecs.mojo`; the
`utils.mojo` dispatch-family banner moved into `DynType` in `dtypes.mojo` where
the adapters live; the `AggFunction` banner moved from the end of
`kernels/aggregate.mojo` up to the trait at `:854`; four `# Main` banners
deleted.

**Still open:**

- `kernels/tests/test_execution.mojo` + `test_execution_gpu.mojo` → `marrow/tests/`.
  They import `...execution` — three levels up and out of their own package,
  left behind by the A4 refactor.
- The two `execution.mojo` files and the two `struct Filter` are namespaced and
  correct; the only option is a rename, and neither name is wrong. **Recommend
  leaving both** and noting them in `architecture.md`.

---

# Part 5 — Deletion and over-abstraction

## §3 — single-use functions

**Angle A — inline all of them.** 20 functions, each with one caller.
*Costs:* several genuinely name a step (`_radix_sort_indices`, `_floor_civil`,
`_promote_operands`) and read better as names. Inlining them is a net loss.

**Angle B — inline only where the split earns nothing. (Recommended.)**
`_format_ns` (`testing/bench.mojo:457`, defined *after* its only caller at
`:344`); the three-helper chain in `string.mojo:523/536/546`, all called ~150
lines away; the four single-use helpers in `hashing.mojo`. Leave the rest.

**Angle C — leave all, and treat the list as documentation** of where the file
boundaries are soft.

**Two items need a decision regardless of angle:**
- `_rapidhash_bool_masked` (`hashing.mojo:206`) has **no call site and no
  pointer-taking site**. Either dead, or a gap in masked-hash coverage for
  boolean columns. Check against `test_hashing.mojo` before deleting.
- `read_metadata` (`parquet/reader.mojo:2384`) is public API whose only caller is
  `parquet/tests/test_metadata.mojo:39`. Options: delete and have the test go
  through the real path; keep and document it as a public entry point; or find
  out whether `ParquetFile` should be using it.

## §4 — over-abstraction

**`trait Join` (`join.mojo:384`)** — three abstract methods, one conformer
(`HashJoin[hasher]`, `:440`), one commented out (`SortMergeJoin`, `:829`). Its
own docstring says "operators use concrete types directly".
- *A — delete it*, make `HashJoin` free-standing. Nothing depends on the
  abstraction; it costs a reader one indirection.
- *B — keep it* as a written-down intent that a second algorithm is expected.
  The commented-out `SortMergeJoin` says someone meant it.
- *What decides it:* whether sort-merge join is on the roadmap. `backlog.md` §0.5
  puts join completeness (**M3.1/M3.2**) in **Won't — post-M1 by construction**.
  On that evidence, A.

**`trait ByteSource` (`parquet/source.mojo:20`)** — one conformer (`MappedFile`).
- *A — delete.* *B — keep*: a second source (in-memory, object store) is
  plausible and the engine roadmap points that way.
- *What decides it:* whether remote/object-store Parquet is in scope. Unlike
  `Join`, this one has a plausible second implementation. **Lean keep.**

**`trait WindowKernel` (`values.mojo:2199`) and `trait ListValue` (`:2539`)** —
one conformer each.
- **Do not touch either.** Windows are AOT-only, which §0 already flags as
  violating the "one engine, two drivers" invariant (**M2.3**); deleting the trait
  now would have to be undone when that is fixed. `ListValue` is the same story
  for list ops in the runtime lane.

**The `AggFunction` split** — trait in `kernels/aggregate.mojo:854`, all four
conformers in `expr/aggregates.mojo`, consumed inside kernels by
`GroupBy.apply[F: AggFunction]` (`groupby.mojo:444`).
- *A — leave.* The layering is correct: kernels owns the protocol, expr populates
  it, and a kernel consumes it. Only the stranded banner is wrong (see Part 4).
- *B — move the trait to `expr/aggregates.mojo`.* Would invert the dependency —
  `groupby.mojo` would import `expr`, breaking the "`marrow/kernels` is a verified
  DAG with no up-edges into `expr`" property recorded in `backlog.md` §8. **Ruled
  out.**

## §1.10 and §1.11 — no proposal

**§1.10 (`parquet/format.mojo`'s 12 Thrift read loops):** leave. A wire-format
decoder where per-field explicitness is the point; both reference
implementations generate the equivalent. The one worth acting on is the
*inconsistency* — `read` is not a trait requirement while `write` is
(`ThriftWritable`, `:286`), and five of the twelve structs do not conform at all.
Either add `read` to the trait and conform all twelve, or drop `ThriftWritable`
and let both be conventions. Currently it is half of each.

**§1.11 (the four `_dispatch` narrowing adapters):** structurally forced — a
closure type cannot be generic over its own trait bound. Per §1.4's findings,
`slice`'s seven `Self(...)` bodies and `validity()`'s seven copies should move
into this section, since they fail for the same class of reason.

---

# Decision table

**Landed 2026-08-16** and removed from the table: the `arrays.mojo` slice comment
(§1.4/3.4), the Python `_Tabular` base (§1.13), the `bench_bitmap` duplicate
(§1.12 outlier), the `test_parquet.mojo` move, the `load_word_le` move, and all
six orphaned banners. See `CHANGELOG.md`.

| Item | Recommended | Gate | Independent? |
|---|---|---|---|
| §1.12 temp paths | suite-wide `mkstemp`, ~15 sites, or leave | tests | yes |
| Part 4 — exec-test move | `kernels/tests/` → `marrow/tests/` | import check | yes |
| §4 `trait Join` | A — delete | judgement | yes |
| §3 single-use | B — selective | judgement | yes |
| §1.7 builder validity | B — `Bitmap.extend_validity` | tests | yes |
| §1.8 c_data release | A — slot helpers | tests | yes |
| §1.5 alloc preamble | B — `ExecContext.alloc_*` | tests | **pairs with §2.3** |
| `utils.mojo` split | A if §1.5 B is taken, else B | import check | **pairs with §1.5** |
| §1.6 filter loop | benchmark first; C as fallback | `bench_filter` | yes |
| §1.3 interval pairing | B now, A if the spike passes | spike + size | yes |
| §1.2 `*Column` | **A regardless**; B if size-neutral | size | yes |
| §1.1 validity delegation | spike first; A if it fails | spike + size | **spike shared with §1.1/§1.2/§1.3** |
| §1.12 fixtures (A/B) | open — depends on shipped-path tolerance | — | yes |
| §1.10 Thrift | consistency only, not dedup | — | yes |

**Two spikes gate five items and should run first:**

1. **Sub-trait default methods on `Value`** — does a default returning
   `Self.State` recurse? Settles §1.1 and informs §1.2.
2. **`Self.K.Interval` as an associated type on a kernel parameter** — settles
   §1.3 Angle A.

Both are ~30 lines and answer questions no amount of reading resolves.

**Everything in the top eight rows of the table is independent, needs only
tests, and could land in any order.**

# Simplification wave — verified plan

**Status: proposal.** Every line below was verified against the code at
`8cc9723` on 2026-08-17 by seven parallel experiments, three of which compiled
and ran tests in isolated worktrees. **On approval this document is folded into
`docs/backlog.md` §5 and §0 and deleted** — the backlog is the only place that
says what is open, and a second status document is exactly the drift the backlog
header warns about.

The goal of the wave is unchanged: strong abstraction foundations, and the
duplicated and messy parts removed, before feature work resumes.

---

## 0. What the experiments changed

The backlog's §5 schedule does not survive verification intact. Five of its
thirteen cards had a false or vacuous premise; two collide with designs §7
already rejected; one is settled and should be deleted.

| Card | Backlog says | Verified |
|---|---|---|
| **S2** | "a date predicate prunes nothing in the fused lane" | **Vacuous** — the fused lane cannot *express* a date predicate. The real defect is `bound_column`: a fused join on a temporal column **raises**. |
| **S4** | 15 structs, "four casts never see `ctx`" | **Six** never see `ctx`, and there are **two** behavioural defects, not one — `TemporalCast` drops `safe` too. An existing test encodes the bug. |
| **S6** | "deletion is the only remaining option" | **Premise false.** `write_repr_to` is a stdlib `Writable` member whose default is field reflection, not `write_to`. Deleting all 26 is a user-visible Python regression. |
| **S8** | "no new cycle" | **False.** It closes `execution → buffers → views → execution`. The acyclic alternative is the design §7 rejected. **Needs your decision.** |
| **S11** | two placement moves | The `equal_any` half **creates a `numeric ↔ compare` cycle**. The `Grouping` half is good and better-motivated than the card argues. |
| **S13** | 17 submodules, 8 re-exported | **19 named, 7 re-exported**, and `interval.mojo` is missing from the docstring entirely. |
| **S14** | ~15 paths, 4 files | **119 paths, 14 files**, and the fix is not `mkstemp` — Mojo tests cannot reach pytest's `tmp_path`. |
| **S17** | scheduled last | **Already landed** at `0e552a7`. Delete the row. |
| **§5.1 bullet 1** | "needs a spike … expect no" | **Settled: no.** Measured. Delete the bullet. |

Three items were promoted **into** the wave that the backlog does not schedule,
and one was demoted out of it.

---

## 1. Group A — correctness, and each already has a failing test

These come first because M1's gate is "results cross-checked against DuckDB".
Every card here now opens with a test that has been **written, run, and watched
fail** — the backlog's own requirement, previously unmet.

### A1 — `cast()` drops arguments on the way to the kernel (was S4) — **M**

Four failing cases, run at `130.66 s`:

```
test_s4_decimal128_float_overflow_safe_does_not_raise FAILED
test_s4_decimal32_rescale_overflow_safe_does_not_raise FAILED
test_s4_timestamp_downscale_truncates_under_safe      FAILED
test_s4_timestamp_upscale_overflow_under_safe         FAILED
4 failed, 57 passed in 130.66s
```

A fifth case asserts the defect *positively* and passes: the value is destroyed,
not merely unchecked.

**Verified shape.** 15 structs across six signatures, exactly as the card says.
`(array,to,safe,ctx)`×5, `(array,to)`×5, `(array,to,ctx)`×2, `(array,to,safe)`×1,
`(array,safe,ctx)`×1, `(array,ctx)`×1.

**Corrections.** *Six* structs have no `ctx` at all — `NumToString`,
`BoolToString`, `BinaryLikeCast`, `FixedSizeBinaryCast`, `NullCast`,
`DecimalCast` — not four. `safe` is dropped on seven arms, of which five are
total conversions (a signature wart) and **two are real defects**:

- `DecimalCast` (`cast.mojo:1020`) — its own docstring concedes it at `:751`,
  *"Arithmetic is unchecked (wrapping / truncating)"*.
- `TemporalCast` (`cast.mojo:1028`) — **not on the card.** ms→s silently
  truncates; s→ns silently overflows int64. Arrow C++ raises on both
  (`ShiftTime`, `scalar_cast_temporal.cc:45-100`).

**Landmine.** `test_timestamp_unit_downscale` (`test_cast.mojo:242`) asserts the
buggy truncation under the default `safe=True`. The suite encodes the bug; fixing
S4 makes that test wrong, and it must be rewritten to `safe=False` in the same
commit.

**Design.** `trait CastKernel(Kernel)` with one abstract
`dispatch(array, to, safe=True, ctx=serial())`. `BinaryKernel`
(`kernels/numeric.mojo:58`) is the precedent — an abstract `dispatch` with
defaulted arguments on a family trait, proven to work. Five structs change only
their conformance clause; ten widen `dispatch`. Arms that cannot use an argument
name it and say why in the docstring; `NumToBool`/`StringToBool` gain `to` and
should *assert* on it, turning a positional invariant into a check.

**Constraints that bite, and are not visible from the card.**

- **`ctx.stripe` bodies may not raise** (§0). `TemporalCast._scale` goes through
  `apply(..., ctx)` → `_cpu_striped`, so the overflow check **cannot** live in
  the lane. It must follow `NumericCast.apply`'s existing serial-checked
  precedent, whose comment already states the reason.
- **Do not write a shared `_cast_dispatch[K: CastKernel]`.** A closure type
  cannot be generic over its own trait bound; the narrowing adapter inlines into
  every arm — measured at +662,740 bytes. `cast()` stays a hand-written ladder.
- **Instantiation doubling is the one real size risk.** If `DecimalCast` branches
  runtime `safe` into two comptime instantiations the way `NumericCast` does, it
  doubles bodies already instantiated over ~256 `FromN × ToN` pairs. Mitigation
  is arrow-rs's: one body with a runtime branch, elided by a comptime precision
  proof (`input_precision + delta_scale <= output_precision`) — the direct
  analogue of marrow's existing `NumericCast.needs_check[In, Out]()`.

**Polarity warning.** arrow-rs's `safe=true` means *null on failure*; Arrow
C++/PyArrow's means *raise*. Marrow follows C++/PyArrow, which invariant 3
requires. Port arrow-rs's structure, not its semantics.

**Size.** ~80 lines for the trait, ~200 for the kernel bodies, ~300 total.
The card's "M" is right, but for the bodies, not the trait.

**Verification trap.** The gate may be structurally unable to catch a regression
here — §0 records that stubbing `cast_array` out of the expression layer left a
gate binary byte-identical. **Before believing a 0.00% delta, run
`nm -C <gate> | grep -i 'DecimalCast\|TemporalCast'`.** If the symbols are
absent, say so in the commit message instead of showing a green checkmark.

### A2 — `is_in` never verifies key equality — **NEW, S, not in the backlog**

`membership.mojo:61-78` calls `probe_hashes` and stops. Its own docstring says
*"`true` where the value's **hash** is present in the set"*. So `is_in` has a
genuine rapidhash-64 false-positive rate that `join` does not — `join` verifies
through `EqKernel` in `SwissHashTable.probe` (`hashtable.mojo:619-626`).

A wrong answer with no error, and `IsIn` is a fused expression node, so it is
reachable from both lanes.

**Decided (D3): verify, matching `join`.** Reuse the existing seven-line
`Take` + `EqKernel` block from `SwissHashTable.probe` rather than inventing a
mechanism. **Must land before B2's optional half**, which deletes the method
those lines live in — see D3 for the sequencing.

### A3 — the fused lane cannot bind a temporal column (was S2) — **S**

Three failing cases, then green after the fix, with no regressions:

```
before:  3 failed, 345 passed, 24 skipped in 415.35s
after:   349 passed, 24 skipped in 323.26s
```

**The card's premise is vacuous.** "A date predicate prunes nothing in the fused
lane" is true, but the fused lane **has no temporal comparison operator and no
temporal literal** — comparison lives only on `NumericValue` (`values.mojo:676-691`)
and `StringValue` (`:1681-1696`), `TemporalValue` (`:2514`) has none, and
`builders.mojo:63-77` has no temporal `lit`. So `TemporalColumn.prune` is
unreachable; adding it alone buys zero pruning.

**The reachable defect is `bound_column`, and it is worse than pruning.** With
the `-1` default, `Relation.join` (`relations.mojo:454`/`:463`) **raises**
*"join: key must be a column reference, got a computed expression"* — a fused
plan cannot join on a date or timestamp column at all — and `Relation.aggregate`
(`:355`) names the key `key0`. Neither is on the card.

**Measured asymmetry**, on a real 4-row-group date32 file, predicate `d > 2500`:
runtime lane keeps **2 of 4**, fused lane keeps **4 of 4**. So the lanes do
disagree; the card is right about that much.

**Not an M1 wall-clock item.** M1.5 signs off through the lazy/Python frontend,
which builds `col("name")` with no dtype → the **runtime lane**, which already
prunes. Seven of the 43 ClickBench queries do range-filter `EventDate`, but they
reach it through the lane that works. This is an M1.6 (AOT DSL) item.

**List columns: `bound_column` yes, `prune` no.** Parquet declares LIST and MAP
column orders **undefined** (`parquet.thrift:1101-1102`) and statistics hang off
leaf `ColumnMetaData` only. Arrow C++ guards on `is_leaf()`
(`file_parquet.cc:175-178`), arrow-rs on `is_nested()` (`arrow/mod.rs:462-476`),
DataFusion says outright *"PruningPredicate does not support pruning on nested
fields yet"*. `Interval.unknown()` is the **correct** answer for a list, not a
placeholder — record why, with the citations. `bound_column` is a syntactic
question and should be answered: the hash kernel accepts list keys
(`hashing.mojo:204`), so a grouped list column is named `key0` for no reason.

**Size: measured 0 bytes.** `query_streaming` `__text` 1,423,236 before and
after, every other section identical, symbol counts identical.

**Follow-up this card does not contain**, and should be filed separately rather
than smuggled in: a `TemporalLiteral` carrying a `T` *instance* (unlike
`NumericLiteral` — `TemporalColumn` stores only `_name`, so `col(n, timestamp(s, "UTC"))`
discards the dtype instance), a `TemporalCompare` breaker node, and comparison
operators on `TemporalValue`. Also: `TemporalExtract` (`:2590`) has no `prune`,
so `col("ts").year() > lit(2100)` — the one date-ish predicate the fused lane
*can* build — prunes nothing despite year-of being monotone.

**Latent trap for whoever writes the literal.** `Interval._compare_scalar` bails
to "maybe" when `a.type() != b.type()` (`interval.mojo:126-128`), and
`TimestampType` carries unit and tz as runtime fields — so `timestamp[s]` ≠
`timestamp[us]` ≠ `timestamp[us, UTC]`, and a literal with the wrong unit
silently prunes nothing. Sound, but it is the exact silent-disable failure M1.0
fixed.

### A4 — `RecordBatch`/`Table` validate nothing — **NEW, S, promoted from §8**

Verified: `RecordBatch.__init__` (`tabular.mojo:38`) and `Table.__init__`
(`:437`) check **none** of the three invariants — field count vs column count,
column *i*'s dtype vs field *i*'s dtype, equal lengths. The free
`record_batch(columns, names=)` (`:404`) checks only count.

**Reachable from the public Python API**, which §8 does not say.
`marrow.record_batch(data, schema=...)` routes to `_build_from_arrays_with_schema`
(`python/bindings/tabular.mojo:105-113`), which constructs with zero checks.
Measured against the pinned pyarrow, all three malformed cases raise or coerce
there and are accepted silently here — a live violation of invariant 3.

**Consequence traced to an out-of-bounds read**, not just a wrong answer.
`to_struct_array()` (`:373-386`) takes `length` from column 0 alone;
`BinaryLikeArray.slice` (`arrays.mojo:929-944`) is non-raising, does not clamp,
and *eagerly recounts nulls* — so a short column reads past its buffer inside
`slice` itself, before any kernel runs.

**Fix follows the tree's own recorded precedent** (`arrays.mojo:225-232`):
`debug_assert` in both constructors (free at `-O3`, live under `-D ASSERT=all`,
which is how every test runs) plus a raising `validate()` called at the **trust
boundaries** — the three Python builders and the free `record_batch()`, where a
*user* supplies the pairing. **Not** a raising `__init__`: that cascades through
six non-raising callers, which is the cascade `arrays.mojo:225-232` explicitly
refuses and the backlog decided against on 2026-08-16.

No existing test depends on the current permissiveness — all 13 `RecordBatch(...)`
sites in `test_tabular.mojo` are well-formed. ~40–60 lines. Measure
`query_streaming`; expect ≈0.

---

## 2. Group B — foundations

### B1 — move the aggregate catalog down to `kernels/` — **S–M**

The strongest structural item in the wave, and better than §5.1 rates it.

**The "expression" half of `marrow/expr/aggregates.mojo` is empty.** Not one of
its 555 lines references `Value`, `DynValue`, `BoxedValue`, `AggExpr`, a
`Relation` or a `Processor`; the file has zero intra-`expr` imports. The whole
file is misfiled. **507 of 555 lines move into `kernels/` with zero new imports**
and create no `kernels → expr` edge — every symbol they need is already defined
in or imported by `kernels/aggregate.mojo`.

**`kernels` already needs the catalog.** `GroupBy.apply[F: AggFunction]`
(`groupby.mojo:447`) is a public kernel API parameterized on a kernels trait
whose only conformers live in `expr/` — so a kernels user cannot call it without
importing the expression layer. That is why **three kernel test files** import
`...expr.aggregates`. §8's "`marrow/kernels` is a verified DAG with no up-edges
into `expr`" is true for the 20 production modules and **false once tests count**;
this move makes it true tree-wide.

**Precondition §5.1 omits.** `FoldedAggregates.grouped`/`whole`/`_named` (48
lines) return `RecordBatch` and import `..tabular`. Moving them verbatim trades
`tabular ⟷ expr.aggregates` for `tabular ⟷ kernels` — strictly worse, since it
relocates the cycle into the size-gated layer. Strip `RecordBatch` out of those
three methods **first**; it costs nothing, because **both callers in
`tabular.mojo` already discard the schema `FoldedAggregates` builds** and rebuild
their own labels (`:308` rebuilds every field; `:340` uses only `.dtype`).

**Placement.** `FoldedAggregates` goes in `kernels/groupby.mojo`, not
`kernels/aggregate.mojo` — the latter closes an `aggregate ⟷ groupby` cycle for
nothing.

**Two traps.** Do not add the catalog names to `kernels/__init__.mojo` — they
collide conceptually with `values.mojo:2302`'s `comptime Sum = Reduction[SumKernel, _]`
and worsen the inconsistency C4 exists to fix. Do not merge `AggFold` into
`AggFunc` while you are in there — the file records that measuring **+3.2 MB
(+24%)**.

**Independent of M1.3's open decision** (eager shortcuts vs `execute(plan)`),
which stays open either way. §5.1's "worth doing before M1.3" is right for a
different reason than it gives: doing it first means M1.3 does not have to decide
anything to be unblocked.

**Size: predicted 0 bytes** — no new call edge; DCE works on the call graph and
every moved function keeps its exact callers and callees. Measure
`query_streaming_agg` (the only gate exercising `resolve_agg`) and
`query_streaming_agg_fused`. **Any non-zero delta means the reachability model is
wrong** and must be understood, not absorbed — and explicitly must not be
laundered through a re-baseline.

### B2 — enforce `SwissHashTable`'s build→probe lifecycle — **XS**

Verified, with two corrections to §8. It is **not reachable in-tree**: all six
instantiation sites (§8 says "two clients"; it is four modules / six sites)
either only insert or always build first, and `JoinProcessor` guards on an
`Optional`. And "reads them unconditionally" is overstated — `probe_hashes` reads
`_offsets` only on a `_find_slot` hit, so probing a *fresh* table is harmless.

But when it does fire it is **memory-unsafe, not merely wrong**:
`alloc_uninit(0)` returns a valid non-null zero-size pointer (measured), and
`Buffer.unsafe_get` has no bounds check even under `-D ASSERT=all` — `_check_bounds`
exists but is called only from `__getitem__`/`__setitem__`. Garbage offsets flow
into a `Take.apply` wild gather.

**Unstated second variant a fix must cover:** build → insert → probe. `_offsets`
is sized from the earlier build, so a newly-inserted bucket id indexes past it.

**Fix:** one `var _built_buckets: Int` (init `-1`), set at the end of
`build_hashes`, plus a `debug_assert` at the **top** of `probe_hashes` — exact
for both variants, zero cost in the inner loop, compiled out at `-O3`. ~10 lines,
one file, the pattern `arrays.mojo:225-232` already documents.

Separately available and worth ~55 lines: lifting key-equality verification out
of `hashtable.mojo` by deleting `probe()` and inlining its seven lines into its
two callers in `join.mojo`. That removes the `numeric` and `filter` inbound edges
and leaves `SwissHashTable` as slots + CSR + hashing. **No comparator injection
is needed, so neither §0 closure limit applies.** It touches ~20 test/bench call
sites, so it is optional and separable from the assert.

---

## 3. Group C — free subtractions, verified and re-sized

Land in any order except where noted. Each was re-counted; the backlog's numbers
were wrong more often than right.

| ID | Item | Verified | Size |
|---|---|---|---|
| **C1** | **`Partition.original_row` is dead** (`partition.mojo:112`). Zero callers in `marrow/`, tests, `benchmarks/`, `python/`; the only other hit is a docstring using a different identifier. `Partition` is not re-exported. | exact | XS |
| **C2** | **Move `kernels/tests/test_execution{,_gpu}.mojo` → `marrow/tests/`.** Both import `...execution` and nothing from `kernels`. Needs 2 line edits (`...execution` → `..execution`). **Case-name collision checked: none** — the suite has zero duplicate case names today. Do before C5 touches `kernels/tests/`. | exact | XS |
| **C3** | **`c_data.mojo` release-slot helpers.** Three structs confirmed — the third is `CArrowArrayStream` (`:1563`/`:1570`); `CArrowDeviceArray` embeds a `CArrowArray` and is not a fourth. Six copies exactly, bodies byte-identical. A 7th instance at `:666` is a dictionary null-test — **do not fold it in**. Trait form is ruled out (a trait default body cannot name `self.release`), so: two module-private free functions taking the pointer, six surviving one-liners, one copy of the arithmetic. Stays inside the `unsafe_ptr` boundary. | exact | S |
| **C4** | **Make `kernels/__init__` say what it does.** Actual: **7** submodules re-exported, not 8; docstring names **19**, not 17; **`interval.mojo` is absent from the docstring entirely**. `mk.concat` failing was confirmed by compiling (`package 'kernels' has no declaration 'concat'`). The re-exported set is exactly "what the Python bindings surface" — a real principle, just unwritten. **Document the boundary; do not export the rest.** The sole consumer (`python/bindings/compute.mojo`) already bypasses the package for 4 imports, which is the evidence. | understated | S |
| **C5** | **`Grouping` → `kernels/grouping.mojo`** (the good half of S11). `core.mojo` is 70 lines imported by **14** modules, and its `Int32Array` import exists *solely* for `Grouping`. Moving it leaves a pure trait module and lets `groupby.mojo` and `distinct.mojo` drop `.core` entirely. A new leaf is correct — `groupby.mojo` would close `distinct → groupby → aggregate → distinct`. **Not cosmetic.** | verified | S |
| **C6** | **`Bitmap.extend_validity`.** Five sites confirmed byte-identical (`builders.mojo:699, 828, 1025, 1232, 1373` — drifted +3 from the card). The block is **9 lines, not 11**, and `reserve` is **not** part of it, so "reserve-then-propagate" is inaccurate. §0's trait-cannot-name-a-field means the helper takes explicit arguments (~5 of them), so the net saving is **~25 lines, not 45**. "Not a hot path" is the wrong justification — it is on the concat and group-by key paths — but the conclusion holds, because it is O(1) per bulk append. | verified, re-sized | S |
| **C7** | **Temp paths in tests** (was S14). **119 occurrences across 14 files**, not ~15 across 4 — the card misses the two largest by an order of magnitude (`test_writer.mojo` 43, `test_reader.mojo` 26) and three files outside `parquet/tests/`. **The fix is not `mkstemp`**: these are Mojo modules and `conftest.py` passes the driver no temp dir. Needs a `tmp_path(name)` helper in `marrow/utils/testing.mojo` plus a session temp dir exported from `conftest.py` as an env var and cleaned in `pytest_sessionfinish`. Two sites are not plain literals, and `test_buffers.mojo:830` deliberately names a **non-existent** file to assert `mmap_file` raises — it needs a path inside the temp dir that is never created. Fully disjoint from every source item; can run on a parallel branch. | understated ~8× | M |

### C8 — `write_repr_to` (was S6): **rescope, do not delete all 26** — **S**

The card's premise is false, and this is the single most important correction in
the batch. `write_repr_to` **is** a member of stdlib `trait Writable`, and its
default is **reflection over fields**, not `write_to`. Verified by compiling a
struct that defines only `write_to`:

```
(1, 3)                  # String(p)
P(x=Int(1), y=Int(3))   # repr(p)  <-- reflection, NOT write_to
```

So deletion does not unify repr; it turns repr into a field dump.

- **22 of the 26 are verbatim `self.write_to(writer)`** and can go.
- **Three carry real distinct behaviour**: `StringScalar` (`scalars.mojo:327`,
  adds the quotes), `Field` (`dtypes.mojo:519`, `Field(name=…, nullable=…)`),
  and `DynScalar` (`:862`, the dispatch that delivers both). `NullScalar` is
  behaviourally identical to its `write_to` and can go.
- **Three Python-bound types regress**: `Field`, `DataType` and `ExecContext`
  register no `__repr__` and fall through to the stdlib `_tp_repr_wrapper`,
  which calls `write_repr_to`.
- **Possible compile failure, not just output drift**: the reflection default
  requires every field to be `Writable`, which is unverified for `ExecContext`
  (`Optional[DeviceContext]`) and `Bitmap`/`Buffer` (`ArcPointer[Allocation]`).
  `precompile`, not pytest, is the gate here.

If the goal is genuinely "one repr", the honest change is to register explicit
`__repr__` for the four unbound Python types *first*.

---

## 4. Decisions — **taken 2026-08-17**

All three resolved. Each is recorded with its alternative, because each reverses
or upholds something already written down.

### D1 — S8's allocation helper — **DECIDED: `buffers.mojo`, and amend §7**

The card's "no new cycle" reasoning is wrong. `marrow/execution.mojo` currently
imports **nothing** from marrow — it is a leaf. Adding `Buffer`/`Bitmap` to
`ExecContext` closes `execution → buffers → views → execution`. "views already
imports both" is irrelevant; the new edge is on `execution`.

The acyclic alternative is `Buffer.alloc_for[T](ctx, n)` in `buffers.mojo`
(`buffers → execution`, and `execution` is a leaf) — **which §7 explicitly
rejected** for pointing the tree's lowest-level module at device policy.

**Decision: `Buffer.alloc_for[T](ctx, n)` in `buffers.mojo`.** The §7 objection —
that it points the tree's lowest-level module at device policy — is aesthetic;
the cycle is structural, and §8 tracks cycles as debt. **§7's rejection row must
be amended, not silently contradicted**: rewrite it to record that the placement
was re-opened on 2026-08-17 because the `ExecContext` alternative was found to
close `execution → buffers → views → execution`, which the original row did not
know. A rejected-designs list that is quietly overruled is worse than no list.

Note the helper takes `ctx` as a parameter, so `buffers.mojo` names the policy
type without owning the policy — which is the narrowest form of the thing §7
objected to.

Two things that ride along either way: the 10 sites are **not one preamble** (7
buffer / 3 bitmap, and two use `alloc_zeroed`), so it is two helpers or a flag;
and **the GPU arm of the two `alloc_zeroed` sites calls `alloc_device`, which
does not zero** — a latent bug centralising would fix. This is **not** the
`_arith[K]` shape: `alloc_uninit[T]`/`alloc_device[T]` are already instantiated
at exactly the same set of `T`s.

### D2 — `equal_any`'s move — **DECIDED: hold the card**

`numeric.mojo:49` imports `StringEqKernel` solely for `equal_any`, so the move
does delete the `numeric → string` edge. But `EqKernel.apply(StructArray, …)`
(`:617-645`) calls `equal_any` twice and **cannot move** — Mojo cannot add a
static method to `EqKernel` from another module. Result: a `numeric ↔ compare`
cycle replacing an acyclic edge.

**Decision: hold it.** The move makes the dependency graph worse, not better, and
S1 has just finished removing cycles. **`equal_any` stays in `numeric.mojo`;
`kernels/compare.mojo` is not created.** Only the `Grouping` half of S11 survives,
as C5.

This is a *decision*, not a deferral — it belongs in §7's rejected-designs list
with the reason (`EqKernel.apply(StructArray, …)` calls `equal_any` and cannot
move, because Mojo cannot add a static method to `EqKernel` from another module),
so the duplication audits stop pulling on it. Revisit only if the `StructArray`
row-equality relocates for an independent reason.

### D3 — `is_in`'s hash-only semantics — **DECIDED: verify, matching `join`**

**Decision: add the `EqKernel` verification.** `is_in` is reachable from both
expression lanes and is a silent wrong-answer path; a ~2⁻⁶⁴ error rate is still a
wrong answer with no error, and Arrow's `is_in` is exact. Matching `join`'s
existing verification also means one behaviour to reason about across the two
`probe_hashes` consumers rather than two.

Cost is a `Take` + `EqKernel` on probe hits only — the same shape
`SwissHashTable.probe` already pays (`hashtable.mojo:619-626`), so the
implementation is the existing seven lines rather than a new mechanism.

**Sequencing with B2.** B2's optional half deletes `probe()` and inlines those
seven lines into its two `join.mojo` callers. If both land, `is_in` becomes a
third caller of the same inlined block — so either do A2 first and let B2's
inlining cover three call sites instead of two, or keep `probe()` and drop B2's
optional half. **Do not do them in the other order**, or A2 will be written
against a method B2 is about to delete.

**Benchmark gap worth noting:** `membership` has no bench (backlog §5's
test-coverage list says so), and per §0 *"an operator with no benchmark has no
performance"* — so the verification's cost will be unmeasurable unless one is
written. That is a reason to write one, not a reason to defer the correctness fix.

---

## 5. Demoted, and settled

**The `DynRelation` planner split moves out of the wave and into M1.1.** I
proposed it as a pre-feature foundation; the evidence says otherwise. It is real
— of `DynRelation`'s 445 lines, **74% is plan construction/binding and only 18%
erasure mechanics** — but extracting it is a 410-line two-file move with zero
behaviour change and an expected **0-byte** size delta, and nothing in M1.2–M1.6
unblocks (`python/bindings/` has no contact with `relations.mojo`). On its own it
is tidying.

It should be M1.1's *first commit*, because M1.1 needs the binder callable from
`optimize.mojo` or it will duplicate it. Recommended shape is **Option B** — keep
the fluent surface (invariant 3, and M1.3 is specified over it), extract only the
binder. **Option A** (verbs → free functions) costs ~174 call-site rewrites and
partially reverts commit `53f7be3`, which moved `execute` *onto* the box
deliberately.

Two constraints that dominate M1.1 and are recorded nowhere:

1. **`DynRelation` exposes no children and cannot easily gain them.** A recursive
   `DynRelation → DynRelation` pass needs `children()`/`with_children()`, and
   both put `DynRelation` inside its own trampoline field type — the exact
   recursive-struct rejection `Relation.with_predicate`'s docstring documents.
   **M1.1's real first task is a spike on breaking that recursion.** A
   `downcast`-based escape does not work either: a downcast cannot name
   `ParquetScan`'s comptime `leaves` parameter, so it silently rebuilds a narrow
   scan as the full-ladder one.
2. **A rewrite rule that *constructs* nodes breaches invariant 1.**
   `DynRelation.__init__[T]` pins `T.to_processor`, so instantiating it for
   `Aggregate` makes `AggregateProcessor` + `groupby` + `hashing` reachable. If a
   pushdown rule rebuilds an `Aggregate`/`Join`/`Sort`, **linking the optimizer
   links every operator into every plan.** The existing `with_predicate` rewrite
   is not ad-hoc — it is the only DCE-safe shape available, and that property
   must survive into `optimize.mojo`.

Also: the 0-row dtype probe costs a measured **+16,528 bytes** (`b16a6aa`), which
is why six of the gate binaries hand-build nodes. An optimizer that re-binds
after every rewrite makes them all pay it.

**§5.1's `expr/values.mojo` validity/state delegation is settled as "no" —
delete the bullet.** Measured, not argued. See §6.

---

## 6. New §0 entries — measured, not inferred

**A trait default is size-free; the boilerplate is the cost.** Hoisting the 22
verbatim `referenced_columns` copies to a `UnaryValue` default compiles clean
(345 passed / 24 skipped) and measures **+0 bytes on `query_streaming_agg_fused`,
+128 on `query_exprs`**. The +128 is *not* the trait mechanism — it is
+48 bytes per boxed instantiation of a reached node, caused by the forced
`.copy()`. **But the source cost is +128/−70 lines: the dedup makes the file 58
lines longer.** §0's argument holds; the item is still not worth doing.

Three compiler limits, each a hard error, none previously recorded in this form:

1. **A trait default body cannot name a field of `Self`** —
   `error: '_Self' value has no attribute 'a'`. The operand must be reached
   through a by-value accessor requirement. *This also rules out C3's trait
   form outright, since those bodies name `self.release`.*
2. **A ref-returning accessor is not expressible at trait level** —
   `cannot return 'self's origin, because it might expand to a RegisterPassable
   type`. Same origin-widening wall as `arrays.validity()`. Hence the `.copy()`.
3. **A sub-trait default returning `Self.State` does not recurse — it fails to
   reduce**: `cannot implicitly convert '_Self.Operand.State' value to
   '_Self.State'`. `rebind` cannot bridge it, because `rebind` takes its argument
   by implicitly-copyable borrow and `State` is not. **This is the §5.1 question,
   answered: impossible, not merely costly.**

Corollary: **a struct parameter does not satisfy an associated-type requirement**
(`struct Neg[A: Leafy](UnaryDirect)` → `required member 'A' is not specified`).
The trait needs a differently-named member the struct binds explicitly.

---

## 7. Stale facts to fix while landing

- **`baseline.json` is stale-high.** `query_streaming` records 1,484,652;
  HEAD measures **1,423,236** (−4.1%). `query_streaming_agg_fused` records
  1,417,476; HEAD measures **1,371,312** (−46,164 — exactly S5's `DynArray.slice`
  saving). The baseline was reset at `0e552a7` and S5 landed after it, so
  `check_gate.py` currently reads green for the wrong reason.
- **CLAUDE.md's `marrow/testing/` path does not exist.** `TestSuite` and
  `Benchmark` live in `marrow/utils/testing.mojo`.
- **§3 M1.1 cites `relations.mojo:273` for `referenced_columns()` — it does not
  exist on `DynRelation` at all**, only on `Value`/`DynValue`/`BoxedValue`. Its
  claimed "consumer" therefore is not one.
- **§8's module graph overstates `relations`' edges by two.** Four imports in
  `relations.mojo` are dead: `Interval` (:50), `PruneStats` (:51), `DynArray`
  (:52), `DynValue` (:54).
- **§0's "both `cast_array` calls" is stale** — `expr/dynamic.mojo` has six, at
  `:176, :178, :192, :205, :207, :337`.
- Backlog line numbers stale across the board: `values.mojo:2452`/`:2565` →
  `:2561`/`:2674`; defaults `:449`/`:460` → `:419`/`:430`; §8's
  `relations.mojo:291`/`:406-736` → `:153`/`:266-576`; `docs/architecture.md:375-397`
  cites four stale lines.
- **`aggregates.mojo:35` and `:306` say `resolve_agg` lives in
  `marrow.expr.dynamic`.** It is at `:191` of that same file.
- **S16 and S17 rows should be deleted** — S17 landed at `0e552a7`, S16 evaporated
  (the card says so itself).

---

## 8. Order, and the test plan

**Order is driven by file overlap, not by importance.**

1. **C1, C2** — disjoint, minutes each. C2 before C5 touches `kernels/tests/`.
2. **C3** — `c_data.mojo` alone.
3. **C4** — `kernels/__init__.mojo` alone, and it must precede C5: the boundary
   decision governs whether the new module is exported.
4. **C5** — 7 files incl. `core.mojo`.
5. **D1** — `Buffer.alloc_for[T](ctx, n)`, plus the `alloc_zeroed` GPU zeroing
   fix and the §7 amendment. Carries the batch's only size gate; **land alone**
   so a delta is attributable.
6. **C6** — `buffers.mojo` + `builders.mojo`, after D1 (both touch `buffers.mojo`).
7. **C8** — widest and most mechanical; cheapest to rebase, so last.
8. **B1** — adjacent to C5 (both touch `kernels/aggregate.mojo`'s imports).
9. **A1, A3, A4** — correctness group, independent of the above.
10. **A2 → B2**, in that order and not the reverse: A2 reuses the seven lines
    B2's optional half deletes. Landing A2 first lets B2 inline across three call
    sites instead of two.
11. **C7** — fully disjoint (test files + `conftest.py`); parallel branch, any time.
12. **Re-baseline** — after the wave, so the wave's own cost is measured against
    the current baseline first. Note §7: the baseline is *already* stale-low
    relative to HEAD, so this is a correction, not an absorption.

**Test plan.** Four whole-directory selections cover everything. Never narrow —
N files cost about what 1 costs, and a narrow selection reaching
`RapidHashKernel.dispatch` **hangs the compiler**. D1 touches `hashing.mojo` and
C5 touches `core.mojo` (which `distinct.mojo:29` imports), so both land squarely
in that hazard.

```bash
pixi run -e dev precompile                    # after EVERY item, ~15 s
pixi run -e dev pytest marrow/tests           # C2 C3 C6 C7 C8 A4
pixi run -e dev pytest marrow/kernels/tests   # A1 A2 B1 B2 C1 C4 C5 D1
pixi run -e dev pytest marrow/expr/tests      # A3 C7
pixi run -e dev pytest marrow/parquet/tests   # C7 (bulk)
pixi run -e dev pytest python/marrow/tests    # C4 C8 A4
```

**`precompile` is the primary gate for C2, C5 and C8** — all three are
pure-compile risks (a moved symbol, a changed import depth, a deleted trait
member), and it surfaces every error in one ~15 s pass where pytest elaborates
all of marrow before saying anything.

**Size gates**: D1 (`query_arith`, `query_dynvalue`), B1 (`query_streaming_agg`,
`query_streaming_agg_fused`), A1 (`query_streaming_agg_fused`, **plus the `nm`
reachability check**), A4 (`query_streaming`). A3 is measured at 0. Always build
one binary directly and read `size -m` → `Section __text`; never the ~10 min
sweep, never the 16 KB-quantized file size.

---

## 9. Operational note

The three build experiments ran in git worktrees with the 20 GB pixi environment
symlinked in (`ln -s …/marrow/.pixi .pixi`), which works and is what made three
concurrent compile-and-test agents possible. **All three worktrees were created
at `ac4b8a0`, ~600 commits behind the branch head**, and each agent had to
discover and fix that itself. Any future parallel run should pin the base
explicitly.

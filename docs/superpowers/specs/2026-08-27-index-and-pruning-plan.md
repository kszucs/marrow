# Index and pruning — the plan

**Supersedes the recommendation of `2026-08-25-pruning-indexing-findings.md`.**
That document's §1–§5 stand and are the evidence base for this one; its §6–§8
are replaced. Four independent design passes (storage, Mojo mechanics, YAGNI,
extensibility) were run against it on 2026-08-26, and every disputed fact below
was checked against the tree rather than taken from a report.

The order is **index layer first, pruning second** — and before either, the
Python frontend has to move onto `marrow.expr`, because nothing downstream is
measurable until it does.

---

## 0. Evidence base, corrected

Five claims that shaped the earlier design are false. They are recorded here
because each one was believed, acted on, and cost design work.

| claim | reality |
|---|---|
| "`SortingColumn` is in `format.mojo`, read nowhere" | **Absent.** `RowGroup` has three fields and `r.skip(f.type)`s everything else (`marrow/parquet/format.mojo:1227-1253`); `grep sorting_column marrow/` is empty. The writer hardcodes `cix.boundary_order = 0  # UNORDERED` (`marrow/parquet/writer.mojo:758`). **Marrow can observe orderedness in neither direction.** |
| "a token index prunes 63.4% of granules on `URL LIKE '%google%'`" | **Right number, wrong index, and unattachable.** ClickHouse's `SplitByNonAlphaTokenizer::nextInStringLike` clears the accumulator on every unescaped `%` and ends `return !bad_token && !token.empty()` (`ITokenizer.cpp:146-192`), so `'%google%'` yields **zero tokens**. The 63.4% was measured with substring containment — *ngram* semantics. It needs `ngrams(col, n)`. |
| "`hypothesis` / Pando need nothing new — they fit" | ClickHouse **deleted** `hypothesis`. `MergeTreeIndexLegacyHypothesis.h:7` — *"Data insertion and index usage will throw an exception, suggesting to drop the index."* |
| "min/max at 8192-row granularity prunes 57.7%" | **Simulated, not achievable.** `hits_0.parquet` reports `has_column_index: False`, `has_offset_index: False`, `created_by: parquet-cpp 1.5.1-SNAPSHOT`, two row groups of 450,560 and 549,440 rows. At its real boundaries: `CounterID = 62` prunes 1 of 2; `URL LIKE '%google%'` and `Title LIKE '%Google%'` prune **0 of 2**; everything else 0. **The findings doc's 1.04x ceiling stands.** |
| "the Python frontend runs the old lane" | It runs **nothing**. `python/marrow/__init__.py:480` imports `.exprold`; the file is `python/marrow/expr.py`. `PYTHONPATH=python python -c "import marrow"` raises `ModuleNotFoundError`. Broken since `ebd4c4c` (2026-08-24), 9 commits, 28 test files. |

Two facts that survived and now carry the plan:

- **Granularity, not index kind, is the dominant variable.** The same min/max
  summary prunes 57.7% on `CounterID` and 0% on `RefererHash` at identical
  granularity, and 0% on `UserID` at 500K row groups versus 95.1% at 8192 rows.
  Layout and page sizing dominate index choice.
- **Scalar blooms add +0.0%** over min/max on the real compound queries Q40/Q41.
  Standalone they look like 57.7%; conjoined with `CounterID = 62` they add
  nothing, because the queried hashes occur 54,687 and 59,176 times per million
  rows. They are not needle lookups.

Reproduce with `benchmarks/pruning/measure_prunability.py` (Step A5).

### The budget

`query_expr2_streaming` is **1,358,480** bytes of `__text` at
`threshold_pct: 0.5` — **6,792 bytes** for index *and* pruning together
(`benchmarks/binary_size/baseline.json`). Historical comparisons from this tree:
a shared generic helper cost **+115,600**; a `Variant`-based erasure cost
**450,112** on `query_join`; a runtime-lane tag switch cost **+1,807,168**.

### The ranking this plan does not overturn

Projection pushdown measured **3.6x** and row-group windowing **1.6–4.7x**
(`docs/backlog.md:668-671`, `marrow/exprold/execution.mojo:434-447`). Neither
needs a statistic. Pruning's entire verified ceiling on this workload is
**1.04x**. All four design passes, given the same numbers, independently ranked
those two above the whole pruning subsystem. This plan is worth executing for
the architecture and for workloads marrow does not yet have — not because it is
the largest available win.

---

## 1. Step 0 — the Python frontend onto `marrow.expr`

**Why first.** Every number after this is otherwise unmeasurable. The ClickBench
harness, the 28 Python test files and `LazyTable` all sit on `exprold`; a change
to the new lane moves nothing a benchmark can see. This step is also the largest
in the plan, and saying so up front is the point.

### 0.1 — Repair the import (unblocks everything, ~1 line)

`ebd4c4c` renamed the Mojo packages `expr` → `exprold` and updated
`python/marrow/__init__.py:480` to match, but `python/marrow/expr.py` — which
holds `LazyTable`, `memtable`, `read_parquet`, `_HAVE_EXPRESSIONS` — was never
renamed.

Rename `python/marrow/expr.py` → `python/marrow/exprold.py`. This restores
`import marrow` and re-enables 28 test files **on the old lane**, which is the
baseline every later measurement is taken against. Land it alone.

- Verify: `PYTHONPATH=python pixi run -e dev python -c "import marrow"`.
- Then: `pixi run -e dev pytest --python -v` — record the pass count. This
  number is the port's acceptance criterion.
- `python/marrow/tests/test_lazy.py:8` imports `_HAVE_EXPRESSIONS` from
  `marrow.exprold`; it follows the rename.

**A regression test belongs here**: `import marrow` must be exercised by a test
that does not run from the repo root. The breakage survived 9 commits because
`marrow/` (the Mojo source) shadows `python/marrow/` as an implicit namespace
package when CWD is the repo root, so `import marrow` silently succeeds and
resolves to the wrong thing.

### 0.2 — The migration, measured

`expr` is not a near-complete lane needing a few gaps closed. Measured
2026-08-27:

**17 of the 18 value node types `golden/prelude.mojo` imports do not exist in
`expr`** — `BoolToNum`, `Coalesce`, `EndsWith`, `FillNull`, `ILike`, `IsNull`,
`Like`, `Lower`, `NotNull`, `NumToBool`, `NumToString`, `NumericCast`,
`StartsWith`, `StringLength`, `StringToNum`, `Strip`, `Upper`. Only `CaseWhen`
is present.

| | `exprold` | `expr` |
|---|---|---|
| non-test lines | ~7,300 | ~3,700 |
| params / CLI | 564 — `ParamCell`, `ParamDecl`, registry, `PathSpec`, `parse_params`, `render_usage`, `render_describe` | 159 — `Param[T]` alone |
| aggregates | 554, incl. `FoldedAggregates`, distinct, approx-distinct | 369 |
| scan side | pruning, page pushdown, row-group windowing | none |
| window functions, `NullPredicate`, `Coalesce`/`FillNull` | present | absent |

Neither lane has subqueries (`2026-08-21-subqueries-design.md` was never
implemented) or `Between`, so neither is a regression.

**Consumers to re-point:** 3 Python binding modules · 12 of the 14 binary-size
gates · `golden/prelude.mojo` and 154 cases · `marrow/tabular.mojo:23`, a real
up-edge from a core module (`from .exprold.aggregates import FoldedAggregates`)
· 3 kernel test files.

### 0.3 — It is a translation, not a copy

The lanes have different protocols, and `expr`'s is the better one. Porting is
rewriting each node in the new idiom:

| `exprold` | `expr` |
|---|---|
| `OutType` / `OutShape` / `State` | `Type` / `shape` / `Bound` |
| `referenced_columns()` | `columns()` |
| `state(batch)` | `bind(batch, bindings)` |
| `validity(batch)` **and** `state_validity(batch, state)` | one `validity(bound)` |
| `render()` | `write_to[W: Writer]` |

**The two-validity-method split must not come across.** `comptime/core.mojo:275-286`
records it as what made `exprold` evaluate `coalesce`/`nullif`/`case_when` twice
per fused pass. A node that needs data-dependent validity belongs with the
Kleene family, which materialises through a kernel, not with the fusing families.

### 0.4 — Two decisions taken

**The pruner is written once, in the new architecture** (§2, §4) — not ported
and then replaced. `expr` therefore has no pruning until Phase B lands, which is
already true today, so nothing regresses relative to `expr`; what regresses is
the *frontend*, once it moves off `exprold`. That window is the cost of not
writing the pruner twice, and it is bounded by Phase B.

**The params/CLI system is in scope.** `marrow compile` and
`python/marrow/compile.py` keep working, and `exprold` deletes in one piece
rather than surviving as a stub.

### 0.5 — Order, and what each step is proved by

Each step is independently green and independently reviewable. Golden is the
forcing function: it cannot be re-pointed until the nodes it imports exist, and
when it is, 154 cases either pass or name exactly what is missing.

| | step | proved by |
|---|---|---|
| 1a | casts — `NumericCast`, `NumToString`, `StringToNum`, `BoolToNum`, `NumToBool` | `marrow/expr/comptime/tests/` |
| 1b | strings — `Lower`, `Upper`, `Strip`, `StringLength`, `StartsWith`, `EndsWith`, `Like`, `ILike` | same |
| 1c | null handling — `IsNull`, `NotNull`, `Coalesce`, `FillNull` | same, plus: one `validity`, no second method |
| 2 | aggregates — `FoldedAggregates`, distinct, approx-distinct, window | resolves `marrow/tabular.mojo:23` |
| 3 | params / CLI | `python/marrow/tests/test_compile.py` |
| 4 | **re-point `golden/`** — prelude + 154 cases | 154 cases green on `expr` |
| 5 | re-point the 3 Python binding modules, drop `compile.py` back to parity | 676 Python tests green |
| 6 | index layer + pruning (§2–§4) | §2's acceptance test, then rows decoded 1,000,000 → 549,440 |
| 7 | re-point the 12 binary-size gates; re-baseline | `pixi run binary_size` |
| 8 | delete `marrow/exprold/` | `precompile` clean; nothing references it |

**The size gate is unreadable between steps 1a and 7**, because 12 of its 14
gates measure the old lane. Until step 7, measure only against
`query_expr2_streaming` and `query_expr2_agg_fused`, which already track `expr`.

## 2. Phase A — the index layer

`marrow/expr/index.mojo`, with **zero edges into the rest of `marrow/expr/`**.
It imports `marrow.scalars`, `marrow.dtypes`, `marrow.schema` and — for the
reader bridge only — `marrow.parquet`. It does not import `logical.mojo`,
`physical.mojo`, `comptime/` or `runtime/`. That sink property is what makes
Phase A independently testable and what keeps either expression lane out of a
binary that only wants statistics.

### A1 — `ParquetFile` narrowings (prerequisite, standalone)

Every public entry point is whole-file and the per-chunk narrowings are private,
which CLAUDE.md forbids calling from outside:

```
2264:  def _trusted_leaf(...)                                  ← private
2289:  def statistics(self) -> List[List[ColumnStatistics]]    ← whole file
2311:  def _chunk_page_index(...)                              ← private
2334:  def page_index(self) -> List[List[PageIndex]]           ← whole file
2365:  def page_bounds(self) -> List[List[List[PageBounds]]]   ← whole file
```

A three-column predicate on `hits_0` therefore decodes **210 column chunks to
use 6**. Add:

```mojo
def column_statistics(self, row_group: Int, leaf: Int) raises -> ColumnStatistics
def page_index_for(self, row_group: Int, leaf: Int) raises -> PageIndex
```

and reimplement `statistics()` / `page_index()` / `page_bounds()` on top. Lands
alone, breaks nothing, useful on its own merits.

### A2 — The value types

These are the expensive-to-change part of the whole design. A mechanism can be
added later against a ⊤ default without touching anything shipped; a value
type's shape cannot — changing it touches every node in both lanes plus the
`DynValue` trampoline table. **Five fields are paid for now and left unused.**

```mojo
struct Bounds:
    """What one key can hold over one granule. Erased, because an index's dtype
    is always runtime — it comes off `LeafColumn`, not off an expression, and
    `ColumnStatistics` already holds `Optional[DynScalar]`."""
    # value  ⊤ = neither bound present;  ⊥ = `empty`
    var lo, hi:               <scalar + presence flag, NOT Optional>   # see R-3
    var lo_open, hi_open:     Bool     # (1) Lt vs Le; IN, BETWEEN, transform inversion
    var empty:                Bool     # (2) ⊥ — an all-null page; `x > 10 AND x < 5`

struct Verdict:
    var maybe_true:  Bool
    var maybe_false: Bool              # (3) carried from day one, left at ⊤
    var relaxed:     Bool              # (4) the query was widened — see below
    var rows: Optional[RowRanges]      # (5) None = "no finer than this granule"
```

Why each, and what it costs to add later:

1. **open/closed bounds.** Needed independently by `IN`, `BETWEEN`, and any
   transform inversion. ClickHouse's `Range` carries exactly these
   (`Core/Range.h:52-53`); marrow's `Interval` has closed bounds only
   (`kernels/interval.mojo:39-41`).
2. **⊥.** An all-null Parquet page is exactly detectable via
   `ColumnIndex.null_pages`, and `page_bounds` already refuses to trust its
   min/max, emitting `None/None` (`reader.mojo:2390-2403`). Under a ⊤-only
   lattice that page **prunes nothing**, though every comparison over it is
   definitely NULL. A free exact prune the vocabulary otherwise forbids.
3. **`maybe_false`.** One bit now, a signature change later. `NOT` is then
   *correct* immediately — `~⊤ = ⊤` — and gets sharper for free whenever a
   kernel starts filling the false side in.
4. **`relaxed` — the genuine soundness hole.** The "every fact defaults to ⊤"
   rule covers the **data** side. Nothing covers the **query** side. A widened
   query — an inverted transform, a tokenizer that drops boundary tokens — is
   sound for `maybe_true` and **unsound for `maybe_false`**. ClickHouse carries
   the bit per RPN atom and forces `can_be_false = true` when it is set
   (`KeyCondition.cpp:5581-5589`), with a comment naming the bug: a false
   negative on `match(...)` makes `NOT match(...)` skip a matching granule.
   Without this bit, adding *any* approximate index later silently arms a wrong
   skip under negation.
5. **`rows`.** `None` is exactly today's behaviour. With it, Sieve,
   learned indexes, bisected sorted pages and eventually posting lists all fill
   a field. Without it, each of them changes `fold`'s return type across both
   lanes and the trampoline table.

**Shape decision: the fold takes a context, not one summary.** `fold(pred,
one_summary)` forecloses bloom-refining-minmax, which is exactly how ClickHouse
composes them — `checkInHyperrectangle(hyperrectangle, ColumnIndexToBloomFilter)`
(`KeyCondition.h:132-136`). Composition happens *below* the fold, per key, which
is also what structurally forecloses the "AND-ing masks is unsound for a
sub-term index" defect (findings §5).

### A3 — `Facts`, `DynFacts`, `RowRanges`

```mojo
trait Facts(Copyable, Movable):
    """What one granule's index can say. A granule is whatever built this
    decided it was — a row group, a page, a mark, a file — and `Facts` never
    says which, because whoever constructed it already knows which rows it
    covers. That is why there is no `parts()` and no container-identity problem.

    One method. Every future question arrives as another method with a ⊤
    default, so no existing implementation is ever broken by one.

    `raises` and takes `self`: an implementation may seek, decompress or consult
    a cache inside it. Nothing above assumes the answer was materialised, which
    is the whole of "disk-backed indexes, read on the fly"."""

    def bounds(self, key: String) raises -> Bounds:
        return Bounds.unknown()
```

`DynFacts` erases it with `ArcPointer[NoneType]` + per-type thin trampolines,
the `DynRelation` shape (`logical.mojo:330-364`) — **never a `Variant`**, which
would name every index kind in one place and cost the DCE property that keeps
`kernels::sort` out of a binary that never sorts.

`RowRanges` is a sorted disjoint interval list over file-absolute row offsets —
the one coordinate system, so row groups, pages and marks intersect as plain
interval arithmetic. It converts to `RowSelection` only at the reader boundary
(`reader.mojo:2107` is the ABI). **`RowSelection.union` must be added**
(`reader.mojo:2437-2529` has `intersect` and no `union`); its absence is exactly
why DataFusion discards any multi-column page conjunct rather than handling `OR`.

### A4 — `RowGroupFacts`

Off `ParquetFile.column_statistics`. Keyed by the **file leaf** schema, not the
scan's projected schema — holding the leaf schema inside the fact is the fix for
the `_leaf_map` confusion `exprold` handled at the call site. D13's repeated-leaf
null-count bug (findings §3) becomes a construction-time refusal: answer
`unknown` for a leaf with `max_repetition_level > 0`.

### A5 — Acceptance

Reproduce `exprold`'s verdict on `CounterID = 62` over `hits_0` — **1 of 2 row
groups, 450,560 rows pruned** — by driving `Facts` directly with a hand-written
probe, **without importing `exprold` and without a pruner existing**. That test
is the seam, demonstrated before anything consumes it.

Plus: preserve the measurement script as `benchmarks/pruning/measure_prunability.py`
so §0's table is reproducible, and add the `align_of` / `size_of` assertion on
`Bounds` that findings §7.3 asks for.

---

## 3. The size gate, between the phases

Add the `DynValue` slot returning ⊤, wire the trampoline, **call nothing**, and
measure. One build against `query_expr2_streaming` and `query_expr2_agg_fused`,
with `query_join` as a drift control (this machine moves ±8% per case).

This answers findings §7.2 — the open question whose reviewer was stopped
mid-run. If the slot alone spends most of the 6,792-byte budget, `fold` belongs
somewhere other than the box, and that is much cheaper to learn now than after
Phase B is written.

---

## 4. Phase B — pruning

1. **`Value.fold(facts: DynFacts, bindings: Bindings) raises -> Verdict`**, ⊤
   default on the trait — both lanes, one method. That total default is the only
   soundness rule in the system: a node added tomorrow cannot be forgotten by a
   pruner written today.
2. **`DynValue`** gains one thin-fn slot, the existing shape.
3. **Overrides**: `Column`, `Literal`, `Param`, the six comparisons, `And`/`Or`
   in `expr/comptime/`; one `_tag` ladder in `RuntimeValue.fold` — whose
   docstring already reserves the field: *"How this node prints and prunes.
   Never how it evaluates."*
4. **`Param` folds to ⊤ when unbound**, never to its default. Pruning at plan
   time with a different value than execution uses is a wrong skip (findings §7.5).
5. **`read_plan`**, ported from `marrow/exprold/execution.mojo:383-410`. Two
   nested loops, coarse to fine.
6. **The seam is a constructor, not a rewrite.** `ParquetScan(path,
   schema).filter(pred)` returns a `Filter` over a scan that also carries the
   predicate as pruning metadata. The `Filter` stays — pruning is conservative,
   so the exact predicate must still run, which makes this correctness-neutral by
   construction. `Relation` stays at two methods; `with_predicate` is not
   resurrected (`logical.mojo:9-22` deleted it and says why).

**Measurement.** The only honest one is the same plan with `_prune` set versus
`None` — same operators, same code path, one field different. The deterministic
number is rows decoded, 1,000,000 → 549,440, and that is a correctness assertion,
not a benchmark. Expect ≈1.6–1.8x on that single query because decode dominates.
Do **not** report a suite-level speedup; the honest suite ceiling is 1.04x.

---

## 5. Deliberately not built

Each of these was designed, costed, and cut. Recorded so they are not
re-derived.

| | why not | what it would take |
|---|---|---|
| `Granulation` as a type | row groups and pages are two nested `for` loops; a third granulation is what would earn the type | a struct of sorted starts |
| `Summary` as a trait | every conformer is used by exactly one index; it becomes a trait the day it needs `serialize`/`deserialize` | pure addition |
| `Index` as a trait | zero methods have two implementations. An index is *whoever constructs a `Facts`* | additive, ⊤ defaults |
| `KeyExpr` matching | no expression-keyed index exists. And it is the **wrong shape**: Iceberg's `Transform::project` *changes the operator* (`Lt`→`LtEq`, binary→`IN`, `transform.rs:570-611`), so the eventual form is `project(atom) -> Optional[(query, relaxed)]`, a transformer — which is why `relaxed` is paid for now | a `KeyMap` trait |
| `ordered()` / bisection | **no source**: `sorting_columns` is unread, `boundary_order` is written as 0. And it belongs on the *index*, not the granulation — orderedness is a joint property of (granulation, key, data) | `order() -> {Unordered, Asc, Desc}` |
| `refine` / extent mode | the +0.0% bloom result is direct evidence that per-column facts are not the bottleneck on the compound queries. Precondition is also stricter than "superset": ClickHouse turns a truncated set into ⊤ (`MergeTreeIndexSet.cpp:428-429`) because a *subset* is unsound | a method with a ⊤ default |
| probe mode 3 (posting) | the stated reason — "a different boolean algebra" — was **wrong**; lifted pointwise over row sets it is the same `BoolMask` algebra. The real reason is allocation. `Verdict.rows` is the door | a second driver |
| token/ngram index | **not attachable** — no `like` atom in either lane | see §6 |
| scalar bloom | +0.0%, and already read and written | nothing |
| runtime/join filters | no joins in the measured queries; and the probe scan is sealed inside `Pipeline._ops` with no `children()` | a registry parameter on `to_operator` |

---

## 6. The cheapest way to de-risk the architecture

Not to refine the factorization further. **Add a `like` atom to the runtime
lane and re-measure with ngram semantics.**

The runtime lane's atoms are `and case_when coalesce column eq ge gt le literal
lt ne not or xor` — no `like`, `in`, `is_null` or `cast`. The 63.4% figure is
the single number justifying the entire extension surface, and it is currently
an inference: it was measured with substring containment, and a splitting
tokenizer yields zero tokens for `'%google%'`. Roughly 30 lines converts it into
a measurement — and decides whether the eventual key is `tokens(col)` or the
parameterized `ngrams(col, n)`, which in turn decides whether comptime key
matching must compare a parameter.

---

## 7. Sequencing

**Step 0.1 is done** (`436837f`): the Python package imports again, 16 collection
errors → 676 passing. The migration order is §0.5; step 6 there expands into the
index and pruning steps below.

| | step | gate |
|---|---|---|
| 1a–3 | port the nodes, aggregates and params (§0.5) | per-group tests; `precompile` clean |
| 4–5 | re-point `golden/` and the Python bindings | 154 golden cases; 676 Python tests |
| A1 | `ParquetFile` per-(rg, leaf) narrowings | whole-file methods reimplemented on top |
| A2–A4 | value types, `Facts`, `DynFacts`, `RowRanges`, `RowGroupFacts` | `align_of` assertion; capacity reserved on every `List[Bounds]` |
| A5 | acceptance: 450,560 rows pruned, no pruner in existence | test passes without importing `exprold` |
| — | **size gate**: `DynValue` slot, calling nothing | delta against 6,792 bytes |
| B | `fold`, the overrides, `read_plan`, the scan seam | rows decoded 1,000,000 → 549,440 |
| 7–8 | re-point the 12 gates, re-baseline, delete `marrow/exprold/` | `pixi run binary_size`; `precompile` clean |
| — | `like` atom + re-measure | 63.4% becomes measured or falsified |

**Two items outrank everything above them from step A onward**: projection
pushdown (3.6x) and row-group windowing (1.6–4.7x) in the new lane. Step 0 is
what makes both of them measurable too, which is the strongest argument for
doing it first.

---

## 8. Open risks

| | risk | settled by |
|---|---|---|
| R-1 | cost of the sixth `DynValue` trampoline slot | the size gate, §3. Do it before writing any index |
| R-2 | does a trait **default** returning `Bounds[Self.Type]` reduce for every conformer? R1 is proven for a *static* default on a kernel (`kernels/aggregate.mojo:165-186`, returning `PrimitiveScalar[Self.AccType[V]]`, and `PrimitiveScalar` is not `ImplicitlyCopyable`); an *instance* default on a value trait is untested | ~40 lines + `precompile`. Fallback is mechanical: declare it abstract, write a four-line ⊤ body per node |
| R-3 | `Optional[Scalar[T.native]]` is align-32 for `Decimal256` — the `c2bb828` `Variant` miscompile class, which produces **no diagnostic** and which ASAN perturbs rather than reveals | explicit presence flags instead of `Optional`; `align_of` assertion; capacity reserved on every `List` of it |
| R-4 | `comptime KeyType: DataType` is a **declaration error** — `DataType` has no `native` | use `PrimitiveType` wherever a `native` is read |
| R-5 | a generic method and an associated type both make a trait unboxable — a trampoline slot has a concrete, non-generic signature | `Facts.bounds` is non-generic and returns a concrete type; `DynFacts` is a concrete struct, and its one genuinely open question gets a thin-fn slot when a producer exists |
| R-6 | per-column page granularity dissolves under AND (intersection of row sets is exact regardless of page boundaries) but **not under OR** | adopt DataFusion's restriction — single-column conjuncts only (`page_filter.rs:144,163-166`) — and say so in the docstring, until `RowSelection.union` exists |

`comptime` deletion of an unusable index via `P.columns()` was investigated and
**is not spellable**: `columns()` is an instance method reading `self._name`, and
`col("a", int64)` builds that `String` from a runtime argument, so the predicate
*value* is not a comptime parameter. The trampoline pattern already delivers the
DCE property without it. A type-level test remains possible and is not needed yet.

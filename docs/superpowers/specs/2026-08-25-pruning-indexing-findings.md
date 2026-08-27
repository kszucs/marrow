# Pruning and indexing — preliminary findings

**Status: §1–§5 stand and are the evidence base. §6–§8 are superseded by
`2026-08-27-index-and-pruning-plan.md`**, which was written after four
independent design passes (storage, Mojo mechanics, YAGNI, extensibility) and
which corrects five claims that were believed and acted on here or in the
discussion around it — see its §0.

This records what was measured, what the reference implementations actually do,
and where four adversarial reviews left the design. It supersedes the analysis
in `2026-08-24-expr2-pruning-pushdown-design.md`, which predates the
`expr2` → `expr` rename and builds on the `_eval` function pointer that
`d8ccc48` removed.

**§1's numbers were re-verified on 2026-08-27 and hold.** In particular the
`prefix`/`contains`/`match` row and the page-level row are properties of *this
file* (`parquet-cpp 1.5.1-SNAPSHOT`, two row groups of 450,560 and 549,440, no
ColumnIndex, no OffsetIndex), not of the format. A later measurement simulating
8,192-row granules produced much larger figures — 57.7% on `CounterID`, 63.4% on
`URL LIKE '%google%'` — which are **not reachable by any reader on this file**
and are statements about page sizing (`marrow/parquet/writer.mojo:63,396`) and
about a rewritten fixture. At the file's real boundaries: `CounterID = 62`
prunes 1 of 2 groups; both `LIKE` predicates prune 0 of 2. **The 1.04x ceiling
below is correct.**

---

## 1. What pruning is worth here, measured

`~/Workspace/ClickBench/data/hits_0.parquet` — the file marrow's benchmark
actually reads — inspected with pyarrow:

```
rows 1,000,000   row_groups 2   cols 105   parquet-cpp 1.5.1-SNAPSHOT
CounterID     rg0 (17, 38)      rg1 (38, 62)
EventDate     rg0 (15901,15901) rg1 (15901,15901)     <- whole file is one day
UserID        rg0 spans int64   rg1 spans int64
RefererHash / URLHash             both span nearly the full int64 range
has_column_index = False   has_offset_index = False   bloom_filter_offset = None
```

| capability | value on this workload |
|---|---|
| row-group min/max | **one** predicate: `CounterID == 62` prunes rg0, 450,560 of 1,000,000 rows |
| range pruning on `EventDate` | zero — single day |
| point lookups on hashes | zero — both groups span the range |
| bloom | zero — file has none |
| page-level | zero — no ColumnIndex/OffsetIndex |
| `prefix` / `contains` / `match` | zero — string columns are `binary` + cast, every LIKE is `%…%` |
| inverted / partition / vector | zero — none exist; `ParquetScan` holds a single `String` path |

Ceiling: Q37/38/41/42 are 366+362+301+288 = 1,317 ms of a 13,935 ms suite. A
perfect 45% cut of their scan portion is ≈590 ms ≈ **4.2%**, about **1.04x**.

For scale, in the same tree: **projection pushdown measured 3.6x**
(`docs/backlog.md:668-671`) and **row-group windowing 1.6-4.7x**
(`marrow/exprold/execution.mojo:434-447`). The new lane has neither. Both are
larger wins than the entire pruning subsystem and neither needs a statistic.

**No wall-clock number for pruning exists anywhere in the tree.** It has
correctness tests (`marrow/exprold/tests/test_pruning.mojo`,
`test_pushdown.mojo`) and zero benchmarks.

---

## 2. What the reference implementations do

Three systems, two designs. They differ because of how many index kinds each
has to serve.

### DataFusion and Polars — rewrite into an ordinary expression

Neither has a pruning algebra. Both rewrite the predicate into an expression
over statistics *columns* and evaluate it with the normal engine.

- Polars, `crates/polars-io/src/predicates.rs:351-398`: builds a one-row frame
  of `len`, `{col}_min`, `{col}_max`, `{col}_nc` and calls
  `evaluate_with_stat_df`, returning a bitmap.
- DataFusion, `datafusion/pruning/src/pruning_predicate.rs`: same, and
  explicitly vectorized — *"suitable for pruning 1000s of containers"*.
  `x = 5` becomes `x_min <= 5 AND x_max >= 5`.

**Why it does not transfer directly.** DataFusion vectorizes to prune
thousands of containers; marrow has **two**. Materializing 105 columns ×
(min, max, null_count) = 315 Arrow arrays of length 2 to vectorize a
2-element evaluation costs more than the scalar walk it replaces. It also needs
an expression→expression rewriter, which `marrow/expr/logical.mojo:9-18`
explicitly declines to carry.

### ClickHouse — RPN plus a per-index condition protocol

`KeyCondition` compiles the predicate to a stack machine of atoms plus
`FUNCTION_AND` / `FUNCTION_OR` / `FUNCTION_NOT` / `ALWAYS_FALSE` /
`FUNCTION_UNKNOWN` (`src/Storages/MergeTree/KeyCondition.h:304-398`), folded
with a two-bit `BoolMask`:

```cpp
// BoolMask.h:14-16
operator&  { can_be_true && m.can_be_true,  can_be_false || m.can_be_false }
operator|  { can_be_true || m.can_be_true,  can_be_false && m.can_be_false }
operator!  { can_be_false, can_be_true }
```

Two bits, not one — which is what makes `NOT` composable.

Indexes are **four** traits (`MergeTreeIndices.h`): `IMergeTreeIndex`
(declaration), `IMergeTreeIndexGranule` (the serialized on-disk payload, with
`serializeBinary` / `deserializeBinary` and a version), `IMergeTreeIndexAggregator`
(writer side, `update(block, pos, limit)` — incremental, resumable mid-block),
and `IMergeTreeIndexCondition` (reader side, `mayBeTrueOnGranule`). The
Granule/Aggregator split is what an on-disk index format needs; `build(batches)`
is only correct when index granularity equals the write unit.

Analysis is split by frequency: `createIndexCondition` runs **once per query**
(`:295`), `mayBeTrueOnGranule` **once per container** (`:167`).

Pruning order: `filterPartsByPartition` → `markRangesFromPKRange` →
`filterMarksUsingIndex` → read → PREWHERE → WHERE. `KeyCondition` is built at
storage-read time, *after* the optimizer has fixed the plan.

**Take:** ClickHouse's shape fits marrow's stated future (inverted, partition,
dataset indexes); DataFusion's fits marrow's present (statistics only, two
containers).

---

## 3. Six safety bugs — fixed in `9a491ce`

All were live, independent of any pruning design, and each produced a wrong
skip once a predicate reached the statistics.

| # | defect | consequence |
|---|---|---|
| D11 | statistic byte length never validated before an unaligned wide load | **out-of-bounds read**, did not trap; arbitrary bound |
| D9 | INT96 decoded as `int64` — low 8 bytes are nanos-*of-day*, values are epoch nanos | every INT96 range predicate pruned every row group |
| D10 | NaN bounds not rejected on read; `_three_way` reports unordered as *equal* | `x < 5` with `x_min = NaN` pruned a matching group |
| D13 | repeated leaves count nulls per *element*, `num_rows` per *row* | `list<int64>`, 100 rows of `[null,5,7]` → prune everything |
| D14 | leaf stats keyed by the bare Parquet element name | `struct<x>`'s bounds handed to a top-level `x` |
| D12 | `column_orders` written but never read | spec: without it, min/max meaning is undefined |

**Settled, not a bug:** `StringSlice` compares bytes **unsigned**, matching
Parquet's BYTE_ARRAY ordering, so a bound containing a byte ≥ 0x80 is not
inverted.

**Still open:** `Interval._three_way` (`marrow/kernels/interval.mojo:111`) is
still wrong for a NaN reaching it by another path — D10 removes NaN at the
source only. And `FileMetaData.write` synthesizes field 7 from the schema
rather than writing `self.column_orders`, so a footer read *without* column
orders and re-written silently gains them.

---

## 4. Design evolution

Four shapes were considered. Recording the dead ends so they are not re-derived.

**(i) Walk the predicate calling interval kernels.** Hardcodes one index kind
into the traversal. A bloom filter cannot answer `<`; an inverted index cannot
answer min/max. Rejected.

**(ii) Two layers** — statistics by expression-rewrite, opaque indexes by a
condition protocol. Rejected: two mechanisms for one job, and the seam between
them has to be maintained.

**(iii) One `Index` protocol, statistics as an implementation.**
`Predicate` (lane-neutral closed IR) + `Index.prune(Predicate, Bindings) -> Bitmap`.
This is what the four reviews attacked. Four structural defects, below.

**(iv) Abstract interpretation** — the current proposal, §6.

---

## 5. What the reviews found

Four adversarial reviews: soundness, extensibility, YAGNI, binary size. The
last was **stopped mid-run** and its question is unanswered.

### Fatal to shape (iii)

- **`NOT` over a conservative default prunes everything.** `NOT (x LIKE '%foo%')`
  → child is `Unknown` → rewrites to `true` → `NOT true = false` → every row
  group pruned, zero rows returned. DataFusion refuses to recurse into `NOT`
  at all (`pruning_predicate.rs:1455-1464`). marrow's current pruner is safe
  only by accident — a `not` node has one child and falls past the two-child
  ladder into `Interval.unknown()` (`marrow/exprold/dynamic.mojo:682`).
- **The third state has nowhere to go.** The rewritten expression is genuinely
  three-valued; `Bitmap` has two states. A row group written with statistics
  disabled evaluates to `NULL`; extracting via `.values()` — which CLAUDE.md's
  style guide nudges toward — reads the data bit under a null slot, in practice
  `0`, and prunes it. A test suite built on marrow-written files never catches
  this.
- **Pages are not a shared container space.** Column `a` may have 2 pages and
  `b` 5, with unrelated row boundaries. "One row per container" presumes they
  partition identically. DataFusion discards any page predicate touching more
  than one column (`page_filter.rs:144-168`); marrow's current code folds once
  per column with the others withheld (`marrow/exprold/execution.mojo:342-377`).
- **AND-ing masks is unsound for a sub-term index.** `x = 5 OR y = 7`: bloom
  disproves `x = 5` → bit 0; minmax says maybe → bit 1; AND → prune a group
  full of matches. ClickHouse documents exactly this
  (`KeyCondition.h:112-129`) and built an atom-position merge subsystem for it.

### Interface defects

- **Missing two-phase split.** One `prune(Predicate, …)` re-derives analysis per
  container. A text index's tokenization and regex folding belong in a
  per-query phase.
- **`Bitmap` is a regression.** `exprold` already returns a `RowSelection` for
  page-level pruning. One bit per container cannot express "rows 100-200 of
  group 3".
- **`containers() -> Int` loses container identity.** If index A's bits address
  files and B's address row groups, combining them is a silent type error.
- **No range parameter.** A sorted/PK index must enumerate every container
  instead of binary-searching — O(n) where ClickHouse is O(log n).

### Futures tested against the protocol

| future | verdict |
|---|---|
| MinMax, Bloom | fit |
| Partition / manifest | fits after container identity; transform inversion (`toYYYYMM(ts)`) is index-internal |
| Inverted / full-text | container-level half fits; phrase, exact-read and ranking do not. `prefix/contains/match` is the SQL surface, not the index's query — CH's atoms carry `TextSearchMode {Any, All, Phrase}` and token lists |
| **Vector / ANN** | **wrong shape.** CH's attempt produced `mayBeTrueOnGranule` that throws `LOGICAL_ERROR` (`MergeTreeIndexVectorSimilarity.cpp:516-519`); the real entry point is a separate virtual returning row offsets, driven by `ORDER BY … LIMIT`, not by the predicate |

### Attacks that failed — do not re-derive

- marrow's `and`/`or` are **already Kleene** (`marrow/kernels/boolean.mojo:140-190`)
  and the comparison kernels already propagate nulls
  (`marrow/kernels/numeric.mojo:101,504,672`). This is the single semantic fact
  the whole rewrite approach rests on, and it is already true.
- One-sided **page** bounds are sound under Kleene AND. The row-group/page
  asymmetry does not by itself produce a wrong skip; the row group is merely
  more conservative.
- Empty row groups prune correctly (`len = 0`, `nc = 0` → `0 != 0` → false).
- Signed zero is a non-issue — the writer normalizes and IEEE `==` treats
  `-0.0 == +0.0`.
- `IN` / `NOT IN` decompose soundly into `OR` of `=` / `AND` of `!=`.

---

## 6. Current proposal — abstract interpretation

> **Superseded by `2026-08-27-index-and-pruning-plan.md`.** The abstract-interpretation
> shape survived; the `Abstract` struct below did not. `Optional[DynScalar]` in a
> struct held in a `List` is the `c2bb828` alignment miscompile class (open
> question 3 below, now answered: use explicit presence flags). The `trait Stats`
> below is unboxable as written — a generic method cannot be a trampoline slot.
> And the domain is missing two fields whose absence is a *soundness* hole rather
> than a precision one: a `relaxed` bit for query-side widening, and ⊥ for an
> all-null page. Read the plan's §2 instead.

Pruning is the same expression evaluated in a different domain: concrete is
rows → values, abstract is one container → what its values could be. The
soundness condition becomes **local and per-operation** — the abstract
operation must over-approximate the concrete one — rather than a global
"only ever says no" enforced by hand-written invariants.

```mojo
struct Abstract:                  # a lattice element; unknown() is ⊤
    var maybe_true: Bool          # both flags, so NOT is expressible
    var maybe_false: Bool
    var lo: Optional[DynScalar]
    var hi: Optional[DynScalar]

trait Stats:                      # facts about one container
    def bounds(self, col) raises -> Optional[Tuple[DynScalar, DynScalar]]:
        return None
    def contains(self, col, v) raises -> Optional[Bool]:   # bloom
        return None
    def nulls(self, col) raises -> Optional[Int]:
        return None

# one method on the value traits, both lanes:
def fold(self, stats: DynStats, bindings: Bindings) raises -> Abstract:
    return Abstract.unknown()     # total default — the only soundness rule
```

Three defects of shape (iii) dissolve rather than needing patches:

- **`NOT`** — a domain that cannot complement soundly returns ⊤. A property of
  the domain, not a De Morgan rule bolted on.
- **Bloom under `OR`** — `⊤ ∨ anything = ⊤`, handled *inside* the fold, so a
  sub-term index structurally cannot wrongly prune.
- **The operand/predicate split** — gone. Everything is a lattice element.

**Lane fit.** Each node interprets itself, which is what both lanes already do
for `columns()`, `name()` and `dtype()` — so neither lane is bent to fit. The
return type is concrete, not generic in the domain, so `DynValue` needs **one**
thin fn pointer and no new erasure box. Fusion is untouched: `fold` runs once
at lowering, `bind`/`lane` are unaffected. `Interval` is `Abstract` minus
`maybe_false`, so the nine existing interval kernels become the fold bodies.

**Indexes become `Stats` implementations**, not a trait with a lifecycle:
`RowGroupStats` answers bounds and nulls, `BloomStats` answers `contains` and
inherits ⊤ for the rest, a future text index answers whatever it can.
Composition happens *below* the fold, which is why the AND-ing defect cannot
recur.

---

## 7. Open questions

1. **Page granularity.** Different columns have different page boundaries, so a
   single per-container `Stats` does not model pages. `exprold` folds once per
   column with the others withheld; that works here but is the one place the
   abstraction is not clean.
2. **Binary size** of a defaulted `fold` across every comptime node type.
   Unmeasured — that reviewer was stopped. `columns()` returning `List[String]`
   is precedent that the shape is tolerable, but `Abstract` is heavier.
3. **`Abstract` alignment.** If it or anything it contains ends up in a
   `Variant` whose largest member is not its most-aligned member, it re-arms
   the miscompile fixed in `c2bb828`. Needs an `align_of` assertion in a test.
4. **Runtime-lane promotion.** `_compare` casts operands at *evaluation*
   (`marrow/expr/runtime/values.mojo:288`), so `fold` sees un-promoted
   children. Pruning must promote identically or refuse mixed widths.
5. **Parameter defaults.** An index must never resolve a `Param` from its
   default: plan-time file pruning would then use a different value than
   execution. An unbound `Param` is ⊤.

---

## 8. Recommendation

Sequence, by measured value per line:

1. **Done** — the six safety bugs (`9a491ce`). Prerequisites for pruning being
   sound at all, and worth fixing regardless.
2. **Projection pushdown** (3.6x measured) and **row-group windowing**
   (1.6-4.7x measured) in the new lane. Both larger than pruning's entire
   ceiling; neither needs a statistic.
3. **Pruning** on the abstract-interpretation shape, after questions 1 and 2
   above are resolved.

Three decisions cost nothing now and are expensive later, whatever shape wins:
carry both `maybe_true` and `maybe_false`; keep negation out of any reified
predicate domain; and let every absent statistic enter as SQL `NULL`, never a
sentinel and never a synthesized zero.

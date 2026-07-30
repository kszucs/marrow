# Binary size: comptime (AOT) vs. erased-AOT vs. hybrid vs. runtime relational plans

Four files implement the exact same query — `SELECT a, name FROM orders
WHERE a > b` — over the same 5-row in-memory batch, producing identical
output:

- **`query_comptime.mojo`** — the fully-monomorphized layer from
  `marrow.aot.relations` (`Table`, `Column`, `Project`, `Filter`). The whole plan
  is one nested generic type; `.execute(batch)` compiles straight to column
  loads, a SIMD comparison, and a filter call. No tag dispatch, no vtables.
- **`query_erased_aot.mojo`** — the "option 1" layer from `marrow.aot.erased`:
  the relational operators are plain **runtime** structs over `List[DynValue]`
  (a walkable, rewritable plan tree, *not* a `*Es` type pack), but each value is
  a **fused-only** box (`DynValue`) that trampolines into the concrete node's
  own `execute()` — no `eval()` tag interpreter. The operators execute
  themselves single-shot (no `Planner`/`RelationProcessor`). Tests whether a
  runtime plan tree can keep the comptime binary size.
- **`query_hybrid.mojo`** — relational *structure* stays runtime/type-erased
  (`marrow.dyn`'s `DynRelation`, `Planner.build()`, the pull-based
  `RelationProcessor` pipeline — same as `query_runtime.mojo`), but the
  *predicate* is a comptime-typed `Gt(Column, Column)` node
  (`marrow.aot.values`) boxed into a runtime `Expr` via the `FUSED` tag
  (`Expr(gt_node)`), so evaluating it is a direct call into the fused
  vectorize loop rather than a walk through `Expr.eval()`'s tag interpreter.
- **`query_runtime.mojo`** — the existing type-erased layer end to end:
  `in_memory_table(batch).filter(...).select(...)` then `execute(plan)`,
  predicate built from `col("a") > col("b")` and evaluated by `Expr.eval()`'s
  tag interpreter.

All four are compiled with `-O3 -g0`, then `strip`ped, so the comparison is
release, no-debug-info code — the fairest apples-to-apples measurement of
what actually ships.

## Run it

```bash
pixi run binary_size
```

Builds all four, strips them, and prints the size/symbol table plus the
per-module symbol breakdown below. `benchmarks/binary_size/compare.py` is a
plain Python script (no dependencies beyond `mojo`, `nm`, `size`, and `strip`
on `$PATH`) — read it directly if you want to change what gets measured.

## ⚠️ Read this before quoting a number below

**Sizes here must be the `__text` *section*, not the `__TEXT` segment and not the
file size.** Both of the latter are padded up to a page boundary — 16 KB on
Apple Silicon — so they move in 16,384-byte steps. Measured 2026-07-29: a change
that added **1,728 bytes** of code moved the stripped file by 16,504 and the
segment by exactly 16,384, while the symbol count went *down* by one.

`compare.py` now reports and ratios on `__text`. Two consequences for what
follows:

- **The 2026-07-05 table below is in the old, page-quantized metric.** Every
  `__TEXT` figure in it is an exact multiple of 16,384, which is the tell. Its
  binaries (`query_comptime`, `query_erased_aot`, `query_hybrid`) also no longer
  have `.mojo` sources, so they cannot be re-measured — treat the table as
  historical and directional.
- **"`query_hybrid` and `query_runtime` have the exact same `__TEXT`" does not
  establish that fusing the predicate saved zero bytes.** Equal page counts are
  compatible with any difference up to 16 KB. The live gates show exactly this:
  `query_dynvalue` and `query_runtime` differ by 16 bytes of file and **384
  bytes of `__text`**. The claim needs re-measuring before it is repeated.

## Current baselines (osx-arm64, 2026-07-29) — `__text`, code only

`query_streaming` is the floor: one `InMemoryTable -> Filter -> Project` with a
column reference and a comparison. Every other gate is that shape plus one thing,
so the delta column is what the thing costs in an AOT binary.

| gate | `__text` | Δ vs floor | what it adds |
|---|---:|---:|---|
| `query_streaming` | 1,251,672 | — | the floor: `col`, `>` |
| `query_arith` | 1,259,316 | +7,644 | fused `+ - *` |
| `query_scan` | 2,285,232 | +1,033,560 | `ParquetScan` — the whole reader |
| `query_sort` | 3,684,276 | +2,432,604 | `Sort` + top-K |
| `query_streaming_agg_fused` | 3,776,820 | +2,525,148 | `Aggregate`, comptime aggs |
| `query_join` | 3,819,060 | +2,567,388 | `Join` |
| `query_exprs` | 3,892,916 | +2,641,244 | string, conditional, membership, cast, temporal |
| `query_streaming_agg` | 4,150,836 | +2,899,164 | `Aggregate`, runtime-named aggs |
| `query_dynvalue` | 5,266,164 | +4,014,492 | the erased lane (was the tag interpreter) |
| `query_runtime` | 5,266,548 | +4,014,876 | interpreter + runtime plan |

Re-measure one gate without paying for the sweep (ten `-O3` builds, ~20 min):

```
pixi run binary_size query_scan
```

`query_streaming` is always built, since it is the ratio baseline.

### Why the metric had to be fixed first

`query_arith` is the demonstration. Against the floor it is **+7,644 bytes of
code** — and **16 bytes *smaller* by stripped file size** (1,324,168 vs
1,324,184). The old metric would have reported fused arithmetic as free, or
faintly negative. It is neither.

## Result (osx-arm64, Mojo 1.0.0b3.dev2026070506) — historical, page-quantized

| binary | unstripped | stripped | symbols | symbols (stripped) | `__TEXT` |
|---|---:|---:|---:|---:|---:|
| `query_comptime` | 710,320 B (710 KB) | 250,120 B (250 KB) | 247 | 24 | 229,376 B |
| `query_erased_aot` | 734,128 B (734 KB) | 250,136 B (250 KB) | 256 | 24 | 229,376 B |
| `query_hybrid` | 11,652,992 B (11.1 MB) | 7,734,104 B (7.7 MB) | 3,360 | 52 | 7,651,328 B |
| `query_runtime` | 11,649,328 B (11.1 MB) | 7,734,088 B (7.7 MB) | 3,353 | 52 | 7,651,328 B |

**~30.9x smaller, stripped**, for the fully-monomorphized version. `size` on
the stripped binaries confirms the gap is genuinely in compiled code, not
just symbol-table noise — the `__TEXT` (executable code) segment column above
tells the same story as the file-size columns. (The comptime binary grew ~16 KB
vs. earlier revisions when named columns dropped their baked `index` in favor of
resolving the position by name against the batch schema — that links
`Schema.get_field_index`'s comparison code, a small, deliberate trade for one
column type per dtype and a unified `Table[…]` / `col(…)` surface.)

## The hybrid result is the interesting one

**`query_hybrid` and `query_runtime` have the *exact same* `__TEXT` size —
fusing the predicate alone saved zero bytes of compiled code**, even though
its own comparison genuinely runs fused (no per-element tag dispatch for
that one operation). The ~4 KB unstripped difference and 7 extra symbols are
just the one additional trampoline function (`_fused_eval_tramp_bool` et
al., visible below as `marrow::aot::values` going from 0 to 4 symbols) —
noise next to a 7.65 MB `__TEXT` segment.

This is the useful negative result: **the size payoff comes almost entirely
from erasing the *relational* layer (`marrow.dyn`), not the scalar/value
layer.** `in_memory_table(batch).filter(predicate).select(...)` still walks
through `Planner.build()`, which links in every `RelationProcessor` kind
(`Scan`, `Filter`, `Project`, `Aggregate`, `Join`, `ParquetScan`, ...)
regardless of which ones this query actually uses, plus the `DynRelation`
vtable/trampoline machinery. `Expr.eval()`'s own op branches (`ADD`, `SUB`,
`MUL`, `DIV`, `EQ`, `NE`, `LT`, `LE`, `GT`, `GE`, `AND`, `OR`, `NEG`, `ABS`,
`NOT`, `IS_NULL`, `IF_ELSE`, `CAST`, `LENGTH`) are a comparatively small slice
of that — boxing away the one comparison this query needs doesn't remove any
of the *other* branches from the compiled function, and the relational
executor around it dwarfs the interpreter either way.

`query_comptime` avoids all of it: `Project[*Es]` and `Filter[Input, Pred]`
are generic structs, so the compiler only ever instantiates the exact node
types this one query uses (`Column`, `StringColumn`, `Gt`) — no relational
processor pipeline, no `Expr` tag interpreter, no vtables, nothing unused to
strip because there was never a branch to begin with.

## The erased-AOT result is the payoff

`query_erased_aot` is the interesting *positive* result. Its plan is a
**runtime** object — `Project`/`Filter` are plain structs over
`List[DynValue]`, walkable and rewritable, not a `*Es` type pack — yet its
`__TEXT` is **229,376 B, byte-identical to `query_comptime`** (1.0x, versus
30.9x for the runtime path). Making the plan a runtime, pushdown-friendly tree
cost *zero* compiled code.

Two properties, together, are what buy it — and the per-module table shows both
holding:

1. **The value box is fused-only.** `DynValue` trampolines straight into each
   node's own fused `execute()` and carries no `eval()` tag-switch, so
   `Expr.eval()` is never reachable. `kernels::arithmetic` (0, vs 371) and
   `kernels::compare` (0, vs 74) confirm the per-op/per-dtype interpreter is
   simply absent.
2. **The driver is closed.** `Project`/`Filter` execute themselves single-shot;
   there is no `Planner` referencing every processor kind, so
   `kernels::join`/`groupby`/`hashing` (0/0/0, vs 11/17/141) and all of
   `dyn::*` (0) never link.

With *both* open surfaces gone, the big shared buckets that only collapse when
neither is reachable (`kernels::execution` 9 vs 667, `views` 2 vs 455, `arrays`
39 vs 376) fall to their comptime levels. The entire cost of the runtime plan
tree is the 17 symbols in `aot::erased`/`aot::relations`/`aot::values` (5/7/5) —
the box, its trampolines, the two column types, and `Gt`. This is the empirical
proof that **rewritability and ~250 KB binaries are decoupled**: the size win is
a property of the closed driver + fused-only values, not of encoding the plan in
the type system. Contrast `query_hybrid`, which fused the value but kept the
open driver and saved nothing — the two experiments bracket exactly which half
matters.

### Per-module symbol counts (unstripped)

Counting distinct symbols whose mangled name references each module (a
symbol can match more than one bucket, since Mojo names embed nested generic
type params — this is a proportional breakdown, not a strict partition):

| module | `query_comptime` | `query_erased_aot` | `query_hybrid` | `query_runtime` |
|---|---:|---:|---:|---:|
| `kernels::execution` | 9 | 9 | 667 | 667 |
| `dtypes` | 58 | 62 | 566 | 562 |
| `views` | 2 | 2 | 455 | 455 |
| `arrays` | 39 | 39 | 376 | 376 |
| `kernels::arithmetic` | 0 | 0 | 371 | 371 |
| `builders` | 1 | 1 | 89 | 89 |
| `kernels::hashing` | 0 | 0 | 141 | 141 |
| `kernels::compare` | 0 | 0 | 74 | 74 |
| `kernels::filter` | 10 | 10 | 66 | 66 |
| `kernels::join` | 0 | 0 | 11 | 11 |
| `kernels::groupby` | 0 | 0 | 17 | 17 |
| `kernels::boolean` | 0 | 0 | 13 | 13 |
| `scalars` | 0 | 0 | 20 | 20 |
| `buffers` | 10 | 10 | 34 | 34 |
| `dyn::executor` | 0 | 0 | 29 | 29 |
| `dyn::relations` | 0 | 0 | 34 | 34 |
| `dyn::values` | 0 | 0 | 16 | 13 |
| `aot::values` | 0 | 5 | 4 | 0 |
| `aot::relations` | 0 | 7 | 0 | 0 |
| `aot::erased` | 0 | 5 | 0 | 0 |

Why the biggest buckets are so lopsided: `comptime_query` only ever
instantiates the *exact* concrete types this one query needs —
`Column[Orders,"a",Int64Type]`, `StringColumn[Orders,"name"]`, `Gt[...]` —
nothing else exists at compile time, so nothing else gets generated.
`query_hybrid`/`query_runtime` go through `DynArray`, which erases the dtype
to a runtime tag — any code operating on an `DynArray` (`Expr.eval()`,
`DynRelation`'s processors, kernel dispatch functions) can't know at compile
time which dtype it'll see, so the compiler generates a full typed
instantiation *per supported dtype*, "just in case." `kernels::execution`
(667, the CPU/GPU dispatch layer under every kernel) and `kernels::arithmetic`
(371, even though this query does zero arithmetic) are the clearest examples:
`Expr.eval()`'s `ADD`/`SUB`/`MUL`/`DIV` branches are reachable code
regardless of whether this specific query ever hits them, so `add`/
`subtract`/`multiply`/`divide` each get compiled for every numeric dtype.
`kernels::join`/`groupby`/`hashing` (11/17/141, all zero for `query_comptime`)
are the same story one level up: `Planner.build()`'s exhaustive per-node-kind
dispatch makes `AggregateProcessor`/`JoinProcessor` reachable — and therefore
compiled in — even though this query never aggregates or joins anything.

See `marrow/aot/relations.mojo`'s module docstring for the "closed vs. open
erasure boundary" framing this demonstrates, and
`docs/aot-relations-design.md` for the full design.

# Binary size: what each relational/expression feature costs an AOT binary

`query_streaming.mojo` is the floor: `SELECT a, name FROM orders WHERE a > b`
over a 5-row in-memory batch, built from `marrow.expr.relations`'s
self-executing nodes (`InMemoryTable`/`Filter`/`Project`, no central planner)
with a fused comptime predicate (`col("a") > col("b")`, boxed via `BoxedValue`
from `marrow.expr.values`). Every other gate in this directory is that same
shape plus exactly one feature, so the `__text` delta against the floor is
what that one feature costs:

- **`query_arith.mojo`** — adds fused arithmetic (`Add`/`Sub`/`Mul`).
- **`query_exprs.mojo`** — adds string (`Like`), conditional (`Coalesce`),
  membership (`IsIn`), cast and temporal (`Year`) nodes.
- **`query_sort.mojo`** — adds `Sort` + top-K `Limit`.
- **`query_join.mojo`** — adds an equi-`Join` (`SwissHashTable`, `rapidhash`,
  radix partitioning).
- **`query_scan.mojo`** / **`query_scan_typed.mojo`** — the leaf is a
  `ParquetScan` instead of an `InMemoryTable`; `_typed` additionally pins the
  scan's column set at comptime (`leaf_of[Int64Type]() | leaf_of[StringType]()`).
- **`query_streaming_agg_fused.mojo`** / **`query_streaming_agg.mojo`** — add
  `Aggregate`, with the aggregate identity resolved at comptime
  (`AggFunc.of[NumericAgg[K, V]]()`) versus by runtime name (`AggFunc("sum")`,
  the shape the Python/ibis frontend uses).
- **`query_dynvalue.mojo`** — the same fat relational nodes as
  `query_streaming.mojo`, but the predicate is built the "runtime" way
  (`col("a") > col("b")` via operators) and boxed into `DynValue`
  (`marrow.expr.values`) instead of being constructed as a fused node directly.
- **`query_runtime.mojo`** — the full type-erased entry point end to end:
  `marrow.expr`'s `in_memory_table(batch).filter(...).select(...)` then
  `plan.execute()`.

There is no comptime-only / fully type-erased pair of binaries left to
contrast against these any more (see "Read this before quoting a number
below") — the current set of gates instead brackets *individual features*
against the one floor.

All gates are compiled with `-O3 -g0`, then `strip`ped, so the comparison is
release, no-debug-info code — the fairest apples-to-apples measurement of
what actually ships.

## Run it

```bash
pixi run binary_size
```

Builds every gate above, strips them, and prints the size/symbol table plus a
live per-module symbol-count breakdown (add gate names as arguments to build
only those, e.g. `pixi run binary_size query_join`; `query_streaming` is
always included as the ratio baseline). `benchmarks/binary_size/compare.py` is
a plain Python script (no dependencies beyond `mojo`, `nm`, `size`, and `strip`
on `$PATH`) — read it directly if you want to change what gets measured.

`benchmarks/binary_size/check_gate.py` is the CI gate: it rebuilds the subset
of gates recorded in `benchmarks/binary_size/baseline.json` and fails if any
of them grew `__text` by more than the recorded threshold. That JSON file,
not this document, is the enforced source of truth for regressions — see its
`_comment` field and `.github/workflows/binary_size.yml`.

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

## Historical note (osx-arm64, Mojo 1.0.0b3.dev2026070506) — page-quantized, not reproducible

Before `marrow.aot` and `marrow.dyn` were folded into today's
`marrow.expr.{values,relations,dynamic}`, this directory ran a four-way
comparison to answer one question: does a runtime, rewritable plan tree have
to pay for its type-erasure, or can it stay as small as a fully-monomorphized
one? The four binaries were `query_comptime` (the monomorphized layer, no
tag dispatch and no vtables at all), `query_erased_aot` (a runtime plan tree
of fused-only value boxes, no interpreter and no central planner),
`query_hybrid` (a runtime plan + runtime interpreter, but with the one
predicate fused), and `query_runtime` (fully type-erased plan and
interpreter, the ancestor of today's `query_runtime.mojo`).

| binary | unstripped | stripped | symbols | symbols (stripped) | `__TEXT` |
|---|---:|---:|---:|---:|---:|
| `query_comptime` | 710,320 B (710 KB) | 250,120 B (250 KB) | 247 | 24 | 229,376 B |
| `query_erased_aot` | 734,128 B (734 KB) | 250,136 B (250 KB) | 256 | 24 | 229,376 B |
| `query_hybrid` | 11,652,992 B (11.1 MB) | 7,734,104 B (7.7 MB) | 3,360 | 52 | 7,651,328 B |
| `query_runtime` | 11,649,328 B (11.1 MB) | 7,734,088 B (7.7 MB) | 3,353 | 52 | 7,651,328 B |

The table is in `__TEXT` (page-quantized — see the warning above), and none of
these four `.mojo` sources exist any more, so it cannot be re-measured or
re-verified; treat it as directional only. What it showed: `query_hybrid` and
`query_runtime` came out byte-for-byte identical, so fusing only the
*predicate value* while keeping a central planner/interpreter saved nothing.
`query_erased_aot`, which kept the plan as a runtime, walkable tree but made
the value box fused-only *and* let each node execute itself with no central
planner, landed within a few hundred bytes of the fully-monomorphized
`query_comptime` — i.e. rewritability and small binaries turned out to be
independent, and the win comes from a closed (non-exhaustive) driver plus
fused-only values, not from encoding the plan in the type system.

That conclusion is still the architecture today (`marrow.expr.relations`'s
self-executing nodes, `BoxedValue`'s fused-only box) — see
`docs/architecture.md` for the current, maintained "erasure boundary = fusion
boundary" framing, rather than this retired four-binary experiment.

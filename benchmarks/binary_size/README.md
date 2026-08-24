# Binary size: what each relational/expression feature costs an AOT binary

`query_streaming.mojo` is the floor: `SELECT a, name FROM orders WHERE a > b`
over a 5-row in-memory batch, built from `marrow.exprold.relations`'s
self-executing nodes (`InMemoryTable`/`Filter`/`Project`, no central planner)
with a fused comptime predicate (`col("a") > col("b")`, boxed via `BoxedValue`
from `marrow.exprold.values`). Every other gate in this directory is that same
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
  (`marrow.exprold.values`) instead of being constructed as a fused node directly.
- **`query_runtime.mojo`** — the full type-erased entry point end to end:
  `marrow.expr`'s `in_memory_table(batch).filter(...).select(...)` then
  `plan.execute()`.
- **`query_param.mojo`** — `query_scan_typed`'s exact query and plan, with the
  scan path and the predicate's right operand late-bound (`param("src",
  string)`, `param("min-a", int64)`) instead of literal, and
  `plan.execute_cli()` in place of `print(...execute())`. Isolates what a
  `param()` node plus the `execute_cli` entry point (argv parsing, `--help` /
  `--describe`, output-format dispatch) cost over a literal-bound scan.

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
  compatible with any difference up to 16 KB. The live gates show an even
  starker version of the same trap: as of 2026-08-05, `query_dynvalue` and
  `query_runtime` have **byte-identical stripped files** (4,232,744 each) while
  their `__text` differs by **1,600 bytes** (4,080,564 vs 4,078,964). File size
  agreeing exactly still doesn't mean the compiled code is the same. The claim
  needs re-measuring before it is repeated, and the exact byte counts drift
  every time either binary's dependencies change — see `baseline.json` for
  numbers a CI run has actually checked.

## Current baselines (osx-arm64, 2026-08-05) — `__text`, code only

`query_streaming` is the floor: one `InMemoryTable -> Filter -> Project` with a
column reference and a comparison. Every other gate is that shape plus one thing,
so the delta column is what the thing costs in an AOT binary.

**`benchmarks/binary_size/baseline.json` is the machine-readable, enforced
source of truth** for the four gates CI checks (`query_streaming`,
`query_join`, `query_streaming_agg_fused`, `query_streaming_agg`) — it is
regenerated by `check_gate.py --update`, so it cannot go stale as prose the way
this table can. The table below covers every gate (including the three
CI doesn't check) for the feature-by-feature comparison; if the two disagree,
`baseline.json` is right and this table needs re-running.

| gate | `__text` | Δ vs floor | ratio | what it adds |
|---|---:|---:|---:|---|
| `query_streaming` | 1,332,456 | — | 1.0x | the floor: `col`, `>` |
| `query_arith` | 1,342,448 | +9,992 | 1.0x | fused `+ - *` |
| `query_scan_typed` | 1,839,632 | +507,176 | 1.4x | `ParquetScan`, column set pinned at comptime |
| `query_scan` | 2,367,336 | +1,034,880 | 1.8x | `ParquetScan` — the whole reader |
| `query_sort` | 3,698,676 | +2,366,220 | 2.8x | `Sort` + top-K |
| `query_streaming_agg_fused` | 3,786,228 | +2,453,772 | 2.8x | `Aggregate`, comptime aggs |
| `query_join` | 3,860,660 | +2,528,204 | 2.9x | `Join` |
| `query_exprs` | 3,880,756 | +2,548,300 | 2.9x | string, conditional, membership, cast, temporal |
| `query_runtime` | 4,078,964 | +2,746,508 | 3.1x | interpreter + runtime plan |
| `query_dynvalue` | 4,080,564 | +2,748,108 | 3.1x | the erased lane (was the tag interpreter) |
| `query_streaming_agg` | 4,159,796 | +2,827,340 | 3.1x | `Aggregate`, runtime-named aggs |

Re-measure one gate without paying for the sweep (ten `-O3` builds, ~20 min):

```
pixi run binary_size query_scan
```

`query_streaming` is always built, since it is the ratio baseline.

### `query_param` — what a late-bound parameter costs (2026-08-19)

Measured with `pixi run binary_size query_param query_scan_typed` against a
freshly rebuilt `query_scan_typed`, since the tree has drifted since
2026-08-05 (`query_streaming` alone moved from 1,332,456 to 1,449,860 in that
time) — the pair below is internally consistent (same build), not a diff
against the stale table above.

| gate | `__text` | Δ vs `query_scan_typed` |
|---|---:|---:|
| `query_scan_typed` (fresh) | 2,025,432 | — |
| `query_param`, writers unconditional | 2,794,420 | +768,988 |
| `query_param`, writers gated (`CLI_WRITERS_ENABLED`) | 2,222,132 | +196,700 |

Far past the ~20 KB a parameter alone should cost (a `NumericParam`/
`StringParam` node is structurally a literal plus a pointer dereference
resolved once per batch — see `values.mojo`). Investigating the first number:
`_write_parquet_output` / `_write_ipc_output` in `marrow/exprold/relations.mojo`
pulled in the Parquet writer, the IPC writer, and the codec layer behind them
unconditionally, even though neither gate program calls either format. Both
are now gated behind `comptime CLI_WRITERS_ENABLED =
get_defined_bool["MARROW_CLI_WRITERS", False]()` — the same opt-in posture as
`GPU_ENABLED` — which recovered 572,288 bytes and zeroed out the
`marrow::parquet::writer` symbol bucket for the default (flag-off) build.
`-o out.parquet` / `-o out.arrow` raise instead of writing unless the binary
is built with `-D MARROW_CLI_WRITERS=true`; `--format table` / stdout is
unaffected either way.

The remaining +196,700 bytes is not writer-linkage — a symbol-level diff
(`nm -n` address-delta) attributes it to `execute_cli`'s own body (22,672 B:
the argv loop, `--help`/`--describe` short-circuit, `-o`/`--format` parsing),
`parse_params` (13,728 B: `atol`/`atof` argv-to-scalar conversion), the
`ParamCell`/`register_param`/`PathSpec` machinery (~7 KB), and — the largest
single piece, ~87 KB — a **second instantiation** of the `DynArray`/
`DynScalar` dispatch ladder and `SIMD.write_to`, because `_write_cli_output`'s
`print(result)` is a different call site from `query_scan_typed`'s inline
`print(...execute())` (see CLAUDE.md's closure-duplication note — each call
site of a closure-taking dispatcher is its own specialization). That is the
fixed cost of routing output through `execute_cli` at all, not a marginal
per-parameter cost, and it is out of scope for the writer-gating decision this
task exists to make; it was not gated further.

### Why the metric had to be fixed first

`query_arith` is the demonstration. Against the floor it is **+9,992 bytes of
code** — and **16 bytes *smaller* by stripped file size** (1,406,744 vs
1,406,760). The old metric would have reported fused arithmetic as free, or
faintly negative. It is neither.

## Historical note (osx-arm64, Mojo 1.0.0b3.dev2026070506) — page-quantized, not reproducible

Before `marrow.aot` and `marrow.dyn` were folded into today's
`marrow.exprold.{values,relations,dynamic}`, this directory ran a four-way
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

That conclusion is still the architecture today (`marrow.exprold.relations`'s
self-executing nodes, `BoxedValue`'s fused-only box) — see
`docs/architecture.md` for the current, maintained "erasure boundary = fusion
boundary" framing, rather than this retired four-binary experiment.

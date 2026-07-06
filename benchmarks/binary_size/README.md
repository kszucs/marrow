# Binary size: comptime (AOT) vs. hybrid vs. runtime relational plans

Three files implement the exact same query — `SELECT a, name FROM orders
WHERE a > b` — over the same 5-row in-memory batch, producing identical
output:

- **`query_comptime.mojo`** — the fully-monomorphized layer from
  `marrow.aot.relations` (`Table`, `Column`, `Project`, `Filter`). The whole plan
  is one nested generic type; `.execute(batch)` compiles straight to column
  loads, a SIMD comparison, and a filter call. No tag dispatch, no vtables.
- **`query_hybrid.mojo`** — relational *structure* stays runtime/type-erased
  (`marrow.dyn`'s `AnyRelation`, `Planner.build()`, the pull-based
  `RelationProcessor` pipeline — same as `query_runtime.mojo`), but the
  *predicate* is a comptime-typed `Gt(Column, Column)` node
  (`marrow.aot.values`) boxed into a runtime `Expr` via the `FUSED` tag
  (`Expr(gt_node)`), so evaluating it is a direct call into the fused
  vectorize loop rather than a walk through `Expr.eval()`'s tag interpreter.
- **`query_runtime.mojo`** — the existing type-erased layer end to end:
  `in_memory_table(batch).filter(...).select(...)` then `execute(plan)`,
  predicate built from `col("a") > col("b")` and evaluated by `Expr.eval()`'s
  tag interpreter.

All three are compiled with `-O3 -g0`, then `strip`ped, so the comparison is
release, no-debug-info code — the fairest apples-to-apples measurement of
what actually ships.

## Run it

```bash
pixi run binary_size
```

Builds all three, strips them, and prints the size/symbol table plus the
per-module symbol breakdown below. `benchmarks/binary_size/compare.py` is a
plain Python script (no dependencies beyond `mojo`, `nm`, `size`, and `strip`
on `$PATH`) — read it directly if you want to change what gets measured.

## Result (osx-arm64, Mojo 1.0.0b3.dev2026070506)

| binary | unstripped | stripped | symbols | symbols (stripped) | `__TEXT` |
|---|---:|---:|---:|---:|---:|
| `query_comptime` | 693,712 B (694 KB) | 233,608 B (234 KB) | 246 | 24 | 212,992 B |
| `query_hybrid` | 11,653,040 B (11.1 MB) | 7,734,104 B (7.7 MB) | 3,360 | 52 | 7,651,328 B |
| `query_runtime` | 11,649,200 B (11.1 MB) | 7,734,088 B (7.7 MB) | 3,353 | 52 | 7,651,328 B |

**~33.1x smaller, stripped**, for the fully-monomorphized version. `size` on
the stripped binaries confirms the gap is genuinely in compiled code, not
just symbol-table noise — the `__TEXT` (executable code) segment column above
tells the same story as the file-size columns.

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
regardless of which ones this query actually uses, plus the `AnyRelation`
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

### Per-module symbol counts (unstripped)

Counting distinct symbols whose mangled name references each module (a
symbol can match more than one bucket, since Mojo names embed nested generic
type params — this is a proportional breakdown, not a strict partition):

| module | `query_comptime` | `query_hybrid` | `query_runtime` |
|---|---:|---:|---:|
| `kernels::execution` | 9 | 667 | 667 |
| `dtypes` | 58 | 566 | 562 |
| `views` | 2 | 455 | 455 |
| `arrays` | 39 | 376 | 376 |
| `kernels::arithmetic` | 0 | 371 | 371 |
| `builders` | 1 | 89 | 89 |
| `kernels::hashing` | 0 | 141 | 141 |
| `kernels::compare` | 0 | 74 | 74 |
| `kernels::filter` | 10 | 66 | 66 |
| `kernels::join` | 0 | 11 | 11 |
| `kernels::groupby` | 0 | 17 | 17 |
| `kernels::boolean` | 0 | 13 | 13 |
| `scalars` | 0 | 20 | 20 |
| `buffers` | 10 | 34 | 34 |
| `dyn::executor` | 0 | 29 | 29 |
| `dyn::relations` | 0 | 34 | 34 |
| `dyn::values` | 0 | 16 | 13 |
| `aot::values` | 0 | 4 | 0 |
| `aot::relations` | 0 | 0 | 0 |

Why the biggest buckets are so lopsided: `comptime_query` only ever
instantiates the *exact* concrete types this one query needs —
`Column[Orders,"a",Int64Type]`, `StringColumn[Orders,"name"]`, `Gt[...]` —
nothing else exists at compile time, so nothing else gets generated.
`query_hybrid`/`query_runtime` go through `AnyArray`, which erases the dtype
to a runtime tag — any code operating on an `AnyArray` (`Expr.eval()`,
`AnyRelation`'s processors, kernel dispatch functions) can't know at compile
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

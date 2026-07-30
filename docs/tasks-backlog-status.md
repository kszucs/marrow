# Backlog status — the index

The one place that says what is actually open. Every entry was checked against
the code, not read off a header. **Last verified: 2026-07-30.**

- `docs/tasks-code-quality.md` — Q tasks (dedup, layering, soundness)
- `docs/tasks-execution-engine.md` — T tasks (M1 = ClickBench)
- `docs/tasks-expr-kernels-layering.md` — L tasks (expr/kernels layering)
- `docs/tasks-expr-simplification.md` — duplication inside `marrow/expr`
- `docs/tasks-type-coverage.md` — V tasks (map, view layouts)
- `docs/tasks-aggregate-followups.md` — group-by specifics

> **Re-verify before trusting this.** These status lines have been wrong before —
> a 2026-07-27 sweep found six tasks marked open that were already done, and on
> 2026-07-30 this index and `tasks-code-quality.md` disagreed about five tasks
> (Q0.6, Q0.8, Q1.1, Q1.2, Q2.4) that the index had closed and the task file
> still listed as open. Check with `grep`, not with a header.

Resolved tasks are **deleted, not struck through** — git history has them.

---

## Open — quality

| task | what is left |
|---|---|
| **Q0.5** | `project`/`aggregate` derive output dtypes by executing against a 0-row batch (`relations.mojo:339`). A fused value's `OutType` is statically known, so it is unnecessary for that lane; ~16 KB, and it retires a hand-built `benchmarks/binary_size` exception. |
| **Q2.1** | No type reaches into another's `_`-prefixed fields: expose `DynType`'s variant, add `Allocation.is_device()`/`is_host()`. |
| **Q2.3** | Validity plumbing: `bitmap_and` should take `Optional[BitmapView]` — today's signature yields offset-misaligned validity for sliced inputs. Layout-preserving. |
| **Q2.5** | Aggregates: kernels as the only representation. Large — do in a worktree, gated by Q6.1. |
| **Q3.2 / Q3.3 / Q3.4 / Q3.5** | Free-function elimination: core+memory, Parquet+IPC (`ipc.mojo` still has 11, and `_slice_body` copies byte-by-byte), the Python layer, and expr. |
| **Q4.1** | Missing value types: `Grouping`, `JoinKind` (its "emits right columns?" predicate is re-derived three times with different membership), `JoinIndex`, `BuildPartition`. |
| **Q4.3 / Q4.4 / Q4.6** | Parquet leaf visitor (8 drifting type ladders), `ipc.mojo` → package, and Q4.6 — a Parquet scan still doubles a minimal AOT binary. |
| **Q6.1** | Cross-engine aggregate benchmark with the AOT path measured. Gates every Q2.5 round. |
| **L2** | Split `values.mojo` — 3,153 lines. The *lane dependency* it was named for is gone; only file size remains. |
| **L3** | `AggFunc`'s late binding: make it honest or remove it. |
| **L6** | A `Scan` trait above the file formats. **Do it before adding CSV/IPC sources, not after.** |
| **V0–V6** | Map is half-shipped (no `MapScalar`, no IPC in either direction, no `cast` arm); the binary-view and list-view layouts are absent. |
| **D1 / D2** | Accepted, not scheduled — both need array-internal changes the layout freeze forbids. |

**Q4.5 needs a test, not work.** Its core landed 2026-07-30: `Value.prune`
exists and the box delegates to it, so a *fused* predicate can skip row groups
where the box previously answered `unknown()` for every fused node. But all
seven cases in `test_pruning.mojo` build an **erased** predicate, so the fused
path is unasserted. Cheapest item on this page.

---

## Open — features (M1 / ClickBench)

Almost entirely sequential, and now the whole critical path.

| task | state |
|---|---|
| **T3.1** — optimizer v1 | **Not started.** No `optimizer.mojo`. The scan side is ready: a `ParquetScan`'s schema *is* its projection, so projection pushdown is a schema rewrite, and the predicate it carries drives row-group and page skipping. |
| **T3.2** — Python lazy bindings | **Not started.** |
| **T3.3** — ibis-flavoured `Table`/`Column` | **Not started.** `python/marrow/` has only `__init__.py`, `compute.py`, `parquet.py`. |
| **T3.4 / T3.5** | ClickBench through the lazy plan — the M1 acceptance gate, blocked on everything above. |

---

## Closed 2026-07-30 — the expression-layer refactor

`TagValue`, the 41-tag interpreter, is deleted. `dynamic.mojo` 1,087 → 113
lines. A runtime-built expression is now made of the *same* nodes the fused lane
uses: `col("a") + col("b")` is an `Add[DynValue, DynValue]`.

    query_dynvalue   5,236,148 -> 3,956,596   -1,279,552  -24.4%
    query_streaming  1,266,040 -> 1,303,028      +36,988   +2.9%

The fused gates grew ~37 KB, ~30 KB of which is `Value.prune` — the capability
noted above. Perf vs `BASELINE.md`: 57/57 rows, nothing attributable.

It closed **L8** outright, reduced **L2** to a file-size question, closed **L7**
(`Kernel` now carries `error`/`expect_same_length`/`expect_same_dtype` and every
named kernel conforms), and delivered **Q4.5**'s core. Full account in
`docs/tasks-step3-expression-nodes.md`.

Also closed and removed: Q0.0, Q0.2, Q0.3, Q0.4, Q0.6, Q0.7, Q0.8, Q1.1, Q1.2,
Q1.3, Q1.4, Q2.2, Q2.6, Q2.4, Q3.1, Q4.2 (dropped), T0.7, T2.4, L1, L4, L5, L9.

---

## Constraints worth knowing before planning

Each of these cost real time to find and invalidates an approach that looks obvious.

- **`Buffer` requires 64-byte pointer alignment.** So `read_at` cannot return a
  sub-`Buffer` at an arbitrary file offset, and neither can IPC's `_slice_body`: Arrow IPC
  pads buffers to 8, not 64. A source owns *one* whole-file `Buffer` and hands out
  `BufferView`s. This constraint has now blocked two separate designs.
- **An operator with no benchmark has no performance.** T2.4's per-row-group scan shipped
  a **4.7x** regression that every test passed through, because nothing benched the scan
  operator — `bench_parquet.mojo` covers `read_table`, which takes a different path.
  `marrow/expr/tests/bench_scan.mojo` now covers it. Two traps inside that: a benchmark
  whose captured `path` was not `keep()`-alive after `b.iter[call]()` silently measured
  **nothing** (it reported 17,774 GElems/s — check throughput for physical plausibility
  before believing a flat A/B), and the pytest-benchmark table prints **mixed units** per
  row, so compare `--benchmark-json` medians (seconds), never the table's numbers.
- **A fixed per-call cost hides until you change how often the call happens.**
  `ParquetFile.read` built a `CompressionLibs` per worker and the first decompress
  `dlopen`s the codec library. Invisible at one read per file; 4.7x at one read per row
  group. When splitting one big operation into many small ones, audit what the operation
  set up *once*.
- **The binary-size gate's headline number is quantized to 16 KB.** Apple Silicon uses
  16 KB pages, so a stripped binary's *file size* — what `compare.py`'s ratio table and
  every figure quoted so far report — moves in 16,384-byte steps. A real +1,728-byte change
  showed up as +16,504 with one *fewer* symbol. Measure `size -m <binary>` → `Section
  __text`; treat any recorded file-size delta near 16 KB as a page crossing, not code.
  Folded into Q0.8.
- **A generic wrapper around an already-erased dispatch is not free.** Folding Q0.4's
  twelve promote-then-dispatch sites in `TagValue.eval` into one
  `_arith[K: BinaryNumericKernel]` helper cost **+115,600 bytes** on `query_dynvalue`
  (file size; the real code delta is smaller — see the 16 KB quantization note above), while
  writing the same four lines inline in each arm cost a fraction of that.
  A parameterised method is instantiated per kernel and each instantiation carries its own
  copy of what it touches, so tidying an erased ladder into a generic helper can cost
  several times the thing it was tidying. Measure before assuming a refactor is neutral —
  and measure *one* gate binary directly (`mojo build -O3 -g0 -I . …query_dynvalue.mojo`,
  ~2.5 min) rather than the whole `pixi run binary_size` sweep (~10 min).
- **Reachability intuitions about the interpreter are usually wrong; stub and measure.**
  The kernel-layer version of Q0.4 was reverted partly for making "the cast fanout
  reachable from every erased binary dispatch". In the expression layer that cost is
  literally zero — `cast` is already linked in by `TagValue`'s `CAST` arm, and stubbing
  both `cast_array` calls out left the gate binary byte-identical.
- **`origin_of(a, b)` is an origin *union*** (used by `Span.__merge_with__`). It is what
  lets a function return values borrowed from either of two storages — the thing
  `ImmUntrackedOrigin` was papering over in the Parquet decode path.
- **A comptime *conditional type* carries no trait conformance** and does not reduce at a
  return site, even inside a `comptime if` that selected the branch; `rebind` does not
  rescue it. This blocks Q0.4's promote-at-construction design.
- **A capturing closure's type is parameterised by its creating scope**, so it cannot be
  stored in a struct field and outlive that scope. Every stored callback must be `thin`.
- **`ctx.stripe` bodies may not raise**, and widening it miscompiles: the parameter form of
  `sync_parallelize` that accepts a raising worker needs an implicitly-capturing closure,
  whose captures are silently not made. Watch for "assignment was never used" warnings on
  buffers the body writes — that is the tell.
- **Benchmarks here vary 10–18 % run to run.** Interleave repeats across refs (nesting them
  concentrates machine drift on whichever ref is measured last and *invents* regressions),
  use five or more, and compare ranges. `bench_add_int32_1m_auto` asks for all physical
  cores and is the least trustworthy number in the suite.
- **`precompile` and `pytest` are mutually exclusive** unless the artifact is cleared —
  fixed by writing to `.precompile/`, but worth knowing if an old checkout misbehaves.

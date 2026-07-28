# Backlog status — quality + features

A cross-cutting snapshot of the two task backlogs, verified against the code rather
than read off their own headers. **Last verified: 2026-07-28.**

- `docs/code-quality-tasks.md` — the Q/L tasks (dedup, layering, soundness)
- `docs/execution-engine-tasks.md` — the T tasks (M1 = ClickBench)

> **Re-verify before trusting this.** Both backlogs' status lines have been wrong before —
> on 2026-07-27 a sweep found six tasks marked open that were already done, and three
> separate scope estimates in Q1.2 were wrong by an order of magnitude in the same
> direction. Check with `grep`, not with the header.

---

## Closed recently

**Quality.** Q0.2, Q0.3, **Q0.4**, Q1.1, Q1.2, Q1.3, Q2.4, Q2.5 step 2, Q3.4's headline,
Q0.7, part of Q3.3, and layering L1/L4/L5/L9. Q0.6 was *measured* and closed as a net
increase (sharing the binary dispatch default costs more lines than it saves). L7's premise
no longer holds — `Kernel` carries `error()` and the argument checks, so it is not a
name-only marker.

**Features.** Wave 0 and Wave 1 appear complete: the `ByteSource` seam (T0.2), Kleene
logic, and the conditional / membership / string / temporal / aggregate kernels. Wave 2 is
complete — T2.1–T2.3b (fused and dynamic wiring, Sort/Limit/TopK, Aggregate + HAVING) plus
**T2.4** (per-row-group streaming scan, projection-into-scan, chunk-relative `PageReader`).

---

## Open — quality

| task | what is left |
|---|---|
| **Q0.5** | A fused value's `OutType` is statically known, so `project`/`aggregate` need not probe it by executing against a 0-row batch. Worth 16,528 bytes on the fused gate and retires the hand-built `benchmarks/binary_size` exception. Also the residual half of Q0.4: the interpreted lane promotes at *execution*, so a `DynValue` tree's output dtype is still only knowable by running it. |
| **Q3.1 tail** | `conditional.mojo` (11 free fns, no struct), `temporal.mojo` (18, incl. `String`-keyed `date_trunc` and 9 delegators only tests call), `membership.mojo` (5). |
| **Q3.3 tail** | `ipc.mojo` still has 12 free fns. `_slice_body` still copies each column buffer byte-by-byte; a memcpy needs a new `Buffer` factory, since `unsafe_ptr()` is restricted to `buffers`/`views`/`c_data`. |
| **Q4.1** | Missing value types: `Grouping`, `JoinKind` (its "emits right columns?" predicate is re-derived three times with different membership), `JoinIndex`, `BuildPartition`. |
| **Q4.2** | F2 has ~30 operators F1 lacks and nothing enforces parity — the same defect class as Q0.4, and the one most likely to bite when adding expressions. |
| **Q4.3–Q4.5** | Parquet leaf visitor (8 drifting type ladders), `ipc.mojo` → package, fused `prune` (the AOT frontend cannot prune row groups at all). |
| **L2 → L6** | Extract `AnyValue` from `values.mojo` (the comptime lane still imports the runtime lane), then a `Scan` trait. **Do L6 before adding CSV/IPC sources, not after.** |
| **L8** | Decompose `DynValue` — 41 tags, 7 fields, two of them overloaded. Large; schedule alone. |
| **D1 / D2** | Accepted, not scheduled: both need array-internal changes, which the layout freeze forbids. |

---

## Open — features (M1 / ClickBench)

This is the critical path, and it is almost entirely sequential.

| task | state |
|---|---|
| **T3.1** — optimizer v1 | **Not started.** No `optimizer.mojo` exists. The scan side is now ready for it: a `ParquetScan`'s schema *is* its projection, so projection pushdown is a schema rewrite, and the predicate it already carries drives row-group and page skipping. |
| **T3.2** — Python lazy bindings | **Not started.** |
| **T3.3** — ibis-flavoured `Table`/`Column` | **Not started.** `python/marrow/` has only `__init__.py`, `compute.py`, `parquet.py` — no lazy surface. |
| **T3.5** — ClickBench through the lazy plan | The M1 acceptance gate; blocked on everything above. |

---

## Suggested order

1. ~~**T2.4**~~ and ~~**Q0.4**~~ — both done 2026-07-28.
2. **Wave 3 chain**: T3.1 → T3.2 → T3.3 → T3.5. Genuinely serial, and the bulk of M1 —
   now the whole critical path.
3. Dedup (Q3.1/Q3.3 tails, Q4.x) and layering (L2 → L6) as capacity allows — none of it
   blocks M1.

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
- **A generic wrapper around an already-erased dispatch is not free.** Folding Q0.4's
  twelve promote-then-dispatch sites in `DynValue.eval` into one
  `_arith[K: BinaryNumericKernel]` helper cost **+115,600 bytes** on `query_dynvalue`
  (5,438,904 → 5,554,504); writing the same four lines inline in each arm cost +16,528.
  A parameterised method is instantiated per kernel and each instantiation carries its own
  copy of what it touches, so tidying an erased ladder into a generic helper can cost
  several times the thing it was tidying. Measure before assuming a refactor is neutral —
  and measure *one* gate binary directly (`mojo build -O3 -g0 -I . …query_dynvalue.mojo`,
  ~2.5 min) rather than the whole `pixi run binary_size` sweep (~10 min).
- **Reachability intuitions about the interpreter are usually wrong; stub and measure.**
  The kernel-layer version of Q0.4 was reverted partly for making "the cast fanout
  reachable from every erased binary dispatch". In the expression layer that cost is
  literally zero — `cast` is already linked in by `DynValue`'s `CAST` arm, and stubbing
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

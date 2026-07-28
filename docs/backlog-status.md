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

**Quality.** Q0.2, Q0.3, Q1.1, Q1.2, Q1.3, Q2.4, Q2.5 step 2, Q3.4's headline, Q0.7,
part of Q3.3, and layering L1/L4/L5/L9. Q0.6 was *measured* and closed as a net increase
(sharing the binary dispatch default costs more lines than it saves). L7's premise no
longer holds — `Kernel` carries `error()` and the argument checks, so it is not a
name-only marker.

**Features.** Wave 0 and Wave 1 appear complete: the `ByteSource` seam (T0.2), Kleene
logic, and the conditional / membership / string / temporal / aggregate kernels. Wave 2's
T2.1–T2.3b are in — fused and dynamic wiring, Sort/Limit/TopK, Aggregate + HAVING.

---

## Open — quality

| task | what is left |
|---|---|
| **Q0.4** | The two expression lanes disagree on `int64 + float64`: the fused algebra promotes, the interpreted one raises `dtype mismatch`. Promote-at-construction is blocked (see *Constraints*), so scope it to the interpreted lane. **The only correctness gap left.** |
| **Q0.5** | A fused value's `OutType` is statically known, so `project`/`aggregate` need not probe it by executing against a 0-row batch. Worth 16,528 bytes on the fused gate and retires the hand-built `benchmarks/binary_size` exception. |
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
| **T2.4** — streaming Parquet scan | **Not started.** `ParquetScanProcessor` reads the whole file into one `Table` and slices it, so memory scales with file size. Needs per-row-group decode plus `columns=` pushdown. Prerequisite: remove `ParquetFile._span()` and make `PageReader` chunk-relative. |
| **T3.1** — optimizer v1 | **Not started.** No `optimizer.mojo` exists. |
| **T3.2** — Python lazy bindings | **Not started.** |
| **T3.3** — ibis-flavoured `Table`/`Column` | **Not started.** `python/marrow/` has only `__init__.py`, `compute.py`, `parquet.py` — no lazy surface. |
| **T3.5** — ClickBench through the lazy plan | The M1 acceptance gate; blocked on everything above. |

---

## Suggested order

1. **T2.4**, beginning with `_span()` removal — a listed M1 dependency, and the parquet
   code is currently well understood.
2. **Q0.4** — the last correctness gap, and small once scoped to the interpreted lane.
3. **Wave 3 chain**: T3.1 → T3.2 → T3.3 → T3.5. Genuinely serial, and the bulk of M1.
4. Dedup (Q3.1/Q3.3 tails, Q4.x) and layering (L2 → L6) as capacity allows — none of it
   blocks M1.

---

## Constraints worth knowing before planning

Each of these cost real time to find and invalidates an approach that looks obvious.

- **`Buffer` requires 64-byte pointer alignment.** So `read_at` cannot return a
  sub-`Buffer` at an arbitrary file offset, and neither can IPC's `_slice_body`: Arrow IPC
  pads buffers to 8, not 64. A source owns *one* whole-file `Buffer` and hands out
  `BufferView`s. This constraint has now blocked two separate designs.
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

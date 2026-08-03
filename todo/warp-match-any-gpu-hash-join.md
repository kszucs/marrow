# Explore `warp.match_any()`/`match_all()` for GPU hash join / group-by

## Correction to the original note

The first pass at this idea (written right after the b3 changelog scan)
claimed marrow's hash-join/group-by kernels "already run GPU probe/build
paths." That's wrong — checked more carefully:

- `marrow/kernels/hashtable.mojo` (`SwissHashTable`, the SIMD-group Swiss
  table every join/groupby is built on) imports `std.gpu.host.DeviceContext`
  but never calls anything from it — `grep -c DeviceContable hashtable.mojo`
  shows exactly one match, the import line itself. Dead import.
- `marrow/kernels/join.mojo`'s `ctx` parameter is always
  `ExecutionContext.parallel(self._num_threads)` — CPU multi-threading, not
  GPU. Its own docstring says the GPU context slot is a "future GPU
  acceleration hook" — i.e. not implemented.
- `marrow/kernels/aggregate.mojo`'s only GPU involvement is delegating
  simple reductions to `views.reduce` (sum/min/max over a single array) —
  unrelated to hash-based grouping.

So: **there is no GPU hash join or GPU group-by today.** `warp.match_any()`
/ `warp.match_all()` (new in b3 — portable same-value lane masks: NVIDIA
`match.any.sync`, AMD ballot fold, Apple shuffle emulation) can't be slotted
into an existing kernel. This note is about whether a GPU hash-join/group-by
would be worth building *and* would want these intrinsics — two decisions,
not one.

## What exists today (the thing a GPU port would replace/parallel, not patch)

`SwissHashTable` in `hashtable.mojo` is a from-scratch Swiss-table
implementation, CPU-only, SIMD-group matching with pipelined probing:

```
Hash Function  →  Partitioner  →  SwissHashTable  →  Operator (join / groupby)
```

Entry points: `insert_hashes`, `build_hashes`, `probe_hashes`, plus
`insert`/`build`/`probe` wrappers. `RadixPartitioner` splits rows across
partitions by hash before they reach the table, presumably to bound
per-partition working-set size for cache locality — the same reason a GPU
version would want partitioning too, probably per-threadblock rather than
per-CPU-core.

## Where `match_any`/`match_all` would actually help, if this gets built

The plausible use is inside a GPU probe kernel: when a warp of threads is
probing the same hash bucket (or a set of buckets that collide into the
same warp), `match_any()` gives you — for free, without shared-memory
traffic — a mask of which lanes in the warp are looking at equal keys. That
can shortcut redundant global-memory key comparisons when many probe rows
in a warp share a key (skewed join keys, or a `GROUP BY` on a
low-cardinality column). This is a real, well-known GPU hash-table
optimization pattern in principle — I have not verified it against any
GPU-Swiss-table reference implementation, and marrow's CPU Swiss table's
specific probing sequence (SIMD group matching) may or may not map cleanly
onto a warp-level equivalent.

## What the actual spike is

Not "add match_any to hashtable.mojo" — it's:

1. Decide whether a GPU hash-join/group-by path is worth building at all
   for marrow's workloads before anything else. This is a much bigger
   design question than the language-feature note it started as — probably
   deserves its own design doc rather than a todo item, if the answer is yes.
2. If yes: prototype a minimal GPU probe kernel (even a toy one, independent
   of `SwissHashTable`) using `warp.match_any()` for intra-warp key dedup,
   and benchmark it against the existing CPU `SwissHashTable::probe_hashes`
   on a skewed-key workload, since that's the specific case where this
   would pay off — a uniform-key workload probably won't show a difference.
3. Only then decide whether it's worth integrating into
   `kernels/join.mojo` / `kernels/hashtable.mojo` for real, following
   whatever GPU dispatch convention `views.reduce`/`views.apply` already
   established (`ExecutionContext.gpu(ctx)`, `has_accelerator_support[...]`
   gating, etc. — see `marrow/views.mojo`).

## Status

Speculative, two levels removed from "ready to prototype." The precondition
(GPU hash join existing) isn't met yet — resolve that design question
first, independently of whether `match_any`/`match_all` end up being useful
inside it.

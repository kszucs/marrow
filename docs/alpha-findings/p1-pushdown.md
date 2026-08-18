# P1 — Projection pushdown, and making the plan IR walkable

Branch: `worktree-agent-a5de174c4a527ce7a` (off `alpha` @ `b130391`).

## The problem, restated from the measurement

`ParquetScan`'s schema *is* its projection — the scan reads only the columns its
schema names — but **nothing ever narrowed it**. A lazy plan built by the Python
frontend starts from `read_parquet(path)`, whose scan schema is the *file's*
schema, and no rewrite ever removed a column from it. So every ClickBench query
against `hits_0.parquet` (1M rows x 105 columns) decoded all 105 columns,
`COUNT(*)` included.

That is the flat ~260 ms floor in the numbers this task started from, and it is
why the marrow/polars ratio inverted with query shape: Q34 (`GROUP BY URL`,
which polars must also read a big string column for) was 4.4x, while Q1
(`count(*)`, which polars answers from the footer) was 455x. The 100x rows were
not measuring a slow engine; they were measuring 104 columns read for nothing.

## What was already there

`referenced_columns()` — "the column names this node reads" — was **fully
implemented** and had no consumer. Verified rather than assumed:

- `values.mojo`: declared abstract on the `Value` trait (line 405), so the
  compiler forces every conformer to answer. All 35 fused nodes do, and the
  answers are right: the three column leaves return their own name, the two
  literals return `[]`, and every composite unions its children (`CaseWhen`
  unions all three).
- `dynamic.mojo:583`: `DynValue` returns its payload name for a `column` tag and
  otherwise the deduped union over `_kids`. Every runtime node stores its
  operands in `_kids` — `is_in`'s value set and `cast`'s target dtype are
  payloads, not children — so nothing is missed.
- `values.mojo:605`: `BoxedValue` forwards through a trampoline.

So the analysis was complete at the *expression* level and absent at the
*relation* level. This work is the `Relation`-level consumer.

## The traversal decision

**Decided: add read-only traversal (`children()`) *and* a rewriting virtual
(`with_projection`), rather than either alone.** The reasoning, and why the
obvious "just add `inputs()`" is not sufficient on its own:

`DynRelation` erases nodes behind `ArcPointer[NoneType]` plus function-pointer
trampolines. The layer already records (in `Relation.with_predicate`'s docstring)
that a trampoline **field** whose function type mentions `DynRelation` makes the
struct recursive and Mojo rejects it. That is a constraint on *fields*, not on
methods — and it is why `with_predicate` returns `Optional[ArcPointer[NoneType]]`
and lets the caller keep its own trampolines and swap only the pointer.

Testing that boundary was the first thing this work did, because the whole
design hangs on it:

- `var _virt_children: def(ArcPointer[NoneType]) thin -> List[DynRelation]` —
  **accepted.** A `List[DynRelation]` return is fine; the recursion check trips
  on the erased type appearing in a field's *own* representation, and a list
  behind a pointer does not.

So read-only traversal is cheap and was added. But it does **not** on its own
enable a rewrite: reading a node's children tells you nothing about how to
rebuild that node around new ones, and erasure means a generic rewriter cannot
construct a node whose type it does not know. A generic `inputs()`-driven
optimiser therefore needs, at minimum, `inputs()` + `with_inputs()` +
a per-child `required_columns()` — three virtuals, of which the third *is* the
entire rewrite logic anyway.

The single `with_projection(needed)` virtual collapses those three into one, and
each node's implementation is five lines of its own concrete code. It also
**does not add a third rewrite protocol**: it is `with_predicate`'s protocol,
verbatim — same erased return, same "same concrete type, swap the pointer" rule,
same reason. That mattered: the alpha findings note the layer already had *two*
incompatible mechanisms and that the other one (kind-tag + `downcast[Sort]`)
already caused a correctness bug with a parameterised `ParquetScan`. Adding a
third would have been the finding repeating itself.

`ParquetScan.with_projection` also inherits the fix that motivated
`with_predicate`: `Self.leaves` is in scope inside the node, so the rewritten
scan keeps its comptime `LeafSet`. A `downcast`-based rewriter could not have
named it, which is exactly the bug the docstring records.

`children()`, not `inputs()`, because **`Aggregate.inputs` already names the
aggregate value expressions** — the trait method would have collided on that one
struct (`invalid redefinition of 'inputs'`). The name is the only thing that
changed; the capability is the one the findings asked for.

## The rewrite

`DynRelation.optimize()` seeds the rewrite with the root's own column names and
walks down. Each node widens or replaces the set according to what it reads:

| node | column set passed to its input |
|---|---|
| `Filter` | `needed` ∪ predicate's columns |
| `Sort` | `needed` ∪ all key expressions' columns |
| `Limit` | `needed` unchanged |
| `Project` | drops the outputs not in `needed`, then passes the **surviving** values' columns |
| `Aggregate` | discards `needed`; passes keys' ∪ aggregate inputs' columns |
| `Join` | *not rewritten* — see below |
| `ParquetScan` | terminates: schema narrowed to the set, in the file's field order |

Two properties are what make it safe, and both have tests:

**The root's output schema never changes.** `optimize()` seeds `needed` with the
root's own columns, and a schema-passthrough node (`Filter`/`Sort`/`Limit`) can
only narrow to a subset its parent asked for. So a plan that emits all 105
columns still reads all 105 — narrowing only ever happens *below* a node whose
schema is its own (`Project`, `Aggregate`), where the width beneath it is
invisible.

**A scan never narrows to nothing.** `COUNT(*)` is `lit(1).count()`, which
references no column; a zero-column read produces zero-row batches, and
`ParquetScanProcessor._load_next_window` reads a zero-row batch as end-of-file.
So the empty case keeps one column and picks the narrowest fixed-width one
(`DynType.byte_width()` is 0 for the variable-width types, which is precisely the
set to avoid) — reading one `uint8` beats reading one `binary`. `Project` has the
same guard for the same reason.

`Project` dropping unread outputs is what makes the rewrite transitive:
`select("x", "y").aggregate(sum(y))` reads only `y`, because the input set is
taken from the values that *survive*, not from every value the projection was
written with.

### Why `Join` is excluded

`Join` holds `left_key_indices` / `right_key_indices` — **positions** into its
children's schemas, fixed when the node was built. Narrowing a child would leave
the plan joining on whichever column happened to land at that index: a silently
wrong answer, not an error. Pushing a projection through a join means
recomputing those indices and the output schema, which is a separate rewrite with
its own conditions. `Join` therefore inherits the default `with_projection`
(`None` — leave the subtree alone) and both sides stay whole. It does implement
`children()`, so it is walkable.

## Predicate pushdown

Left as it was, deliberately. Today's rule — push into a `ParquetScan` only when
the `Filter` sits directly on one — is what every ClickBench query hits anyway
(`read_parquet(...).filter(...)`), so extending it buys nothing measurable here,
and each extension has a correctness condition that is not free:

- through `Limit`: **unsound.** The predicate is pruning metadata, so pushing it
  below a `Limit` changes which row groups the `Limit` counts from.
- through `Project`: sound only for pass-through columns. A projection may rename
  (`rename` lowers to `Project`), and a renamed predicate column whose new name
  collides with a *different* column below prunes on the wrong statistics — again
  a wrong answer rather than an error. It needs a per-column
  `bound_column`-and-same-name check.
- through `Sort` and a nested `Filter`: sound, and the two cheap ones.

Conjunct splitting is the real precondition for good pruning and it is **not**
done: `Filter` holds one `predicate`, so an `AND` cannot be partially pushed, and
detecting an `AND` means matching `DynValue._tag == "and"` on the runtime lane
with no fused-lane equivalent. That is the next step, and it wants the two-lane
question answered rather than a tag test bolted on.

## Binary size

`pixi run python3 benchmarks/binary_size/check_gate.py`, both runs on this
machine, base tree vs this branch. `baseline.json` was **not** touched.

```
before                           baseline     measured      delta      pct
query_streaming                 1,484,652    1,439,756    -44,896  -3.024%
query_join                      1,507,836    1,464,616    -43,220  -2.866%
query_streaming_agg_fused       1,417,476    1,388,848    -28,628  -2.020%
query_streaming_agg             1,932,404    1,903,988    -28,416  -1.470%
query_dynvalue                  4,871,156    4,887,476    +16,320  +0.335%
OK: no gate grew more than 0.5%.

after                            baseline     measured      delta      pct
query_streaming                 1,484,652    1,446,032    -38,620  -2.601%
query_join                      1,507,836    1,465,704    -42,132  -2.794%
query_streaming_agg_fused       1,417,476    1,391,800    -25,676  -1.811%
query_streaming_agg             1,932,404    1,906,876    -25,528  -1.321%
query_dynvalue                  4,871,156    4,893,748    +22,592  +0.464%
OK: no gate grew more than 0.5%.
```

**Verdict: passes.** But read it the way the alpha findings insist on — a gate
that passes is not the same as no regression, because `check_gate.py` compares
against a recorded `baseline.json` that four of the five gates now sit *below*.
The branch-to-branch number is the honest one:

| gate | before | after | delta | pct |
|---|---:|---:|---:|---:|
| `query_streaming` | 1,439,756 | 1,446,032 | **+6,276** | +0.436% |
| `query_join` | 1,464,616 | 1,465,704 | **+1,088** | +0.074% |
| `query_streaming_agg_fused` | 1,388,848 | 1,391,800 | **+2,952** | +0.213% |
| `query_streaming_agg` | 1,903,988 | 1,906,876 | **+2,888** | +0.152% |
| `query_dynvalue` | 4,887,476 | 4,893,748 | **+6,272** | +0.128% |

+1 KB to +6.3 KB, at most +0.44%. That is the shape of two extra trampolines
instantiated per `Relation` type plus the per-node rewrite bodies — the cost of
concrete code, not of an open dispatcher. It is deliberately **not** a shared
generic rewriter: the one measured precedent for that here (`variant_dispatch`)
cost +662,740 bytes on a single gate, because a narrowing adapter closure inlines
into every arm of every instantiation. Every node writes its own five-line
`with_projection` instead, and the erasure stays closed.

## ClickBench

`pixi run -e bench python python/marrow/tests/bench_clickbench.py` — 43 queries
x 3 engines x 5 repeats, engines interleaved per repeat, medians. Base tree and
this branch measured back to back on the same machine.

```
                marrow      polars       duckdb    marrow vs polars
before        13 913.6 ms   785.6 ms   1 596.9 ms       17.7x
after          3 870.0 ms   774.2 ms   1 607.2 ms        5.0x
```

**3.6x faster overall, and 17.7x -> 5.0x against polars.** The two untouched
engines moved by -1.5% and +0.7%, which is the normalisation this box needs: the
machine did not get faster between the runs, so the marrow delta is the change.

Per query the shape is exactly what the diagnosis predicted — the queries whose
ratio was absurd were the ones reading 104 columns for nothing, and they are the
ones that collapsed:

| query | before | after | speedup | vs polars, before -> after |
|---|---:|---:|---:|---|
| q01 `count(*)` | 271.1 ms | **9.9 ms** | 27x | 482x -> 21x |
| q02 | 275.8 ms | **6.1 ms** | 45x | 256x -> 6.8x |
| q04 | 262.2 ms | **8.5 ms** | 31x | 176x -> 5.8x |
| q05 `nunique(UserID)` | 266.3 ms | **13.4 ms** | 20x | 68x -> 3.4x |
| q20 | 265.8 ms | **6.3 ms** | 42x | 277x -> 7.2x |
| q34 `GROUP BY URL` | 316.6 ms | **104.0 ms** | 3.0x | 4.5x -> **1.5x** |
| q35 | 317.0 ms | **105.8 ms** | 3.0x | 4.3x -> 1.5x |
| q40 | 396.6 ms | **151.7 ms** | 2.6x | 7.8x -> 3.0x |
| q30 (90 fused sums) | 611.0 ms | 353.6 ms | 1.7x | 182x -> 117x |
| q24 `SELECT *` | 546.6 ms | **550.4 ms** | 1.0x | 5.3x -> 5.1x |

Two rows are worth reading closely, because together they are the argument that
the rewrite does what it claims and only that:

- **q24 did not move, and must not.** It is `SELECT * ... ORDER BY EventTime
  LIMIT 10` — the plan genuinely emits all 105 columns, `optimize()` seeds the
  column set with the root's own schema, and nothing narrows. A pushdown that
  "sped up" `SELECT *` would be dropping data.
- **q34 is now 1.5x polars**, from 4.4-4.5x. That query was already the honest
  end of the old range because polars must read the big `URL` column too; now
  both engines read one column and the gap is kernel-to-kernel.

What is left is real work rather than scan waste. The remaining outliers are q30
(116.8x — 90 fused `SUM(ResolutionWidth + k)` over one column, arithmetic-bound
and untouched by this change), q23/q24 (`SELECT *`-shaped), and q21/q22 (`GROUP
BY` over a wide value column).

## Correctness

`pixi run -e bench pytest python/marrow/tests/test_clickbench.py` — every query
cross-checked against DuckDB, each in its own subprocess.

```
before   85 passed, 1 skipped, 11 warnings in 26.47s
after    85 passed, 1 skipped, 11 warnings in 15.59s
```

Identical, and the run itself got 40% faster. This is the load-bearing check: a
pushdown that drops a needed column produces a *wrong answer*, not an error, so
43 queries diffed against a second engine is what catches it.

Also green:

- `pixi run -e dev pytest marrow/expr/tests` — **390 passed, 24 skipped** (the
  whole directory: `test_plan`, `test_pushdown`, `test_streaming`, `test_parity`,
  `test_join`, `test_aggregates`, `test_runtime`, `test_values`, …).
- `pixi run -e dev precompile` — 0 errors, 0 warnings.

Twelve new cases in `marrow/expr/tests/test_pushdown.mojo`. Ten assert the
**narrowed scan schema**, reached by walking `children()` from the root rather
than by a `downcast` chain — a test that only checks answers passes when nothing
was pushed at all. Three of them also execute against a real file, so the rewrite
is pinned to producing the same rows as the plan it replaced:

- `test_relation_children_walks_the_plan` — traversal, root to leaf, no downcast
- `test_projection_pushdown_narrows_scan_to_selected_column`
- `test_projection_pushdown_keeps_predicate_columns`
- `test_projection_pushdown_keeps_sort_keys`
- `test_projection_pushdown_keeps_group_keys_and_agg_inputs`
- `test_projection_pushdown_is_transitive_through_project`
- `test_projection_pushdown_leaves_full_scan_alone` — and the root schema is
  identical before and after the rewrite
- `test_projection_pushdown_keeps_one_column_for_count_star`
- `test_projection_pushdown_stops_at_a_join`
- `test_projection_pushdown_preserves_results`
- `test_projection_pushdown_count_star_counts_every_row`
- `test_projection_pushdown_groups_on_the_narrowed_scan`

## Left undone

- Predicate pushdown remains non-recursive; conjunct splitting is not
  implemented (reasons above).
- Projection is not pushed through `Join` (reason above).
- `InMemoryTable` is not narrowed — its batch is already resident, so there is
  no I/O to save, and narrowing it would change a leaf's schema for no gain.
- `write_to` still renders one shallow label per node, so `explain()` prints a
  single line rather than a tree. `children()` is the primitive that makes a
  recursive renderer possible; writing it was out of scope here.

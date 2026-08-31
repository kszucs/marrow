# marrow — open work

**The only document that says what is open.** Everything under `docs/*.md` and
`docs/superpowers/` was consolidated here on 2026-08-30 and the source files
deleted: 33 design documents, plans, audits and research notes, 15,694 lines.
They were audited file by file against the tree first — what had shipped, what
was false, what was superseded — and only two categories survived into this
file:

1. **work someone may still choose to do**, and
2. **constraints that bind that work** — measured dead ends, compiler limits and
   traps, each of which cost real time to find and each of which invalidates an
   approach that looks obvious.

Nothing here describes something that is already built. For what marrow *does*,
read the code, `CLAUDE.md`, `golden/COVERAGE.md` (the best statement of engine
capability) and `CHANGELOG.md`. For why a rejected design was rejected, see
§4 — that section exists so the same ideas stop being re-proposed.

The deleted set is recoverable in full: `git show 8f365d14:docs/` lists it, and
`git log --diff-filter=D --name-only -- docs/` finds any single file.

---

## 1. Open work

Ordered by value. Nothing below is started.

### 1.1 Correctness — known-wrong answers

**Integer `//` and `%` follow Python, not SQL.** `FloordivKernel` and
`ModKernel` forward Mojo's operators: Python floors and takes the *divisor's*
sign, SQL truncates toward zero and takes the *dividend's*. So `-1 // 3` is `0`
in SQL and `-1` here, and `-1 % 3` is `-1` there and `2` here. Division by zero
returns a number where SQL returns NULL — the kernels substitute 1 for a zero
divisor inside a SIMD lane that can neither raise nor null.

Recorded as three strict xfails: `golden/cases/math_floordiv_truncates_toward_zero.mojo`,
`math_modulo_sign_follows_dividend.mojo`, `math_integer_division_by_zero.mojo`.
A fix needs a sign correction **and** a validity mask derived from the divisor.
It was invisible because `math_mod_int64` asks `((n % 3) + 3) % 3` — the one
form both rules agree on.

**`count(*)` desugars to `count(lit(1))`.** `builders.mojo` returns
`lit(1, int64).count().alias("count_star")`, whose `columns()` is empty — so
anything that prunes by "what does this read" prunes every column, and
`RecordBatch.num_rows()` returns **0** when there are none (`tabular.mojo`).

**Projection pushdown landed 2026-08-31 and handles it rather than fixing it:**
`ColumnPruning` never narrows a source to zero columns, keeping the first when
the demand is empty. The desugaring is unchanged, so the trap still waits for
the next thing that reads `columns()` and believes the answer.

### 1.2 The published guides document APIs that do not exist

The only user-visible item here. `docs/guide/expressions.qmd` documents
`from marrow.expr import col, lit, parquet_scan` — there is no such Python
module — and calls the engine "pull-based" when `physical.mojo` is push/drain.
`docs/guide/compile.qmd` documents `plan.execute_cli()`, deleted with the CLI
entry point. These are on the rendered site.

**`README.md` came off this list on 2026-08-30**, when the Python query API was
restored: its lazy-query section had described `morsel_size` and `strictness`
arguments that no longer exist, claimed `explain()` "renders one node, not the
tree" when plans render recursively, cited two deleted docs, and carried a
40/43 ClickBench claim whose harness went with them. The guides are still
wrong, and `docs/guide/expressions.qmd`'s import line is now *nearly* right —
the names are `marrow.col` / `marrow.lit` / `marrow.read_parquet`, not
`marrow.expr.*`.

### 1.3 Latent compiler hazards

**~15 more `t"…{dtype}"` sites under `marrow/kernels/`.** A t-string
interpolating a recursive `Writable` value inside a function-level recursion
cycle deadlocks the compiler — see §4 for the full finding. Three sites were
fixed; the rest are untested and sit in the same shape.

**File the deadlock upstream.** `docs/repros/tstring_recursive_writable_cycle.mojo`
is a 55-line reproducer with no marrow import for a compiler hang that emits no
diagnostic. Modular should have it.

### 1.4 Engine capability the golden corpus measures as missing

`golden/COVERAGE.md` is authoritative and machine-checked: 278 cases, of which
**85 carry `-- skip mojo`** because marrow has no API for them. Their bodies are
never compiled, so they are proposals rather than verified spellings. In rough
order of how often a real query wants them: set operations (UNION/EXCEPT/
INTERSECT), window functions (7 cases), 13 aggregates (median, quantile,
first/last, arg_min/arg_max, string_agg, bool_and/or, corr/covar, mode,
skewness), 16 string functions (substr, replace, regexp, lpad, position,
split_part, concat), 13 temporal (date_diff, age, strftime/strptime, last_day,
make_date, epoch, iso_week), nested types (struct field, map lookup, list
element/slice/contains, unnest), decimals, `DISTINCT ON`, GROUPING SETS/ROLLUP/
CUBE, and cross/non-equi/asof joins.

Three of these need only a node over a kernel marrow already has: `bool_and`/
`bool_or` over `AnyKernel`/`AllKernel`, `list_contains` over
`ArrayContainsKernel`, and `IN` over `is_in`.

### 1.5 One kernel reachable from no expression node

`ArrayContainsKernel` (`kernels/nested.mojo`) is correct and tested but wired
to nothing. Either add the node or delete the kernel; do not leave a public
kernel with no consumer.

`ConcatKernel` (`kernels/string.mojo`) came off this list on 2026-08-30: the
runtime lane's `add` tag dispatches to it when the operands turn out to be
strings, which is the call its own docstring already said it existed for. The
comptime lane still has no `+` on `StringValue` -- a typed string column
concatenates only by leaving the lane.

### 1.6 ~~Top-K is a dead parameter~~ — done 2026-08-31

`TopN` in `marrow/expr/optimizer.mojo` rewrites `Limit(Sort(x))` to a bounded
sort, and `SortOperator` now passes `sort_indices(limit=)`. Two details the
original entry got right and one it did not:

- The `Limit(Filter(Sort(x)))` hazard is real, and the rule requires the
  `Limit` to be **directly** above the `Sort` — which makes it unrepresentable
  rather than merely avoided. `PushFilterBelowSort` runs first and moves the
  common offender out of the way, so adjacency is less restrictive than it
  sounds.
- The bound is `offset + length`, not `length`: the `Limit` still skips
  `offset` rows of the ordered result.
- The bound applies to the **primary-key pass only**. Multi-key sort composes
  stable passes from least to most significant key, so truncating an earlier
  pass drops rows the later keys still have to order. The original entry did
  not mention this and it is the one way to get `TopN` silently wrong.

It did **not** need "a `row_limit` channel through `Pushdown` and a per-node
rule table" — that was the shape the demand-lattice design would have required.
It is a field on `Sort` and one rule.

### 1.7 Structure

**Seven files over 1,000 lines** (`expr/optimizer.mojo` joined at ~1,100, and
is the one file that should stay whole — it exists so every rule is readable in
one place), with seams identified: `kernels/aggregate.mojo`
(1,573 — fold algebras | `AggState`, whose storage/driver/policy should split
three ways | the `AggKernel` conformers), `expr/logical.mojo` (**1,672** — the value
layer | `Relation`+`DynRelation`+the fluent verbs | the eight relation nodes),
`kernels/cast.mojo` (1,246), `expr/physical.mojo` (1,281 — wire types |
contract | `Pipeline` | the operators), `comptime/core.mojo` (1,212 — machinery
| the family traits, ~85% fluent surface), `kernels/filter.mojo` (1,163 — two
structs of ~520 lines each; split by layout, not by struct).

Deliberately not done: the churn would swamp review of the behaviour changes it
would ride alongside. `Groups` and the decimal casts were split out because
they had hard internal boundaries.

**`DynBounds`/`compare_dyn`/`PruneStats.dyn_bounds` stay in `pruning.mojo`.**
Moving them to `runtime/` — their only consumer — means exposing
`PruneStats._cols` or adding an accessor for one caller. `PruneStats` hosts
three readings of itself; moving one and leaving two is arbitrary, and trading
encapsulation for a file boundary is not a simplification.

### 1.8 Test and infrastructure gaps

- **No cross-lane parity test.** "One engine, two drivers" was enforced by a
  `test_parity.mojo` across four axes; it went with the previous expression
  package and has no replacement. The invariant is currently unenforced.
- **`HashGrouper` has no dedicated test** — `kernels/tests/test_groupby.mojo`
  was deleted without replacement.
- **CI has not run since 2026-05-11.** `test.yml` calls a deleted task, the docs
  job cannot pass, and the binary-size gate is not in CI at all — which is how a
  +55% size regression once survived ten commits.

### 1.9 Measurement trap: `==` on an erased array

`DynArray.__eq__` routes through `ArrayData.__eq__`, which compares the
**layout** — dtype, length, null count, *offset*, validity, whole buffers,
children — and says so in its own docstring: "two layouts holding the same
values at different offsets are not equal here". Two consequences bite in
tests, and both cost time on 2026-08-30:

- **A filtered result over-allocates its buffer**, so comparing one to a fresh
  `array([...])` of the same values answers False. `PrimitiveArray[T].__eq__`
  loops over the valid elements and is immune; the erased one is not. Assert
  element by element, or compare typed arrays.
- **A value under a null is unspecified.** A kernel computes every lane and
  masks afterwards, so `add(a, b)` holds `a[i] + b[i]` under a null where a
  builder writes 0. Whole-array `==` sees the difference.

CLAUDE.md's "prefer `assert_true(result == expected)`" is about
`PrimitiveArray[T].__eq__` and holds there. It does not transfer to `DynArray`.

### 1.10 Python binding limits, measured 2026-08-30

Audited against the `std.python.bindings` surface at Mojo
1.1.0.dev2026083005 while restoring the Python query API. Each was verified by
reading `mojo/stdlib/std/python/{bindings,_python_func}.mojo` **and** by
running the built `.so`; none is a marrow bug, and each dictates a shape the
bindings currently have.

- **`PythonTypeBuilder.bind` installs four slots** -- `tp_new`, `tp_init`,
  `tp_dealloc`, `tp_repr` -- and nothing else. `def_method` fills the type's
  `tp_dict`, not a CPython slot, so a registered `__str__` is reachable as
  `obj.__str__()` and **not** as `str(obj)`, which falls back to `tp_repr`.
  Measured: `str(expr)` returns `"<marrow.Expr: gt(a, 1)>"` while
  `expr.__str__()` returns `"gt(a, 1)"`. Every Python wrapper that wants the
  real text calls `._binding.__str__()` explicitly (`LazyTable._plan_text`).
  The same limit is why operators live in Python: a registered `__add__` would
  never fire for `+`, and a registered `__eq__` would never fire for `==`.

- **No `__eq__` reaches `tp_richcompare`, and none is registered anyway.**
  `ma.int32() == ma.int32()` is `False` and `ma.array([1,2]) == ma.array([1,2])`
  is `False` -- both identity comparisons. The Python suite compares dtypes by
  `str()` for exactly this reason. Fixing it needs a slot, not a method.

- **`def_property` does not exist.** The `tp_getset` slot is unexposed (there
  is a design doc for it in the modular tree, unlanded). So `Array.type()` and
  `RecordBatch.num_rows()` are methods where PyArrow has properties. That is a
  real divergence from the "follow PyArrow closely" rule and it is not
  currently fixable from this side.

- **A dotted type name sets `__module__` but breaks attribute lookup.** CPython
  3.14 emits `DeprecationWarning: builtin type X has no __module__ attribute`
  once per registered type -- 11 per import today. Passing `"probe.Dotted"` to
  `add_type` does set `__module__` correctly, but `finalize(module)` uses the
  same string as the module *attribute* key, so the type lands at
  `vars(m)["probe.Dotted"]` and `m.Dotted` stops resolving. Verified both ways.
  The warning cannot be silenced without an upstream change that splits the
  `PyType_Spec` name from the attribute name. Worth reporting.

- **`PyObjectFunction` supports 8 positional arguments, kwargs, and a typed
  self** -- more than the bindings use. The typed-self form takes
  `Pointer[T, MutAnyOrigin]` as its first parameter and downcasts
  automatically, which would delete the `py_self.downcast_value_ptr[T]()` line
  at the top of most binding functions and most of `helpers.mojo`'s `pymethod`
  factory family. Not adopted here: it is a mechanical sweep across ten binding
  modules and belongs in its own change. The kwargs form is deliberately *not*
  adopted -- keyword sugar lives in pure Python by project rule.

- **Statistics pruning is unreachable from Python.** `DynRelation.filter` has
  two overloads: the erased one takes `DynValue` and reads every row group, and
  the pruning one takes `V: Value & Prunable` and captures the concrete type.
  A `PythonObject` has already erased that type by the time it crosses the
  boundary, so `Plan.filter` can only reach the first. `RuntimeValue` *does*
  conform to `Prunable`, so what is missing is a way to carry the unerased
  value across, not the pruning itself -- the `Expr` box holds a
  `RuntimeValue` and could call the pruning overload directly.

### 1.11 Undocumented subsystems

Nine substantial pieces of the codebase have no design document and never did:
the whole Parquet subsystem (nine modules, ~380 KB), the Arrow IPC layer, the C
Data Interface, the GPU execution model, `utils/argparse.mojo` (769 lines),
`kernels/groups.mojo`/`bounds.mojo`/`cast_decimal.mojo`, `Dispersion`, the
`comptime/temporal.mojo` nodes, and the `_drop` destructor trampoline on every
erased box. Listed so that "there is no doc" is not mistaken for "there is no
feature".

---

## 2. Standing constraints

Each of these cost real time to find and invalidates an approach that looks
obvious. Read before planning anything.

### Architectural invariants (gate every merge)

1. **Small-binary DCE.** Preserve the closed-erasure property: no open
   dispatchers, fused-only value boxes, closed per-dtype kernels. Gate on
   `pixi run binary_size`.

   **Amended 2026-08-31.** `DynRelation` is now a `Variant`, which does name
   all nine relation nodes in one place — the thing this invariant forbids.
   That is deliberate and its price is bounded by *where* the variant is used:
   `isa`/`get` instantiate nothing per member, so inspection is free, while
   `to_operator` stays on a per-type trampoline. Routing lowering through the
   variant's ladder instead measured **+348%** of `__text` on `query_streaming`,
   because it makes every node's operator — and therefore `kernels::sort`,
   `kernels::join` and the Parquet reader — reachable from any plan. The
   working rule is narrower than the original: **a closed type set may be a
   variant; what must never go through its ladder is anything that reaches a
   kernel.**
2. **One engine, two drivers.** No feature may exist in only one lane. This was
   enforced by a `test_parity.mojo` across four axes — op names read off the
   kernel, pruning, values, aggregates through a keyless plan — which went with
   the old expression package and **has no replacement under
   `marrow/expr/tests/`**. Until one exists the invariant is unenforced; a new
   op should still add both lanes.
3. **PyArrow-shaped naming** in the core types and the bindings.
4. **Code quality is an acceptance criterion**, not a follow-up. Behaviour lives
   on the type or trait, not in free functions. A wave does not open until the
   prior wave's quality pass is green — it is a gate, not cleanup-later.

### Do not change

- **Array, scalar and builder layout.** Adding methods and accessors is fine;
  adding, removing, reordering or re-typing fields is **out of scope, not
  deferred**. Note this constraint has been read too broadly before: B13 (a
  sliced `BoolArray` double-applying its offset) was filed as blocked by it,
  when the fix was deleting a redundant `+ self.offset` and touched no field at
  all. Check whether a fix actually needs a layout change before deferring it.

### Measurement traps

- **The binary-size gate's file-size number is quantized to 16 KB.** Apple
  Silicon uses 16 KB pages, so a stripped binary's *file size* moves in 16,384-byte
  steps — a real +1,728-byte change once showed up as +16,504 with one *fewer*
  symbol. Measure `size -m <binary>` → `Section __text`.
- **Measure one gate binary directly** (`mojo build -O3 -g0 -I . …query_dynvalue.mojo`,
  ~2.5 min), not the whole `pixi run binary_size` sweep (~10 min).
- **A generic wrapper around an already-erased dispatch is not free.** Folding
  twelve promote-then-dispatch sites into one `_arith[K]` helper cost
  **+115,600 bytes**; writing the same four lines inline in each arm cost a
  fraction. A parameterised method is instantiated per kernel and each
  instantiation carries its own copy of what it touches.
- **A runtime switch over tags is the anti-pattern.** Rewriting the runtime lane
  as one `_eval` switch over ~70 tags cost **+1,807,168 bytes of `__text`
  (+45.7%)** on `query_dynvalue`, because every arm became reachable from every
  node. The fn-pointer `EvalFn` (`dynamic.mojo:209`) exists for this reason.
- **Reachability intuitions about erased paths are usually wrong; stub and
  measure.** Stubbing the `cast_array` calls out of the expression layer left
  the gate binary byte-identical. (The card that recorded this said "both";
  `expr/dynamic.mojo` has **seven** — `:187, :189, :203, :216, :218, :359,
  :373` — so re-count before repeating the experiment.)
- **An operator with no benchmark has no performance.** T2.4's per-row-group scan
  shipped a **4.7x** regression that every test passed through, because nothing
  benched the scan operator.
- **A benchmark whose captured value is not `keep()`-alive after `b.iter[call]()`
  measures nothing** — one reported 17,774 GElems/s. Check throughput for
  physical plausibility before believing a flat A/B.
- **The pytest-benchmark table prints mixed units per row.** Compare
  `--benchmark-json` medians (seconds), never the table's numbers.
- **Benchmarks here vary 10–18% run to run.** Interleave repeats across refs
  (nesting them concentrates machine drift on whichever ref is measured last and
  *invents* regressions), use five or more, compare ranges.
- **`views._reduce_dispatch` staying on `sync_parallelize` is deliberate, not an
  unconverted leftover** — it is the fourth documented one, and its comment says
  so. Routing it through `ctx.stripe` forces the serial arm to allocate a
  one-slot partials buffer it does not need: `sumint64_1k` 0.19-0.20 → 0.30-0.32
  µs, `sumfloat64_1k` 0.23-0.24 → 0.34-0.37 µs (five interleaved repeats,
  disjoint ranges). A serial fold should not pay for the merge's scratch.
- **A fixed per-call cost hides until you change how often the call happens.**
  `ParquetFile.read` built a `CompressionLibs` per worker and the first
  decompress `dlopen`s the codec library — invisible at one read per file, 4.7x
  at one read per row group.

### Trait and compiler limits measured 2026-08-17

Each is a hard error, and each rules out an approach that looks obvious.

- **A trait default body cannot name a field of `Self`** —
  `error: '_Self' value has no attribute 'a'`. The operand must be reached
  through a by-value accessor requirement.
- **A ref-returning accessor is not expressible at trait level** —
  `cannot return 'self's origin, because it might expand to a RegisterPassable
  type`. Same origin-widening wall as `arrays.validity()`; this is what forces
  the `.copy()`.
- **A sub-trait default returning `Self.State` does not recurse — it fails to
  reduce**: `cannot implicitly convert '_Self.Operand.State' value to
  '_Self.State'`. `rebind` cannot bridge it, because `rebind` takes its argument
  by implicitly-copyable borrow and `State` is not. **This settles §5.1's
  validity/state delegation question as impossible, not merely costly.**
- **A struct parameter does not satisfy an associated-type requirement** —
  `struct Neg[A: Leafy](UnaryDirect)` gives `required member 'A' is not
  specified`. The trait needs a differently-named member the struct binds
  explicitly.
- **A trait default is size-free; the boilerplate is the cost.** Hoisting the 22
  verbatim `referenced_columns` copies to a `UnaryValue` default compiles clean
  and measures **+0 bytes on `query_streaming_agg_fused`, +128 on `query_exprs`**
  — and the +128 is not the trait mechanism but +48 bytes per boxed
  instantiation, from the forced `.copy()`. It is still not worth doing: the
  dedup makes the file **58 lines longer**.

### Measurement traps found in the alpha wave (2026-08-18)

Each of these produced a **confident wrong conclusion** before it was caught.
Sources: `git show c0831f5^:docs/alpha-findings/<name>.md`.

- **A sampled profile tells you where time goes *under the profiler*.** `sample`
  and Instruments inflate image load/unload, so anything dominated by `dlopen`,
  `dlclose` or loader work is over-weighted. Measured on one `-O1 -g` build:
  **37.83 ms/run under the sampler, 7.85 ms/run without it** — a ~5x
  over-attribution that produced the claim "80% of `COUNT(*)` is `dlopen`". The
  real saving from fixing it was ~0.9 ms/query. Confirm a profile-derived
  hypothesis with a wall-clock A/B before believing its percentage.
  (`o1-codec-caching`)
- **The profile build is `-O1`; the benchmark build is `-O3`.** `scripts/profile.py`
  rebuilds with debug info because `-O3` inlines away the frames you are reading.
  So a trace shows *where*, never *how much* — absolute numbers come from
  `bench_clickbench.py`.
- **Perturbing one `-O3` unit can move an unrelated kernel.** An exact-size
  `self.tokens.reserve(n)` — an upper bound, apparently free — cost **+43% on
  `bench_contains_1m`**, which shares no code with the changed type, plus +35%
  and +21% on other untouched scan paths. Reproduced two runs per side; vanished
  when the one line was removed. **Only the drift controls caught it**: without
  benchmarks the change cannot touch, a 20.9x win would have shipped with a 40%
  regression underneath. This is the concrete case for CLAUDE.md's "always
  include rows the change cannot touch". (`o3-string-alloc`)
- **A passing size gate is not "no regression".** `check_gate.py` compares to the
  recorded `baseline.json`, **not** to the branch under test. Four of five
  recorded values sat above the tree on *both* branches, so gates read as
  shrinking 2.4-4.1% while the branch-to-branch measurement showed every gate
  **grew** ~16 KB. Ask which question you are answering. (`g3-regression-check`)
- **ASAN is not usable as evidence on this tree.** A deliberate-overflow probe
  built with the harness's own flags **hangs before producing a report**. This is
  stronger than CLAUDE.md's existing "ASAN can hide a heap bug": here it cannot
  be run at all. (`f1-distinct-segfault`)
- **`pixi run -e dev python script.py` does not rebuild `libmarrow.so`** — only
  pytest's `conftest.py` does. The natural A/B (checkout old, run script,
  checkout new, run script) therefore measures the *new* library twice. It
  produced a confident "no improvement, revert it" on a change that was a 20.7x
  win, and separately made an already-fixed bug still look broken. Rebuild with
  `pixi run build_python` between variants, or drive the comparison through
  pytest. (`o2-cast-utf8`)
- **A mass failure at exactly the harness deadline is the harness, not your
  change.** Five separate runs reported every case failed with empty messages at
  precisely 1800.0s. It nearly caused a correct fix to be reverted. Distinguish
  the two shapes: a *slow* unit burns CPU; a *wedged* one freezes — check that
  accumulated CPU `time` is advancing, not just `%cpu`.
- **A clean `mojo precompile marrow` is not evidence a test file will build.** It
  compiles the library, not the test's instantiations. Both compiler hangs in §2
  are invisible to it. (`o2-cast-utf8`, `h2-nested-equality-wedge`)

### Compiler and platform facts

- **`Buffer` requires 64-byte pointer alignment**, so `read_at` cannot return a
  sub-`Buffer` at an arbitrary file offset, and neither can IPC's `_slice_body`:
  Arrow IPC pads to 8, not 64. A source owns *one* whole-file `Buffer` and hands
  out `BufferView`s. This has blocked two separate designs.
- **A comptime conditional type carries no trait conformance** and does not
  reduce at a return site, even inside a `comptime if` that selected the branch;
  `rebind` does not rescue it.
- **A capturing closure's type is parameterised by its creating scope**, so it
  cannot be stored in a struct field and outlive that scope. Every stored
  callback must be `thin`.
- **A narrow test selection can hang the compiler where a wide one does not**,
  and the diagnostic advice for it is a trap. See §2 *`dispatch` hangs a narrow
  unit*. Two things cost hours on 2026-08-17: `mojo run` leaves an **idle parent
  process pinned at ~0:07.9 CPU** while a child does the real work, so
  `pgrep -x mojo | head -1` (and the harness's own "compare elapsed against CPU
  time" hint) reports a frozen CPU clock and looks exactly like a deadlock —
  sum CPU across *all* mojo processes instead, and it climbs past 17 CPU-minutes.
  And `ps -eo comm` prints `mojo` for the parent but the **full path** for the
  child, so `awk '$2=="mojo"'` silently misses the one that matters.
- **`ctx.stripe` bodies may not raise**, and widening it miscompiles: the
  parameter form of `sync_parallelize` that accepts a raising worker needs an
  implicitly-capturing closure whose captures are silently not made. Watch for
  "assignment was never used" warnings on buffers the body writes.
- **`origin_of(a, b)` is an origin union**, which is what lets a function return
  values borrowed from either of two storages.
- **A closure type cannot be generic over its own trait bound**, so a *shared*
  dispatch loop would have to bind `func` on `Movable` and let the caller narrow
  through an extra closure. That adapter inlines into every arm: it measured
  **+662,740 bytes (+31.9% of `__text`)** on `query_streaming_agg_fused`. Each
  erased box writes its own `isa` ladder instead — do not refactor them back
  onto a common helper.
- **Two closure arguments to the same call may not both capture mutably**, nor
  mut+imm over one origin. An API taking several closures over shared mutable
  state must thread that state through as an explicit `mut` parameter of each
  closure — which, when there are several callbacks over one object, is a trait
  (`parquet`'s `LeveledSink`). A state *struct* handed to separate closures does
  not work: it makes them parametric over the enclosing generic parameter.
- **macOS needs the Metal toolchain installed separately** —
  `xcodebuild -downloadComponent MetalToolchain`. Without it every GPU test dies
  with `Metal Compiler failed to compile metallib`, which reads like a Mojo bug
  and is not one (a marrow-free three-line `elementwise` program fails
  identically). This was tracked as B25 and blamed on a backend crash for
  months; resolved 2026-08-16, `test_views_gpu` is 15/15.
- **A trait requirement cannot name a field**, which is why `slice()` (7 bodies)
  and `validity()` (7 byte-identical bodies) in `arrays.mojo` are unfactorable.
  `validity()` returns `Optional[BitmapView[origin_of(self.bitmap._value)]]` — the
  return type names a field, so it is neither expressible as a requirement nor
  reachable from generic code over `A: Array`. The `ArrayData` round-trip default
  that would replace the seven `slice` bodies makes every typed `slice`
  **raising**, and keeping `slice` non-raising was decided 2026-08-16. The same
  wall blocks a `trait ValidityBuilder` default (needs `self._bitmap`) and a
  `CReleasable` trait for `c_data`'s release slot (needs `self.release`). These
  are language limits, not design debt — stop re-filing them as opportunities.
  **Probed directly on 2026-08-17** for `validity()`, and it fails twice over:
  declaring the requirement with a self-wide `origin_of(self)` is rejected
  because Mojo does not widen an implementation's narrower origin at the
  conformance site, and `NullArray` cannot implement it at all — it is all-null
  with no bitmap, and `None` already means all-valid. Even with the origin
  solved it would be a promise 2 of 9 conformers could not keep, which is the
  `to_device` shape the abstraction audit called the tree's one leaky
  abstraction.
- **Erase into a trait whose members are all runtime methods**, and only where
  the conformance is *honest* **and** has a consumer outside its own loop. A
  comptime member has no execution point at which to raise, so a box can only
  supply a plausible constant and the failure mode is a **wrong answer, not an
  exception**. Four sound-but-unconsumed conformances were removed for the second
  half of that rule at zero binary cost; see `docs/dyn-conformance-removal.md`.
- **A trait *default method* is not the `_arith[K]` shape.** The +115,600-byte
  trap above is about a *generic wrapper instantiated per kernel*; a trait default
  is instantiated once per conforming struct, exactly as the hand-written copy
  was, so DCE sees the same reachability graph. If that holds, deduplicating
  behind a trait default is source-level only. **It is an argument, not a
  measurement** — one ~30-line spike settles it for every item in §5.1's first
  bullet, and should run before any of them.
- **`.claude/worktrees/` contains two stale worktrees** (`docs-revamp`, `q25`) holding pre-Q2.5 `AGG_*` and `reinterpret_array` code.
  Exclude them from every grep or you will get false positives.

---

---

## 3. The compiler deadlock — fixed 2026-08-30, kept because the shape recurs

`raise Self.error(t"unsupported dtype {dt}")` deadlocked the Mojo compiler:
~7 s of CPU, then parked forever at 0% CPU with flat RSS, main thread in
`semaphore_wait_trap`, **no diagnostic**. `String("unsupported dtype ", dt)` —
same message, same value — builds.

**Minimal trigger: a t-string interpolating a recursive `Writable` value inside
a function-level recursion cycle.** Three ingredients, each independently
necessary; remove any one and it builds in 5 s. Reproducer with no marrow
import: `docs/repros/tstring_recursive_writable_cycle.mojo`.

It had been open since 2026-08-17 under three separate backlog entries, and two
confident diagnoses were wrong before this one:

- **not** the `List`+`Variant` layout corruption — `DynScalar` measures 80/8 at
  HEAD and re-introducing the broken 96/32 shape left the deadlock unchanged;
- **not** the `dispatch` ↔ `apply(StructArray)` instantiation cycle — removing
  the `is_struct()` arm entirely still hung;
- **not** the erased ladder's fan-out width, which is irrelevant.

Two working rules came out of it:

1. **The compiler wedges *before* it diagnoses.** An ordinary type error
   produces an identical hang with an empty log. Always `precompile` first;
   "it still hangs" can just mean "it does not compile".
2. **Deadlock vs slow is CPU *time*, not elapsed.** Frozen CPU time plus flat
   RSS is a deadlock. A runaway elaboration burns CPU and grows memory.

The mechanism is still **unexplained** — the toolchain is stripped, and the
t-string/`String(...)` difference is inferred from the A/B, not observed.


## 4. Rejected and replaced designs

Each row is a design that was written down, then **not** built — because the tree
built something different and better, or because the premise turned out to be
wrong. They are here so nobody re-litigates them from a stale document. Every
citation below was checked against the code, not copied from the design it
replaces.

### Simplification wave (2026-08-17)

- **`equal_any` → a neutral `kernels/compare.mojo` — rejected (D2).**
  `numeric.mojo:49` imports `StringEqKernel` solely for `equal_any`, so the move
  *does* delete the `numeric → string` edge. But `EqKernel.apply(StructArray, …)`
  (`:617-645`) calls `equal_any` twice and **cannot move**, because Mojo cannot
  add a static method to `EqKernel` from another module. The result is a
  `numeric ↔ compare` **cycle replacing an acyclic edge** — worse, immediately
  after S1 finished removing cycles. Revisit only if the `StructArray`
  row-equality relocates for an independent reason.
- **Amended (D1): the rejection of `Buffer.alloc_for[T](ctx, n)` in
  `buffers.mojo` is overruled.** The original row rejected it for pointing the
  tree's lowest-level module at device policy. It was re-opened on 2026-08-17
  because the `ExecContext` alternative turned out to close
  `execution → buffers → views → execution`, which the original row did not know.
  A rejected-designs list that is quietly contradicted is worse than no list, so
  the reversal is recorded here rather than left implicit.

### Alpha wave (2026-08-18)

- **`mojo-regex` — rejected on correctness, not on version drift.** It builds
  against our pinned Mojo (13/14 of its own tests pass) so the expected blocker
  was not the real one. It **never enters an optional group**: `(?:www\.)?` is
  skipped, so Q29 returns `www.example.com` where pyarrow and CPython return
  `example.com`; 10 of 25 `sub()` cases disagree with CPython. Minimal repro:
  `sub("(?:foo)?bar", "B", "foobar") -> "fooB"`. It survived upstream because
  their tests use `(?:` seventeen times and never with a trailing `?`. Adopting
  it would have converted an honest 42/43 gap into a **silently wrong answer on
  the majority of rows**, since `www.`-prefixed referers dominate `hits`. See
  M2.6 for the replacement.
- **A hand-written `extract_host` for Q29 — rejected.** It buys a DEVIATED row,
  generalises to nothing, and gets deleted the moment real regex lands.
- **Over-allocating buffers so "64-byte padded" becomes literally true —
  rejected, and the premise was wrong anyway.** `Columnar.rst:264-273` says pad
  "to a length that is a multiple of 8 or 64 bytes", i.e. *round the size up* —
  exactly `align_up(bytes, 64)`, which is byte-for-byte Arrow C++'s
  `PoolBuffer::RoundCapacity`. `arrow::AllocateBuffer(64)` allocates 64 bytes
  with zero slack. Marrow was already conformant; the false step was inferring
  that padding implies slack past the logical end. Over-allocating was rejected
  because **it cannot deliver the invariant**: FOREIGN buffers come from the
  producer, and pyarrow allocates exactly 64 bytes for a 512-row bitmap — the
  guarantee would hold on half the buffers and make the other half harder to
  find. Memory cost was explicitly *not* the deciding argument.
- **Rewriting `DynArray.__eq__` to dispatch once — tried, reverted.** O(n) arms
  instead of O(n²) and exactly as strict, and it compiles; but the hang in §2 is
  recursion through `ListLikeArray.__eq__`, not the squared ladder, so it fixes
  nothing. Not left in as an unmeasured change to a hot, size-gated file.
- **A fan-out threshold on probe rows for the parallel join — tried, measured,
  removed.** It would have *doubled* the regression: within the partitioned
  layout, fanning out beats serial partitions at **every** size, 8192 rows
  included (194 us vs 388 us). The expensive thing is the partitioning, not the
  `sync_parallelize`. What fixed it was `_DEFAULT_RADIX_BITS` 6 -> 4 — the
  64-partition default came from a *one-shot* 10M sweep, and morsel streaming
  pays it per call.
- **`inputs()`-based optimizer traversal — rejected for `children()` +
  `with_projection()`.** Both were superseded 2026-08-31: the optimizer walks
  plans through `Relation.traverse(f)`, one method per node that applies `f` to
  its own children and rebuilds itself. No `children()`, no `with_projection()`,
  and no ladder over node types in the optimizer.
- **Building a temporal literal through the storage integer and relabelling with
  `relabel_array` — rejected on measurement.** It avoids six
  `PrimitiveBuilder`/`PrimitiveArray` monomorphisations (+23,668 bytes on
  `query_streaming`) but drags in `DynArray.from_data`'s 30-arm ladder that these
  binaries do not otherwise link: **+106,276 bytes, 4.5x worse**. The trick is
  right in `kernels.cast`, where `from_data` is reachable anyway; it is wrong in
  `scalars.mojo`.

### Sort

- **Permutation refinement (`getPermutation` / `updatePermutation` /
  `EqualRanges`) was never built.** Multi-key sort is a **column-oriented LSD**:
  `SortIndices.multi` (`sort.mojo:449`) stable-sorts by the *least*-significant
  key, then for each more-significant key gathers the column under the running
  permutation, sorts that, and composes (`sort.mojo:493-515`). There is no
  equal-range bookkeeping anywhere. *Why it stands:* one stable pass per key
  yields the same order without tracking ranges, and it reuses `Take` and the
  single-column sorters unchanged. *Its one cost is real*: the composition
  depends on every pass being stable — the `stable` flag was accepted and never
  forwarded, so this silently returned wrong orders until B1 was fixed by making
  the comparison path's comparator break ties on the original row index. It also
  re-gathers each key column per pass (**FU-6**).
- **8-bit radix passes / 256-bucket histograms were replaced by 11-bit passes /
  2048 buckets** (`_BITS_PER_PASS` `sort.mojo:76`; `bucket_count`, `:252`).
  *Why:* 6 passes instead of 8 for 64-bit keys with a histogram that still fits
  L1 per thread — measured ~6.7× at N=10M against 8-bit serial; 16-bit thrashes
  L1 (`sort.mojo:77-86`).
- **The comparison-vs-radix cutoff is not `N < 64`.** It is
  `_RADIX_THRESHOLD = 32_768` (`sort.mojo:59`). *Why:* PDQsort measured faster
  all the way to ~28K on int64; below the threshold the pair-buffer setup
  dominates. Decimal128/256 take the comparison path at any size, because their
  native width exceeds the UInt64 radix key.
- **`argsort` as a free function plus a `SortOptions` struct** became the
  `SortIndices` kernel struct (`sort.mojo:344`) with `dispatch`/`apply`/`multi`
  and loose parameters. Per-column `nulls_first` is still wanted — §6.

### Group-by

- **Of the seven designed groupers, exactly one shipped, and it is not one of
  them.** There is a single type-agnostic `HashGrouper` over a `SwissHashTable`
  (`groupby.mojo:43`). `DirectMapGrouper`, `PackedKeyGrouper`,
  `RowEncodedGrouper`, `TwoLevelGrouper` and `SpillingGrouper` do not exist, and
  neither does `kernels/row.mojo`. *Why:* the Swiss table's SIMD probing made
  per-key-type tables not worth their combinatorics; the design's own premise —
  "ClickHouse has 40+, so we need several" — did not survive contact with one
  table that handles every key type through `StructArray`.
- **The dispatch axis is inverted.** The design selects a grouper by **key
  type** (bool/uint8 → direct map, ≤16 fixed bytes → packed, else row-encoded).
  The shipped `_choose_strategy` (`groupby.mojo:402-412`) never looks at the key
  type at all: it selects on **row count and a sampled cardinality estimate**.
  Reading the design as a guide to the code inverts the whole decision.
- **`GROUP_THREAD_LOCAL` (`groupby.mojo:312`) is a strategy the design never
  proposed** — a DuckDB-style thread-local partial aggregation that splits by
  row range and merges partials, chosen for large *low*-cardinality inputs where
  radix cannot use more threads than there are distinct keys. Conversely
  ClickHouse's 256-bucket two-level merge, the design's headline recommendation,
  was not built; `GROUP_RADIX` uses 64 partitions (`RADIX_BITS = 6`,
  `groupby.mojo:304`).
- **The `group_id(keys) -> PrimitiveArray[uint32]` public API does not exist.**
  The entry point is the `GroupBy` struct (`groupby.mojo:321`) with
  `aggregate[A]` / `apply[F]` / `aggregate_columns`. The companion `unique` was
  never written and is still wanted — **M2.2**.

### Joins

- **`JoinHashTable` with an intrusive `_chain_next` collision list was never
  built.** Neither identifier appears anywhere in the tree. It was replaced by
  `SwissHashTable` plus a **CSR index** — `_offsets` / `_rows`, built after
  insertion and grouped by bucket (`hashtable.mojo:76-87`) — so a probe walks
  `[offset[bid], offset[bid+1])` contiguously (`:523-532`) instead of chasing
  pointers. *Why:* same ALL-strictness multi-match capability, sequential memory
  access, and one shared table type for join and group-by.
- **`IndexPairs` is not a struct.** It is
  `comptime IndexPairs = Tuple[Int32Array, Int32Array]` (`join.mojo:201`) — two
  Arrow arrays rather than two `List[Int32]`, which is what lets per-partition
  results merge by buffer memcpy.
- **The `PartitionedOp[T]` trait + `partition_apply` driver, recorded as
  "non-trivial in Mojo's generics, tracked as a follow-up", shipped** as
  `RadixPartitioner.map_partitions` (`partition.mojo:288`). It is done, not
  deferred; do not schedule it again.

### Abstractions and deduplication

Folded in from the abstraction, organizational and duplication audits
(2026-08-17), which were deleted once their open items became §5's `S`-IDs.
Every row is an approach that looks obvious and does not work.

- **One `Column[T: DataType]` struct** replacing the four `*Column` nodes. It
  would have to satisfy `NumericValue.lane` (returns `SIMD[T.native, W]`) and
  `StringValue.lane` (returns `String`) at once — different return types, and a
  `comptime if` does not rescue it, per §0's conditional-type note.
- **Promoting `write_repr_to` to an `Array`/`ArrowScalar` requirement.** Was the
  filed fix; the conformance removal made it unreachable — a requirement now binds
  only the nine typed arrays and nine typed scalars, leaving the erased handles
  (the only ones callers hold) still forwarding to `write_to`. Deletion is what
  remains, and is **S6**.
- **`Buffer.alloc_for[T](ctx, n)` in `buffers.mojo`** for the ten-site GPU-or-host
  preamble. It points the tree's lowest-level module at device policy. The
  decision belongs on `ExecContext`, which *is* the policy type and already owns
  `GPU_ENABLED` after `5b14bfa` — **S8**.
- **Moving the `AggFunction` trait down to `expr/aggregates.mojo`** to reunite it
  with its four conformers. It inverts the dependency: `groupby.mojo` would import
  `expr`, breaking the verified "`marrow/kernels` has no up-edges into `expr`"
  property. The catalog moving the *other* way is the live option — §5.1.
- **A generic `_header_equal[A: Array](a, b)`**, and the `unsafe_get`-on-`Array`
  fix that was filed for `__eq__`'s five copies. Both die on §0's
  trait-cannot-name-a-field limit; the element loops are also genuinely different
  (three index types, and `unsafe_get` raises for some arrays and not others).
  What was reachable landed instead (2026-08-17): the shared validity helper
  was inlined into all seven `__eq__` bodies, which removed two popcounts and
  a bit-by-bit loop rather than removing lines.
- **An embedded `ArrayHeader` field** collapsing `length`/`nulls`/`offset`/
  `bitmap` across six arrays. Forbidden by §0's *Do not change* — array layout.
- **`kernels/interval.mojo`'s placement — examined and upheld.** Consumed
  exclusively by `expr/` and it never touches an array, which usually means
  misfiled; the module docstring argues it, and the argument holds. Its one real
  consequence is that `IntervalKernel(Kernel)` inherits `expect_same_length` /
  `expect_same_dtype` it can never call, which is the `Named`-split finding in
  §5.1. Recorded so it is not re-opened.
- **The two `execution.mojo` files and the two `struct Filter`** are namespaced
  and correct. The only option is a rename and neither name is wrong. Leave both.

### Decimal

- **The custom `Int256 { low: UInt128, high: Int128 }` struct is unnecessary and
  was never written.** Its premise — "Mojo has no native `DType.int256`" — is
  false: `Decimal256Type = _DecimalType[DType.int256]` (`dtypes.mojo:252`), and
  `Decimal128Type` likewise uses `DType.int128` (`:251`). `struct Int256` has
  zero occurrences.
- **Decimal *arithmetic* was never built at all** — not rejected, just never
  started. It is **M3.7**, and the design's result-type rules are the part worth
  keeping.

---

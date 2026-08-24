# AOT Compiled Queries vs. JIT: Research Notes and Benchmark Results

> **Dated record — read the measurements as history, the argument as current.**
> Verified against `b2e7dae`, 2026-08-03.
>
> **The subject file no longer exists.** Everything below describes
> `marrow/faszom.mojo`, which was renamed to `marrow/exprold/lane.mojo` and then to
> **`marrow/exprold/values.mojo`** (`d70ad3e`). The benchmark file was
> `marrow/bench_faszom.mojo`, renamed to `marrow/exprold/tests/bench_fused.mojo`
> and since deleted — so **the reproduction command in "Benchmark Results"
> cannot be run**, and the code sketches (`FilterExpr`, `GtExpr`, `Add`,
> `Column`, `execute(expr, n)`) name types the tree no longer has. The current
> node vocabulary is in `docs/architecture.md`.
>
> **The binary-size figures are superseded.** The 52 KB / 1.1 MB pair and the
> **21×** ratio in "Binary Size: AOT vs. Full Engine" predate every later change
> to the expression layer, including the deletion of the tag interpreter. The
> live gate is **`benchmarks/binary_size/`**, measured as `__text` (never
> stripped file size — it is page-quantized to 16 KB on Apple Silicon). Its
> current table gives `query_runtime` 5,266,548 against a `query_streaming`
> floor of 1,251,672 — a ratio of about **4.2×**; with the floor re-measured at
> HEAD (1,302,900, per the CHANGELOG) it is about **4.0×**. Either way the
> order of magnitude the 21× implied is gone. Note `docs/backlog.md` **I4**
> tracks re-baselining those numbers, so treat them as approximate. The
> external comparisons (DuckDB 35 MB, Polars 188 MB, Spark 460 MB) were never
> re-measured and are indicative only.
>
> **There is no runtime code generation anywhere in the tree.** No JIT, no
> LLVM-at-runtime, no emitted machine code — the only "codegen" in the codebase
> is Mojo's GPU codegen and Thrift's, neither of which is query compilation.
> Everything in "Can We Add Runtime Specialization to AOT?" that is described as
> *already present* still is (`compressed_store`'s popcount dispatch, literal
> folding through comptime parameters, the null-free fast path); everything
> described as a *proposal* remains unbuilt.
>
> **What is kept, and why.** The literature survey (Hyper / Umbra / Excalibur),
> the AOT-vs-JIT comparison, and the "why you can't link just the part you need"
> argument are the intellectual justification for marrow's two-lane
> architecture — a monomorphized AOT lane beside a small erased runtime lane —
> and they are recorded nowhere else in the repo.

## What We Built

`marrow/faszom.mojo` implements compile-time kernel fusion using Mojo's
parametric type system. A query plan is encoded as a Mojo type:

```mojo
var expr = FilterExpr(
    data.copy(),
    AndExpr(
        GtExpr(Add(Column(a.copy()), Column(bv.copy())), Column(c.copy())),
        LtExpr(Add(Column(d.copy()), Column(ev.copy())), Column(f.copy())),
    ),
)
keep(execute(expr, n))
```

`FilterExpr[T, Pred]` is a type parameter, not a runtime value. `execute` is a
monomorphized function — LLVM sees the full predicate tree inlined at compile
time and applies `-O3` across the entire loop.

### Single-pass fused filter

The filter implementation uses a 64-element outer block loop:

1. Evaluate predicate for `W` elements → `SIMD[DType.bool, W]`
2. Pack `W` bool lanes into bits of a `UInt64` word via `_pack_bools[W]` (compile-time unrolled `@parameter for`)
3. Repeat `64 // W` times to fill a full 64-bit word
4. Call `BufferView.compressed_store(src, word: UInt64)` — adaptive:
   - ≤ 24 set bits → sparse CTZ scatter
   - > 24 set bits → dense byte-chunked branchless copy

No `BoolArray` is allocated. No bitmap is written or read. The predicate result
lives entirely in registers between steps 1 and 4.

---

## Benchmark Results (Apple M-series, int32)

Run with: `pixi run -e dev pytest marrow/bench_faszom.mojo --benchmark`

### 1-op baseline — `a == b`

| Size | Fused | Dispatch | Speedup |
|------|-------|----------|---------|
| 100k | 8.4µs | 7.4µs | 0.88× |

Both paths are 1 allocation, 1 memory pass — parity is expected. Tiny fused
overhead confirms the expression-tree wrapper is essentially free at scale.

### 3-op expression — `(a + b) == (c + d)`

| Size | Fused | Dispatch | Speedup |
|------|-------|----------|---------|
| 100k | 13.4µs | 30.9µs | **2.3×** |
| 1M | 139.8µs | 298.4µs | **2.1×** |
| 10M | 1.43ms | 3.05ms | **2.1×** |
| 100M | 14.3ms | 32.4ms | **2.3×** |

Dispatch allocates 3 intermediate arrays and does 3 memory passes.
Fused allocates 1 array and does 1 memory pass.

### 5-op expression — `(a * b + c) == (d * a - b)`

| Size | Fused | Dispatch | Speedup |
|------|-------|----------|---------|
| 100k | 29.4µs | 55.6µs | **1.9×** |
| 1M | 307.5µs | 533.3µs | **1.7×** |
| 10M | 3.15ms | 5.39ms | **1.7×** |
| 100M | 32.2ms | 55.4ms | **1.7×** |

Dispatch allocates 5 intermediate arrays. Speedup is slightly lower than 3-op
because compute-per-element is higher (more FLOPs per byte read), reducing
the relative weight of allocation overhead.

### Filter Group A — `WHERE a + b > c` (50% selectivity)

| Size | Fused | Dispatch | Speedup |
|------|-------|----------|---------|
| 100k | 21.3µs | 24.4µs | **1.1×** |
| 1M | 219.1µs | 232.0µs | **1.1×** |

Modest gain: filter is memory-bandwidth-bound either way. Savings come from
eliminating 1 intermediate array + bitmap pack/unpack, not from compute.

### Filter Group B — `WHERE a + b > c AND d + e < f` (~25% selectivity)

| Size | Fused | Dispatch | Speedup |
|------|-------|----------|---------|
| 100k | 24.7µs | 39.9µs | **1.6×** |
| 1M | 248.3µs | 389.7µs | **1.6×** |

Dispatch allocates 4 intermediate arrays + 1 BoolArray before any rows are
eliminated. Fused evaluates both sub-predicates together and scatters in one
pass. The larger intermediate footprint explains the bigger gain over Group A.

---

## JIT Systems from the Literature

### Hyper (Neumann, VLDB 2011)

- Translates relational pipelines to LLVM IR, JIT-compiles with LLVM backend
- Compilation cost: **10–100ms**, grows **super-linearly** with query complexity
- Added *adaptive execution* later: interpret LLVM IR immediately via ORC JIT,
  compile optimized native code in background, swap transparently

### Umbra FlyingStart (Kersten et al., VLDB 2021)

- Replaced Hyper's LLVM backend with a custom single-pass IR → x86 emitter
- No register allocation, no instruction scheduling — just direct instruction
  emission
- Compilation cost: **~50ms** (down from 10–100ms for Hyper)
- Trade-off: faster compilation but lower code quality than LLVM -O2/-O3

### Excalibur (Gubner & Boncz, PVLDB 2022)

- Adaptive JIT: starts in vectorized execution, replaces hot pipeline fragments
  with JIT-compiled specialized variants
- Uses a **Multi-Armed Bandit (MAB)** search to find the best execution tactic
  per fragment without exhaustive search
- Fragment cache (64 entries): **26× faster** with warm cache vs. no cache —
  directly quantifying the JIT compilation tax
- Outperforms Umbra on TPC-H SF50: +50% (Q1), +80% (Q6), +20% (Q9)
- Outperforms DuckDB by 7–21×, MonetDB by 29×
- **Open source**: https://github.com/t1mm3/db_excalibur

Both Hyper-style *Typer* (data-centric compilation) and Tectorwise (vectorized
interpretation) achieve **"roughly equal performance on most queries"** per the
CMU lecture — the compilation advantage is primarily about eliminating
intermediate materialization, not about instruction-level improvements.

---

## AOT vs. JIT Comparison

| Property | Hyper | Umbra | Excalibur | **Marrow faszom** |
|----------|-------|-------|-----------|-------------------|
| Specialization scope | Per pipeline | Per pipeline | Per hot loop | **Per expression tree** |
| Code generation cost | 10–100ms LLVM JIT | ~50ms custom emit | ~10ms per fragment | **0ms (AOT)** |
| Optimizer quality | LLVM -O2 | None (fast emit) | LLVM -O3 (fragments) | **LLVM -O3 (whole binary)** |
| Runtime constant folding | Yes | Yes | Yes | Compile-time Literal only |
| Handles ad-hoc queries | Yes | Yes | Yes | No (plan must be known at build time) |
| Fragment caching | No | No | Yes (MAB) | Implicit (binary IS the cache) |

---

## Where AOT Compilation Shines

### Short-running queries (OLTP / point lookups)

JIT compilation cost is 10–100ms for Hyper. For a query that *runs* in 5ms,
that is 2–20× overhead. Hyper added adaptive execution (interpret first,
compile in background) specifically to handle this. Umbra FlyingStart cut it to
~50ms — still unacceptable for sub-millisecond queries. AOT cost is 0ms.

### Repeated query patterns (dashboards, ETL, ML feature pipelines)

Excalibur's 26× speedup from its fragment cache is the clearest data point:
that gap is the JIT tax, paid once and amortized. AOT is "infinitely warm cache
with zero miss cost" — the same benefit without warmup or eviction. For
analytics running the same 10 query shapes thousands of times per day, AOT is
strictly better.

### Optimizer quality over latency

Umbra's FlyingStart deliberately forgoes register allocation and instruction
scheduling to achieve 50ms emit. AOT uses full LLVM -O3 uniformly across the
entire binary, including paths that Excalibur's adaptive JIT would only
optimize for hot fragments.

### Streaming / micro-batch analytics

JIT latency amortizes over long-running queries. For 100ms micro-batches, even
50ms compilation is catastrophic. AOT has flat, predictable per-call latency.

### Embedded / edge / serverless

No LLVM runtime required. No JIT memory buffers. Deterministic cold-start.

---

## Where JIT Wins

### Novel runtime constants

If the query optimizer observes at runtime that a join's build side has 47
distinct values and emits a perfect-hash probe, AOT cannot replicate this
without pre-compiling all possible cardinality variants. Multi-versioning covers
finite, enumerable specialization spaces; it cannot cover open-ended value spaces.

### Ad-hoc SQL

A query parsed from a string at runtime cannot become a Mojo type at compile
time. Dynamic queries must use the interpreted dispatch path.

---

## Can We Add Runtime Specialization to AOT?

Yes — and we already do a form of it. The key insight: "runtime specialization"
in JIT systems is really two steps — (1) observe runtime values, (2) emit
specialized code. AOT cannot do (2) for new values but can fully cover (1) by
pre-compiling all useful variants and dispatching at runtime.

### Already present

`BufferView.compressed_store(src, sel_bits: UInt64)` measures `popcount` at
runtime and dispatches between:
- Sparse path (CTZ scatter, ≤24 set bits per 64-element block)
- Dense path (byte-chunked branchless, >24 set bits)

This is runtime specialization — two AOT-compiled paths, runtime selection.

### Literal constant folding — already better than JIT

`Literal[val]` is a Mojo type parameter. The comparison threshold `100` in
`GtExpr(Column(a), Literal(100))` is folded by LLVM -O3 into an immediate
instruction in the comparison loop. JIT systems only get this if the optimizer
pushes constants through to the generated IR; we get it unconditionally.

### Selectivity-adaptive filter variants

Pre-compile three strategies, measure in the first N blocks, dispatch for the
remainder:
- `< ~5%` → pure CTZ scatter, skip popcount checks
- `5–75%` → current adaptive path
- `> ~75%` → bulk copy + mask, skip compressed_store overhead

This covers the main Excalibur win on TPC-H Q1/Q6 without any runtime code
generation.

### Null-density specialization

Static form already exists: if `bitmap is None`, validity checks are skipped
entirely. Runtime form: track null count over the first chunk; switch to the
null-free path for the rest if density is below a threshold.

### CPU feature dispatch

AOT-compile NEON and SVE (or SSE4/AVX2/AVX-512) variants, detect at process
start via CPUID, set a function pointer once. Standard technique, zero overhead
after init.

---

## Key Takeaway

For workloads with a finite, known set of query shapes — ETL pipelines, embedded
analytics, ML feature computation, anything running on a fixed schema — AOT
compilation matches JIT-level code quality at zero runtime compilation latency.
The "warmup" problem that motivated Hyper's adaptive execution and Umbra's
FlyingStart simply does not exist. Our 1.7–2.3× expression fusion speedup and
1.1–1.6× filter speedup are measured against an already-vectorized columnar
baseline; the comparison against row-oriented interpreted engines would show the
~100× improvement documented in the Hyper and CMU literature.

---

## Binary Size: AOT vs. Full Engine

Running the same query (`WHERE a + b > c`, 1M int32 elements) as a stripped
standalone binary:

| System | What's measured | Stripped size |
|--------|----------------|--------------|
| **Marrow fused** (AOT) | single query, exact types, full DCE | **52 KB** |
| **Marrow dispatch** | 3 kernels × 11 numeric types + DynArray | **1.1 MB** |
| **DuckDB v1.5.2 CLI** | full database engine | **35 MB** |
| **libduckdb.dylib** | shared library | **47 MB** |
| **DataFusion** (est.) | Rust query engine CLI | **~20–40 MB** |
| **Polars native `.so`** | Rust engine + Python extension | **188 MB** |
| **Spark 4.1 JARs** | all JARs, no JVM | **460 MB** |
| **PySpark full dist** | JARs + Python + scripts | **479 MB** |

**Umbra** has no public binary; as a full C++ database engine it would sit in
the 30–80 MB range alongside DuckDB.

The fused binary is **21× smaller than our own dispatch binary** and **~700×
smaller than DuckDB**. Both gaps compound at the same root cause.

### Why you can't link just the part you need

Every system in the table builds its execution path around **runtime
polymorphism**:

- **DuckDB**: the query optimizer constructs a physical plan as a tree of
  `PhysicalOperator*` nodes at runtime and calls `Execute()` through a vtable.
  The dispatch table for every built-in operator (40+) must be present because
  the planner picks which one to instantiate after parsing the SQL string.

- **DataFusion**: Rust trait objects — `Arc<dyn PhysicalExpr>`,
  `Arc<dyn ExecutionPlan>`. Rust's linker can dead-code-eliminate unused
  generic monomorphizations, but vtable slots are filled at runtime so every
  `dyn PhysicalExpr` implementation is live as far as the linker can see.

- **Polars**: same pattern — the lazy plan is a runtime `Expr` enum dispatched
  through a physical planner. Rust, but same trait-object dispatch story.

- **Spark**: JVM classloader loads operator classes by string name at plan
  execution time. No static analysis is possible; every class in every JAR is
  potentially loadable.

Even with static linking + LTO, you'd keep nearly the full DuckDB binary.
The liveness chain flows through a `char*` SQL string → parser →
string-keyed function registry → function pointer → operator implementation.
Every link in that chain looks live to the linker.

### The structural difference

In those systems the **query is data** — a string, an AST, a plan object —
interpreted by a runtime engine that must be fully present. In faszom the
**query is a type**:

```
FilterExpr[Int32Type, GtExpr[Add[Column[Int32Type], Column[Int32Type]], Column[Int32Type]]]
```

The Mojo/LLVM compiler has a closed-world view of exactly which code paths are
reachable. There is no planner, no function registry, no vtable. Every branch
for every other type and every other operator is provably unreachable and is
eliminated before the binary is written to disk.

---

## References

- Neumann, T. "Efficiently Compiling Efficient Query Plans for Modern Hardware."
  PVLDB 4(9), 2011. https://www.vldb.org/pvldb/vol4/p539-neumann.pdf
- Gubner, T. & Boncz, P. "Excalibur: A Virtual Machine for Adaptive Fine-grained
  JIT-Compiled Query Execution based on VOILA." PVLDB 16(4), 2022.
  https://www.vldb.org/pvldb/vol16/p829-boncz.pdf
- Gubner, T. "Charting the Design Space of Query Execution using VOILA." VLDB 2021.
  https://t1mm3.github.io/assets/papers/vldb21.pdf
- CMU 15-721 Advanced Database Systems, Lecture 7: Query Compilation.
  https://15721.courses.cs.cmu.edu/
- Excalibur source code: https://github.com/t1mm3/db_excalibur

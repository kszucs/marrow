# Heterogeneous Execution Design

Design notes on GPU/CPU architecture, synchronization, parallelism, kernel
fusion, and the path to efficient heterogeneous query execution in marrow.
These notes were produced by reading the Mojo stdlib source alongside the
marrow codebase in May 2026.

---

## GPU backend support

Three GPU backends are supported by the Mojo stdlib's `DeviceContext`:

| Backend | API string | Notes |
|---|---|---|
| NVIDIA | `"cuda"` | Full feature set |
| AMD ROCm | `"hip"` | Full feature set |
| Apple Silicon | `"metal"` | No raw device pointers; no `enqueue_host_func` |

`is_apple_gpu()` is used throughout the GPU stdlib (thread IDs, warp ops, memory,
sync barriers) so Apple Silicon is a fully first-class target. The one limitation
is `unsafe_ptr()` on a `DeviceBuffer` which raises on Metal, and
`enqueue_host_func` which is CUDA-only.

Apple Silicon memory is **unified** — CPU and GPU share the same physical RAM.
There are no explicit host-to-device copies for GPU buffers on Apple, which makes
CPU/GPU data sharing especially cheap.

Metal arch strings must be validated before compiling GPU kernels. A malformed
target such as `"metal:2-metal4"` causes a hard compile failure deep inside
`simd_width_of`. The helper `has_accelerator_support[*dtypes]()` in
`marrow/utils.mojo` handles this validation and also filters out dtypes Metal
does not support (e.g. `float64`). Always call it before entering a GPU branch.

---

## DeviceStream and DeviceContext

A `DeviceContext` is a handle to a single GPU stream of execution. A
`DeviceStream` is an additional queue on the same device. Both implement the
`_FunctionEnqueuer` trait, so both can launch kernels via `enqueue_function`.
The difference is only which C runtime call is used:

- `DeviceContext` → `AsyncRT_DeviceContext_enqueueFunctionDirect`
- `DeviceStream` → `AsyncRT_DeviceStream_enqueueFunctionDirect`

**Within one stream**: operations execute in FIFO order. No explicit sync is
needed between consecutive GPU kernels on the same `DeviceContext` or
`DeviceStream`.

**Across streams**: operations may run concurrently or in any order. Use
`DeviceEvent` to express cross-stream dependencies:
`stream_a.record(event)` then `stream_b.wait(event)`.

`_elementwise_impl_gpu` (and by extension `views.apply`) only accepts a
`DeviceContext`, not a `DeviceStream`. To dispatch independent GPU column
operations onto separate streams you must call `ctx.enqueue_function` directly.
`ctx.create_stream()` creates a fresh `DeviceStream`; `ctx.stream()` returns the
context's default stream as a `DeviceStream` value.

---

## Synchronization

### GPU kernels are asynchronous

`elementwise[process, gpu_width, target="gpu"](length, ctx)` — the call that
backs `views.apply` — enqueues a kernel and returns to the CPU immediately. It
does **not** call `ctx.synchronize()`. Chaining multiple `apply` calls on the
same `DeviceContext` pipelines them on the GPU stream without any CPU stall
between them.

### Where blocking actually occurs

| Operation | Blocks CPU? | Reason |
|---|---|---|
| GPU `apply` (elementwise) | **No** | enqueues kernel, returns immediately |
| GPU `reduce` | **Yes** | calls `to_cpu()` which is an implicit D2H sync |
| `Buffer.to_cpu()` | **Yes** | memory copy syncs the stream |
| `ctx.synchronize()` | **Yes** | explicit barrier |
| CPU `sync_parallelize` | **Yes** | fork-join, always blocks |
| CPU `vectorize` | **Yes** | single-threaded, synchronous |
| Pipeline: Filter → Project (same device, same stream) | **No** | stream ordering |
| Pipeline barrier: Aggregate, Join build phase | **Yes** | needs full input |

`reduce` in `views.mojo` is the critical constraint for SQL aggregates. The GPU
kernel runs async but the `dev_buf.to_cpu(ctx)` call at the end of
`_reduce_dispatch` forces a sync. For a pipeline like:

```
apply[compare]  →  apply[filter]  →  reduce[sum]
```

the first two `apply` calls are fully async and pipelined on the GPU. `reduce`
forces a single sync at the end. Do not call `reduce` per morsel — accumulate
partial sums on the device and sync once at `finish()`.

### The pattern for CPU/GPU overlap

Since GPU `apply` is async, the CPU is free to process other work while the GPU
is running:

```mojo
# Enqueue GPU work — returns immediately
apply[op](gpu_col_a, ctx=gpu_ctx)         // queued on GPU stream

# CPU works concurrently while GPU runs
sync_parallelize[process_col_b](workers)  // CPU blocks; GPU does not

// Only sync once, after both finish
gpu_ctx.device.value().synchronize()
```

---

## `DeviceContext` as a unified task scheduler

### Batch-enqueue all column ops, single `synchronize()`

`enqueue_cpu_function` and `enqueue_cpu_range` submit coroutines to the AsyncRT
thread pool and return immediately. This means you can enqueue one closure per
column, then call `synchronize()` once at the end of the batch — all columns
execute concurrently (bounded by the thread pool):

```mojo
var cpu_ctx = DeviceContext(api="cpu")

for col in batch.columns:
    cpu_ctx.enqueue_cpu_function[lambda: apply_kernel(col)]()

cpu_ctx.synchronize()  # one barrier drains all column tasks
```

The GPU path already works this way: each `views.apply` call enqueues a kernel
async onto the GPU stream and returns. A single `gpu_ctx.synchronize()` after
all columns are enqueued is the right pattern there too.

**One caveat**: `enqueue_cpu_range` allocates N coroutine handles in a loop
before calling `AsyncRT_DeviceContext_enqueueHostFunctionRange`. Very large N
creates allocation pressure. Keep granularity at the column level (one task =
one column), not the element level.

### Inter-task dependencies — queue-level fences, not per-task futures

A `DeviceContext` is a **single ordered queue**, not a DAG scheduler. All tasks
submitted to the same context execute in FIFO order, so sequential dependencies
within one context are automatic — enqueue A then B and B always observes A's
output.

For cross-context dependencies the primitive is `enqueue_wait_for`:

```mojo
ctx_b.enqueue_wait_for(ctx_a)
# all subsequent ops on ctx_b wait for ctx_a's queue to drain to this point
ctx_b.enqueue_cpu_function[next_stage]()
```

This is a fence against the *entire* queue of `ctx_a` up to the point of the
call. There is no way to express "task B depends on task A but not task C" when
A and C are both on `ctx_a`. For true DAG dependencies you would need one
`DeviceContext` per logical task, which is expensive.

For marrow's use cases — "GPU filter runs before CPU aggregate merge" — the
queue-level fence is the right granularity.

### Cross-device CPU↔GPU dependencies

`enqueue_wait_for` calls `AsyncRT_DeviceContext_enqueue_wait_for_context` with
two opaque handles. There is no device-type check — the API is symmetric across
CPU and GPU contexts. This enables full CPU↔GPU chaining:

```mojo
var gpu_ctx = DeviceContext()           # Metal/CUDA
var cpu_ctx = DeviceContext(api="cpu")

# GPU produces a filtered column
gpu_ctx.enqueue_function[filter_kernel](...)

# CPU merge step waits for the GPU buffer to be ready
cpu_ctx.enqueue_wait_for(gpu_ctx)
cpu_ctx.enqueue_cpu_function[merge_bitmaps]()

# GPU consumer waits for CPU to finish staging
gpu_ctx.enqueue_wait_for(cpu_ctx)
gpu_ctx.enqueue_function[gpu_aggregate](...)

cpu_ctx.synchronize()
gpu_ctx.synchronize()
```

On Apple Silicon this is particularly cheap because CPU and GPU share physical
memory — `enqueue_wait_for` only needs to stall the queue, not copy data.

### Practical RecordBatch dispatch pattern

```
batch arrives
  │
  ├─ cpu_ctx.enqueue_cpu_range[apply_predicate](num_cpu_cols)  # parallel CPU
  │
  ├─ gpu_ctx.enqueue_function[gpu_kernel](...)                  # async GPU
  │
  ├─ cpu_ctx.enqueue_wait_for(gpu_ctx)                         # CPU waits for GPU result
  │
  ├─ cpu_ctx.enqueue_cpu_function[merge_results]()             # merge
  │
  └─ cpu_ctx.synchronize()  +  gpu_ctx.synchronize()           # one barrier each
```

The key is that `cpu_ctx.enqueue_wait_for(gpu_ctx)` is itself non-blocking on
the CPU — it inserts a dependency into the CPU queue so the merge closure
doesn't start until the GPU queue has drained to that point, but the calling
thread continues immediately.

---

## `sync_parallelize` and the CPU thread pool

`sync_parallelize` in the stdlib does:

```mojo
var cpu_ctx = ctx.or_else(DeviceContext(api="cpu"))
cpu_ctx.enqueue_cpu_range[func_wrapped](count=num_work_items)
cpu_ctx.synchronize()
```

When called without a `ctx`, it creates `DeviceContext(api="cpu")` via
`AsyncRT_DeviceContext_create` on every call. The AsyncRT runtime maintains a
**persistent thread pool** — the `DeviceContext` is a lightweight handle into
it, not the pool itself, so threads are not created or destroyed per call. The
overhead is the handle creation, not thread lifecycle.

The important limitation is the pattern itself: it is always **fork-join**. Work
is submitted, all workers complete, the calling thread unblocks. There is no
persistent work queue where you can produce morsels on one thread and consume
them on a pool of worker threads concurrently.

To reduce handle-creation overhead, store a persistent CPU `DeviceContext` in
marrow's `ExecContext` and pass it through to `sync_parallelize`.

---

## Parallelism levels

```
Table (full dataset)
  └── Column (contiguous array, N rows)
        └── Morsel (~64K rows, pipeline unit)
              └── SIMD chunk (8–16 elements)
```

### CPU-only

**Intra-morsel parallelism** splits a 64K morsel across threads. 64K int32 =
256KB ≈ L2 cache size. Each thread gets ~32KB on an 8-core machine, which fits
in L1. This is only worthwhile for compute-heavy operations (hashing, string
ops). For memory-bandwidth-bound operations — which covers most column arithmetic
and comparisons — threads contend on the shared memory bus and bandwidth
saturates at 2–4 threads regardless.

**Inter-morsel parallelism** is the stronger win. Multiple threads each pull
independent morsels through the same pipeline, each maintaining their own
partial aggregate state. No intra-morsel synchronization is needed. This is the
model used by DuckDB (morsel-driven parallelism).

The current pull-based executor is serial at the morsel level. `FilterProcessor.pull()`
fetches one morsel, processes it, returns. To add inter-morsel parallelism:
- Replace the single-threaded loop with a shared atomic morsel counter
- Clone the processor pipeline per worker thread
- Merge per-thread partial aggregates at the end with a single lock or reduction

For bandwidth-bound pipelines, do not parallelize inside a morsel. Parallelize
across morsels.

### GPU-only

**64K elements is too small for a GPU kernel launch.** Reference numbers for
Apple Silicon (~200 GB/s bandwidth) and NVIDIA H100 (~3.35 TB/s):

```
64K int32 = 256 KB
  Apple M-series: 256 KB / 200 GB/s ≈ 1.3 μs execution
  Kernel launch overhead: ~2–5 μs
  → launch overhead > execution time

1M int32 = 4 MB
  Apple M-series: 4 MB / 200 GB/s ≈ 20 μs execution
  → reasonable GPU utilization
```

**Do not split columns into 64K morsels before dispatching to GPU.** Process
the entire column (or as much as fits in device memory) in one kernel launch.
The `gpu_threshold = 1_000_000` field on the executor's `ExecContext` is
the right guard for this — columns below threshold go to CPU, columns above go
to GPU. It is currently unused in the planner.

For GPU aggregation: keep partial sums on device through all input. Call `reduce`
(which forces a D2H sync) only once at query completion, not once per morsel.

### Heterogeneous CPU + GPU together

The dispatch rule per column:
- `len(col) >= gpu_threshold` → GPU (async enqueue)
- `len(col) < gpu_threshold` → CPU (`sync_parallelize` or `vectorize`)

Both enqueues happen before any sync. A single `ctx.synchronize()` at the morsel
output boundary collects both. On Apple Silicon the cost of "uploading" a column
to the GPU is zero — both sides already see the same memory.

For **morsel-level CPU/GPU pipelining** (GPU processes morsel N while CPU
processes morsel N+1):
- Pre-allocate two sets of device buffers (ping-pong)
- Morsels must be independent — true for scan + filter + project, not for joins
- After processing both, merge partial aggregates with one sync

---

## Architecture gaps (as of May 2026)

The kernel layer (`execution.mojo`) and view layer (`views.mojo`) are
well-designed for heterogeneous dispatch. The gap is in the executor
(`expr/executor.mojo`), which does not thread execution context through its
processor tree.

### Two disconnected `ExecContext` types

`execution.mojo` defines:

```mojo
struct ExecContext:
    var num_threads: Int    // serial / multi(N) / auto
    var device: Optional[DeviceContext]

    def is_gpu() -> Bool
    def wants_parallel(n, min_parallel_size) -> Bool
    def resolved_num_threads() -> Int
```

All arithmetic, compare, and view kernels accept this type. It has `serial()`,
`parallel()`, `gpu()`, and `auto()` factory methods.

`expr/executor.mojo` defines a separate type:

```mojo
struct ExecContext:
    var device_ctx: Optional[DeviceContext]
    var num_cpu_workers: Int
    var morsel_size: Int      // default 65_536
    var gpu_threshold: Int    // default 1_000_000 — currently unused
```

The planner carries the query-level one but never converts it to the kernel-level
type and never passes it to processors.

### `ValueProcessor.eval` has no execution context

```mojo
trait ValueProcessor:
    def eval(self, batch: RecordBatch) raises -> AnyArray: ...
```

Every `BinaryProcessor.eval()` calls `add(l, r)` with an implicit
`ExecContext.serial()`. The GPU path that exists in all kernels is
unreachable from the executor. The `DISPATCH_CPU / DISPATCH_GPU / DISPATCH_AUTO`
constants defined on `Binary` expression nodes are imported by the executor but
never read by the planner or processors.

### Minimal wiring fix

```mojo
// Step 1: add ctx to the trait
trait ValueProcessor:
    def eval(self, batch: RecordBatch, ctx: KernelCtx) raises -> AnyArray: ...

// Step 2: thread it through each processor
struct BinaryProcessor(ValueProcessor):
    def eval(self, batch: RecordBatch, ctx: KernelCtx) raises -> AnyArray:
        var l = self.left.eval(batch, ctx)
        var r = self.right.eval(batch, ctx)
        if self.op == ADD:
            return add(l, r, ctx)   // now reaches GPU path
        ...

// Step 3: Planner converts query-level ctx → kernel-level ctx
// applying gpu_threshold to decide the device per column
```

---

## Kernel fusion

### Why it matters

Each `views.apply` call is one kernel launch. A predicate like
`WHERE a > 5 AND b < 10` currently produces:
1. `apply[compare]` on column a → intermediate bitmap, launch 1
2. `apply[compare]` on column b → intermediate bitmap, launch 2
3. `apply[and_]` on the two bitmaps → merged bitmap, launch 3
4. `filter` using bitmap → output columns, launch 4

A fused kernel would do all four in one pass with no intermediate buffers.

### JIT is required for true register-level fusion of runtime plans

`DeviceFunction` and `compile_function` in the Mojo stdlib are **compile-time
AOT constructs** — the kernel function is a compile-time parameter baked into
the binary via `CompiledFunctionInfo`. Mojo has no user-facing JIT API.

GPU `elementwise` kernels require the `process[W](idx)` body to be a
`@parameter` function — fully inlined and statically known at compile time.
Dispatching through `AnyValueProcessor._virt_eval` (a function pointer) inside
a GPU kernel is not possible.

This is the same constraint that led RAPIDS/cuDF to use Python-based JIT
(Numba, `torch.compile`) to generate fused GPU kernels from runtime expression
trees.

### What is achievable without JIT

**CPU — loop fusion without register fusion**: one `vectorize` pass that chains
through the `AnyValueProcessor` tree via function pointers. One pass through
memory, no intermediate `AnyArray` allocations. The compiler will not inline
across function pointers so SIMD register fusion does not occur, but eliminating
intermediate allocation and the extra memory passes is a meaningful win for
bandwidth-bound workloads.

**Pre-compiled physical operator library**: AOT-compile fused kernels for the
patterns covering ~90% of real SQL. The `Planner` pattern-matches subtrees at
plan-build time and replaces matched nodes with `PrecompiledProcessor` wrappers.
Unmatched trees fall back to the interpreter, which remains correct.

```mojo
// Compiled once — single kernel, no intermediate buffers
def scan_filter_sum[T: DType](
    col: BufferView[T, _],
    threshold: Scalar[T],
    ctx: ExecContext,
) raises -> Scalar[T]:
    @parameter
    @always_inline
    def process[W: Int, rank: Int, alignment: Int = 1](idx: IndexList[rank]):
        var i = idx[0]
        var val = col.load[W](i)
        var mask = val > SIMD[T, W](threshold)
        partial_acc += (val * mask.cast[T]()).reduce_add()
    _apply_dispatch[T, has_accelerator_support[T](), process](len(col), ctx)
    ...
```

High-value patterns to compile first (in priority order):
1. `scan + compare_filter` — eliminates 1 kernel + 1 intermediate bitmap
2. `scan + filter + aggregate` — covers the most common SQL shape end-to-end,
   eliminates all intermediate arrays
3. `scan + project (arithmetic) + filter` — computed predicates like
   `WHERE a * 2 > b`
4. `scan + multi_predicate_filter` — AND/OR of multiple column comparisons

---

## Where parallelism should actually happen

There are two places in the current architecture where parallelism can be
introduced: inside each kernel call (intra-morsel, via `_apply_dispatch`) and
across morsels at the executor level. Only one of them makes sense today.

### Intra-morsel kernel parallelism — does not make sense at 64K rows

When `_apply_dispatch` receives a 64K-row morsel and `ctx.wants_parallel(65536)`
fires (the threshold is 32768), it calls `sync_parallelize` to split that morsel
across threads. The numbers do not work:

```
64K int32 = 256KB — fits entirely in L2 cache
Single-thread vectorized processing at L2 bandwidth: ~25μs
sync_parallelize overhead (wakeup, dispatch, barrier): ~10–50μs
→ overhead is 40–200% of the actual work
```

For bandwidth-bound operations — which covers nearly all column arithmetic,
comparisons, and filter — threads contend on the same memory bus and bandwidth
saturates at 2–4 threads regardless of how many are launched. The
`min_parallel_size=32768` threshold in `wants_parallel` is too low by roughly
an order of magnitude for 64K morsels to benefit from intra-morsel threading.

Note: `is_gpu()` already returns `False` from `wants_parallel`, so the GPU path
correctly avoids CPU threading. But `_apply_dispatch` still launches a GPU
kernel per 64K morsel, which is equally wrong for different reasons (see GPU
section above).

### Inter-morsel parallelism at the executor — makes sense, not yet implemented

The right level is one thread per morsel, each thread running an independent
copy of the pipeline. Each thread's 256KB morsel is cache-local, there is no
shared state during processing, and partial aggregates are merged once at the
end. This is the DuckDB morsel-driven parallelism model and it scales correctly
with core count.

The current executor does not exploit this because `RelationProcessor.pull()` is
not thread-safe and there is only one pipeline instance. The structural fix is:
- Shared atomic offset counter in `ScanProcessor`
- Per-thread pipeline clone built by `Planner`
- Single merge step after all threads exhaust their morsels

### Removing morsel splitting entirely

A cleaner alternative is to remove the morsel split and pass full columns
directly to kernels, letting `_apply_dispatch` handle parallelism at the right
scale.

**What changes:** `ScanProcessor` returns the entire `RecordBatch` in one
`pull()` call. `FilterProcessor`, `ProjectProcessor` each call their kernel once
on the full column. The pull-based pipeline structure stays, it just produces
one large batch instead of N small ones.

**What you gain:**

- CPU `sync_parallelize` is now justified: 1M int32 = 4MB, processing time
  ~400μs at memory bandwidth, thread overhead ~10–50μs → overhead is ~3–12%.
- GPU `elementwise` is now efficient: 1M elements amortizes the 2–5μs kernel
  launch overhead across ~20μs of work on Apple Silicon.
- `_apply_dispatch` already handles any size correctly — no kernel code changes.
- The executor becomes simpler: no offset tracking, no morsel accumulation in
  `read_all()`.

**The trade-off — intermediate memory:**

Without morsel splits, all pipeline stages hold full-column intermediate results
simultaneously. For a `Filter → Project → Aggregate` pipeline on a 10M-row,
8-column float32 table:

```
filter predicate mask:    10M bits ≈ 1.2 MB
filtered output columns:  up to 10M × 8 × 4B = 320 MB worst case
project output:           similar
```

With 64K morsels only ~256KB of intermediates are live at once. On Apple Silicon
with unified memory this is less painful than on discrete GPU. For large tables
or memory-constrained environments morsels remain necessary.

**When removing morsels is wrong:**

- **`LIMIT` queries**: with morsels you can stop after the first N matching rows.
  Without morsels, the full table is scanned even for `LIMIT 10`.
- **Tables larger than device memory**: morsels are the mechanism that allows
  streaming data through a fixed working set.
- **Streaming output**: first results cannot be returned until the full scan
  completes.

**Practical recommendation:**

Make morsel size configurable per device rather than hardcoding 64K:

```mojo
struct ExecContext:
    var cpu_morsel_size: Int   // 256K–1M: large enough for sync_parallelize
    var gpu_morsel_size: Int   // full column or gpu_threshold rows
```

For the current marrow workload (analytics on in-memory data, no LIMIT), setting
`cpu_morsel_size = len(table)` is equivalent to removing morsels and gives all
the performance upside. `LIMIT`-aware early termination and memory-bounded
streaming can be added later without changing the kernel layer.

The key insight: the 64K figure was chosen to fit a morsel in L2 cache, but
kernels already tile at SIMD-register width internally via `vectorize`. The
morsel does not need to be cache-sized. The only real constraints on morsel size
are available memory for intermediates and streaming/early-termination
requirements — both are better served by a large configurable chunk than by a
fixed 64K.

---

## Recommended implementation order

1. **Thread kernel `ExecContext` through `ValueProcessor.eval`**
   Unlocks the GPU path that already exists in all kernels.
   Lowest effort, highest immediate impact.

2. **Make morsel size configurable; increase it dramatically**
   Add `cpu_morsel_size` and `gpu_morsel_size` to `ExecContext`.
   Set CPU default to 256K–1M rows so `sync_parallelize` is justified.
   Set GPU default to the full column (or `gpu_threshold`) so kernel
   launch overhead is amortized. The 64K default is wrong for both devices.

3. **Enforce `gpu_threshold` in the planner**
   Route columns below threshold to CPU, above to GPU.
   Required before any heterogeneous execution is correct.

4. **Persistent CPU `DeviceContext` in `ExecContext`**
   Store at query start, reuse across all `sync_parallelize` calls.
   Avoids `AsyncRT_DeviceContext_create` overhead per morsel.

5. **Inter-morsel CPU parallelism** (if morsel splitting is kept)
   Shared atomic morsel counter + per-thread pipeline clones.
   Only worthwhile once morsel size is large enough that intra-morsel
   threading is not justified (i.e. after step 2).

6. **Heterogeneous column dispatch**
   Async-enqueue large columns to GPU, process small columns on CPU
   concurrently, single `ctx.synchronize()` at batch output boundary.

7. **Morsel-level CPU/GPU pipeline overlap**
   GPU processes morsel N while CPU scans + processes morsel N+1.
   Requires ping-pong device buffer allocation.
   Only meaningful once morsel size is large enough for GPU utilization.

8. **Pre-compiled fused operator library**
   Start with `scan_filter_aggregate`. Add patterns as profiling identifies
   the next bottlenecks.

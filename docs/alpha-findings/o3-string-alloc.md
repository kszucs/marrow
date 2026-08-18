# O3 — the string allocation bottleneck in ClickBench q21

ClickBench q21 (`SELECT COUNT(*) FROM hits WHERE URL LIKE '%google%'`) was the
largest single query cost in the suite. The brief attributed it to buffer
growth — `_alloc_bytes` at 25.8% and `List::_realloc` at 15.7% of a sampled
tree — and guessed at an output `List` that reallocs instead of reserving from
a known size.

**The allocation share was right and understated; the shape was wrong.** There
is no growing output buffer. The allocations are a whole `LikePattern`
object — a token `List[Int]`, a literal `List[UInt8]` and a `String` — built
and thrown away **once per row**.

---

## 1. What the profile actually says

`pixi run profile --sample clickbench-q21`, 20 repeats, 7 603 main-thread
samples, bucketed by self time:

| share | bucket |
|---|---|
| **49.5%** | tcmalloc / AsyncRT allocator internals |
| **6.3%** | `std::memory::alloc*` |
| 19.6% | `StringSpan` compare / find — the actual matching |
| 10.0% | other libmarrow |
| 7.1% | kernel wait / sync |
| 3.5% | snappy decompress |
| 1.7% | `List` ops |

So **~56% of the query is the allocator**, against ~20% doing the string
matching it exists to do — worse than the 41% / 15% in the brief.

Attributing allocator *inclusive* time to the nearest non-allocator caller
gives the shape:

| share | immediate caller |
|---|---|
| 32.1% | the inlined `FilterProcessor` pipeline |
| 15.4% | `List::_realloc`, `T = List[Int]` |
| 8.5% | a `StringSpan x StringSpan -> Bool` function in libmarrow |

Two of these name the culprit outright. `List[Int]` is the type of
`LikePattern.tokens`, and nothing else in the query pipeline uses one.
A `StringSpan x StringSpan -> Bool` function that *allocates* is
`LikeKernel.predicate[o1, o2](s, pat) -> Bool` — which is
`LikePattern[False](pat).matches(s)`.

## 2. The defect

`StringPredicateKernel.apply` (array x array) is a per-element loop over
`Self.predicate`:

```mojo
for i in range(n):
    if left.is_valid(i) and right.is_valid(i):
        if Self.predicate(left.unsafe_get(UInt(i)), right.unsafe_get(UInt(i))):
            data.set(i)
```

That is correct for `startswith` / `contains` / `==`, which have no per-call
set-up. `LikeKernel.predicate` is different: it has to *compile* its pattern
before it can match, so this loop rebuilt a token list, a literal buffer and a
`String` for every row.

Nothing in the kernel layer forced that shape onto q21 — the expression layer
did. `marrow/expr/dynamic.mojo` wires `like` through
`_string_binary[LikeKernel]`, which calls the array x array `K.dispatch`, and a
literal operand is evaluated by `DynValue._literal` as
`payload[DynScalar].repeat(batch.num_rows())`. So a *constant* pattern reaches
the kernel as n identical rows, and every one of the n compiles was redundant.

`marrow/expr/values.mojo` (the AOT lane) already avoids this: it calls
`apply_scalar`, which `LikeKernel` overrides to compile once. Only the runtime
lane — the one the Python bindings use, and therefore the one ClickBench
measures — took the slow path.

## 3. The fix

`_match_arrays[L, R, ignore_case]` in `marrow/kernels/string.mojo`, which
`LikeKernel` and `ILikeKernel` install as their array x array `apply`. It
remembers the last pattern text and recompiles only when it changes:

```mojo
var pat = right.unsafe_get(UInt(i))
if not primed or StringSlice(current) != pat:
    compiled = LikePattern[ignore_case](pat)
    current = String(pat)
    primed = True
if compiled.matches(left.unsafe_get(UInt(i))):
    data.set(i)
```

A memo rather than a broadcast special case: the constant shape collapses to a
single compile, a genuinely varying right operand still recompiles on every
change, and the check costs one comparison against a string already in cache.
No array, scalar or builder layout changed; nothing outside `string.mojo` did.

## 4. Numbers

`pixi run -e dev pytest marrow/kernels/tests/bench_string.mojo --benchmark`,
**Min of 5 rounds**, two runs per side, best of the two. `length`, `contains`,
`upper`, `like_general` and every scalar-pattern LIKE case are drift controls —
this box drifts +-8% per case, and sibling compiles pushed load average to 18
during the session, so nothing here is reported without them.

| benchmark | before (ms) | after (ms) | change |
|---|---|---|---|
| `bench_like_array_1m` | 250.90 | 12.00 | **20.9x** |
| `bench_like_array_dense_1m` | 242.38 | 9.85 | **24.6x** |
| `bench_like_array_sparse_1m` | 239.15 | 11.37 | **21.0x** |
| `bench_like_array_100k` | 25.27 | 1.14 | 22.2x |
| `bench_like_array_10k` | 2.652 | 0.111 | 23.9x |
| `bench_ilike_array_100k` | 38.14 | 3.46 | 11.0x |
| *controls* | | | |
| `bench_length_1m` | 0.069 | 0.069 | 1.00x |
| `bench_contains_1m` | 7.332 | 7.358 | 1.00x |
| `bench_like_general_100k` | 6.927 | 6.893 | 1.00x |
| `bench_like_dense_1m` (scalar) | 4.880 | 4.820 | 1.01x |
| `bench_like_sparse_1m` (scalar) | 7.653 | 7.477 | 1.02x |
| `bench_like_scalar_1m` | 8.170 | 7.804 | 1.05x |
| `bench_ilike_scalar_100k` | 2.938 | 2.893 | 1.02x |
| `bench_upper_100k` | 73.92 | 74.46 | 0.99x |

The dense/sparse pair is what separates the two possible diagnoses, and it
answers cleanly. **Before**, dense and sparse cost the same (242.4 vs
239.2 ms): the cost tracked the row count, not the match count — per-row
set-up. **After**, they separate (9.8 vs 11.4 ms) and land on their
scalar-pattern counterparts (4.9 and 7.5 ms): the cost now tracks how far into
each string the substring search has to scan, which is what a compare-bound
implementation looks like.

End to end, ClickBench `hits_0.parquet` (1M x 105) at `-O3`, 5 interleaved
repeats, medians. Because absolute times moved with load, the honest figure is
the **marrow / duckdb ratio**, which the interleaving makes a within-run
quantity — and which is flat on the untouched q01 and q13 across all four runs:

| query | before | after | marrow/duckdb before | after | normalised |
|---|---|---|---|---|---|
| q21 | 780.8 / 764.4 ms | 91.6 ms | 3.7x / 4.0x | 1.0x | **3.9x** |
| q22 | 806.9 / 635.7 ms | 101.6 ms | 3.6x / 4.3x | 1.1x | **3.6x** |
| q23 | 1658.5 / 1239.1 ms | 162.3 ms | 3.9x / 4.4x | 0.9x | **4.6x** |
| q01 *(control)* | 25.1 / 22.0 ms | 11.0 ms | 3.4x / 3.6x | 3.8x | 1.0x |
| q13 *(control)* | 88.8 / 85.7 ms | 39.0 ms | 2.7x / 2.7x | 2.8x | 1.0x |

Against polars, q21 went from 7.8x to **2.2x** and q23 from 10.8x to **3.0x**.
q23 now runs *faster* than duckdb (0.9x).

## 5. A measurement result worth keeping

The first version of the fix also reserved the token list up front
(`self.tokens.reserve(n)` in `LikePattern._compile`, an exact upper bound,
saving four geometric reallocations per compile). It looked free.

It cost **+43% on `bench_contains_1m`, +35% on `bench_like_sparse_1m` and
+21% on `bench_like_scalar_1m`** — paths whose source it does not touch at all.
`ContainsKernel.apply_scalar` shares no code with `LikePattern`. The regression
reproduced across two runs per side and disappeared entirely when the single
`reserve` line was removed, with everything else unchanged; it is a codegen or
layout effect of perturbing one `-O3` compilation unit, not a semantic one.
The reserve is also worthless *after* the memo, since `_compile` now runs once
per array rather than once per row, so it was dropped.

Two things follow. Adding a line that cannot logically affect an unrelated
kernel can still cost that kernel 40%; and the only reason this was caught is
that the benchmark set included controls the change was not supposed to move.
Without `bench_contains_1m` and `bench_length_1m` in the same run, a 20x win
would have shipped with a 40% regression hidden underneath it.

## 6. Still open — and not ours to fix

**`DynValue._literal` splats a constant into a full-length array.**
`payload[DynScalar].repeat(batch.num_rows())` builds n copies of `"%google%"`
per morsel: a fresh offsets buffer, a fresh values buffer, and n
`BinaryLikeBuilder.append` calls. In the profile that is the 4.2%
`BinaryLikeArray` conversion plus the 2.9% `BinaryLikeBuilder.append` self
time, and their allocations. The memo makes it harmless for *matching*, but the
array is still built and thrown away every morsel.

The fix belongs in `marrow/expr/dynamic.mojo`, which this agent does not own:
`_string_binary` should recognise a literal right operand and call the kernel's
`apply_scalar` — which every `StringPredicateKernel` already has, and which
`LikeKernel`/`ILikeKernel` already override — instead of `dispatch`. That is
the same choice `marrow/expr/values.mojo:2070` already makes in the AOT lane,
and it would remove the splat for `startswith`, `endswith`, `contains` and the
string comparisons too, not just LIKE.

Residual gap after this change: `bench_like_array_1m` (12.0 ms) is still ~1.5x
`bench_like_scalar_1m` (7.8 ms) on identical data, and that difference is the
per-row validity check, offset load and memo comparison on a right operand that
carries no information. Routing constants to `apply_scalar` would close it.

## 7. Verification

- `pixi run -e dev precompile` — 0 errors, 0 warnings.
- `pixi run -e dev pytest marrow/kernels/tests/test_string.mojo` — 39 passed.
- `pixi run -e dev pytest marrow/expr/tests/test_strings.mojo
  marrow/expr/tests/test_runtime.mojo` — 56 passed.
- `pixi run -e bench pytest python/marrow/tests/test_clickbench.py` —
  85 passed, 1 skipped.
- `pixi run python3 benchmarks/binary_size/check_gate.py` — OK. Four gates
  shrank; `query_dynvalue` +22 592 bytes (+0.464%), inside the 0.5% threshold,
  which is the cost of the `_match_arrays` instantiations. `baseline.json`
  untouched.

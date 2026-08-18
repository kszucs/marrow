# F1 — the ClickBench Q11/Q12 SIGSEGV

Branch `alpha`, worktree
`/Users/kszucs/Workspace/marrow/.claude/worktrees/agent-a650287548b6407b7`.

The brief named `COUNT(DISTINCT)` in a grouped context as the distinguishing
ingredient. **It is not involved at all.** Neither is the group-by, the
aggregate cluster, the Parquet page-skip path, or the distinct sketches. The
defect is a one-element heap overflow in a SIMD-adjacent primitive in
`marrow/views.mojo`, reachable from *any* `filter` over a primitive column.

---

## 1. Root cause

**`marrow/views.mojo:251` — `BufferView.compressed_store_dense` writes
`popcount(sel_bits) + 1` elements where its caller allocated
`popcount(sel_bits)`.**

```mojo
var offset = 0
comptime for i in range(8):
    var byte = (sel_bits >> UInt64(i * 8)) & 0xFF
    var b = byte
    var k = 0
    comptime for bit in range(8):
        self.unsafe_set(offset + k, src.unsafe_get(i * 8 + bit))  # <-- always stores
        k += Int(b & 1)
        b >>= 1
    offset += Int(pop_count(byte))
```

The routine is *branchless on purpose*: it stores the lane first and only then
consults the lane's selection bit, so an unselected lane writes a value the next
selected lane overwrites. That is correct for every lane below the highest set
bit. For every lane **above** it there is no next selected lane, so the write
stays — at element index exactly `popcount(sel_bits)`, one past the packed
output.

The overflow becomes a *heap* overflow at `marrow/views.mojo:361`, inside
`BufferView.filter`:

```mojo
var buf = Buffer.alloc_uninit(out_len * size_of[Scalar[Self.T]]())
var dst = buf.view[Self.T](0, out_len)
...
out_pos += dst.slice(out_pos).compressed_store(self.slice(i), sel_word)
```

`out_len` is the pre-counted set-bit total, so on the **last** selection word
`out_pos + popcount(sel_word) == out_len` and the extra store lands at element
`out_len` — outside `buf`.

The last link is `marrow/buffers.mojo:444`:

```mojo
def _aligned_size[T: DType](length: Int) -> Int:
    return math.align_up(length * size_of[T](), 64)
```

`CLAUDE.md` says "All buffers are 64-byte aligned **and padded**". They are
aligned; they are **not** padded. When `length * size_of[T]` is already a
multiple of 64 there is no slack whatsoever, and the one-element store goes
straight into the neighbouring allocation. For `int64` that is every output
length divisible by 8 — one filter call in eight, on average.

### Why it is a SIGSEGV with no message rather than a clean abort

The corrupted bytes are gperftools tcmalloc's singly-linked freelist `next`
pointer inside the adjacent free block. Nothing notices at the time of the
write. The process dies later, at an unrelated allocation, in
`tcmalloc::SLL_Next` — which is why every crash report points at
`Buffer::alloc_uninit` under `AggregateProcessor::pull` and why the shape looked
like an aggregate bug. Four macOS `.ips` reports from separate runs all fault on
the same value, `KERN_INVALID_ADDRESS at 0x0000000100000026`: a data word where
a pointer should be.

```
tcmalloc::SLL_Next(void*)                                   | libAsyncRTRuntimeGlobals.dylib
tcmalloc::SLL_TryPop(void**, void**)                        | libAsyncRTRuntimeGlobals.dylib
tcmalloc::ThreadCache::FreeList::TryPop(void**)             | libAsyncRTRuntimeGlobals.dylib
marrow::buffers::Buffer::alloc_uninit[...]                  | libmarrow.so
marrow::expr::execution::AggregateProcessor::pull(...)      | libmarrow.so
marrow::expr::relations::Sort::to_processor(...)_closure_1  | libmarrow.so
marrow::expr::execution::DynProcessor::collect(...)         | libmarrow.so
```

The stack is where the heap was *found* broken, never where it was broken.

### Why it looked selectivity- and column-dependent

`compressed_store` only takes the dense path when `popcount(sel_word) > 24`,
so a selection word must be **dense** to trigger it — and it must be the last
contributing word, and the output length must land on the no-slack boundary for
that dtype. That triple conditional is the whole reason the bisection table read
as noise:

| predicate on `hits_0.parquet` | rows kept | result |
|---|---|---|
| none | 1,000,000 | ok — no filter runs |
| `MobilePhone >= -32768` (all rows) | 1,000,000 | ok — every word is all-ones, run-merged by memcpy, `compressed_store` never called |
| `MobilePhone == 31337` / `WatchID == 1` | 0 | ok — both row groups pruned by statistics, no morsel produced |
| `IsRefresh <> 0` | 256,201 | ok |
| `SearchPhrase <> ''` | 69,354 | ok |
| `MobilePhoneModel <> ''` | 19,643 | **SIGSEGV** |
| `MobilePhone <> 0` | 27,160 | **SIGSEGV** |
| `Age = 31` | 433,099 | **SIGSEGV** |

Note the two ends: an all-ones mask never reaches the dense path, and a mask
with 0 surviving rows never reaches it either. It is the *middle*, clustered
case that hits it — which is exactly the shape of a real predicate.

---

## 2. The fix

`marrow/views.mojo`, in the adaptive `compressed_store`:

```mojo
var cnt = Int(pop_count(sel_bits))
if cnt <= sparse_threshold or self._length <= cnt:
    self.compressed_store_sparse(src, sel_bits)
else:
    self.compressed_store_dense(src, sel_bits)
```

`BufferView` already carries its own length, and `dst.slice(out_pos)` reports
exactly the elements still available, so the check is local, exact and free. It
demotes precisely one word per `filter` call — the last one — to the sparse
CTZ path, which is at most 64 iterations once per call.

The alternative, over-allocating by one element in `BufferView.filter`, was
rejected: it leaves `compressed_store_dense` a loaded gun for the next caller.
The contract is now written into both docstrings.

### Regression test

`marrow/tests/test_views.mojo::test_bufferview_compressed_store_dense_stays_in_bounds`
— a sentinel element immediately past a view sized exactly to the popcount.
Before the fix it fails with the *value of the last source lane* in the sentinel
slot, which is the overflow itself, not a proxy for it:

```
At ./marrow/tests/test_views.mojo:394:17: AssertionError: `left == right` comparison failed:
   left: 64
  right: -99
```

`test_bufferview_filter_last_word_stays_in_bounds` covers the same shape
end-to-end through `BufferView.filter`. It passes both before and after — the
overflow wrote *correct* data one element too far — and is kept as a
correctness guard on the path that now changes strategy, not as a detector.

---

## 3. Ruled out, with evidence

Each of these cost a build or a run; none of them is the bug.

- **`COUNT(DISTINCT)` and the grouped distinct kernels.** Replacing it with
  `count` still crashes; so does `aggregate(by=[], c=lit(1).count())` with no
  group key and no distinct at all.
- **The group-by.** The minimal reproduction has `by=[]`.
- **Multi-morsel accumulation in `AggregateProcessor`.** It is where the crash
  *surfaces* (the next allocation), never where it originates. A zero-row
  morsel is handled correctly — `FilterProcessor.pull` loops until a morsel has
  rows.
- **Parquet page-level skipping (`RowSelection`).** Disabled it outright
  (`_read_plan` returning `None` for the selections) and rebuilt: all three
  crashing cases crash identically. This was my leading hypothesis and it was
  wrong.
- **Parquet row-group pruning.** A predicate `pruning.mojo` cannot use crashes
  the same way.
- **The Parquet reader generally.** `morsel_size` changes the crash and cannot
  change decoding: 8192 crashes, 65536 and 1,000,000 do not, on the identical
  file and projection. Morsel size only slices already-decoded row groups.
- **`ASSERT=all`.** Rebuilt `libmarrow.so` with it; no assertion fires. The
  overflow goes through `unsafe_set` on a `BufferView`, which is unchecked by
  construction.
- **AddressSanitizer.** Not usable as evidence here: a trivial
  `unsafe_alloc(16)` + 200-byte overflow probe, built with the same flags
  `conftest.py` uses, hangs before producing any report. `CLAUDE.md`'s warning
  that a clean ASAN run is not evidence is, if anything, understated.
- **`lldb`.** Cannot attach to the pixi CPython on this machine ("Not allowed to
  attach to process"). The macOS `.ips` crash reports in
  `~/Library/Logs/DiagnosticReports/` are a complete substitute and were the
  single most useful tool in this investigation — see `repros/showips.py`.

---

## 4. Structural notes on the aggregate cluster

Four agents have flagged this cluster (`docs/alpha-findings/README.md` §1) and I
was pointed at it as the prime suspect. **It is not implicated in this defect**,
and that is worth recording precisely because the crash signature framed it.
What the investigation does say about it:

- `AggregateProcessor::pull` appears at the top of every crash report for a bug
  that lives two layers below it, because it is the first thing after a long
  filtered scan to allocate. Any heap corruption anywhere in the scan/filter
  pipeline will name it. Anyone reading a stack trace here should treat
  "`AggregateProcessor::pull` + `alloc_uninit` + tcmalloc" as *"the heap is
  broken"*, not as a lead.
- The buffering in `pull` (every morsel's group ids and evaluated value columns
  held until the drain completes) is what makes the window between corruption
  and detection wide and the failure look nondeterministic. It is not wrong, but
  it is why this took a bisection rather than a stack trace.

The one structural finding this investigation adds is **not** in the aggregate
layer:

> **`CLAUDE.md` and `views.mojo` both assert that Arrow buffers are "64-byte
> padded", and `Buffer._aligned_size` only aligns.** Two separate pieces of code
> are written against the padding: this one (a write, now fixed) and
> `BitmapView.load_bytes`/`store_bytes`, whose docstring says verbatim "Safe
> because Arrow buffers are 64-byte padded".

`BitmapView.compressed_store` deposits its bits with an unconditional 8-byte
load/OR/store, so on the last word of a filtered bitmap it can read and write up
to 7 bytes past a bitmap whose byte size is an exact multiple of 64 (e.g. any
512-bit output). It is **benign for heap integrity today** — the out-of-range
bytes are written back exactly as read (`x | 0`) — so I have not changed it and
deliberately did not add a test that cannot fail. It is a live hazard for two
reasons worth recording:

1. it is a read-modify-write, so it is a lost-update race the moment two threads
   filter into adjacent allocations;
2. it will be reported by any sanitizer that ever does work here.

The durable fix is to make the documented contract true — have `_aligned_size`
guarantee slack past the logical size rather than merely aligning — but that
changes `len(buffer)` for every buffer in the tree and the default length of
every `Buffer.view()`, so it is not an alpha-week change. Recorded here so the
next person meets the discrepancy in writing rather than in a crash.

---

## 5. Reproductions

The minimal synthetic reproduction is the unit test above; it needs no data
file. The bisection drivers used against the real dataset are kept under
`repros/`:

- `repros/bisect2.py` — projection x predicate x morsel-size x source matrix
  over `~/Workspace/ClickBench/data/hits_0.parquet`. The smallest crashing
  shape is a **two-column** projection:
  `python repros/bisect2.py MobilePhoneModel,UserID mpm_ne_empty 8192`
- `repros/run2.sh` — runs a list of those specs, reporting each exit code.
- `repros/showips.py`, `repros/lastcrash.sh` — decode the macOS crash reports.
- `repros/synth.py`, `repros/synth2.py` — synthetic Parquet files used to test
  (and rule out) the page-skipping and zero-row-morsel hypotheses.

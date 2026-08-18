# G1 — the buffer padding invariant, made real and enforced

Branch `alpha`, worktree
`/Users/kszucs/Workspace/marrow/.claude/worktrees/agent-a26a6b6323c661d4f`.

The brief asked me to decide whether to pad marrow's allocations or drop the
padding claim, and to add debug assertions that would have caught F1's heap
overflow. The short version:

- **The premise that marrow does not pad is wrong.** `Buffer._aligned_size` is
  byte-for-byte what Arrow C++ does. Marrow is already spec-conformant.
- **What is false is the inference** that padding implies *slack past the
  logical end*. It does not, in the spec or in Arrow C++, and three separate
  places in `views.mojo` were written against that inference.
- **Decision: do not pad further; bound the accesses.** Padding cannot deliver
  the invariant, because half the buffers marrow reads are not marrow's.
- The assertions found **a second, unrelated defect**: `BitmapView.load_bits`
  silently dropped rows from any filter over a sliced column.

---

## 1. What the Arrow specification actually says

`~/Workspace/arrow/docs/source/format/Columnar.rst`, § *Buffer Alignment and
Padding*, lines 264–295, verbatim:

> Implementations are **recommended** to allocate memory on aligned addresses
> (multiple of 8- or 64-bytes) and pad (overallocate) to a length that is a
> multiple of 8 or 64 bytes. When serializing Arrow data for interprocess
> communication, these alignment and padding requirements **are enforced**. If
> possible, we suggest that you prefer using 64-byte alignment and padding.
> Unless otherwise noted, padded bytes do not need to have a specific value.

and, on the motivation:

> The recommended padding of 64 bytes allows for using `SIMD` instructions
> consistently in loops without additional conditional checks.

**Recommended vs required, stated rather than blurred:** for in-memory data,
alignment and padding are a *recommendation*. They become *requirements* only
"when serializing Arrow data for interprocess communication" — the IPC
format. Nothing in the columnar spec obliges an in-memory producer to pad, and
nothing obliges a consumer to assume padding.

**The C Data Interface is not IPC and has its own, weaker, rules.**
`CDataInterface.rst` lines 487–489:

> It is recommended, but not required, that the memory addresses of the buffers
> be aligned at least according to the type of primitive data that they
> contain. Consumers MAY decide not to support unaligned memory.

That document does not mention padding **at all**. A buffer arriving through
`__arrow_c_array__` may be unaligned and is certainly not guaranteed to have a
single spare byte.

### What "pad (overallocate)" means numerically

It means *round the allocation length up to a multiple of 64*. It does **not**
mean *append 64 bytes*. Those coincide only when the length is not already a
multiple of 64.

### Arrow C++ does exactly the rounding, and nothing more

`AllocateBuffer(size, pool)` → `PoolBuffer::Resize(size)` → `Reserve(size)` →

```cpp
// cpp/src/arrow/memory_pool.cc:984
static Result<int64_t> RoundCapacity(int64_t capacity) {
  ...
  return bit_util::RoundUpToMultipleOf64(capacity);
}
```

the pool then allocates `new_capacity` bytes, and `ResizePoolBuffer` calls
`buffer->ZeroPadding()`, which memsets `[size_, capacity_)`. `kDefaultBufferAlignment`
is `64` (`cpp/src/arrow/type_fwd.h:759`).

So `arrow::AllocateBuffer(64)` allocates **exactly 64 bytes**: `size_ == 64`,
`capacity_ == 64`, and `ZeroPadding()` memsets zero bytes. An `int64` column of
8 values has no slack in Arrow C++ either.

### And marrow's rule is identical

```mojo
# marrow/buffers.mojo
def _aligned_size[T: DType](length: Int) -> Int:
    return math.align_up(length * size_of[T](), 64)
```

`align_up(n, 64)` **is** `RoundUpToMultipleOf64(n)`. Marrow already implements
the spec's recommendation, at the strongest setting the spec suggests, with the
same semantics as the reference implementation. The only thing marrow does
*not* do is `ZeroPadding()` — `alloc_uninit` leaves the rounded-up bytes
uninitialised, which the spec explicitly permits ("padded bytes do not need to
have a specific value"), though `_aligned_byte_range`'s docstring had claimed
otherwise and now does not.

**Conclusion.** `CLAUDE.md`'s "All buffers are 64-byte aligned and padded" was
*true as a statement about allocation sizes* and *false as a licence to
overstep*, and it was being read the second way. That is the defect: a sentence
that is technically correct and operationally misleading.

---

## 2. The decision: bound the accesses, do not over-allocate

Two honest options were on the table. The literal "pad allocations to a 64-byte
multiple" is a no-op — it is already done — so the real choice was between
**over-allocating beyond the 64-byte multiple** (e.g. `align_up(bytes, 64) + 64`,
guaranteeing 64 bytes of slack) and **dropping the slack claim and bounding
every access**.

I chose the second. The reasoning, in order of weight:

1. **Padding cannot deliver the invariant, because marrow does not own every
   buffer it reads.** A FOREIGN buffer imported over the C Data Interface —
   every pyarrow array handed to a marrow kernel — is allocated by the producer.
   pyarrow allocates through `AllocateBuffer`, so a 512-row validity bitmap is
   *exactly* 64 bytes with nothing behind it. Any code that assumes slack is
   wrong on imported data no matter what marrow's allocator does. Over-allocating
   would buy a guarantee that holds on roughly half the buffers in a real
   pipeline and would make the remaining violations *harder* to find, because
   they would only reproduce on imported data.

   `Buffer.from_foreign` made this worse in writing: it rounded the producer's
   size up to a 64-byte multiple and justified it with "Arrow's spec guarantees
   all exported buffers are padded to multiples of 64 bytes". The C Data
   Interface guarantees no such thing. That docstring is now corrected.

2. **It diverges from Arrow C++ for no compatibility gain.** Anyone comparing
   the two would find marrow allocating more and have to discover why.

3. **F1 already rejected the local form of this** ("over-allocating by one
   element in `BufferView.filter` … leaves `compressed_store_dense` a loaded
   gun for the next caller"). Guaranteed slack silently blesses out-of-bounds
   writes; that is how the next one gets written.

The memory cost was *not* the deciding factor, and I want to be explicit that
it does not carry the argument: `align_up(bytes, 64) + 64` costs a flat 64
bytes per allocation — 0.0008% of an 8 MB `int64` column, and trivial in
absolute terms even on tiny buffers. **If the FOREIGN objection did not exist,
padding would be affordable.** It exists, so it does not help.

### Binary size

`debug_assert` compiles out unless `-D ASSERT` is set and the gates build
without it, so the expectation was no movement. Measured `__text` against
`benchmarks/binary_size/baseline.json` (threshold 0.5%):

| gate | baseline | measured | delta | pct |
|---|---|---|---|---|
| `query_streaming` | 1,484,652 | 1,444,196 | −40,456 | **−2.72%** |
| `query_join` | 1,507,836 | 1,465,320 | −42,516 | **−2.82%** |
| `query_streaming_agg_fused` | 1,417,476 | 1,388,848 | −28,628 | **−2.02%** |
| `query_streaming_agg` | 1,932,404 | 1,903,988 | −28,416 | **−1.47%** |
| `query_dynvalue` | 4,871,156 | 4,891,764 | +20,608 | +0.42% |

`benchmarks/binary_size/check_gate.py` on the final tree: **`OK: no gate grew
more than 0.5%.`** Four gates shrank and the fifth is inside the threshold. The baseline was reset
at `0e552a7`, and `alpha` has moved since (F1's fix among others), so these
deltas are not attributable to this change alone — the point is that the gate
passes and nothing here grew it.

---

## 3. The assertions, and proof that each fires

`debug_assert` is the right tool: it compiles out in release exactly as Arrow
C++'s `DCHECK` does, so production pays nothing. Adding a bounds check to a
method named `unsafe_*` is not a contradiction — the name promises no *runtime*
check, and a debug-build check is how you learn the caller broke the contract.

### 3.1 `BufferView` — the write surface

| Site | Assertion |
|---|---|
| `unsafe_set` | `_check_bounds(index)` |
| `unsafe_get` | `_check_bounds(index)` |
| `store[W]` | `_check_range(index, W)` — `index + W <= length` |
| `load[W]` | `_check_range(index, W)` |
| `gather[W]` | `reduce_min() >= 0 and reduce_max() < length` |
| `compressed_store[W]` (LLVM) | `popcount(mask) <= length` |
| `compressed_store_sparse` | `popcount(sel_bits) <= length` |
| `compressed_store_dense` | `length > popcount(sel_bits)` and `len(src) >= 64` |
| `filter` | postcondition `out_pos == out_len` |

**`unsafe_set` is the one that matters**, and it is precisely the one F1 could
not get `-D ASSERT=all` to catch:

> `ASSERT=all`. Rebuilt `libmarrow.so` with it; no assertion fires. The overflow
> goes through `unsafe_set` on a `BufferView`, which is unchecked by
> construction.

Both views already *had* a `_check_bounds` helper holding a `debug_assert`; the
unsafe and SIMD paths simply never called it. They do now.

**Proof it fires.** Reverting F1's guard in the adaptive `compressed_store`
(`if cnt <= sparse_threshold or self._length <= cnt` → `if cnt <= sparse_threshold`)
and running `marrow/tests/test_views.mojo`:

```
At: ./marrow/views.mojo:152:21: Assert Error: BufferView index 32 out of bounds for length 32
```

Element 32 of a 32-element destination — the one-past-popcount store, named at
its own site, instead of a SIGSEGV inside `tcmalloc::SLL_Next` several
allocations later.

### 3.2 `BitmapView` — split read and write bounds

The two need different bounds, and conflating them is what produced a false
guarantee in the first place.

- **Writes** (`store_bytes`, `store[W]` whole-byte path, `compressed_store`) get
  `_check_byte_range` — exact: `byte_index + count <= (offset + length + 7) >> 3`.
- **Reads** (`load_bytes`, `load[W]`, `load_bits`) get `_check_byte_read_range` —
  the *allocation* bound, `align_up(byte_extent, 64)`. `load[W]` takes an
  unconditional 4-byte load and `load_bits[T]` a `size_of[T]()`-byte one, so
  they legitimately run into the allocation's padding while every lane the
  caller consumes stays inside the view. The looser bound is honest about that
  and still fires on the case worth catching: a view whose byte extent is
  already a multiple of 64 has no padding, so a 512-bit bitmap trips it.

  This bound is **optimistic for FOREIGN buffers** — see §5.1.

`compressed_store` additionally asserts `_offset == 0` (it is byte-addressed and
ignores `_offset`, which nothing said before), `bit_offset + count <= length`,
and that `bits` has nothing set above `count`. Those three compose into the byte
bound: `byte_idx + nbytes == ceil((bit_offset + count) / 8) <= ceil(length / 8)`.

**Proof it fires.** Reverting the tapered store to the old unconditional 8-byte
one:

```
At: ./marrow/views.mojo:647:21: Assert Error: BitmapView byte range [63, 71) out of bounds for a view spanning 64 bytes
```

### 3.3 Postcondition: produced count vs allocated capacity

`BufferView.filter` and `BitmapView.filter` size their destination from a
pre-counted popcount, then fill it. Both now assert that what they wrote equals
what they allocated. This is the assertion class the brief asked for, and it
is the one that caught a **new** bug — see §4.

### 3.4 Cost

Tests build at `-O1` with `-D ASSERT=all`; benchmarks build at `-O3` without it,
so nothing measured for performance is affected. Measured on the fixed
selection `marrow/tests/test_views.mojo` + `marrow/kernels/tests/test_sort.mojo`
(141 cases):

| | wall clock | of which compile |
|---|---|---|
| write surface only (`unsafe_set`/`store`/byte ranges/filter counts) | 135.51 s | 135 s |
| full surface (+ `load[W]`, `gather`, `compressed_store*`, bitmap reads) | 134.05 s | 134 s |

Execution is ≈0.5 s in both; **compilation dominates completely** and the delta
(−1.1%) is noise. Per-lane assertions in `load`/`store`/`gather` did not make
the suite materially worse, so they were kept at the innermost loop rather than
pushed out to the call boundary.

A second, whole-library measurement agrees: a cold `mojo precompile marrow`
(which builds every module under `marrow/`) is **15.45 s** with the assertions
and **15.27 s** with the three changed files reverted to `alpha`, back to back
under identical load. That one does not compile the assertion *bodies*
(`mojo precompile` rejects `-D`), so the 141-case `-D ASSERT=all` figure above
is the load-bearing number; this one bounds the elaboration cost of the added
source.

### The `test_arrays.mojo` timeout is pre-existing

`marrow/tests/test_arrays.mojo` does **not** fit inside the harness's 1800 s
default deadline, alone or paired with `test_buffers.mojo`, and reports every
case as failed with an empty message when it blows through. I ran the
single-variable A/B rather than assume, because a mass failure at exactly the
deadline is easy to mistake for a real regression:

```
with G1's changes           165 failed in 1800.08s (0:30:00)
alpha, three files reverted 165 failed in 1800.06s (0:30:00)
```

Identical. `git checkout 557fc34 -- marrow/views.mojo marrow/buffers.mojo
marrow/kernels/filter.mojo` was the only variable. `test_buffers.mojo` on its
own is 66 passed in 6.29 s, so the cost is entirely `test_arrays.mojo`. Use
`--mojo-timeout` on that file. Worth raising separately: a core test file that
cannot be run at the default deadline is a hole in the suite, and it is not
this change's to fix.

---

## 4. What the assertions found: `load_bits` truncation

This was not in the brief. The postcondition assertion in `BitmapView.filter`
found it.

**`BitmapView.load_bits[T](index)` returned the top `_offset & 7` bits as
zeros.** It issues one unaligned `size_of[T]()`-byte load at
`(_offset + index) >> 3` and shifts right by the sub-byte offset — so with a
non-zero offset, the run's last `_offset & 7` bits live in the byte *after* the
load, and came back unset.

Every `BitmapView` over a **sliced** array carries such an offset, and both
`BufferView.filter` and `BitmapView.filter` read their selection with
`load_bits[uint64]`. The consequence is a **silent wrong answer**: rows dropped
from a filter over a sliced column, no error, no crash.

**Why no existing test caught it.** All five sliced-filter tests in
`marrow/kernels/tests/test_filter.mojo` use 3–5 element arrays, which fit
entirely in `filter`'s tail block, where the tail mask discards the corrupted
high bits before they matter. The bug needs a slice **longer than 64 elements**
with a sub-byte offset, so the bulk loop runs.

**Proof.** Reverting only this fix and running the new tests:

```
At: ./marrow/views.mojo:1332:21: Assert Error: BitmapView.filter produced 138 bits but its destination was sized for 150
```

12 of 150 rows silently discarded.

**Fix.** Fold in the ninth byte when the offset is sub-byte, and only when that
byte is inside the view:

```mojo
var result = raw >> Scalar[T](bit_off)
if bit_off > 0 and byte_idx + size_of[T]() < self._byte_extent():
    var hi = Scalar[T](self._data[unsafe_offset = byte_idx + size_of[T]()])
    result = result | (hi << Scalar[T](NBITS - bit_off))
return result
```

The common case (`bit_off == 0`, any unsliced array) short-circuits on the first
condition, so the hot loop is unchanged.

Regression tests: `test_bounds_bitmapview_load_bits_with_bit_offset`,
`test_bounds_bitmapview_filter_with_bit_offset`,
`test_bounds_bufferview_filter_with_bit_offset` (`marrow/tests/test_views.mojo`)
and `test_filtersliced_multiword_offset`,
`test_filtersliced_multiword_offset_with_nulls`
(`marrow/kernels/tests/test_filter.mojo`).

---

## 5. `BitmapView.compressed_store` — F1's open item, re-verified and fixed

F1 assessed the up-to-7-byte read-modify-write as "benign for heap integrity
today". **That assessment is correct as far as it goes, and I can now say why
rather than assume it**, but it is a weaker statement than it sounds.

*Why the value is preserved.* The overshoot bytes receive `x | high_bytes(bits
<< bit_off)`. `bits` never has anything set at or above `count` (both callers
pass a `pext` result, which is packed to the LSB), and `bit_off + count <= 64`
in the wide branch, so every byte past `ceil((bit_off + count) / 8)` ORs in
literal zero. `x | 0 == x`. Single-threaded, the memory is bit-identical
afterwards — which is exactly why a guard-byte sentinel test **passes both
before and after** and is not on its own a detector.

*Why it still had to be fixed.* Value preservation is not the whole property.

1. It is an out-of-bounds **write** to the allocator. A 512-bit bitmap is
   exactly 64 bytes; the store touches bytes 63–70 of a 64-byte allocation.
   Value-preserving or not, that is 7 bytes of someone else's memory, and if
   the allocation happens to end a page it is a SIGSEGV rather than a
   scribble.
2. It is a lost-update race the moment two threads filter into adjacent
   allocations — F1's point, and still true.
3. Any sanitizer that ever runs here reports it.

*The fix.* Write exactly `ceildiv(bit_offset % 8 + count, 8)` bytes — 1 to 9 —
instead of always 8 (+1). The wide 8-byte path is retained for `nbytes >= 8`,
which is every full 64-bit block, so the hot loop is untouched; only the final
short block of a filter takes the ≤7-iteration byte-wise branch.

The detector is the assertion, not a sentinel: pre-fix,
`test_bounds_bitmapview_compressed_store_zero_slack` trips
`_check_byte_range` (§3.2). The guard-byte test
(`test_bounds_bitmapview_compressed_store_guard_bytes`) is kept as a
correctness guard on a path that now changes strategy, and its docstring says
so — the same role F1 gave `test_bufferview_filter_last_word_stays_in_bounds`.

### 5.1 Not fixed, reported: the read over-reads on FOREIGN buffers

`BitmapView.load[W]` (4-byte load) and `load_bits[T]` (`size_of[T]()`-byte load)
still over-read up to 3 and 7 bytes past a view's last live byte.
`_check_byte_read_range` bounds them by `align_up(byte_extent, 64)`, which is a
true statement about **marrow-owned** buffers and an optimistic one about
FOREIGN buffers, where the producer's allocation may end at the logical byte.

Tapering these loads would cost a branch **per lane** in the hottest bitmap
loops in the tree, which is the one place the brief said to prefer reporting
over doing. **Recommendation:** either taper them behind a
`comptime`-selectable slow path used when the backing `Allocation` kind is
FOREIGN, or have the C Data importer copy any bitmap whose byte length lands on
a 64-byte boundary. The second is cheaper to reason about and affects a tiny
fraction of imported arrays.

---

## 6. Audit — other computed-count destinations

`unsafe_ptr()` is restricted by `CLAUDE.md` to `buffers.mojo`, `views.mojo`,
`c_data.mojo`, `utils/byteorder.mojo` and the Parquet codec layer. That was the
search space, plus every `alloc_*` whose size is a computed count.

**Finding A — `Buffer.view[T]()` with no explicit length weakens the new
assertions.** `view()` defaults `length` to `self._size // size_of[T]()`, i.e.
the **padded** element count, not the logical one. A two-pass count-then-fill
destination viewed without an explicit length is therefore bounds-checked
against up to 63 bytes more than it owns.

`BufferView.filter` gets this right (`buf.view[Self.T](0, out_len)`), which is
why the assertion bites there — and so, on inspection, do
`marrow/kernels/join.mojo:354` and `marrow/kernels/sort.mojo:572`, which I had
expected to be offenders and are not. Fixed here, in the file with the history:

- `marrow/kernels/filter.mojo` — the string/binary values destination
  (`alloc_uninit(total_bytes)`) and both `child_idx_buf` destinations
  (`alloc_uninit(total)`) now pass their computed length, plus two
  `pos == total` / `dst_byte_pos == total_bytes` postconditions.

Still open, reported rather than changed:

- `marrow/kernels/filter.mojo:241`, `:369`, `:428`, `:826`, `:877` — offsets and
  fixed-width destinations viewed without a length. Same mechanical fix.
- `marrow/kernels/cast.mojo:873`, `:1006` — same shape; that file is owned by an
  unmerged branch and was left alone.
- `marrow/kernels/hashtable.mojo:453` writes its CSR rows through
  `Buffer.unsafe_set` rather than a view. That method had **no bounds check at
  all**, which is the `BufferView.unsafe_set` defect one layer down; it now
  calls `Buffer._check_bounds[T]`, as does `Buffer.unsafe_get`. The bound is
  the padded element count, so it catches gross overruns and not a one-element
  overstep — the reason to pass explicit lengths to `view()` rather than rely
  on it.

**Finding B — `BitmapView.store[W]` is inconsistent about `_offset`.** The
`W % 8 == 0` branch is byte-addressed and ignores `_offset`; the `W < 8` branch
applies it. Nothing documented the split. A `debug_assert(self._offset == 0)`
now pins the whole-byte branch to the offset-0 destinations it is built for,
and no in-tree caller violates it — but the two branches of one method meaning
different things by `bit_index` is a latent trap of exactly the `load_bits`
kind.

**Finding C — `BitmapView.compressed_store`'s `bit_offset` is not logical.** It
is measured from `_data`, not from the view's `_offset`, despite reading like a
logical index. Now asserted.

**Finding D — `mmap_file` is sound but for a different reason.** It rounds the
Buffer's logical size up to 64 and argues that `mmap` rounds the mapping up to
a page and bytes past EOF read as zero. That is correct on every platform
marrow targets, and unlike `from_foreign` it is a statement about memory that
actually exists. Left alone; the docstring already says why.

**Finding E — `marrow/utils/compression.mojo` uses `unsafe_ptr()` and is not on
`CLAUDE.md`'s permitted list.** The list says `marrow/parquet/utils.mojo` "left
the list because it moved to `marrow/utils/compression.mojo`" — the destination
was never added. Documentation drift, not a defect: it is the codec layer the
rule intends to permit.

**Not a finding.** `views._apply_packed_dispatch` deliberately over-writes past
`length` on the GPU arm, clamped to `length + 64 / size_of[In]`. It is safe and
self-consistent, and for a reason that does not involve slack: the launch covers
at most `align_up(length, gpu_width)` bits with `gpu_width` a power of two no
wider than 64, and a `Bitmap` of `length` bits allocates
`align_up(ceildiv(length, 8), 64)` bytes, which always covers
`ceildiv(align_up(length, 64), 8)`. The docstring now says this instead of
citing padding. `views.apply` for bitmaps (`_apply` unary/binary) computes its
tail read bound exactly and was verified in-bounds by hand.

---

## 7. Test coverage

`marrow/tests/test_views.mojo` grew from 69 to 82 cases. The matrix varies:

- **Selection shape** — nothing selected; all selected; lowest lane only;
  highest lane only; a single interior lane; low half (F1's shape); alternating
  with the top lane clear and with it set; 48-of-64; a dense interior run.
  Recording *why* the ends are safe was worth it: an all-ones word writes its
  last lane at index 63 *before* consuming bit 63, and a word with its highest
  bit set has a real home for the trailing store, so neither ever oversteps.
  Only the dense-middle-with-clear-top shape does.
- **Sparse/dense threshold** — 23 / 24 / 25 set bits. The fix changed which path
  runs, so the boundary is now behaviour-critical and is pinned rather than
  implicit.
- **Element width** — int8 / int16 / int32 / int64 / float64.
- **Destination slack** — exactly `popcount` (dangerous) and `popcount + 1`.
- **Multi-word** — `filter` at every width, sized so the output is always a
  64-byte multiple (zero slack).
- **BitmapView** — zero-slack last byte; guard bytes; the 9-byte straddle; every
  bit offset 0–7 × count 1–16 (128 cells, the new tapered path); a ragged
  517-bit length; `filter` producing exactly 512 bits; sub-byte-offset views.

Every new test was verified to **fail** against the pre-fix behaviour by
reverting each of the three fixes in turn (§3.1, §3.2, §4). No test was kept
that passes both before and after, except
`test_bounds_bitmapview_compressed_store_guard_bytes`, which is labelled a
guard rather than a detector for the reason given in §5.

---

## 8. Verification

```
marrow/tests/test_views.mojo                                  82 passed in   8.20s
marrow/tests/test_buffers.mojo                                66 passed in   6.29s
marrow/tests/test_c_data.mojo                                 42 passed in  37.48s
marrow/kernels/tests/test_filter.mojo                         54 passed in  37.50s
marrow/kernels/tests/test_filter.mojo + test_boolean.mojo     65 passed in  42.60s
marrow/tests/test_views.mojo + kernels/tests/test_sort.mojo  141 passed in 134.05s
python/marrow/tests                            491 passed, 55 skipped in 2.36s
                                               (after a 246 s libmarrow.so rebuild)
mojo precompile marrow                                 0 errors, 0 warnings
pixi run binary_size                                   gate green, see §2
```

`test_c_data.mojo` and the Python suite are the ones that matter for §5.1: both
drive **FOREIGN** buffers imported from pyarrow through the bitmap read paths,
and `_check_byte_read_range` does not fire on either. That is evidence the
optimistic bound is not being violated on the covered paths, not a proof that
it cannot be.

`marrow/tests/test_arrays.mojo` is the one file left unverified, and the reason
is the pre-existing 1800 s harness ceiling documented in §3.4 — it does not fit,
on `alpha` either. Its coverage of `BufferView.unsafe_get`/`unsafe_set` is
indirect but real, so it should be run under `--mojo-timeout` before this is
considered fully checked.

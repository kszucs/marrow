# O2 — The `binary` → `string` cast was paying for UTF-8 validation on every query

ClickBench's Parquet `BYTE_ARRAY` columns arrive as `binary`, and marrow's
string kernels are bound on `StringLikeType`. So every string query in
`python/marrow/tests/clickbench.py` goes through the helper

```python
def s(name):
    return col(name).cast(STRING)
```

on 28 of the 105 columns. Profiling put **11.5% of q34** and **3.8% of q21** at
`marrow/kernels/cast.mojo:813` — the body of `BinaryLikeCast._check_utf8`.

## What the cost was

`BinaryLikeCast.apply` relabels with no allocation whenever the source and
target offset widths agree, which `binary` → `string` does. The cast itself is
free. The entire cost sat in the guard above it:

```mojo
comptime if safe and bytes_to_text:
    Self._check_utf8(array)
```

and `_check_utf8` was a scalar loop over every element — `unsafe_get(i)` →
`StringSlice` → `.as_bytes()` → `_is_valid_utf8`, one call per row, on every
query, re-validating an immutable buffer it had validated the previous time.

Measured through the Python API on a rebuilt `libmarrow.so`, that loop ran at
**4.2 GB/s** on pure-ASCII data and **5.5 GB/s** on ClickBench's real `URL`
column. The relabel it guards is ~14 ns, i.e. constant.

## The fix — two whole-buffer fast paths in front of the exact loop

Both are *sufficient* conditions that fall through to the original per-element
loop when they do not hold, so the accept/reject decision is unchanged and only
the cost of reaching it moves.

1. **All-ASCII** (`_all_ascii`). Every byte `< 0x80` makes every element valid
   UTF-8 no matter where the offsets cut the buffer and no matter what a null
   slot holds, because ASCII is a subset of UTF-8 closed under slicing. Four
   SIMD accumulators for throughput, reduced once per 4 KiB so a buffer that
   *fails* the test bails near the first non-ASCII byte instead of after a full
   wasted pass.

2. **Valid window + element starts on character boundaries.** If the whole byte
   window validates *and* no element begins on a continuation byte
   (`0b10xxxxxx`), each element is a whole number of characters and therefore
   validates on its own.

Path 2's window check is itself block-skipping (`_validate_utf8_window`) rather
than one `_is_valid_utf8` call, and that turned out to be the part that matters
for real data — see the next section.

The boundary scan is load-bearing rather than belt-and-braces. **A validator
that only looks at the byte window is strictly weaker than the loop it
replaces**: `"é"` is `0xC3 0xA9`, and split across two adjacent elements the
concatenation is valid UTF-8 while each half on its own is malformed. A
window-only check accepts it; the loop rejects it.
`test_binary_to_string_split_character_raises` pins exactly that case.

The fall-through matters equally. A null slot may hold arbitrary bytes, so a
whole-window check can *fail* on an array the loop accepts; falling back rather
than raising is what stops that becoming a false rejection —
`test_binary_to_string_null_slot_bytes_are_not_validated` pins it.

## The real column is not ASCII, and that decided the design

The obvious fix is the all-ASCII path alone. It is also not enough, and the data
says why:

| column | value bytes | non-ASCII bytes |
|---|---|---|
| `URL` | 88 562 192 | 3 988 061 (**4.5%**) |
| `SearchPhrase` | 3 528 017 | 91.0% |
| `Title` | 138 409 995 | 80.1% |

`URL` — the column q21 and q34 are built on — **never takes the all-ASCII
path**. With only path 1 implemented, q21 and q34 did not move at all.

What rescues it is that those bytes are *clustered*, not spread evenly:

| block size | pure-ASCII blocks (skippable) |
|---|---|
| 16 B | **92.7%** |
| 64 B | 86.2% |
| 128 B | 80.7% |

So `_validate_utf8_window` walks the window a SIMD block at a time, skips the
pure-ASCII blocks outright, and hands only the impure regions to
`_is_valid_utf8`. Safe for the same reason path 1 is: a pure-ASCII block cannot
contain any part of a multi-byte sequence, so no sequence straddles one, and
every region handed over begins and ends on a character boundary.

## Measured

Kernel level, through the Python API, **rebuilding `libmarrow.so` for each
variant** (see the trap below):

| data | pre-fix | post-fix | |
|---|---|---|---|
| synthetic pure-ASCII, 72.7 MB | 17.19 ms (4.2 GB/s) | **0.83 ms** (87.8 GB/s) | **x20.7** |
| ClickBench `URL`, 88.5 MB | 16.01 ms (5.5 GB/s) | **6.14 ms** (14.4 GB/s) | **x2.6** |

End-to-end, `bench_clickbench.py`, the two builds run back to back in the same
session, 43 queries x 3 engines x 2 repeats interleaved:

| | median delta |
|---|---|
| queries with **no** string cast (drift control, n=15) | **+0.4%** |
| queries **with** a string cast (n=27) | **-4.5%** |

The control sitting at +0.4% is what makes the -4.5% attributable rather than
drift. Headline rows:

| query | pre | post | delta |
|---|---|---|---|
| q21 (`URL LIKE '%google%'`) | 346.5 ms | 328.7 ms | **-5.1%** |
| q34 (`GROUP BY URL`) | 106.9 ms | 96.7 ms | **-9.5%** |
| q35 | 109.6 ms | 98.7 ms | -9.9% |
| q28 | 102.9 ms | 84.0 ms | -18.4% |
| q37 | 121.4 ms | 105.2 ms | -13.3% |
| q18 | 47.8 ms | 42.9 ms | -10.3% |

41-query total 4 052 ms → 3 905 ms (-3.6%).

## Two traps this task walked into, both worth recording

**1. `pixi run python script.py` does not rebuild `libmarrow.so`.** Only
`conftest.py` does, and only for a pytest session. Measuring "before" by
`git checkout`-ing the old kernel and re-running a plain Python script therefore
measures the *new* library twice. That produced a confident and completely wrong
"pre-fix 5.95 ms vs post-fix 6.25 ms — no improvement" reading, and nearly got
the change reverted as useless. Any before/after through the Python API must run
`pixi run -e bench build_python` between variants.

**2. The Mojo microbenchmark and the shared library appeared to disagree by
3.5x, and did not.** Once each variant was actually built, `bench_cast.mojo`'s
17.61 ms and the shared library's 17.19 ms for pre-fix pure-ASCII agreed to
within 3%. Worth stating because the intermediate reading looked exactly like
the "benchmark harness lies" failure mode and invited the wrong conclusion; the
harness was fine, the `.so` was stale.

Separately, the first benchmark batch ran against five sibling agents compiling
on the same box: **every untouched control row moved by x2.15 to x7.43** between
that batch and the next. Raw before/after ratios across those two batches are
meaningless. All numbers above come from back-to-back runs on a quiet machine
with controls that stayed within +/-1%.

## A compiler wedge found on the way: `DynArray == DynArray`

Worth its own section because it cost an hour and looks like nothing.

The first version of these tests contained one line —
`assert_true(out == cast(out, string))` — comparing two `DynArray` values. With
it, the `test_cast.mojo` driver **never finished compiling**: `mojo` sat at 0%
CPU with 18 s of CPU time consumed after 57 minutes, twice, with a
`modular-crashpad-handler` alongside it. It does not error and it does not
crash; it wedges.

Everything around it stayed healthy, which is what made it hard to see:

- `mojo precompile marrow` was clean, 0 errors and 0 warnings, in seconds.
- An untouched file, `marrow/tests/test_dtypes.mojo`, compiled in **4 s**.
- The same `test_cast.mojo` at the pristine `alpha` commit compiled in **84 s**.
- The `bench_cast.mojo` driver, `-O3`, built and ran fine throughout.

Bisected by putting each removed construct back one at a time, each probe a
separate ~85 s compile:

| construct | result |
|---|---|
| `BinaryBuilder` instantiated in the test file | 71 passed, 86 s |
| `StringSlice(unsafe_from_utf8=Span(local))` appended to a builder | 71 passed, 85 s |
| **`assert_true(out == cast(out, binary))`** — `DynArray` vs `DynArray` | **wedged, hit the 420 s timeout** |

`DynArray.__eq__` has to resolve the active variant member on *both* sides, so
it elaborates the ladder squared. One occurrence in one test case is enough.

Practical consequence for anyone writing tests here: **compare a `DynArray`
against a typed array, or compare `.dtype()` and elements, but do not compare
two `DynArray`s.** `CLAUDE.md` already recommends `assert_true(result ==
expected)` over element loops — that advice is sound for typed arrays such as
`PrimitiveArray[T]`, and this is the exception to it.

Note also that this is invisible to `precompile`, so "the tree compiles" is not
evidence that a test file will build.

## Correction to the brief: Arrow C++ does not validate the buffer

The task described Arrow C++ as validating "the buffer, not element by element."
That is not what the source does. `Utf8Validator` in
`cpp/src/arrow/compute/kernels/scalar_cast_string.cc` has a
`VisitValue(std::string_view)` called once per element, and `UTF8DataValidator`
in `cpp/src/arrow/array/validate.cc` likewise uses `VisitArraySpanInline` per
element. Arrow validates per element too — its `ValidateUTF8Inline` is simply
cheap enough per call that it never needed a buffer-wide pre-check. marrow's
per-element overhead is what made one necessary here, and the split-character
case is precisely why Arrow's per-element choice is the *safe* one: any
whole-buffer shortcut has to prove it is not weakening anything.

## Lever (a) — `safe=` in the expression API — is blocked by file ownership

Note first that the **array**-level API already has it:
`marrow.compute.cast(arr, target, safe=...)` is what the measurements above
drive. The gap is only on the *expression* side, and threading it through needs
three seams:

- `python/marrow/_expr_column.py::Column.cast` — mine
- `python/bindings/expressions.mojo::_expr_cast` — mine
- `marrow/expr/dynamic.mojo::DynValue.cast` — **not mine**

`DynValue.cast(to: DynType)` stores a single `DynPayload(to.copy())`, and its
evaluator `DynValue._cast` calls `cast_array(args[0], payload[DynType])` with
`safe` left at its default. There is no way to carry the flag from the binding
to the kernel without changing that struct, and `marrow/expr/**` belongs to a
sibling agent on this branch, so it was left alone rather than edited across the
boundary. The change, for whoever owns it: give `DynValue.cast` a
`safe: Bool = True` and widen the payload to carry it.

It is also much less interesting than it was. The point of `safe=False` was to
let a user skip a cost that should not have been there; that cost is now 2.6x
smaller on real data and 20x smaller on ASCII, with `safe=True` still the
default and still rejecting malformed input.

Lever (c) — caching validation per buffer — was not attempted. It would need
mutable state on the array structs, which `CLAUDE.md` forbids, and after the
above the per-query cost is a single mostly-skipped pass.

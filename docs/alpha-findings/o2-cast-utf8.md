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

## What the cost actually was

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

The microbenchmark added in `marrow/kernels/tests/bench_cast.mojo` separates the
two costs by running the same relabel with `safe=True` and `safe=False`. The gap
between them *is* the validation:

| case (1M URL-shaped rows) | baseline median |
|---|---|
| `bench_binary_to_string_safe_1m` (validating) | **37.85 ms** |
| `bench_binary_to_string_unsafe_1m` (pure relabel) | **30.06 ns** |
| `bench_string_to_string_relabel_1m` (control, guard compiled out) | **30.75 ns** |

Note the units differ per row — pytest-benchmark scales each independently. The
relabel is ~30 **nanoseconds**; validation was ~37.85 **milliseconds**. Reading
those two rows in the same unit is the whole finding: validation was **~1.26
million times** the cost of the operation it was guarding.

## The fix — two whole-buffer fast paths in front of the exact loop

Both are *sufficient* conditions that fall through to the original per-element
loop when they do not hold, so the accept/reject decision is unchanged and only
the cost of reaching it moves.

1. **All-ASCII.** Every byte `< 0x80` makes every element valid UTF-8 no matter
   where the offsets cut the buffer and no matter what a null slot holds, because
   ASCII is a subset of UTF-8 that is closed under slicing. `_all_ascii` is a
   four-accumulator SIMD OR-reduction over the values window — memory-bandwidth
   bound, no per-element work at all. This is the path `URL` takes, and it is why
   q21/q34 improve.

2. **Valid window + element starts on character boundaries.** If the whole byte
   window validates *and* no element begins on a continuation byte
   (`0b10xxxxxx`), each element is a whole number of characters and therefore
   validates on its own. This covers the genuinely multi-byte columns
   (`Title`, `SearchPhrase`) that path 1 cannot carry.

The boundary scan in path 2 is not optional, and it is the part worth arguing
for. **A validator that only looks at the byte window is strictly weaker than
the loop it replaces.** `"é"` is `0xC3 0xA9`; split across two adjacent elements
the concatenation is valid UTF-8 while each element on its own is malformed. A
buffer-only check accepts it; the loop rejects it.
`test_binary_to_string_split_character_raises` pins exactly this case.

The fall-through matters as much as the fast paths. A null slot is allowed to
hold arbitrary bytes, so a whole-window check can *fail* on an array the loop
accepts. Falling back rather than raising is what keeps that from becoming a
false rejection —
`test_binary_to_string_null_slot_bytes_are_not_validated` pins it.

## Correction to the brief: Arrow C++ does not validate the buffer

The task described Arrow C++ as validating "the buffer, not element by element."
That is not what the source does. `Utf8Validator` in
`cpp/src/arrow/compute/kernels/scalar_cast_string.cc` has a `VisitValue(
std::string_view)` called once per element, and `UTF8DataValidator` in
`cpp/src/arrow/array/validate.cc` likewise uses `VisitArraySpanInline` per
element. Arrow validates per element too — its `ValidateUTF8Inline` is simply
cheap enough per call that it never needed a buffer-wide pre-check. marrow's
per-element overhead is what made one necessary here, and the split-character
case above is precisely why Arrow's choice is the *safe* one and any
whole-buffer shortcut has to prove it is not weakening anything.

## Lever (a) — `safe=` in the Python API — is blocked by file ownership

Threading `safe=` from `Column.cast` to the kernel needs three seams:

- `python/marrow/_expr_column.py::Column.cast` — mine
- `python/bindings/expressions.mojo::_expr_cast` — mine
- `marrow/expr/dynamic.mojo::DynValue.cast` — **not mine**

`DynValue.cast(to: DynType)` stores a single `DynPayload(to.copy())`, and its
evaluator `DynValue._cast` calls `cast_array(args[0], payload[DynType])` with
`safe` left at its default. There is no way to carry the flag from the binding to
the kernel without changing that struct, and `marrow/expr/**` belongs to a
sibling agent on this branch. It was left alone rather than edited across the
boundary.

It is also now close to unnecessary. The point of `safe=False` was to let a user
who knows their data is UTF-8 skip a cost that should not have been there; with
the fast paths in place that cost is largely gone for both ASCII and valid
multi-byte data, and `safe=True` — which still rejects malformed input — stays
the default for everyone. If the expr owner wants it anyway, the change is to
give `DynValue.cast` a `safe: Bool = True` and widen the payload to carry it.

## Measured — and why the raw ratios are a lie

Five sibling agents were compiling on this box, and the two benchmark batches did
not see the same machine. **Every untouched row moved between them**: the
control cases this change cannot possibly affect came out x2.15 to x7.43 faster
in the second batch, median **x2.39**. Raw before/after ratios are therefore
meaningless here; everything below is divided by that factor, exactly as
`CLAUDE.md`'s second benchmark rule requires.

Controls used (untouched by this change): `int32_to_float64_{10k,100k,1m}`,
`int64_to_int32_unsafe_{10k,100k,1m}`, `dispatch_int64_to_float64_{100k,1m}`,
`timestamp_upscale_{100k,1m}`, `string_to_string_relabel_{100k,1m}`,
`binary_to_string_unsafe_{100k,1m}`.

| case | baseline | post-fix | raw | **normalised** |
|---|---|---|---|---|
| `binary_to_string_safe_1m` | 37 849 us | 824.5 us | x45.9 | **x19.2** |
| `binary_to_string_safe_100k` | 2 734.9 us | 65.5 us | x41.7 | **x17.5** |
| `binary_to_string_safe_10k` | 175.0 us | 6.29 us | x27.8 | **x11.7** |
| `dispatch_binary_to_string_1m` | 55 663 us | 788.8 us | x70.6 | **x29.6** |
| `dispatch_binary_to_string_100k` | 2 723.2 us | 65.0 us | x41.9 | **x17.6** |
| `binary_to_string_utf8_safe_100k` (non-ASCII) | 1 347.3 us | 614.2 us | x2.2 | **x0.92** |

Two things to read off this table.

**The ASCII path is now bandwidth-bound.** 1M rows of ~72 bytes is ~72 MB, and
824.5 us of that is **~87 GB/s**. The baseline, normalised to the same machine,
was ~4.5 GB/s. There is no meaningful headroom left in path 1.

**Path 2 buys nothing on its own — and that is not an argument for deleting
it.** At x0.92 the non-ASCII case is unchanged within the +/-8% drift band: the
whole-window validate plus the boundary scan costs about what the per-element
loop cost, which says the Mojo stdlib `_is_valid_utf8` is throughput-bound
rather than call-overhead-bound, so amortising the call overhead wins nothing
once the bytes stop being ASCII.

Path 2 earns its place a different way. Path 1 is a *probe*: when it fails it
has already read the buffer, and without path 2 a non-ASCII column would pay
that failed pass **on top of** the unchanged per-element loop — a real
regression introduced by an optimisation aimed at a different column. Path 2 is
what absorbs it, and x0.92 is the measurement that says it does. The shape to
keep in mind is: **ASCII columns get ~19x, non-ASCII columns get left alone.**

## What was not done

Lever (c) — caching the validation result per buffer — was not attempted. It
would need mutable state on the array structs, which `CLAUDE.md` forbids, and
after the above the per-query cost on the hot path is a single bandwidth-bound
pass rather than something worth memoising.

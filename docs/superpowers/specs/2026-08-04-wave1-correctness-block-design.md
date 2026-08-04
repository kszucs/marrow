# Wave 1 correctness block — Q7.4, Q7.3, B8

**Date:** 2026-08-04
**Branch:** `backlog-wave1`
**Status:** approved, not started

## Context

Twenty-four backlog items are closed on `backlog-wave1`. The re-evaluation that
produced this spec surfaced one fact that is not on the backlog and that colours
everything else:

> `main` is at 2026-07-09. This branch is **513 commits ahead**, and CI has not
> run since 2026-05-11. Every "green" result on this branch was produced on one
> developer machine and verified by nothing else.

That argues for landing the branch and fixing CI (Wave 2) before adding more.
The decision was taken to **continue closing correctness debt first** and merge
afterwards. This spec covers that block. Wave 2 remains the next thing after it,
and the divergence only grows meanwhile — that is an accepted, explicit cost, not
an oversight.

Scope was chosen deliberately:

- **In:** Q7.4, Q7.3, B8. Note B8 grew during design review: the backlog cards it
  as a one-line `byte_width()` guard, but the defect is the `is_primitive()`
  predicate disagreeing with the `PrimitiveType` trait. Still small, but it
  touches a predicate with six call sites and a documented PyArrow-parity claim.
  The backlog card should be reworded when this lands.
- **Out:** B22 (`Buffer.to_device` → `to_cpu` loses data). Real data loss, but
  GPU-only and off the ClickBench path.
- **Out:** B4 (BIT_PACKED Parquet levels). Blocked on hand-assembling a fixture,
  not on the fix — no reference writer emits BIT_PACKED.
- **Out:** B23 (single-file `test_join.mojo` selection deadlocks the toolchain).
  Filed 2026-08-04; a harness concern, not a library defect.

### Standing constraints

Every item below is gated on these, per `docs/backlog.md` §0 and CLAUDE.md:

- `pixi run -e dev precompile` stays at **0 errors, 0 warnings**.
- Benchmark position must not regress; A/B runs must be **interleaved**, never
  all-after-then-all-before. That trap already produced two phantom regressions
  on the sort work in this session.
- AOT `__text` size gates: `query_streaming` **1,309,024**,
  `query_streaming_agg_fused` **3,748,532**, `query_streaming_agg` **4,122,292**,
  `query_join` **3,780,276** (new baseline, no prior reading captured).
- TDD: watch each test fail before writing the fix.
- `benchmarks/` is **outside** `marrow/`, so `precompile marrow` does not compile
  it. Only `pixi run binary_size` does. A clean precompile is not sufficient
  evidence for a change to any public signature.
- Do not select `marrow/kernels/tests/test_join.mojo` alone — it deadlocks
  (B23). Select the directory.

## Approach

**Measure-gated, one commit and one verification per item**, in the order Q7.4 →
Q7.3 → B8. Q7.4 goes first because it changes a trait surface and so needs its
own size-gate reading, isolated from Q7.3, which moves the aggregate gates.

---

## 1. Q7.4 — scalar-RHS string predicates

### Problem

`StringPredicate.prepare` (`marrow/expr/values.mojo`) materializes both operands
unconditionally:

```mojo
var la = into_array(self.l.execute(batch), n).as_string().copy()
var ra = into_array(self.r.execute(batch), n).as_string().copy()
ctx.append(Self.K.apply(la, ra).to_dyn())
```

For a literal right-hand side this allocates an *n*-row `StringArray` holding the
same string *n* times. `Self.K.apply` is then the array × array overload, whose
per-row `predicate` for LIKE/ILIKE constructs a fresh `LikePattern`
(`marrow/kernels/string.mojo:740`).

Two independent wastes:

| Waste | Affects |
|---|---|
| *n*-row splat of a scalar operand | all six predicates sharing `StringPredicate` |
| per-row pattern compilation | LIKE / ILIKE only |

The six are `StartsWith`, `EndsWith`, `StrContains`, `StrEq`, `StrNe`, and
`Like`/`ILike`. `StrEq` with a literal is what `WHERE url = '...'` compiles to,
which is ClickBench-shaped.

`LikePattern` (`string.mojo:557`) and its scalar-pattern overloads (`:746`,
`:769`) already exist and have **no non-test caller**. FU-4 was marked done
because the machinery landed; the optimization never happened.

### Design

One defaulted trait method and one comptime branch.

```
StringPredicateKernel                       # marrow/kernels/string.mojo:297
    + apply_scalar[L: StringLikeType](
          left: BinaryLikeArray[L], pat: StringSlice
      ) raises -> BoolArray
      default body: loop i -> predicate(left[i], pat); no splat
    LikeKernel / ILikeKernel override:
      compile one LikePattern, then _match_pattern

StringPredicate.prepare                     # marrow/expr/values.mojo:1753
    comptime if Self.R.OutShape == 0:       # RHS is scalar-shaped
        evaluate R once -> String
        ctx.append(Self.K.apply_scalar(la, pat).to_dyn())
    else:
        existing array x array path unchanged
```

`OutShape` already distinguishes scalar from columnar operands, so the branch
resolves at elaboration. No new node, no runtime check, no change to `Datum`.

### Boundaries

The kernel keeps owning comparison semantics; the expr layer keeps owning shape.
Neither learns anything about the other. `apply_scalar` is a peer of `apply`, not
a replacement.

### Testing

Red first. The scalar and array paths must agree on:

- `s LIKE 'foo%'` and `s == 'x'` over the same column;
- a **null literal** RHS — the whole output should be null, and the existing
  array path gets this from `Bitmap.intersect`, so the scalar path must too;
- a null LHS row;
- an empty pattern, and a pattern that is entirely wildcards.

Existing `test_string.mojo` and `marrow/expr/tests` results must be unchanged.

### Risk and fallback

A defaulted trait method can instantiate per conforming kernel — six kernels ×
two string widths. Take a `__text` reading before and after and **report it**.
If it regresses meaningfully, fall back to the narrow LIKE-only fix routed
through the existing `:746`/`:769` overloads, which needs no trait change.

### Done when

Both paths agree on every case above; `query_streaming` and `query_join`
`__text` reported; string benchmarks show the scalar path no slower, and
measurably faster for LIKE.

---

## 2. Q7.3 — count's two grouped implementations

### Problem

`CountAgg`'s docstring (`marrow/kernels/aggregate.mojo:1040-1048`) claims twice
that it is the grouped `count` for numeric columns too — *"one implementation
rather than a fold for numbers and a scan for everything else"*. `CountValid.resolve`
(`marrow/expr/aggregates.mojo:152-170`) contradicts it:

```mojo
if value_dtype.is_numeric():
    job[NumericAgg[CountKernel, V]]()   # AggState fold
else:
    job[CountAgg]()                     # validity-only scan
```

Meanwhile the AOT lane (`values.mojo:1905`) does use `K.Grouped`, which is
`CountAgg`. So the two lanes run different code for the same aggregate, and the
documented invariant is false.

### Why the code cannot settle which should win

`CountAgg.grouped` takes a `DynArray` and calls `values.is_valid(i)` per row —
erased dispatch — but guards it with `has_null`. So:

- **null-free column**: `counts[gid] += 1`, no value load, no validity check.
  Favours `CountAgg`.
- **nullable column**: erased `is_valid` per row, where `AggState` pays a typed
  check. Plausibly favours `AggState`.

Both scatter into a per-group slot, so the difference is not scatter-vs-scan as
the card implies. This is an empirical question.

### Measurement

No grouped `count` benchmark exists — `bench_groupby.mojo` covers `sum`, `mean`,
`min`, `max` only. That is itself a gap, since ClickBench is dense with
`COUNT(*)`.

Add `bench_groupby_count_1m_g100k` and a `_nulls` variant, and drive **both**
implementations from the **same binary** via `GroupBy.aggregate[A]` with `A`
bound to each. One binary, interleaved, no rebuild — this avoids the
all-after-then-all-before trap.

Measure at `g100k`, never `g10` (§0 measurement traps).

### Decision rule — fixed before seeing numbers

| Outcome | Action |
|---|---|
| `CountAgg` wins or ties on **both** shapes | Converge the runtime lane onto it; delete the `NumericAgg[CountKernel]` special case; the invariant becomes true. |
| `AggState` wins clearly on nullable | Keep both, and **rewrite the docstring** to state the split and its reason. Two implementations with an honest reason beat one with a false claim. |
| Mixed, or inside noise | Keep both; document; record the erased `is_valid` as why merging is not free. |

### Testing

Both paths must already agree. If no test pins grouped `count` over a nullable
numeric column against a nullable non-numeric column, add it **before** touching
anything.

### Scope boundary

This decides which of two existing implementations `count` uses. It does not
touch Q2.5's `AggState`-widening work.

### Done when

Benchmarks exist and are committed; the decision rule has been applied to real
numbers; the docstring is true either way.

---

## 3. B8 — bool is not a `PrimitiveType`, but `is_primitive()` says it is

### Problem

`DynType.byte_width()` (`marrow/dtypes.mojo:981`) guards on `is_primitive()`,
which includes `is_bool()`, then dispatches:

```mojo
if not self.is_primitive():
    return 0
...
return variant_dispatch[PrimitiveType, func=f](self._v)
```

`BoolType` does not conform to `PrimitiveType` — correctly, since bool is
bit-packed — so the dispatch **aborts the process**. Latent only because every
current caller tests `dt == bool_` first, and `test_dtypes.mojo:116` skips bool.

Verified against PyArrow 23.0.1 on 2026-08-04:

```
pa.types.is_primitive(pa.bool_())  -> True
pa.bool_().bit_width               -> 1
pa.bool_().byte_width              -> ValueError: Less than one byte
```

So both references call bool primitive. marrow should not — see below.

### The real defect is `is_primitive()`, not `byte_width()`

An earlier draft of this spec proposed special-casing bool inside `byte_width()`
and explicitly rejected excluding bool from `is_primitive()`, on PyArrow-parity
grounds. That was wrong. It weighted an external predicate *name* over marrow's
own internal consistency, and it missed what the call sites are already doing.

**Every caller of `is_primitive()` peels bool off first:**

| Site | Shape |
|---|---|
| `kernels/filter.mojo:85` | `if dt == bool_: … elif dt.is_primitive():` |
| `kernels/filter.mojo:607` | same |
| `kernels/hashing.mojo:317` | same |
| `kernels/sort.mojo:454` | same |
| `c_data.mojo:1001` | bool handled by the bit-packed branch above |
| `ipc.mojo:1941` | `dtype.is_bool() or dtype.is_primitive() or …` |

They must, because `is_primitive()` exists to guard a
`variant_dispatch[PrimitiveType]` and `BoolType` does not conform. So the
predicate already means "conforms to `PrimitiveType`" in practice, and the gap to
Arrow's type-id notion is patched by hand six times.

Two details make this concrete rather than stylistic:

- `c_data.mojo:1001` is `elif dtype.is_primitive(): … dtype.byte_width()`. Bool
  reaching that arm would abort **on the C Data import path**. It is prevented
  only by the ordering of the branch above it.
- `ipc.mojo:1941`'s `is_bool() or is_primitive()` is **redundant today**. The
  author treated them as distinct; the implementation does not.

The docstring's own prose — *"numeric, temporal, interval, decimal"* — already
excludes bool. Only the implementation includes it.

### Design

Make the runtime predicate agree with the comptime trait.

```
DynType.is_primitive()      drop `self.is_bool() or`
DynType.is_fixed_size()     was `return self.is_primitive()`
                            now  `return self.is_bool() or self.is_primitive()`
                            (Arrow does treat boolean as fixed-size — 1 bit)
DynType.byte_width()        unchanged; bool is now non-primitive and returns 0
```

`byte_width()` needs **no special case**: the abort disappears as a consequence
of the predicate telling the truth. Every existing `if dt == bool_: … elif
dt.is_primitive():` site behaves identically, since bool was already handled
before the `elif` was reached.

### Divergence from PyArrow, stated deliberately

Verified 2026-08-04: `pa.types.is_primitive(pa.bool_())` is True, and Arrow C++'s
`is_primitive(Type::type)` (`type_traits.h:1150`) lists `Type::BOOL` as its first
case. **marrow deliberately diverges**, because marrow — unlike either reference
— has a `PrimitiveType` *trait* that generic code dispatches on, and a runtime
predicate that disagrees with it is a trap rather than a convenience. Both
references express "primitive" only as a type-id switch, so they have no such
constraint to satisfy.

This must be recorded in the docstring, replacing the current parity claim. If a
Python-facing PyArrow-compatible predicate is ever wanted, the binding layer can
spell it `is_bool() or is_primitive()` — which is exactly what `ipc.mojo:1941`
already writes.

Rejected alternatives:

- **Special-case bool in `byte_width()`.** Treats the symptom; leaves the
  predicate lying and the six hand-patches in place.
- **Make `BoolType` conform to `PrimitiveType`.** Bool has no byte width;
  forcing one is a lie the bit-packed layout does not support.

### Testing

- `test_dtypes.mojo:16` currently asserts `is_primitive()` is True for bool.
  Flip to `assert_false`, and add a comment naming the divergence so a future
  reader does not "fix" it back.
- Un-skip the bool case at `test_dtypes.mojo:116`; assert `byte_width() == 0`.
- Add `assert_true(DynType(bool_).is_fixed_size())` — this is the property the
  change must *not* break, and nothing currently pins it.
- Pin one caller end-to-end: `filter`/`take` over a `BoolArray` must still take
  the bool arm and return a `BoolArray`.

Red first, and expect the red for `byte_width` to be **an abort, not an
assertion failure**. An abort kills the whole `TestSuite` runner, so it presents
as mass failure rather than one bad case — diagnose from the inner runner
summary, not pytest's file-level rollup.

### Done when

`is_primitive()` is False for bool and `is_fixed_size()` is True for it;
`byte_width()` returns 0 without a special case; the docstring records the
divergence and its reason; all six call sites unchanged and passing.

---

## Verification for the block

Per item: `precompile` clean, targeted tests red-then-green, then the suites the
change can reach. At the end of the block, one full pass:

```
pixi run -e dev precompile                       # 0 errors, 0 warnings
pixi run -e dev pytest marrow/tests marrow/kernels/tests
pixi run -e dev pytest marrow/expr/tests marrow/parquet/tests
pixi run -e dev pytest python/marrow/tests
pixi run binary_size                             # all four gates
```

Baseline to beat: **1,951 tests** passing (523 kernels + 807 core/expr + 621
parquet/python).

Each item gets a `CHANGELOG.md` entry and its backlog card removed or corrected.

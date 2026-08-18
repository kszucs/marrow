# C1 — `binary` group-by aborts the process

**Status:** fixed
**Symptom:** `ABORT: ./marrow/arrays.mojo:2572:23: get: wrong variant type`
**Severity:** process abort in a release build, on a first-class Arrow type.

## Reproduction

```python
import marrow as ma, pyarrow as pa
N = 200_000
k = ma.array(pa.array([f"k{i % 1000}" for i in range(N)], type=pa.binary()))
v = ma.array(pa.array(list(range(N)), type=pa.int64()))
ma.record_batch({"k": k, "v": v}).group_by(["k"]).aggregate([("v", "count")])
```

Four aborts, one per worker thread. `string` at the same size is fine;
`binary` below 60,000 rows is fine.

| rows | key dtype | strategy | before |
|------|-----------|----------|--------|
| 1,000 | binary | serial | OK |
| 50,000 | binary | serial | OK |
| 200,000 | binary, 1k groups | **thread-local** | **ABORT** |
| 200,000 | binary, all distinct | radix | OK |
| 200,000 | string, 1k groups | thread-local | OK |

## Root cause

`marrow/builders.mojo:814` (pre-fix):

```mojo
def extend(mut self, arr: DynArray) raises:
    comptime if Self.T.offset == DType.int32:
        self.extend(arr.as_string())
    else:
        self.extend(arr.as_large_string())
```

`BinaryLikeBuilder[T]`'s erased `extend` reconstructed the *source* array's
concrete type from the **builder's own offset width**. Offset width does not
identify a type: `BinaryType` and `StringType` are both 32-bit-offset
`BinaryLikeType`s. So a `BinaryBuilder` fed a `binary` array asked the variant
for `BinaryLikeArray[StringType]`, which it does not hold —
`DynArray.as_type`'s `debug_assert` is compiled out in release, so
`Variant.__getitem__` aborted the process.

The typed leaf immediately below it (`builders.mojo:820`) was already correct:
`def extend[U: BinaryLikeType](mut self, arr: BinaryLikeArray[U])`. It accepts
*any* binarylike source and only reads `U.offset`. The erased wrapper was
strictly narrower than the leaf it called.

`ListLikeBuilder.extend` (`builders.mojo:1015`) had the identical defect —
`MapType` is also a 32-bit-offset `ListLikeType`, so a `map` source into a
`ListBuilder` aborted the same way.

### Why only the thread-local strategy

Of the three grouping strategies, **only thread-local materializes its unique
keys through a builder**:

- `GROUP_SERIAL` and `GROUP_RADIX` both run `GroupBy._by_partition`, which
  records each new group's first-occurrence *row number* and gathers the key
  columns with a single `take` at the end. No builder, no erased `extend`.
- `GROUP_THREAD_LOCAL` runs `HashGrouper` per worker, and
  `HashGrouper._register_new_groups` (`groupby.mojo:154`) appends new key rows
  into a `DynBuilder` per key column.

The strategy is chosen from row count and a sampled cardinality estimate
(`_PARALLEL_MIN_ROWS = 60_000`, `_PARALLEL_ALWAYS_ROWS = 200_000`), which is
why the abort looked like a size threshold rather than a type bug.

### Blast radius beyond group-by

The defect was never group-by-specific — group-by is just where someone
happened to hit it. Anything routed through `DynBuilder.extend` shared it:

- **`concat()`** (`kernels/concat.mojo`) is *entirely* `DynBuilder.extend`.
  Concatenating `binary` arrays aborted. `test_concat.mojo` had 16 cases and
  not one mentioned `binary`.
- **`ChunkedArray.combine_chunks()`** (`arrays.mojo:2294`) delegates to
  `concat`, so a `binary` column in a multi-chunk `Table` aborted on combine.
- **The expression layer's streaming group-by** (`expr/execution.mojo:732`)
  holds a `HashGrouper`, so it aborted on the same input.

`cast.mojo`'s `DynBuilder` use is `append_null()` only, so casting is
unaffected.

## Second defect, found while auditing

`marrow/kernels/numeric.mojo:602` (pre-fix):

```mojo
if left.dtype().is_string() or left.dtype().is_large_string():
    return StringEqKernel.dispatch(left, right).as_bool().copy()
else:
    return EqKernel.dispatch(left, right, ctx).as_bool().copy()
```

`equal_any` is the "equality over an arbitrary dtype" primitive that hash-join
row verification and `nullif` are built on. Its family test was `stringlike`,
so `binary` fell into the **numeric** arm and `dispatch_primitive` raised
`"dispatch_primitive: dtype is not primitive"`. Joining on a `binary` key was
impossible while the identical join on `string` worked:

```
OK    join int64        RAISE join binary
OK    join string       RAISE join large_binary
OK    join large_string
```

This one *raised* rather than aborting, so it was not the alpha blocker — but
it is the same mistake: a family predicate one notch too narrow for the
operation being selected. What decides the kernel here is whether the payload
is variable-width, and `binary` is as variable-width as `string`.

The fix does **not** widen `StringPredicateKernel` to `BinaryLikeType`. That
family's `is_string_like()` guards are deliberate — `LIKE`, `upper` and
`startswith` are text operations, and widening the trait would make
`upper(binary_array)` type-check. Instead `equal_any` dispatches
`binarylike` into a local `_bytes_equal` leaf bound on `BinaryLikeType`,
mirroring `StringPredicateKernel.apply`'s null semantics.

`equal_any` also gained an explicit **same-dtype guard**, which the old code
got for free from the kernel it delegated to. The new arm resolves its comptime
`T` from the *left* dtype and then reads the *right* operand at that same `T`,
so without the guard a mismatched pair would have been one more wrong
`as_type` — the very failure mode being fixed. Worth recording as its own
finding: **the "resolve a type from one operand, apply it to another" shape is
itself the hazard**, and it appears wherever a `dispatch_*` arm downcasts more
than the value it dispatched on. `test_equal_any_mismatched_dtypes_raise`
pins it.

## Kernels audited and found clean

`filter` (`filter.mojo:89`), `take` (`filter.mojo:610`), `sort`
(`sort.mojo:434`) and `rapidhash` (`hashing.mojo:203`) all dispatch
`binarylike` correctly. `count_distinct` and `is_in` hold no `DynBuilder` and
no `HashGrouper`. Parquet's `as_string()` call sites (`codecs.mojo:715`,
`statistics.mojo:246`, `writer.mojo:351`, `writer.mojo:417`) are all guarded by
exact `is_string()` / `is_binary()` ladders. Verified empirically at 200,000
rows for both `binary` and `large_binary`.

---

# Findings: what let this compile

## 1. The offset width was used as a proxy for the type

Both defective sites had the shape *"derive the concrete source type from a
property of the destination."* That is a category error, and it is invisible
at the call site because the wrong answer is still a well-formed type.

The general rule already in CLAUDE.md — **"dispatch on the widest family the
typed leaf accepts"** — would have prevented all three bugs. The leaf accepted
`BinaryLikeType`; the wrapper named a single `StringType`. The rule is stated
for *kernels*; nothing applied it to *builders*, and builders are where two of
the three defects lived. The rule is really about erasure boundaries in
general, not kernels specifically.

### The same shape survives in two constructors (latent, not live)

`ListLikeBuilder.__init__` (`builders.mojo:973`) and
`ListLikeArray.from_arrays` (`arrays.mojo:1313`) both synthesize their dtype
with `comptime if Self.T.offset == DType.int32: list_(...) else:
large_list_(...)`. Instantiated at `MapType` — also a 32-bit-offset
`ListLikeType` — they would label a map as `list<…>`. Neither is reachable
today: `MapBuilder` wraps an inner `ListBuilder` rather than being
`ListLikeBuilder[MapType]`, and `MapArray.from_arrays` is implemented as
`ListArray.from_arrays(...).to_map(...)`. Left alone deliberately — they are
traps, not defects — but they are the same reasoning error and the next
person to instantiate either at `MapType` will find it.

## 2. The obvious fix does not compile — `{mut self}` dispatch closures
##    miscompile in the builders

The natural fix is the one CLAUDE.md prescribes everywhere else:

```mojo
def extend(mut self, arr: DynArray) raises:
    def leaf[U: BinaryLikeType](d: U) raises {mut self, imm}:
        self.extend(arr.as_binary_like[U]())
    arr.dtype().dispatch_binarylike(leaf)
```

**It does not work in a builder.** The `ListLikeBuilder` version fails the
compiler backend outright:

```
marrow/builders.mojo:1035: error: 'kgen.call' op callee argument #1 expected
type '!kgen.pointer<struct<(pointer<none>) memoryOnly>>' but operation
argument has type '!kgen.pointer<struct<(pointer<none>, scalar<bool>) memoryOnly>>'
mojo: error: failed to run the pass manager
```

The generated thunk is typed for a bare pointer while the capture is a
pointer-plus-bool — `ListLikeBuilder`'s child is a `DynBuilder`. The
`BinaryLikeBuilder` version *elaborates and codegens*, then produces a binary
that **crashes on startup** — every case in the file, including ones that
touch none of this. Both are now explicit `if dt.is_…()` ladders.

Two things about this are worth more than the bug itself:

- **`mojo precompile` reports the tree clean in both cases.** It elaborates;
  it does not run the pass manager. CLAUDE.md already says "building is not
  passing" — this is the sharper version: *precompiling is not building*. A
  change to a widely-instantiated generic needs a real test-driver build
  before it can be called compiling, and the 0-error/0-warning precompile
  gate cannot substitute.
- **The failure mode of the second one is a startup crash with no attribution.**
  70/70 cases in `test_builders.mojo` failed with `execution crashed`,
  including `test_bool_builder_zero_length`, which predates every line I
  touched. Nothing in that output points at `BinaryLikeBuilder.extend`. The
  only way to localize it was to revert one closure at a time.

So the ladders in `DynBuilder.__init__`, the parquet encoders and now these two
`extend`s are not stylistic hold-outs from before `dispatch_*` existed — at
least in the builders, they are the only form that compiles. Worth stating in
CLAUDE.md next to the existing closure guidance, which currently reads as
though the dispatch form is always available.

## 3. `as_type` aborts instead of raising, and the guard is compiled out

```mojo
def as_type[T: Array](ref self) -> ref[self._v[T]] T:
    debug_assert(self._v.isa[T](), "as_type: wrong type, holds ", self.dtype())
    return self._v[T]
```

In a release build the `debug_assert` vanishes and `Variant.__getitem__` calls
`abort()`. The consequences:

- A library bug becomes a **process kill** in a user's application. There is no
  exception to catch, no stack, no chance for the caller to fall back.
- Inside `sync_parallelize` it aborts from a worker thread, so the message
  prints N times and the traceback names nothing useful.
- The message says `get: wrong variant type` at `arrays.mojo:2572` — a line
  that is on the *generic* accessor, identical for all ~40 array types. It
  names neither the type held nor the type requested (the `debug_assert` that
  would have is exactly the thing compiled out).

**Recommendation — not applied here, flagging for a decision.** I did not
change `as_type`'s signature: `arrays.mojo` is layout-sensitive, the accessor
is on a very hot path (`.as_int32()` in every kernel inner loop), and making it
`raises` would ripple through every `ref`-returning accessor and every caller.
Two cheaper options, in order of preference:

1. **Make the abort message self-describing.** `Variant.__getitem__`'s abort is
   upstream, but marrow could `abort()` itself with the held and requested
   dtype before indexing, under a *non*-debug check on the erased-box accessors
   only. Cost is one `isa` branch on a path that is already a variant load. This
   turns a 4x-repeated `wrong variant type` into `as_type: holds binary,
   requested string`, which is the entire diagnosis.
2. **Add a raising `try_as_type`** for the handful of call sites that resolve a
   *runtime* dtype, and leave `as_type` as the unchecked fast path for code
   that has already proven the type via `dispatch_*`. The bug class here is
   exactly "code that thinks it proved the type and didn't".

I have added a comment at `as_type` recording that the assert is compiled out
in release and that a wrong downcast is therefore fatal.

## 4. The type lattice permits the mistake silently

`StringType`/`LargeStringType` conform to `StringLikeType`, a sub-trait of
`BinaryLikeType`; `BinaryType`/`LargeBinaryType` conform to `BinaryLikeType`
only. That is the correct lattice. But `as_string()` and `as_binary_like[T]()`
are both total functions on `DynArray` — the narrower one is not rejected for a
binarylike-bound builder, because the builder's `T` never enters the accessor's
type. The mismatch is only detectable at run time, on the value.

There is no cheap fix in the current design: the erased box holds a runtime
type by construction, so *any* downcast is a runtime check. What is available
is to make the unchecked downcasts rarer — every `as_string()` outside a
`dispatch_stringlike` arm is a place where a human asserted a type the compiler
did not verify. There are currently 4 such sites in `marrow/expr/values.mojo`
and 8 in `marrow/parquet/`; the parquet ones are all guarded by an exact
`is_string()` ladder, which is the safe form of the same pattern.

## 5. Test coverage tracked the *type system*, not the *type space*

`BinaryBuilder` and `LargeBinaryBuilder` appeared in **zero** tests before this
change. `binary` appeared in zero `test_concat.mojo` cases. The suite tested
`BinaryLikeBuilder` thoroughly — via `StringBuilder`, the alias that happened to
work. Because `string` and `binary` are the *same generic struct*, coverage of
one reads as coverage of the other, and a parameter-specific bug hides in the
gap.

Where a family has N instantiations and the code branches on the parameter, the
test needs to be parameterized over the family too. The tests added here are
written that way (`_check_bytes_keys[T: BinaryLikeType]`,
`_assert_bytes_concat[T]`, `_extend_roundtrip[T]`), instantiated for all four
of `binary` / `large_binary` / `string` / `large_string`.

## 6. Strategy selection hid a type bug behind a size threshold

The three grouping strategies are chosen by a heuristic on row count and
sampled cardinality, and they do not agree on *how* keys are materialized —
two gather with `take`, one builds. That difference is an implementation
detail with no test asserting the three agree on anything but numeric keys.

`GroupBy` already exposes a `strategy` override for exactly this reason, and
`test_groupby_parallel_matches_serial` uses it — for `int32` only. Extending
that one test across dtype families would have caught this at 3,000 rows
instead of 200,000. The new tests do that.

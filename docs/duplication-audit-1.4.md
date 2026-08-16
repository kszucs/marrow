# §1.4 — `arrays.mojo` slice/`__eq__` duplication: solution proposal

**Status: proposal, nothing implemented.** Companion to `docs/duplication-audit.md`
§1.4, which reported the duplication. This document is the outcome of working
through it: most of what §1.4 proposed turns out to be unreachable, and what is
reachable is a different and better change than the one filed.

Verified against the working tree at `5435f59` + uncommitted changes
(2026-08-16).

---

## 1. What §1.4 claimed, and what survives

§1.4 filed two items: `slice()` written 7 times and `__eq__` written 5 times,
with the fix being "add `unsafe_get` to the `Array` trait so a generic helper
can express the loop."

**That fix does not work, and neither does any variant of it.** Working through
the alternatives (§2) established that `slice` and `validity()` cannot be
factored at all in Mojo today, and that `__eq__`'s element loops are genuinely
different rather than accidentally so. What *is* reachable is a rewrite of
`_validity_equal` that removes real work rather than lines, plus two latent bugs
that the duplication was hiding.

Net effect of the proposal: **−30 lines of redundant prose, two O(n) scans and
one bit-by-bit loop removed from every `__eq__` with nulls, and two
`DictionaryArray` correctness fixes.** The seven `slice` bodies and seven
`validity()` bodies stay.

---

## 2. Ruled out, with reasons

Recorded so none of these is re-litigated from the original card.

### 2.1 A trait-level default `slice` via `ArrayData`

```mojo
trait Array:
    def slice(self, offset: Int = 0, length: Int = -1) raises -> Self:
        return Self(self.to_data().slice(offset, length))
```

This is *expressible* — `__init__(out self, data: ArrayData) raises`
(`arrays.mojo:140`) and `to_data()` (`:164`) are both already trait
requirements, so the default can construct `Self` through the witness. It would
replace 7 of 9 bodies with one.

**Ruled out because it makes every typed `slice()` raising.** `to_data()` and
`__init__(data)` both raise. The trait's own docstring (`arrays.mojo:167`)
states the property being given up: *"Every typed implementation is non-raising
and stays that way — Mojo accepts a non-raising body against a raising
requirement, so typed call sites are unaffected. Only generic code holding
`[T: Array]` has to propagate."* Decision (2026-08-16): keep `slice`
non-raising; if the implementations cannot be unified without giving that up,
they should not be unified.

Two secondary costs, recorded in case the decision is ever revisited:

- `ArrayData` has no `slice` — it would need one (~10 lines, once). That is
  where `_sliced_null_count`'s six positional arguments would stop being a
  footgun, since `ArrayData` holds every field it needs.
- The round-trip is not O(1): `to_data()` builds a `List[Buffer]` and, for
  nested types, a recursive `List[ArrayData]`, and `ArrayData.__init__`
  validates. **This is not disqualifying** — `Array.slice` is a per-morsel /
  per-column operation, not per-element. The 12 `.slice(` hits in `filter.mojo`
  are `BufferView.slice`; the genuine `Array.slice` callers are
  `filter.mojo:545`, `:1048`, `groupby.mojo:532`, `:543`, `:726`,
  `tabular.mojo:133`, `expr/execution.mojo:193`, `:489`, `:598` — all once per
  batch or per column.

### 2.2 A generic `_header_equal[A: Array](a, b)`

Ruled out by the same class of limit. `validity()` returns
`Optional[BitmapView[origin_of(self.bitmap._value)]]` — the return type **names
a field**, so it cannot be a trait requirement, and without it a generic over
`A: Array` cannot reach the validity view.

This also means `validity()`'s seven byte-identical copies (`arrays.mojo:567`,
`816`, `986`, `1136`, `1455`, `1699`, `1856`) are unfactorable — an eighth
duplication §1.4 did not count.

### 2.3 An embedded `ArrayHeader` field

Six of the seven arrays carry exactly `length`, `nulls`, `offset`, `bitmap`
under the same names. Collapsing them into one embedded struct with a
`sliced()` method would make every `slice` a two-liner.

**Forbidden.** `docs/backlog.md` §0, *Do not change*: "Array, scalar and builder
layout. Adding methods and accessors is fine; adding, removing, reordering or
re-typing fields is out of scope, not deferred." It would also touch every
construction site in the tree.

### 2.4 Moving `_validity_equal` to `buffers.mojo`

Considered because it looks like a third member of the
`Bitmap.intersect_views` / `Bitmap.intersect` family (`buffers.mojo:1042`,
`:1074`): same two-optional-operands shape, same `None`-means-all-valid
identity, same four-branch skeleton.

**Ruled out by §3.1.** The rewritten form needs *null counts*, which are
array-level facts, not bitmap-level ones. It stays in `arrays.mojo`.

### 2.5 Unifying the `__eq__` element loops

Not attempted. They are genuinely different, not accidentally so:
`PrimitiveArray` uses `unsafe_get(Int)`, `BinaryLikeArray` `unsafe_get(UInt)`,
`ListLikeArray`/`FixedSizeListArray` a raising `unsafe_get` inside try/except,
`FixedSizeBinaryArray` a byte-wise loop, and `StructArray` has no per-element
loop at all (it compares children). Each is 4–6 lines. Unifying would mean
normalising three index types and the raising-ness of `unsafe_get` across every
kernel that calls it.

---

## 3. The proposal

Five items. Item 4 is a **precondition** of item 1 — see §4.

### 3.1 Rewrite `_validity_equal` around the eagerly-maintained null count

`null_count()` is a **stored field** on every array — `self.nulls` /
`self._nulls`, never computed (`arrays.mojo:548, 831, 983, 1197, 1511, 1768,
1883, 2174`). That is the deliberate A2/B12 choice: `_sliced_null_count`'s
docstring records that marrow rejects Arrow C++'s `kUnknownNullCount = -1` and
polars' `RelaxedCell<u64>` unknown-sentinel precisely so the count is always
known in O(1).

The current implementation does not use it, and pays twice for that.

**Current** (`arrays.mojo:357`):

```mojo
def _validity_equal[
    ao: Origin[mut=False], bo: Origin[mut=False]
](
    length: Int,
    a: Optional[BitmapView[ao]],
    b: Optional[BitmapView[bo]],
) -> Bool:
    if not a and not b:
        return True
    if not a:
        return b.value().count_set_bits() == length
    if not b:
        return a.value().count_set_bits() == length
    ref av = a.value()
    ref bv = b.value()
    for i in range(length):
        if av.test(i) != bv.test(i):
            return False
    return True
```

Two defects, neither visible from any single call site:

1. **The one-sided branches recompute what the caller already knows.** Every
   caller checks `self.null_count() != other.null_count()` immediately before
   calling. If `a` is absent, self has no bitmap, so `self.nulls == 0`, so
   `other.nulls == 0` by that check — and `b.count_set_bits() == length` is
   already established true. An O(n/64) popcount answering a question settled in
   O(1) two lines earlier, on both branches.

2. **The both-present branch is a bit-by-bit loop.**
   `BitmapView.__eq__` (`views.mojo:1002`) does the same comparison with
   word-level XOR, and `Bitmap.__eq__`'s docstring (`buffers.mojo:1117`) records
   the measurement: *"a bit-by-bit loop here measured ~64x slower for the same
   answer."* That conclusion, written down in `buffers.mojo`, never reached
   `arrays.mojo`.

**Proposed:**

```mojo
def _validity_equal[
    ao: Origin[mut=False], bo: Origin[mut=False]
](
    nulls_a: Int,
    a: Optional[BitmapView[ao]],
    nulls_b: Int,
    b: Optional[BitmapView[bo]],
) -> Bool:
    """Do two arrays mark the same *positions* null?

    Equality is a question about null positions, not about how the validity is
    stored (B26). Because `null_count()` is an eagerly maintained field rather
    than a lazily resolved sentinel — see `_sliced_null_count` — "both are
    all-valid" is an integer comparison, whatever bitmap either one happens to
    carry.
    """
    if nulls_a != nulls_b:
        return False
    if nulls_a == 0:
        return True
    if not a or not b:
        return False  # a nonzero null count with no bitmap is malformed
    return a.value() == b.value()
```

- Both popcounts gone, replaced by an `Int` comparison.
- The bit-by-bit loop gone, replaced by `BitmapView.__eq__`'s word-level XOR.
- The `length` parameter is no longer needed — `BitmapView.__eq__` checks its
  own lengths, and every caller has already established the arrays are the same
  length.
- It now *states* B26 rather than implementing it: "a missing bitmap means
  all-valid, which is a value, not a distinguishing representation" is literally
  `if nulls_a == 0: return True`.

Safety of `a.value() == b.value()`: both views come from
`bitmap.value().view(self.offset, self.length)`, so both are offset-applied and
of the array's own length; the callers have already established those lengths
are equal. Bit *i* of each view is position *i*'s validity, so equal bit
patterns is exactly equal null positions.

The `if not a or not b` guard is not dead: `NullArray.null_count()` returns
`self.length` with no bitmap (`arrays.mojo:432`), so a nonzero count without a
bitmap is a legal state somewhere in the type set, and `ArrayData.__init__` does
not enforce the implication. `NullArray` is not a caller today, but the
primitive should stay total.

### 3.2 Update the six callers

Sites: `arrays.mojo:868` (`PrimitiveArray`), `:1082` (`BinaryLikeArray`),
`:1334` (`ListLikeArray`), `:1611` (`FixedSizeListArray`), `:1798`
(`FixedSizeBinaryArray`), `:2008` (`StructArray`).

Each drops its now-redundant `null_count()` check, which the new
`_validity_equal` subsumes. The shared header goes from three checks to two.

**Before** (`PrimitiveArray`, lines 872–880):

```mojo
        if self.length != other.length:
            return False
        if self.null_count() != other.null_count():
            return False
        # Positions, not representation — see `_validity_equal` (B26).
        if not _validity_equal(self.length, self.validity(), other.validity()):
            return False
```

**After:**

```mojo
        if self.length != other.length:
            return False
        # Positions, not representation — see `_validity_equal` (B26).
        if not _validity_equal(
            self.null_count(),
            self.validity(),
            other.null_count(),
            other.validity(),
        ):
            return False
```

The four sites that also compare `dtype` (`:1334`, `:1611`, `:2008`, and
`:1798`'s `byte_width`) keep that check unchanged — it is per-type and does not
belong in the shared primitive.

### 3.3 Fold `BoolArray.__eq__` in as a seventh caller

`BoolArray` (`arrays.mojo:616`) hand-rolls the same semantics differently — a
pairwise `is_valid(i)` loop with no `_validity_equal` call, and so no B26
reference. It is one of the two divergences in the current set.

**Before:**

```mojo
        if (
            self.length != other.length
            or self.null_count() != other.null_count()
        ):
            return False
        for i in range(self.length):
            var lv = self.is_valid(i)
            var rv = other.is_valid(i)
            if lv != rv:
                return False
            if lv and self[i] != other[i]:
                return False
        return True
```

**After:**

```mojo
        if self.length != other.length:
            return False
        # Positions, not representation — see `_validity_equal` (B26).
        if not _validity_equal(
            self.null_count(),
            self.validity(),
            other.null_count(),
            other.validity(),
        ):
            return False
        for i in range(self.length):
            if self.is_valid(i) and self[i] != other[i]:
                return False
        return True
```

`BoolArray.validity()` (`:567`) and `__getitem__` (`:556`) are both non-raising,
so this stays non-raising.

### 3.4 Delete the repeated slice comment — **DONE (2026-08-16)**

The three-line comment repeated above all seven `_sliced_null_count` calls was
deleted; `_sliced_null_count`'s own body comment already said it, and better.
−21 lines, no behaviour change. It is what made the `slice` bodies read as 13
lines instead of ~10 — the copies were most of the apparent duplication §1.4
measured.

### 3.5 Fix the two `DictionaryArray` divergences

Both are latent bugs the duplication hid — nine hand-written implementations,
and the one that opted out of the shared shape is the one that got it wrong.

**(a) `slice` does not recount nulls** (`arrays.mojo:2224`):

```mojo
    def slice(self, offset: Int, length: Int) -> Self:
        return Self(
            dtype=self._dtype.copy(),
            length=length,
            nulls=self._nulls,          # <-- parent's count, for a sub-range
            offset=self._offset + offset,
            ...
        )
```

Every sibling calls `_sliced_null_count`. This violates the eager-recount
invariant (A2/B12) that §3.1 now *depends on*.

**Open question — this fix is not free.** `DictionaryArray` has no `bitmap`
field; its validity lives in `_indices`
(`is_valid(i) → self._indices[].is_valid(self._offset + i)`). Counting a
sub-range means reaching into the indices array, and `DynArray.slice`
(`arrays.mojo:2533`) *raises*. Three options, none obviously best:

1. Make `DictionaryArray.slice` raising. It is the only array whose `slice`
   signature already differs from its siblings, and the trait requirement
   permits raising — but it gives up the same property §2.1 was rejected for.
2. Reach the indices' validity without `DynArray.slice` — e.g. a non-raising
   `null_count_in_range(offset, length)` accessor.
3. Recount at construction rather than at slice.

**(b) `__eq__` ignores `_offset` and `_nulls`** (`arrays.mojo:2256`):

```mojo
    def __eq__(self, other: Self) -> Bool:
        return (
            self._dtype == other._dtype
            and self._length == other._length
            and self._indices[] == other._indices[]
            and self._values[] == other._values[]
        )
```

Two dictionaries differing only in `_offset` compare equal.

**Open question — what the correct semantics are.** Comparing `_indices` and
`_values` structurally is *representation* equality, which is the same mistake
B26 fixed one level up: two dictionaries whose `_values` are permuted encode the
same logical column and compare unequal. Per CLAUDE.md's process rule, settle
this against Arrow C++ (`../arrow/cpp`) and arrow-rs before choosing, rather
than picking the minimal `_offset` patch.

The same shape appears one level further down in
`DictionaryScalar.__eq__` (`scalars.mojo:544`), which compares `_index` as well
as `_decoded`. **Out of scope here** — file with (b) as one scalar/dictionary
equality item rather than folding it into an `arrays.mojo` change.

---

## 4. Ordering

**3.5(a) is a precondition of 3.1.** The rewritten `_validity_equal` trusts
`nulls` to be accurate; `DictionaryArray.slice` is currently the one place that
invariant does not hold. Landing 3.1 first would make `__eq__`'s correctness
depend on something not yet true.

`DictionaryArray.__eq__` does not call `_validity_equal` today, so the coupling
is indirect — but folding it into the shared shape is the obvious follow-up, and
the ordering should assume it.

Suggested sequence:

1. ~~**3.4**~~ — done 2026-08-16.
2. **3.5(a)** — resolve the raising question, restore the invariant.
3. **3.1 + 3.2 + 3.3** — one commit; the callers must move with the signature.
4. **3.5(b)** — after settling semantics against the reference implementations.

---

## 5. Explicitly not done

Both move to the audit's §1.11 ("structurally forced"), alongside the four
`_dispatch` narrowing adapters:

- **`slice`'s seven `Self(...)` bodies.** After 3.4 each is ~8 lines and the
  only shared content is the four-value header. Mojo cannot express partial
  construction, trait field requirements, or (per §2.3) an embedded header
  field.
- **`validity()`'s seven byte-identical bodies.** Return type names a field, so
  it cannot be a trait requirement or reached generically (§2.2).

Neither is a design difference. Both are language limits, and they should be
recorded as such so the audit stops listing them as opportunities.

---

## 6. Testing

`marrow/tests/test_arrays.mojo` — one selection, one compilation unit:

```bash
pixi run -e dev pytest marrow/tests/test_arrays.mojo
```

New cases required, since nothing currently exercises any of these paths:

| Case | Covers |
|---|---|
| Equal null counts, different null positions → unequal | 3.1's `nulls_a == nulls_b` path must still compare bitmaps |
| All-valid *with* a bitmap vs all-valid *without* → equal | 3.1's `nulls_a == 0` early return; the B26 property |
| Two slices at different offsets with the same logical validity → equal | offset-applied views |
| `BoolArray` with nulls, equal and unequal | 3.3 |
| Sliced `DictionaryArray` reports the sub-range's null count | 3.5(a) |
| Dictionaries differing only in `_offset` → unequal | 3.5(b) |

Not needed: a benchmark for 3.1. The ~64x figure is already recorded in
`Bitmap.__eq__`'s docstring from a prior measurement, and the popcount removal
is a strict reduction in work — there is no path where the new form does more.
A null-heavy `__eq__` bench would be nice-to-have, not a gate.

**Size gate:** not expected to apply. `_validity_equal` is a leaf predicate, not
a wrapper around erased dispatch, and the proposal removes code rather than
adding a generic layer — the +115,600-byte `_arith[K]` trap in §0 does not have
this shape. Confirm with `pixi run binary_size` only if 3.5(a) resolves toward
option 2 (a new accessor on `DictionaryArray`).

---

## 7. Open questions

1. **3.5(a):** which of the three options for recounting `DictionaryArray`'s
   nulls on slice — accept `raises`, add a non-raising range accessor, or
   recount at construction?
2. **3.5(b):** what is correct dictionary equality? Settle against Arrow C++ and
   arrow-rs first.
3. **Scope:** should 3.5 land with this work, or be filed as a `backlog.md` §1
   (Wave 1 — Correctness) item on its own? It is bug-fixing, not deduplication —
   but 3.5(a) is a precondition of 3.1, so it cannot be deferred indefinitely.

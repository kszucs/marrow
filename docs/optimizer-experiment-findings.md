# Findings from the optimizer prototype experiment — 2026-08-30

Six findings — four defects, one compiler limit, one open question — surfaced
while three independent optimizer prototypes were built
against `bb1a305f`. **None is caused by the optimizer work**; each was found by
it and reproduced with the prototype code fully dead-code-eliminated. They are
recorded here rather than fixed, so the optimizer work is not held up behind
them.

`docs/backlog.md` is nominally the only place that says what is open. This file
exists because it was asked for separately; fold these entries into the backlog
if that convention should hold.

Provenance: three prototypes — a node-owned protocol (arm A), an RTTI rewriter
(arm B), and a demand lattice (arm C). "Confirmed independently" below means
verified by reading the tree and by running the reference implementation, not
taken from a prototype's report.

---

## 1. `StructArray.slice()` does not slice its children — `.limit(n)` then any expression evaluation breaks

> **FIXED in `a27167aa` (2026-08-31)**, independently of this experiment and
> found the same way — by building the Python query surface. `field()` now
> pushes the parent's offset and length down to the child. The entry is kept
> because the *comptime* half of it was worse than recorded below: that lane did
> not raise, it read elements `[0, len)` of the child and so took the **wrong
> window** whenever the struct's offset was non-zero — a silent wrong answer,
> right by coincidence only at offset 0, which is why every existing test
> passed. `kernels/sort.mojo` had the same exposure and there was no test for
> `field()` on a sliced struct at all.

**Severity: wrong answers / raised errors on a plan shape the engine supports.**
Confirmed independently, and by two prototypes with separate reproducers.

`slice()` (`marrow/arrays.mojo:1971`) records the new `offset` and `length` on
the struct but passes `children=self.children.copy()` **unsliced**:

```mojo
return Self(
    dtype=self.dtype.copy(),
    length=actual_length,
    ...
    offset=self.offset + offset,
    children=self.children.copy(),   # <-- parent offset never applied
)
```

`field(index)` (`:1909`), `field(name)` (`:1921`), `flatten()` (`:1965`) and
`unsafe_get(name)` (`:1902`) then hand back that raw child, so a caller sees the
**parent's** row count. `__getitem__` (`:1928`) *does* apply the offset, so the
two accessors disagree with each other.

Arrow C++ handles exactly this case — `StructArray::field(i)` slices the child
when `data_->offset != 0 || child_data->length != data_->length`. marrow has no
such conditional, while the docstrings on both `field` overloads state they
match PyArrow's API.

**Reference behaviour** (pyarrow 23.0.1, the version pinned in `pixi.toml`):

```python
sa = pa.StructArray.from_arrays(
    [pa.array([1,2,3,4,5]), pa.array(['a','b','c','d','e'])], names=['x','y'])
s = sa.slice(2, 2)
len(s)              # 2
len(s.field(0))     # 2   -> [3, 4]
len(s.flatten()[0]) # 2   -> [3, 4]
```

marrow returns the unsliced child of length 5 for the same shape.

**Impact on the engine.** A batch is a `StructArray`, so `.limit(n)` followed by
any expression evaluation fails. Two independent reproducers, both with no rule
set named:

- `table(b).limit(2).filter(p)` raises
  `to_array: expected a column of 2 rows, got 4`
- the same shape over a 5-row input raises `... got 5`

Reported failure chain: `physical.mojo:770` -> `runtime/values.mojo:413` ->
raise at `physical.mojo:136`. **Not yet verified independently** — the chain is
from a prototype's trace; the layout defect above is verified.

**Why it survived the suite:** every existing test puts `limit` last in the
plan, so nothing evaluates an expression over a sliced batch.

**Fix direction:** apply the parent offset/length in the child accessors, as
Arrow C++ does. Whether to slice eagerly in `slice()` or lazily in `field()` is
a real choice — lazily matches Arrow C++ and keeps `slice()` O(1).

---

## 2. `Buffer.__eq__` compares the padded allocation, including uninitialized bytes

**Severity: latent. Reachable from one caller, and the hot paths already avoid
it.** Confirmed independently.

`marrow/buffers.mojo:917`:

```mojo
def __eq__(self, other: Self) -> Bool:
    if self._size != other._size:
        return False
    ...
    for i in range(self._size // 8):   # <-- whole allocation
```

`_size` is documented at `:408-409` as "always a multiple of 64
(`_aligned_size`)". A buffer holding four `int64` values has 32 logical bytes
and `_size == 64`, so half the words compared lie past the logical end.
`alloc_uninit` explicitly does not zero-fill (`alloc_host` does `memset_zero`),
so those bytes are whatever the allocator last left there.

**Scope — narrower than it first appears.** Two things contain it:

- `PrimitiveArray.__eq__` (`arrays.mojo:843`) deliberately does not use it,
  and says why: *"buffer may be over-allocated in filtered output, so full
  `Buffer.__eq__` would read uninitialized bytes."* It compares elements.
- `BitmapView.__eq__` (`views.mojo:1256`) masks the tail word correctly.

The **only** caller in `marrow/*.mojo` is `ArrayData.__eq__`
(`arrays.mojo:287`), the flat layout produced on demand by `to_data()` for
interop. So `RecordBatch`/`Table`/`PrimitiveArray` equality does not reach it.

**Open question.** One prototype reported `RecordBatch.__eq__` as *not
reflexive* across two executions of one unchanged plan, and attributed it to
this padding read. That attribution does not hold for primitive columns, per the
containment above. The observation was real enough that the prototype worked
around it in tests; **the actual mechanism is unexplained and unreproduced
here.** Worth a focused look before trusting `RecordBatch` equality in a
soundness harness that compares optimized against unoptimized results — which
is precisely what an optimizer needs.

**Fix direction:** compare the logical byte count, not `_size`. `ArrayData` also
compares `offset` for equality, making it stricter than logical equality; that
may be intentional for a physical-layout struct, but it is worth stating.

---

## 3. `pushdown.mojo` claims a plan rewrite is impossible. It is not.

**Severity: a wrong architectural claim in a load-bearing docstring.**
Confirmed by two prototypes independently, with different evidence.

`marrow/expr/pushdown.mojo` states:

> a rewrite is not merely unnecessary but unavailable: `DynRelation(copy=self)`
> copies trampolines bound to `R`, so a rewritten node would have to have the
> same concrete type, and returning `Optional[DynRelation]` from a trampoline
> field makes the struct recursive, which the compiler rejects.

All three self-referential trampoline shapes compile at **0 errors, 0 warnings**
on Mojo 1.1.0.dev2026083005:

```mojo
var _virt_children: def(ArcPointer[NoneType]) thin -> List[DynRelation]
var _virt_with_children: def(
    ArcPointer[NoneType], List[DynRelation]
) thin raises -> DynRelation
var _virt_opt: def(ArcPointer[NoneType]) thin -> Optional[DynRelation]
```

What the compiler rejects is a genuinely **by-value recursive field**. Adding
`var _recursive: DynRelation` to the same struct gives:

```
marrow/expr/logical.mojo:418:8: error: attempt to resolve a recursive reference
    to declaration 'DynRelation.__move_ctor_is_trivial'
```

A function-pointer field is one word regardless of what its signature mentions;
only a stored value is recursive. The docstring recorded the second behaviour as
if it were the first.

**Consequence:** "no rewriter is possible" is not an argument available to any
design. Riding `to_operator` versus materialising a rewritten plan is an
engineering trade-off, not a compiler constraint. The paragraph should be
corrected before it misdirects further work.

---

## 4. Join keys are positional `List[Int]` — FIXED in `1fcf3302`

> **Resolved 2026-08-31.** `Join` now stores key *names*, resolved from the
> caller's indices at construction and back to indices at lowering. The public
> verb still takes indices, so `plan.mojo` and every existing caller are
> unchanged. This was the prerequisite for both column pruning below a join and
> `PushFilterBelowJoin`.


**Severity: a prerequisite, not a bug today. It becomes a silent wrong-answer
bug the moment projection pushdown reaches a join.**

`Join` stores `_left_keys: List[Int]` / `_right_keys: List[Int]`
(`marrow/expr/logical.mojo:1156`), and the public `.join()` verb takes the same
(`:703`). The indices are positions in the child's schema.

Any rewrite that changes a join child's schema silently rebinds them to
different columns. **Projection pushdown is exactly such a rewrite** — it
narrows a scan's output. There is no error; the query returns a join on the
wrong columns.

Two of three prototypes hit this and both chose to **block projection pushdown
below `Join`** rather than risk it. That is the correct conservative call and it
is also where the remaining value is: pushdown that stops above a join misses
the case that matters most on join-heavy workloads.

**Fix direction:** key by name or by `DynValue`, resolved against the child
schema at plan-build time. This is an API-breaking change to `.join()`, so it
wants to land before the optimizer depends on it, not after.

---

## Two non-defects worth recording

**`RuntimeValue` has no arithmetic tags at `bb1a305f`.** `lit(1) + lit(2)` is
not constructible in the runtime lane — there is no `add` tag in
`RuntimeValue.evaluate`, no `__add__`, no `add()` constructor. All three
prototypes independently hit this while implementing constant folding, and all
three folded boolean identities instead. Arithmetic exists only in the comptime
lane. (Work in progress on the main checkout adds this surface; it was not part
of `bb1a305f`.)

**Constant folding cannot reach comptime values, by design.** The optimizer must
never convert a `ComptimeValue` into a `RuntimeValue`: it would destroy the
fusion the lane exists for and link the entire runtime interpreter into an AOT
binary — measured at **+119 percentage points** of `__text` in the RTTI
prototype. The correct boundary is *ask, do not rewrite*: a comptime value
answers `columns()`, a selectivity estimate, and its conjuncts as still-fused
boxes. Structural rules (predicate pushdown, projection pushdown, TopN) are
lane-agnostic; value-inspecting rules (folding, CSE) are runtime-only, and for a
fused predicate LLVM performs the equivalent inside the loop at `-O3`.

---

## 5. A concrete overload does not beat a trait-bound generic overload

**Measured 2026-08-30. Compiles, resolves without ambiguity, and is never
called.** Recorded because the design it kills is the obvious one.

Conjunction splitting — turning `filter(a AND b)` into two stacked filters —
wants to happen where the predicate's concrete type is still visible, which is
the typed `.filter()` verb. `pruning.mojo` sets that precedent: it builds
`PrunePredicate` there rather than adding a slot to `DynValue`, and calls the
placement "the single most important size decision here", because a `DynValue`
trampoline is paid for every projection value, every sort key and every
aggregate input in the program.

So the natural design is a **more specific overload**:

```mojo
def filter[V: Value & Prunable](self, var predicate: V) raises -> DynRelation
def filter[                                     # more specific — never chosen
    L: ComptimeValue, R: ComptimeValue
](self, var predicate: BoolBinary[AndKernel, L, R]) raises -> DynRelation:
    return self.filter(predicate.l.copy()).filter(predicate.r.copy())
```

This **compiles at 0 errors / 0 warnings**, raises no ambiguity diagnostic, and
Mojo selects the *generic* overload. A test asserting the rendered plan contains
two `Filter` nodes gets one:

```
AssertionError: `left == right` comparison failed:
   left: 1
  right: 2
```

**Note `precompile` cannot detect this.** A generic overload is only
instantiated at a call site, so the library compiles clean whether or not the
overload is ever selected. Only an instantiating test shows it — the same trap
CLAUDE.md records as "`precompile` being clean is not evidence that a test file
will build".

**Consequence for conjunction splitting.** It cannot be done at the verb by
overloading. The remaining options, none free:

1. `conjuncts()` on `Value`, defaulting to `[self]` — works, and costs one more
   `DynValue` trampoline, i.e. exactly the cost `pruning.mojo` avoided. Measure
   it before adopting.
2. A distinct verb (`filter_all([a, b])`) — no trait change, no dispatch, but it
   puts the burden on the caller and `a & b` stays unsplit.
3. Leave it to the caller: `.filter(a).filter(b)` already splits, and the
   predicates channel already conjoins stacked filters, so the *pruning* benefit
   is available today without any mechanism at all.

Option 3 is why this is a low-priority gap rather than a blocker: the value
splitting adds over stacked filters is independent pruning per conjunct and the
ability to push one half below a join, and the second of those is blocked on
positional join keys (§4) anyway.

---

## 6. `RecordBatch.__eq__` is not reflexive — FIXED in `371c27f2`

> **Resolved 2026-08-31.** The chain was `DynArray.__eq__` -> `to_data()` ->
> `ArrayData.__eq__` -> `Buffer.__eq__`, which compares `_size // 8` words over
> the 64-byte-aligned allocation — including bytes past the logical end.
> §2 had the mechanism right; what was wrong was concluding it was unreachable
> from `RecordBatch`, since `DynArray.__eq__` takes the `ArrayData` route to
> dodge a compiler deadlock and never reaches `PrimitiveArray.__eq__`, which
> carefully avoids the problem. Fixed by zeroing the alignment padding in
> `alloc_uninit`. `test_executing_one_plan_twice_agrees` passes.


> **Status 2026-08-31: confirmed.** This entry was briefly retracted on the
> strength of a run where the control passed; that retraction was wrong and the
> way it was wrong is the finding's most useful property.

A direct control — execute one unchanged plan twice, compare first by extracted
values and then by `RecordBatch.__eq__`:

```mojo
var first = plan.execute(ctx)
var second = plan.execute(ctx)
assert_equal(_col(first, 0), _col(second, 0))   # passes
assert_true(first == second)                    # FAILS
```

```
AssertionError: values agree but RecordBatch.__eq__ does not
                — the equality is at fault, not the engine
```

**Execution is deterministic; the equality is not.** Every cell agrees. Only
the whole-batch comparison disagrees.

**It is flaky, and that is why it was nearly missed.** The identical case
passed 15/15 when `test_optimizer.mojo` ran alone, and failed when the same
file ran inside `pytest marrow/expr/tests` (141 passed, 1 failed). A comparison
whose answer depends on how much else has been allocated is reading state it
does not own, or comparing physical representation rather than logical value.

**The mechanism is still not pinned down**, and §2's proposed one does not
explain it: `PrimitiveArray.__eq__` (`arrays.mojo:843`) deliberately avoids
`Buffer.__eq__` and compares elements. Candidates worth checking next are the
physical properties `__eq__` *does* compare — null counts, offsets, and whether
a bitmap was allocated at all. `PrimitiveArray.__eq__`'s own comment notes that
"a slice that excludes every null still carries its parent's bitmap", which is
exactly the kind of representational difference two executions could disagree
on while every value matches.

**Why this matters more than an ordinary bug.** Result equivalence between an
optimized and an unoptimized plan is the soundness harness an optimizer rests
on. Built on `RecordBatch.__eq__`, that harness reports failures that are not
real — and, being flaky, reports them only sometimes. `test_optimizer.mojo`
therefore compares **extracted values against hand-written expected rows**, and
keeps this case as a standing failure rather than deleting it, so the bug stays
visible until it is fixed.

### Superseded: the earlier reproduction path
### The original (incorrect) reproduction path

While testing the above, two plans over separately-built but identical
`RecordBatch` inputs compared **unequal** on their results:

```mojo
var split = table(_conj_batch()).filter(a & b)   # typed verb, carries a pruner
var whole = table(_conj_batch()).filter(DynValue(a & b))  # erased verb, no pruner
assert_true(split.execute(ctx) == whole.execute(ctx))     # FAILS
```

Both plans are structurally one `Filter` over an in-memory table, the pruner is
never consulted for an in-memory source, and the inputs are element-wise
identical. The results should be equal and are not.

This is consistent with the non-reflexive `RecordBatch` equality reported in §2,
and it is a smaller reproducer than the one that report came from. It does
**not** confirm the padding mechanism proposed there — `PrimitiveArray.__eq__`
does not reach `Buffer.__eq__` — so the cause is still open.

**Why this matters more than it looks:** result equivalence between an
optimized and an unoptimized plan is the soundness harness the whole optimizer
depends on. If array equality is unreliable, that harness silently reports
failures that are not real, or worse, passes that are not either.

---

## 7. Mojo limits hit while making `DynRelation` variant-backed (2026-08-31)

Five, each of which changed the code. Recorded with exact diagnostics because
none is documented and each cost a compile cycle.

**`_type_is_eq` does not exist.** Matching a generic parameter against a variant
member does not need it — `Variant`'s constructor takes a member directly, so
the whole of `DynArray`'s construction path is:

```mojo
@implicit
def __init__[T: Array](out self, var array: T):
    self._v = Self.VariantType(array^)
```

Reaching for a type-equality test is the wrong instinct; the variant already
resolves it.

**An implicit conversion does not chain through `Optional`.** `DynRelation` has
an `@implicit` constructor from any `Relation`, yet `return Limit(...)` from a
function returning `Optional[DynRelation]` fails with *"cannot implicitly
convert 'Limit' value to 'Optional[DynRelation]'"*. Binding a typed local first
(`var out: DynRelation = Limit(...)`) works. This is why the rules return a
plain `DynRelation` rather than an `Optional`.

**A struct parameter must be qualified inside a static method.** `R.rewrite(x)`
inside `struct Optimizer[R: RuleSet]` gives *"unqualified access to struct
parameter 'R'; use 'Self.R' instead"*.

**`len(String)` is rejected outright**, with a diagnostic explaining that UTF-8
makes a single length ambiguous: use `byte_length()`, `len(s.codepoints())` or
`len(s.graphemes())`.

**A trait default that returns `DynRelation` forces `Writable` onto the trait.**
`DynRelation.__init__` is bound on `Relation & Writable` and `_dispatch`
narrows members with `conforms_to(T, Relation)`, so `Relation` has to imply
`Writable` for the two to meet. All eight nodes already conformed; the effect is
that a future relation node must be printable.

### And one consequence that is not a compiler limit

**A variant closes the type set permanently.** `test_erasure.mojo` defined a
test-local `_RelationProbe` to prove the box did not drop its node's
destructor. A variant cannot hold one — the members *are* the set — so nothing
outside `logical.mojo` can be a relation node any more, not tests, not
`python/bindings/`, not a future extension.

The hazard the probe existed to catch is also gone: a variant destroys its
member at the true type, where `rebind[ArcPointer[NoneType]]` erasure forgot
it. That bug class is designed out rather than tested away, and the test was
replaced with a named case saying so, so the missing coverage does not read as
an oversight later.

---

## 8. An aggregate above a `Limit` returns zero rows — FIXED in `371c27f2`

> **Resolved 2026-08-31.** The hypothesis recorded below was right:
> `Pipeline.drain`'s early termination set `_stage = len(self._ops)` whenever
> *any* stage reported done, skipping every stage above the `Limit`. Anything
> answering only from `drain` — every aggregate, and a `Sort` — was dropped.
> `done` split into `_first_done()` so the drain resumes just above the
> finished stage.


**Severity: silent wrong answer on a plan shape the engine accepts.**
Confirmed 2026-08-31, independent of the optimizer.

```mojo
var plan = table(batch).limit(3).aggregate(
    [col("a", int64).sum().alias("total")]
)
plan.optimize[NoRules]().execute(ctx).num_rows()   # 0, must be 1
```

`AssertionError: an ungrouped aggregate must emit one row — left: 0, right: 1`

An ungrouped aggregate emits exactly one row for any input, including an empty
one: `sum` over no rows is `NULL`, `count(*)` is `0`. Returning *zero rows*
means `SELECT sum(a) FROM (SELECT * FROM t LIMIT 3)` yields nothing at all.

**No optimizer involvement.** The probe runs through `optimize[NoRules]`, so no
rule fires and the plan executes exactly as it did before this work. The
aggregate alone is fine — the same aggregate without the `limit` returns 23 as
expected (`test_optimizer_removes_a_sort_before_an_aggregate`).

**Likely mechanism, not yet confirmed.** `LimitOperator` reports `done` once it
has its rows, which in a push engine is what stops the source early. An
aggregate has nothing to push and answers only from `drain`. If `done`
propagates in a way that skips `drain` on the operators above it, the aggregate
never gets the call it emits from. `physical.mojo`'s `Pipeline` is where that
interaction lives.

**How it was found**, which is the part worth keeping: a test asserted that a
TopN-bounded sort feeding an aggregate returns the right sum, and the
*unoptimized* side returned nothing. A harness that compared the optimized plan
against the unoptimized one would have seen both return zero rows, called them
equal, and passed. Checking each side against hand-written expected values is
what surfaced it.

Standing as a failing test — `test_an_aggregate_above_a_limit_emits_one_row` —
rather than being silenced, so it stays visible until fixed.

---

## 9. What the variant-backed `DynRelation` costs: +348%

Measured 2026-08-31, `pixi run binary_size`, against the baseline recorded in
`a27167aa`.

| gate | baseline | variant | delta |
|---|---:|---:|---:|
| `query_streaming` | 1,484,132 | 6,652,476 | **+348.2%** |
| `query_streaming_agg_fused` | 1,478,768 | 6,654,772 | **+350.0%** |
| `query_expr2_streaming` | 1,472,420 | 6,641,724 | **+351.1%** |
| `query_expr2_agg_fused` | 1,477,872 | 6,648,756 | **+349.9%** |
| `query_join` | 1,633,200 | 6,601,212 | **+304.2%** |
| `query_dynvalue` | 8,462,992 | 9,996,264 | +18.1% |
| `query_streaming_agg` | 12,024,976 | 13,973,800 | +16.2% |

**Arm B's RTTI prototype predicted +358% for naming node types. This measures
+348%** — the estimate was accurate to within 3%, from a different mechanism
(free-standing rules downcasting) than the one finally built (a variant box).
The conclusion generalises: it is *naming the node types in one place* that
costs, not how they are named.

**The cost falls on exactly the binaries the invariant protected.** The fused
AOT gates grow ~350%; the gates that already link an interpreter grow ~17%. The
project's headline comparison was `query_streaming_agg_fused` (1.48 MB) against
`query_streaming_agg` (9.94 MB), a **6.7x** gap that is the AOT lane's whole
argument. After this change the same pair is 6.65 MB against 13.97 MB — **2.1x**.

**What did not regress:** `marrow::expr::runtime` links **0 symbols** in every
fused gate. No AOT binary pays for the runtime interpreter, so the
"never lower a `ComptimeValue` into a `RuntimeValue`" constraint held in the
built artifact, not merely in review.

**The trade, stated plainly.** This buys an optimizer whose rules are readable
in one file, can inspect a node, and can construct one — none of which the
trampoline design could do, and the absence of which is why its rules ended up
scattered across eight `to_operator` methods with nothing to read. It costs the
small-binary property that distinguishes marrow from DuckDB, DataFusion and
polars. Both halves are real; the decision is which one the project is for.

The baseline was **not** re-recorded. The gate should stay red until that
decision is made deliberately.

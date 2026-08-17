# A1 — null ops and `COUNT(*)`: findings

Written while adding `is_null` / `is_valid` / `is_nan` / `is_inf` / `fill_null`
to the runtime lane, `FillNull` to the AOT lane, and `count_star()`.

Everything below is a defect or a simplification opportunity I hit directly.
Line numbers are against the tree at the time of writing.

---

## 1. `DynAgg` is still an interpreter, and it is the exact mistake `EvalFn` was introduced to fix

**This is the most important finding in this document.**

`marrow/expr/dynamic.mojo` has a long, well-earned docstring on `EvalFn`
explaining that naming an operation with a *string* and resolving it through a
central switch cost `query_dynvalue` **+1,807,168 bytes of `__text` (+45.7%)**,
because every kernel arm became reachable from every `DynValue`. The fix was to
make the operation comptime: `__sub__` names `_binary[SubKernel]`, so a program
links exactly the kernels its expressions mention.

`DynAgg`, thirty lines below in the same file, does the thing that docstring
forbids:

```mojo
struct DynAgg(Copyable, Movable, Writable):
    var func: String        # marrow/expr/dynamic.mojo
    var input: DynValue
```

`col("x").sum()` is `self.aggregate("sum")` — a *string*. It is resolved by
`resolve_agg` (`marrow/expr/aggregates.mojo:191`), an 8-arm string ladder, from
`AggFunc.of` (`:312`) and `FoldedAggregates` (`:460`). Both call sites take a
runtime `String`, so **any program that builds any aggregate links all eight
aggregations** — `Sum`, `Product`, `Mean`, `Min`, `Max`, `Count`,
`CountDistinct`, `ApproxCountDistinct`, and every dtype instantiation each of
their `resolve` methods dispatches into.

The scalar lane and the aggregate lane are the same problem and got opposite
answers, in the same file, without the discrepancy being noted anywhere.

**Cleaner shape.** Give `DynAgg` the treatment `DynValue` already got: a thin
fn pointer chosen at the call site, so `.sum()` names `Sum` and nothing else.

```mojo
comptime AggResolveFn = def(DynType, ...) thin raises -> AggFunc
struct DynAgg:
    var resolve_fn: AggResolveFn     # `.sum()` names Sum's resolver
    var name: String                 # render/alias only — never selects
```

`resolve_agg` stays for the genuinely dynamic caller (SQL, a wire frontend)
rather than being on the path of every typed `.sum()`. This mirrors exactly what
the `_tag` string is reduced to on `DynValue`: "drives `render`, `prune` and
`name` only — it never selects a kernel."

I did not do this: it changes `AggFunc`/`FoldedAggregates`, which I do not own.
**It should be measured with `pixi run binary_size` before the alpha ships** —
the `EvalFn` note says this class of mistake was worth 45.7% of `__text` on one
gate, and nothing has measured the aggregate half.

## 2. `COUNT(*)` allocates and scans a column to compute a number the grouper already has

`count_star()` is `lit(1).count()`, and I verified it is *correct*
(`test_count_star_probe_literal_counts_every_row`). It is not cheap.

Per batch, per `COUNT(*)`:

1. `DynValue._literal` does `payload[DynScalar].repeat(batch.num_rows())` —
   an N-element `Int64Array` allocation (8N bytes) of a constant;
2. `CountValid.resolve` sees a numeric dtype and picks
   `NumericAgg[CountKernel, Int64Type]`, which **scans** that column's validity
   to count the valid entries;
3. the answer is N, or per group the group size — which the grouper computed
   when it built the groups.

ClickBench asks for `COUNT(*)` in roughly 30 of 43 queries, so this is on the
hot path of most of the benchmark the alpha is aimed at.

**Root cause is an abstraction gap, not an oversight.** `DynAgg` and `AggExpr`
both require an *input expression*; there is no way to spell an aggregate with
no input. So `COUNT(*)` has to invent a column to count. A `CountStar`
`Aggregation` whose state is a counter advanced by the group size — no input,
no allocation, no scan — cannot currently be represented.

`count_star()` as shipped is the right *interface* (it stops every caller
rediscovering the `lit(1)` trick, and the name is now the thing to optimise
behind), but it is a placeholder implementation and should be labelled as one.

## 3. `fill_null(a, b)` and `coalesce(a, b)` compute the same thing, twice

For two operands these are the same function: "take `a` where valid, else `b`".
`CoalesceKernel.apply` (`marrow/kernels/conditional.mojo:270`) and
`FillNullKernel.apply` (`:378`) are two separate row loops that both build a
`Selection` and call `gather`. The only real difference is that `fill_null`
additionally pins the operands to one dtype and one length.

Keeping both *names* is right — PyArrow and Polars users reach for both, which
is why I added `fill_null` rather than telling callers to write `coalesce` — but
one of the two bodies should be the other's caller. `FillNullKernel.combine`
could be `CoalesceKernel.combine` plus the two `expect_*` checks.

Related: both go through `Selection` + `gather`, which is a **per-row** loop
(`for i in range(sel.length())`) with a per-row `is_valid(i)` call. For a
null-handling op on a primitive column this should be a bitmap operation plus a
masked select, not a scalar gather. That is the single biggest easy win in
`conditional.mojo`.

## 4. A trait default that returns a concrete node type silently forbids overriding

`Value` carried `isnull()`/`notnull()` defaults returning the *fused*
`NullPredicate`. That is a representation decision baked into the shared trait,
and it made the runtime lane's version unwritable:

- a struct method does **not** override a trait default in Mojo — the two become
  competing overloads and the call site reports `ambiguous call to 'is_null'`;
- so `DynValue` could not return a `DynValue` from `is_null()`.

That mattered, because a predicate that leaves the erased lane cannot be
recombined with one that stays in it:
`col("a").is_null() | (col("b") > lit(1))` does not compile —
`BoolValue.__or__` takes a `BoolValue` and a `DynValue` is not one. The erased
lane had null *detection* but not composable null *predicates*, which for a
dataframe frontend is most of the value.

Resolved by moving the two methods off `Value` and onto `DynValue`. This cost
nothing: every caller in the tree was already a `DynValue`, because the AOT lane
always spells the node directly (`IsNull(col("a", int64))`). `NullPredicate`'s
`A: Value` bound is untouched.

**Generalisable rule, worth putting in CLAUDE.md:** a trait default may return
`Self`, a scalar, or a container — but returning *a specific node type* makes the
trait dictate representation to every conformer, and Mojo reports the resulting
conflict at the unrelated call site rather than at the definition.

## 5. Trait-default parameter names leak into conformers' namespaces

Adding `def fill_null[R: NumericValue](...)` to `NumericValue` broke three
unrelated structs:

```
error: name conflict between parameter 'R' in the default trait method
       and a parameter in the struct
  struct NumericBinary[K: BinaryNumericKernel, L: NumericValue, R: NumericValue]
```

A trait default method's parameter name must not collide with a *conformer's*
struct parameter. Every binary operator in `values.mojo` already says `Rhs` — the
convention exists precisely because of this, but it is unwritten, so the next
person to add a trait default will rediscover it. Documented in the method's
docstring; it belongs in CLAUDE.md's "Associated types, traits, reflection" list.

## 6. The parity suite silently under-tested two shipped ops

`test_parity_coalesce` and `test_parity_nullif` used `assert_fused` — the
one-lane helper — whose docstring says it is for "ops the runtime `DynValue`
interpreter does not yet expose ... PENDING T2.2". But `DynValue.coalesce` and
`DynValue.nullif` both exist. The placeholder outlived the gap and nobody
noticed, so two shipped ops had no cross-lane assertion at all.

Upgraded both to real `assert_parity` in this change. **Worth auditing every
remaining `assert_fused` call site** against the current `DynValue` surface; the
helper's docstring is a to-do list that has no owner and no expiry.

## 7. Two aggregate representations, and the erased one is not the box

`AggExpr` (AOT, names the `Aggregation` *type*) and `DynAgg` (runtime, names a
*string*) are parallel structures with no shared trait, so every consumer
handles both. Contrast the scalar layer, where `BoxedValue` is the single type
both lanes erase into and `assert_parity` can therefore take one argument type
for either lane.

There is no `BoxedAgg`. That is why the aggregate half of `test_parity.mojo` is
thin — `test_agg_parity_grouped_sum` has to build two whole plans and compare
output columns, rather than handing two aggregates to one helper. Adding the
box would make aggregate parity as cheap to test as value parity is, which is
probably a precondition for trusting the aggregate lane at alpha.

## 8. Minor: `is_inf` had no expression surface in either lane

`IsInfKernel` and the fused `IsInf` node both existed; nothing exposed them
fluently on the runtime lane, and `NumericValue.isinf()` was the only path. Added
`DynValue.is_inf()` alongside the others — it is one line given `_predicate`, and
leaving it out would have been an arbitrary hole next to `is_nan`.

Also renamed `isnull`/`notnull`/`isnan`/`isinf` to `is_null`/`is_valid`/
`is_nan`/`is_inf`. CLAUDE.md requires following PyArrow naming, PyArrow spells
these `is_null`/`is_valid`/`is_nan`, and the kernels' own `comptime name`
strings were *already* `"is_null"`/`"is_nan"`/`"is_inf"` — the fluent methods
were the only place using the jammed-together spelling.

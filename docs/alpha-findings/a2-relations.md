# A2 — `marrow/expr/relations.mojo`: findings

Written while adding `with_columns` / `drop` / `rename`. Line numbers are against
the file **after** that change.

---

## 1. The two `aggregate` overloads do **not** earn their complexity — delete the second

`relations.mojo:310` — `aggregate(keys, aggs: List[AggExpr])`
`relations.mojo:391` — `aggregate(keys, inputs, aggs: List[AggFunc], names)`

**Verdict: the second overload is redundant, and it is a leaky abstraction.** It
should be deleted and its two call sites moved to the first.

### It is redundant — verified, not argued

I rewrote both of its callers onto the first overload and ran the suite:

- `marrow/expr/tests/test_aggregates.mojo:412` `_fused_sum_max_by_region`
- `marrow/expr/tests/test_aggregates.mojo:467` `test_fused_non_numeric_aggregation`

```
AggFunc.of[NumericAgg[SumKernel, Int64Type]](DynType(int64))   # + parallel inputs[] + names[]
  ->  AggExpr.of[NumericAgg[SumKernel, Int64Type]](fused_col("amount", int64)).alias("total")
```

`25 passed in 132.73s` — including `test_fused_aggregate_matches_the_dynamic_one`,
`test_fused_aggregate_results` (asserts the values 90 and 50) and
`test_fused_non_numeric_aggregation` (asserts the string `min` **and** that the
key column is named `region`). The experiment was reverted; the tree is
unchanged.

Nothing was lost because the convergence point already exists one layer down:
**`AggExpr` (`values.mojo:2159`) is exactly the "one type, two constructors"
seam this needs.** `AggExpr.of[A]` (`values.mojo:2222`) is the comptime
constructor, `AggExpr.__init__(DynAgg)` (`values.mojo:2197`) the by-name one, and
`resolve()` (`values.mojo:2243`) collapses them into one `AggFunc`. `AggFunc`
itself does the same thing again and does it right — `AggFunc.of[A]`
(`aggregates.mojo:293`) versus `AggFunc(name, dtype)` (`aggregates.mojo:304`),
two constructors on **one** type.

So the fused lane already had a first-class spelling. The second overload is a
*third* convergence point for a split that was already resolved twice below it.

### It is leaky — the two overloads contradict each other in the same struct

Overload 1's docstring (`relations.mojo:322-324`) states the design rule:

> An aggregate is written on the expression it aggregates (`col("amount").sum()`),
> so nothing has to be kept in positional correspondence.

Overload 2 is three parallel lists that must be kept in positional
correspondence, plus a fourth that is *indexed across two different things*:
`names[i]` for keys and `names[len(keys) + j]` for aggregates
(`relations.mojo:426-428`). The caller has to count. That is the exact burden
overload 1 exists to remove, reintroduced 80 lines later.

The two hand-written arity checks at `relations.mojo:410-421` are pure
consequence: they exist only because the shape is parallel lists, and overload 1
needs none of them.

### What it was actually reaching for — and still does not deliver

Overload 1 cannot name a **computed** key: `relations.mojo:359-362` names a key
after its source column when `bound_column` answers, else `key0`, `key1`, …
Overload 2 looks like the escape hatch, but it is not a good one — you name keys
by position in a flat list that also covers the aggregates.

Meanwhile the fused lane never needed the escape hatch at all: fused column
leaves implement `bound_column` (`values.mojo:767` `NumericColumn`,
`values.mojo:1729` `StringColumn`), so overload 1 already derives `region` for a
fused key. That is why the rewrite above produced identical schemas.

**Cleaner shape.** Delete overload 2. Give keys the naming affordance aggregates
already have, so both sides of `GROUP BY` are named the same way:

```mojo
rel.aggregate(
    keys=[col("ts").date_trunc("month").alias("month")],   # named like an agg
    aggs=[col("amount").sum().alias("total")],
)
```

That needs `.alias(...)` on a key expression (a thin `KeyExpr`, or reuse the
`out_name` field `AggExpr` already carries), removes the only capability gap,
deletes ~45 lines and two arity checks, and leaves one verb per operation.

### Side effect: the binary-size gate does not gate the API it claims to

`benchmarks/binary_size/query_streaming_agg_fused.mojo:47-55` does not call
either overload — it constructs `Aggregate(...)` **by hand with a hand-written
schema**. That is precisely what the module docstring forbids
(`relations.mojo:23-26`: "Build plans through these, not by constructing nodes
… a hand-built plan can declare a schema its own expressions do not produce").
So the AOT size gate measures a node the plan builder never emits, and deleting
overload 2 would not even register there. The gate should go through the
plan-building API, or it is not gating the plan-building API.

---

## 2. `project` silently downgrades `nullable` and drops field metadata

`relations.mojo:537`:

```mojo
fields.append(Field(names[i], values[i].execute(probe).dtype()))
```

`Field.__init__` defaults `nullable=True, metadata={}` (`dtypes.mojo:487-493`).
`select` (`relations.mojo:283`) instead copies the whole input `Field`.

So for a column declared `nullable=False`, these two produce **different
schemas for the same query**:

```mojo
rel.select("x")                        # x: int64, nullable=False
rel.project(["x"], [col("x")])         # x: int64, nullable=True   <- wrong
```

The dtype probe is right and should stay — it is the only thing keeping a
computed column honest. But it answers a *dtype* question and is being used to
answer a *field* question. For a value that is a bare column reference, the
honest field is the input field.

`with_columns` / `drop` / `rename` as added here copy the input `Field` for every
pass-through column, so they do not have the bug. `project` still does. The fix
is to ask `values[i].bound_column(input_schema)` — the mechanism already exists
and `aggregate` already uses it (`relations.mojo:359`) — and copy the field when
it answers `>= 0`.

---

## 3. Five verbs now build `Project`, four of them by copy-paste

`select` (`:270`), `project` (`:507`), `with_columns` (`:547`), `drop` (`:641`),
`rename` (`:687`). Four of the five build the same thing: a `Project` whose
values are `col(name)` pass-throughs and whose fields are copied from the input.
I added three of those four, so this is partly self-inflicted and worth saying
plainly.

The shared shape is "a list of (source column name, output `Field`)". A private

```mojo
def _passthrough(self, var pairs: List[Tuple[String, Field]]) raises -> DynRelation
```

would let `select`/`drop`/`rename` become their selection logic and one call, and
`with_columns` reuse it for its pass-through slots. I did not do it here because
`select` is existing shared surface and this was not the assignment, but the
duplication is real and it is the natural follow-up.

---

## 4. `join`'s two key-resolution loops are duplicated, and the error is unhelpful

`relations.mojo:456-473`: two near-identical loops, left then right. Both raise
the same string:

```
join: key must be a column reference, got a computed expression
```

It does not say which side, which position, or what the expression was. On a
multi-key join that is a guessing game. One private helper taking the key list,
the schema and a side label would halve the code and let the message name the
offender.

---

## 5. `with_predicate`'s erased-pointer protocol is a documented workaround that will not scale

`relations.mojo:112-131` returns `Optional[ArcPointer[NoneType]]` rather than
`Optional[DynRelation]`, because a field whose function type mentions
`DynRelation` makes the struct recursive. The reason is real and well documented
(same limit as `BoxedValue._resolve_names_fn`, `values.mojo:508-522`).

The cost is that the *caller* must know the protocol: `filter`
(`relations.mojo:300-307`) copies itself, swaps `_data` by hand, and relies on
"same concrete type, so the trampolines still apply". That reasoning is correct
but it is written out longhand at the call site. Today there is one caller. A
second node gaining pushdown means a second hand-rolled pointer swap, and the
invariant that makes it sound is a comment, not a signature.

A private `DynRelation._replacing_data(var ptr) -> DynRelation` would put the
swap and its justification in one place and make the call sites one line.

---

## 6. Minor

- `RELATION_GENERIC` / `RELATION_PARQUET_SCAN` / `RELATION_SORT`
  (`relations.mojo:103-105`) is an open mini-RTTI whose only consumer is the
  top-K fold in `limit` (`:803`). Fine at three values; it grows one constant per
  node that ever needs recognising, and nothing ties a constant to its struct.
- `select` is variadic (`*names: String`) while `drop`, `project`,
  `with_columns` and `rename` take `List[String]`. `select` is therefore the one
  verb you cannot call with a computed column list — which is the 105-column
  problem again, in the verb where it is easiest to hit.

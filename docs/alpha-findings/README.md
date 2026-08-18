# Alpha findings — simplification and abstraction issues

Six agents built the alpha in parallel and each kept a findings log. This is the
synthesis: what they independently converged on, ranked by **strength of
evidence**, not by how bad it sounds.

The individual logs are the primary sources and carry the file:line detail:

| Log | Author's vantage point |
|---|---|
| [`a1-null-ops.md`](a1-null-ops.md) | adding null predicates + `fill_null` to both lanes |
| [`a2-relations.md`](a2-relations.md) | adding `with_columns`/`drop`/`rename` to the plan builder |
| [`b1-expr-bindings.md`](b1-expr-bindings.md) | binding `DynValue`/`DynAgg` to Python |
| [`b2-plan-bindings.md`](b2-plan-bindings.md) | binding `DynRelation` to Python |
| [`d1-binding-delta.md`](d1-binding-delta.md) | binding the methods the first two deferred |
| [`e1-clickbench.md`](e1-clickbench.md) | writing 43 real queries as the first user |

**Evidence tiers used below.** *Measured* — someone ran an experiment and has a
number. *Reproduced* — someone triggered the defect from a test. *Argued* — a
reading of the code, however careful. Treat the tiers differently.

---

## 1. The aggregate cluster — five carrier types, and four agents found it

This is where the abstraction is weakest, and it is the one area every agent
that touched it reported independently without seeing each other's notes.

The cluster carries three pieces of data — *(function, input expression, output
name)* — across `DynAgg`, `AggExpr`, `AggFunc`, `AggFold` and
`FoldedAggregates`.

### 1.1 `DynAgg` duplicates `AggExpr` field-for-field — *argued, three times*

Flagged separately by A1 (§7), B1 (§2.1) and B2 (§5). `AggExpr.__init__(var agg:
DynAgg)` is a copy constructor that *also* re-applies "empty alias ⇒ use the
function name" — a rule `DynAgg` does not apply itself, so `DynAgg.out_name` is
`""` where `AggExpr.out_name` is `"sum"`. B1 had to write that rule a **third**
time in the binding to give Python an honest `name()`.

### 1.2 `DynAgg` is a string-tag interpreter — *the strongest single finding*

A1's headline. `DynAgg` holds `var func: String`, resolved by `resolve_agg`'s
8-arm ladder, so **any program building any aggregate links all eight
aggregations** and every dtype instantiation beneath them.

What makes this sharp rather than stylistic: it sits ~30 lines below a docstring
recording that string-tag dispatch cost `query_dynvalue` **+1,807,168 bytes
(+45.7% of `__text`)** and was removed for exactly that reason. Same file, same
mistake, opposite answers, undocumented.

**Scope correction from measurement.** `marrow::expr::dynamic` contributes **0
symbols** to every fused/AOT target, so the closed-erasure DCE property is
intact and this does **not** leak into the comptime lane. The cost is confined
to the runtime-lane targets (`query_dynvalue`/`query_runtime`, 3.4x the fused
baseline), which are the interpreter by definition.

`benchmarks/binary_size/check_gate.py` on the merged `alpha` — every gate
*shrank*, and the runtime lane did not move at all:

```
gate                             baseline     measured      delta      pct
query_streaming                 1,484,652    1,423,236    -61,416  -4.137%
query_join                      1,507,836    1,454,504    -53,332  -3.537%
query_streaming_agg_fused       1,417,476    1,371,312    -46,164  -3.257%
query_streaming_agg             1,932,404    1,886,260    -46,144  -2.388%
query_dynvalue                  4,871,156    4,871,156         +0  +0.000%
OK: no gate grew more than 0.5%.
```

So the whole alpha — six merged branches, the Python binding layer, three new
plan verbs, five new null ops — cost **nothing** in the AOT lane and shrank four
of five gates. This item is real and worth fixing on its own merits; it is
**not** a release blocker, and nothing in the alpha made it worse.

### 1.3 `AggExpr` is a two-variant sum type spelled as unenforced `Optional`s

B1 (§2.2) and B2 (§5a) independently. Every method is an `if` reconstructing an
invariant the type does not enforce; a malformed one fails late with `unknown
aggregate function: ` on an empty name.

### 1.4 The second `aggregate` overload should be deleted — *measured*

A2's headline, and the best-evidenced item in this document because A2 did not
argue it — it **rewrote both callers onto the first overload and got 25 passing**,
including the fused/dynamic parity test, then reverted. B2 reached the same
conclusion independently (§5d) from the opposite direction: the fused overload
"regresses to parallel lists", taking four of them plus hand-checked length
invariants, which is precisely what `AggExpr` exists to eliminate.

### 1.5 `COUNT(*)` has no representation — *reproduced*

D1's finding, confirmed by E1 hitting it in 28 of 43 queries. `count_star()` is
`lit(1).count().alias("count_star")`, so `function()` returns `"count"` and the
only marker is a default alias that the first `.alias("n")` erases. A plan cannot
be inspected for it, so the obvious optimisation — read the row count from the
Parquet footer — is unavailable. Root cause is 1.3: `DynAgg.input` is mandatory,
so a nullary aggregate must lie about having one. A1 measured the consequence:
it allocates an N-element constant column and scans it to compute a number the
grouper already has.

### 1.6 What is *not* wrong

B1 (§2.3) and B2 both checked and both concluded the `AggFunc`/`AggFold`/
`FoldedAggregates` split **is** justified — the binary-size measurements behind
it are real. The complaint is that nothing in the names says so. Do not
"simplify" this one.

---

## 2. The plan IR cannot be traversed — *argued, high consequence*

B2's headline (§1). `Relation` has no `inputs()`, so a plan cannot be walked —
from Python or Mojo. No EXPLAIN, no rewrite framework, no cost model.

The consequence is already visible: the layer has **two incompatible ad-hoc
rewrite mechanisms**, a kind-tag + `downcast[Sort]` in `limit()` and a per-node
virtual `with_predicate` in `filter()`. The docstrings record that the downcast
approach *already caused a correctness bug* with parameterised `ParquetScan`.

Related: no node's `write_to` renders its children (B2 §2), so `explain()` prints
one shallow label. D1 found and fixed a second-order version of this — `str()` on
a bound type silently returned the derived `repr` because `def_method` fills
`tp_dict`, not the CPython `tp_str` slot, and **every existing assertion was a
substring test, so nothing caught it**.

---

## 3. `project` loses field fidelity — *reproduced, and partly fixed*

A2 found that `project` rebuilds a bare `Field`, dropping `nullable` and
metadata, so `select("x")` and `project(["x"], [col("x")])` produce different
schemas for the same column. D1 then reproduced it from Python:

```
t.select("a")            -> nullable = False   (correct)
t.project(a=t["a"])      -> nullable = True    (widened)
```

D1 fixed the immediate bug by adding `DynRelation.select(names: List[String])`.
The root cause is open, and D1 proposed the fix: `BoxedValue.bound_column(schema)`
already exists and `aggregate` already uses it for exactly this test — if
`project` copied the source field when `bound_column >= 0`, the fidelity gap
closes at the root and three of the four projecting verbs collapse to name
arithmetic. A2 independently noted five verbs now build `Project` by copy-paste.

---

## 4. Silent wrong answers — *reproduced, and the one I would block on*

E1's gap #3. `isin([-1, 6])` against an `int16` column builds an `int64` value
set and returns **zero rows** instead of raising. Same shape for
`binary_col != lit("")`.

Every other gap in this document either errors or crashes. This one returns a
plausible wrong number, which a user cannot detect. It is small to fix and it is
the item I would treat as release-blocking ahead of any missing feature.

---

## 5. One root cause behind four crashes — *reproduced*

`arrays.mojo:2572` (`DynArray.as_type[T]` → `self._v[T]`) aborts the **process**
on a wrong-variant downcast; the `debug_assert` above it does not fire in
release. Four distinct symptoms trace to it:

- grouping by a `binary` key above 200,000 rows (my bisection: 50K binary ok,
  200K binary aborts, 200K string ok — the boundary is `_PARALLEL_ALWAYS_ROWS`)
- ClickBench Q11, Q12 — a *filter* over a multi-morsel scan with a binary column
- ClickBench Q24 — `SELECT *` + `ORDER BY` + `LIMIT`

E1 ranks fixing it as gap #1 and notes it is partly nondeterministic, so a race
or use-after-free rather than a pure dispatch mistake. Being fixed on its own
branch. **This is the difference between 39/43 and 42/43.**

A general form worth keeping (B1 §1.1): `add_type[T]` rejects any struct holding
a function pointer, because deriving `Writable` reflects over every field. So
**any marrow struct that grows a function-pointer field becomes silently
un-bindable**. `AggFunc._grouped_fn` and all three `AggFold` fields are already
in that position.

---

## 6. Smaller, concrete

- `fill_null(a,b)` and `coalesce(a,b)` are the same computation 115 lines apart
  (A1 §3, D1). They differ only in a cast that lives in the `DynValue`
  evaluator, not the kernel, undocumented.
- `Expr` has no `alias()` — only `Agg` does — so expressions cannot carry an
  output name and every projecting verb takes names out of band. This costs the
  polars/ibis `with_columns((expr).alias("x"))` spelling entirely (D1).
- All plan errors flatten to bare `Exception`; the message strings are becoming
  the API (D1).
- `is_nan` on an int64 column fails at *execution* with `dispatch_floating:
  dtype is not floating` — naming the dispatcher, not the operation, column or
  dtype (D1).
- The parity suite silently under-tested `coalesce`/`nullif` behind an expired
  `assert_fused` placeholder; A1 upgraded both, and the remaining `assert_fused`
  sites deserve an audit (A1 §6).
- The lazy frontend cannot request parallel execution — `execute()` is bound with
  a default `ExecContext` while the eager surface spells it `num_threads=` (B2).

---

## Recommended order

1. **`arrays.mojo:2572`** — one root cause, four crashes, 39→42 on ClickBench.
2. **Dtype-mismatch raising** (§4) — the only gap that returns wrong numbers.
3. **Delete the second `aggregate` overload** (§1.4) — already proven green.
4. **`project` copies the source field when `bound_column >= 0`** (§3) — closes a
   fidelity bug at the root and collapses three verbs.
5. **Collapse `DynAgg` into `AggExpr`** (§1.1) — removes a rule written three
   times; prerequisite for giving `COUNT(*)` an honest representation.
6. **`Relation.inputs()`** (§2) — unblocks EXPLAIN and a real rewrite pass.

1 and 2 are alpha work. 3–6 are the shape of the next cycle, and 3 is free.

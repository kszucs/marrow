# D1 — closing the deferred binding delta: what the gaps were made of

Written while binding the five `DynValue` null methods, `count_star`, and
`with_columns`/`drop`/`rename`, and while adding
`DynRelation.select(List[String])`. Prior findings in `b1-expr-bindings.md`
and `b2-plan-bindings.md`; this confirms three of theirs, refutes nothing, and
adds five that only became visible once the delta was actually bound.

---

## 1. Confirmed: `select` variadic-only was a *correctness* bug, not a detour

B2 §4 recorded that `DynRelation.select(*names: String)` cannot be called from a
runtime frontend and that `plan.mojo` routed through `project` instead. It
called the two "the same node". They are not, and the difference is user-visible
from Python today:

```python
# a Parquet file with pa.field("a", pa.int64(), nullable=False)
pa.schema(t.select("a").schema).field("a").nullable   # False — correct
pa.schema(t.project(a=t["a"]).schema).field("a").nullable  # True — widened
```

`select` copies the source `Field`; `project` probes the expression's dtype and
builds `Field(name, dtype)` around it, which defaults `nullable=True` and drops
metadata. A frontend forced onto `project` silently launders every non-nullable
column in the schema. Fixed by adding the `List[String]` overload.

**The general form is more interesting than the fix.** `select`, `drop`,
`rename` and `with_columns` all lower to one `Project` node and differ *only* in
how the output `Field` list is computed — copy the input field, or probe the
expression. That is ~120 lines of near-identical loop across four methods:

| verb | names | fields |
|---|---|---|
| `select` | given | copied from input |
| `drop` | complement | copied from input |
| `rename` | input, renamed | copied, `name` swapped |
| `with_columns` | merged | copied for pass-through, probed for the rest |
| `project` | given | **always probed** |

`with_columns` is the one that already does both, and it decides per column by
asking "is this name in the input schema". `project` could decide the same thing
more precisely with machinery the file already has: `BoxedValue.bound_column
(schema)` returns the column position of a bare column reference and `-1`
otherwise — `aggregate` uses it for exactly this purpose ("a bare column keeps
its own name"). If `project` copied the source field whenever
`values[i].bound_column(schema) >= 0`, then:

- `project` stops laundering pass-throughs, which removes the `select`/`project`
  fidelity difference at its root rather than at one call site;
- `select`, `drop` and `rename` collapse to name-list arithmetic delegating to
  `project`, and the four field-building loops become one.

I did not do it: it changes `project`'s output schema for existing callers
(strictly more faithful, but a behaviour change) and `relations.mojo` was the
only file I was allowed to touch, for one overload. It is the single largest
simplification available in that file.

## 2. New: `COUNT(*)` has no representation — it is encoded as a trick

`count_star()` is `lit[Int64Type](1).count().alias("count_star")`, and the
comment is right that it needs no new kernel. But the *plan* keeps no record
that this is `COUNT(*)`:

```python
count_star().function()      # "count"  — not "count_star"
count_star().input().render()  # "literal"
count_star().render()        # "count(literal) as count_star"
```

The only thing distinguishing `COUNT(*)` from a genuine `COUNT(1)` is the
default alias, which the first `.alias("n")` erases. So a plan cannot be
inspected for it, an optimiser cannot rewrite it (the classic `COUNT(*)` →
row-count-from-metadata rewrite over a Parquet scan is unavailable), and the
ClickBench queries that want it pay a full literal-column materialisation plus a
validity scan per morsel to compute a number the row group header already
carries.

This is a consequence of the shape B1 §2.2 and B2 §5a already flagged: `DynAgg`
is `(func: String, input: DynValue, out_name: String)` and the input is
*mandatory*. There is no "aggregate with no operand", so the nullary aggregate
has to lie about having one. A `func == "count_star"` value with a null input,
or an `Optional[DynValue] input`, would make it representable — and would then
be the natural place to hang the metadata rewrite.

I chose `marrow.count_star()` (a free function returning an `Aggregate`) over a
`("count_star",)` tuple spelling deliberately: the tuple form would have needed
a new arity in `_agg`'s marshalling *and* in `_aggregate_spec`, to express
something the existing `Aggregate` path already carries end to end. The free
function reuses the whole path — keyword rename, positional alias, `.alias()` —
and adds one binding.

## 3. New: `fill_null(a, b)` and `coalesce(a, b)` are the same computation

Both are bound now because both exist on `DynValue`, and both were tested. They
return identical results, and the kernels explain why:
`CoalesceKernel.apply` picks the first valid of N candidates;
`FillNullKernel.apply` picks candidate 0 if valid else candidate 1. At N=2 those
are the same loop written twice, in the same file
(`marrow/kernels/conditional.mojo`), 115 lines apart.

They are not perfectly interchangeable — `fill_null` casts its fill operand
(`col("n").fill_null(lit("x"))` on an int column raises *`cast: cannot parse 'x'
as int64`*) while `coalesce` requires matching dtypes — but that difference is
in the `DynValue` evaluator, not the kernel, and it is undocumented. Either
`FillNullKernel` should be `CoalesceKernel` with N=2, or the coercion difference
should be the stated reason the two exist.

## 4. New: `Expr` cannot be named, so the Python projection surface is
   keyword-only

`Agg` has `alias(name)`. `Expr` has nothing equivalent — `name()` is a *reader*
that returns `""` for anything but a bare column. So an expression cannot carry
its own output name, and every projecting verb has to take names out of band:

```python
t.with_columns(total=t["qty"] * t["price"])          # marrow — keyword-only
t.with_columns((pl.col("qty") * pl.col("price")).alias("total"))   # polars
```

The plan's `with_columns(names, values)` has the same shape, so this is not a
binding artefact. It costs the polars/ibis positional spelling entirely, and it
means `project`/`with_columns` cannot preserve argument order on Python < 3.7
semantics grounds (they do, via `**kwargs` insertion order, but the surface
depends on it). `DynValue.alias(String) -> Self` storing an optional out-name in
the existing `_payload` slot would close it, and `AggExpr` already proves the
plan layer knows how to read one.

## 5. New: `explain()` was returning the plan's `repr`, not the plan

B1 §1.2 documented that `def_method` fills `tp_dict` and not the CPython slots,
using `__str__` as the example. The plan bindings then tripped over it in a way
nothing caught, because every assertion was a substring test:

```python
str(plan_binding)         # '<marrow.Plan: InMemoryTable(...)>'   ← derived repr
plan_binding.__str__()    # 'InMemoryTable(...)'                  ← the method
```

`LazyTable.explain()`, `__str__` and `__repr__` all went through `str(...)`, so
`explain()` returned a bracketed repr and `repr()` returned
`LazyTable\n<marrow.Plan: ...>`. Fixed here by calling `.__str__()` explicitly
and asserting on the *prefix* rather than a substring.

**The generalisable point:** every bound type in `python/bindings/` that
registers `__str__` has this defect latent, and a substring assertion cannot
detect it. `_Wrapper` should own a `_text()` helper that calls the binding's
`__str__` method rather than the builtin, so no wrapper has to remember.

## 6. New: the binding layer flattens every plan error to bare `Exception`

Routing `LazyTable.drop` through the real plan node changed its error type:
Python's emulation raised `KeyError("drop: no such column(s): ['nope']")`, the
Mojo node raises `Error("drop: column 'nope' not found")`, which surfaces as
`Exception`. I kept the Mojo error — one source of truth, and it matches every
other unknown-column path on this surface — but the surface as a whole now has
no typed errors at all: an unknown column, a dtype mismatch, a bad join
strictness and an out-of-range index are all `Exception`, distinguishable only
by message text. Tests here have to `pytest.raises(Exception, match=...)`.

Mojo's `Error` carries only a string, so the mapping has to live in the binding
layer — a small `_raise_as(KeyError, ...)` dispatch on message prefix would be
ugly; an error-code field on `Error` would not. Worth deciding before the
surface grows, because the message strings are becoming API.

## 7. Smaller notes

- **`is_nan` / `is_inf` raise from the dispatcher, naming the dispatcher.**
  `col("n").is_nan()` on an int64 column fails with *`dispatch_floating: dtype is
  not floating`* — no mention of `is_nan`, the column, or its actual dtype. The
  failure is at execution, not plan build, even though the plan-build dtype
  probe (`RecordBatch.empty(schema)`) would have caught it: `project`/`aggregate`
  probe, `filter` does not, so where you get the error depends on which verb you
  wrapped it in.
- **`is_valid` renders as `not_null(a)`.** The method name comes from Arrow, the
  rendered name from `NotNullKernel.name`, and nothing reconciles them. Every
  other bound method round-trips (`is_null` → `is_null`, `fill_null` →
  `fill_null`), so a caller reading `render()` output reasonably assumes it is
  the method name. This is the same naming drift B1 §3.2 catalogued, now with a
  concrete instance in the null family.
- **`with_columns` is the verb that makes the plan layer usable, and it was the
  last one bound.** Before it, adding one derived column to the 105-column
  ClickBench table meant `project(**{c: t[c] for c in t.column_names}, new=...)`
  — which, per §1, would also have laundered all 105 fields to nullable.
- **`marrow/expr/__init__.mojo` re-exports `col`, `lit`, `if_else` but not
  `count_star`, `coalesce` or `case_when`.** The binding imports from
  `marrow.expr.builders` directly, as `plan.mojo` already did for `col`. Not a
  problem, but the `__init__` export list is now a partial view of
  `builders.mojo` rather than its public surface.

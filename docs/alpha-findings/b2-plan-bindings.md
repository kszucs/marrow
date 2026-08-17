# B2 — binding the plan layer: what the abstraction cost

Findings from binding `DynRelation` to Python (`python/bindings/plan.mojo`) and
writing the lazy frontend (`python/marrow/expr.py`). Binding is a good probe: it
can only use the *public* surface, from *runtime* values, so anything that only
works from comptime Mojo shows up immediately.

Ordered by how much they cost, worst first.

---

## 1. The plan IR is not walkable — `Relation` has no `inputs()`

`Relation` declares `with_predicate` / `schema` / `to_processor` / `write_to` /
`kind`. There is **no `inputs()`**. An earlier version of this trait had both
`inputs()` and `exprs()`; they are gone.

The consequence is larger than it looks: **nothing outside a node can traverse a
plan.** No EXPLAIN, no cost model, no rewrite framework, no "which columns does
this plan touch" pass — not from Python, and not from Mojo either. Every
optimisation has to be hand-inlined into the verb that builds the node, which is
exactly what has happened:

- Predicate pushdown lives in `DynRelation.filter`, which calls a **virtual
  `with_predicate` on every node type** that returns
  `Optional[ArcPointer[NoneType]]` — an erased pointer the caller then swaps into
  a copy of itself while keeping its own trampolines. The docstring explains that
  returning `Optional[DynRelation]` makes the struct recursive, which is true.
  But the reason a *rewrite* has to be expressed as a per-node virtual returning
  an erased pointer at all is that there is no tree walk to write it as.
- The top-K fold in `.limit()` works by `self.kind() == RELATION_SORT` then
  `downcast[Sort]()`. That is RTTI-by-enum, and it only works because `Sort` has
  no comptime parameters — the same trick is explicitly *not* available for
  `ParquetScan`, whose docstring records that a downcast silently rebuilt a
  narrow-`LeafSet` scan as the full-ladder one.

So the IR has two incompatible rewrite mechanisms (a kind tag + downcast, and a
virtual returning an erased pointer), one of which is already documented as
having been a correctness bug, and neither generalises. **Adding `inputs()` back
is the single highest-value change to this layer.** With it, pushdown and the
top-K fold both become ordinary bottom-up rewrites and `with_predicate` can go.

## 2. `write_to` renders one node, never its children — there is no EXPLAIN

Every `write_to` in `relations.mojo` prints its own node and stops. `Filter`
prints its predicate; `Sort` prints its keys; none of them print `self.input`.
So the whole plan renders as a single label:

```
>>> t.aggregate(by=["k"], total=("sum","v")).order_by(("total","descending")).head(2)
LazyTable
<marrow.Plan: Sort(keys=[total], limit=2)>
```

That is the *top node only*. The scan, the aggregate and the limit fold are
invisible. The task brief hoped a plan repr would be "effectively a free
EXPLAIN"; it is not, and the Python repr is currently misleadingly shallow —
it looks like a complete answer.

Fix is small and there are two ways: give each `write_to` a recursive call into
its input (a line per node), or add `inputs()` (§1) and write one indented
renderer in the binding. The second is better and subsumes this.

## 3. `DynRelation` cannot be registered as a Python type

`mb.add_type[DynRelation]("Plan")` does not compile:

```
constraint failed: Could not derive Writable for DynRelation —
member field `_virt_with_predicate` does not implement Writable
```

`add_type` wants a `write_repr_to`. `DynRelation` declares only `write_to`, so
the compiler tries to *derive* the missing one, and derivation walks the
struct's fields — straight into the function-pointer trampolines. **Every
`Dyn*`/erased box in the tree has this property**, so this will recur for
`DynValue`, `DynAgg`, `DynProcessor`, anything else someone tries to bind.

Worked around with a one-field `Plan` wrapper struct in `plan.mojo` that
forwards `write_to` and supplies `write_repr_to`. The real fix is two lines on
`DynRelation` (`def write_repr_to[W: Writer](self, mut writer: W): self.write_to(writer)`),
exactly as `RecordBatch` and `Table` already do. Worth doing tree-wide for the
erased boxes rather than growing a wrapper per binding.

## 4. `select` is variadic-only, so a dynamic frontend cannot call it

`DynRelation.select(self, *names: String)`. A runtime `List[String]` cannot be
splatted into a Mojo variadic (CLAUDE.md records the same limit for
`VariadicPack` forwarding). Every dynamic caller — the Python bindings, a SQL
parser, anything reading a projection off the wire — is locked out of the one
verb that exists for exactly that job.

Routed through `project(names, values)` with `col(name)` values instead. **The
two are not quite equivalent**, which is a small latent defect:

- `select` does `fields.append(schema.fields[idx].copy())` — the source field,
  so `nullable` and field metadata survive.
- `project` does `Field(names[i], values[i].execute(probe).dtype())` — a fresh
  field, so `nullable` defaults to `True` and metadata is dropped.

So projecting a non-nullable column by name widens it to nullable depending on
which verb you used. Either `select` should take a `List[String]` overload, or
`project` should preserve the source field when the expression is a bare column.

## 5. The aggregate cluster is five carrier types for three pieces of data

This is the part the brief flagged, and it is the right suspect. Trace
`col("x").sum().alias("total")` to a running aggregate:

| Type | Holds |
|---|---|
| `DynAgg` | `func: String`, `input: DynValue`, `out_name: String` |
| `AggExpr` | `out_name`, `input: BoxedValue`, `_unresolved: Optional[DynValue]`, `_func: String`, `_of: Optional[fn]` |
| `AggFunc` | `name`, `out_dtype`, `is_mergeable`, `_grouped_fn` |
| `AggFold` | `_whole_fn`, `_partials_fn`, `_merge_fn` |
| `FoldedAggregates` | `List[AggFunc]` + `List[AggFold]` |

plus the `AggFunction`/`Aggregation` traits, four implementations
(`NumericFold[K]`, `OrderPreserving[Op]`, `CountValid`, `DistinctCount[exact]`)
and the `resolve_agg` name switch. That is a lot of machinery for what a user
wrote as *(function name, input expression, output name)*.

Several of the individual splits are well argued in their docstrings, with
binary-size measurements attached (`AggFold` split out of `AggFunc` = +3.2 MB
avoided; resolving the fold eagerly = +1.2x avoided). Those I would not touch.
The problems below are not about the splits.

### 5a. `AggExpr` is a `Variant` hand-rolled as three unenforced `Optional`s

`_unresolved: Optional[DynValue]`, `_func: String`, `_of: Optional[fn]` encode a
two-case sum type:

- fused lane: `_of` set, `_func` empty, `_unresolved` None
- dynamic lane: `_of` None, `_func` set, `_unresolved` set

Nothing enforces it. `resolve()` is `if self._of: ... else: AggFunc(self._func, dtype)`,
so an `AggExpr` with **neither** set is constructible and fails late and
confusingly — `AggFunc("", dtype)` reaches `resolve_agg` and raises
`unknown aggregate function: ` with an empty name. A `Variant` of the two
payloads would make the invalid state unrepresentable and delete a field.

### 5b. The dynamic lane stores its input expression twice

```mojo
self.input = agg.input.copy()
self._unresolved = agg.input.copy()
```

Both copies of the same `DynValue`. `input_for()` prefers `_unresolved` when
present, so on the dynamic path **`input` is written, never read** except by
`write_to`. Two allocations per aggregate for one expression.

### 5c. The empty-alias rule is implemented in two places

`DynAgg` treats an empty `out_name` as "use the function name", and
`AggExpr.__init__(var agg: DynAgg)` re-implements the same rule
(`if not out_name: out_name = agg.func.copy()`). Two owners for one default.

### 5d. `DynRelation` has two `aggregate` overloads, and the fused one regresses to parallel lists

- `aggregate(keys: List[BoxedValue], aggs: List[AggExpr])`
- `aggregate(keys, inputs: List[BoxedValue], aggs: List[AggFunc], names: List[String])`

The second takes **four parallel lists** with hand-written length invariants
(`len(names) != len(keys) + len(aggs)` raises; `len(inputs) != len(aggs)` raises).
Keeping N lists positionally aligned at runtime is precisely the failure mode
`AggExpr` was introduced to remove — the first overload's docstring says so:
"nothing has to be kept in positional correspondence".

The fused lane does not need this. `AggExpr.of[A](input)` already exists and is
already what `Reduction.alias` calls, so a fused caller can build
`List[AggExpr]` and use the first overload. I would delete the four-list
overload.

### 5e. `is_mergeable` is on the box that cannot merge

`AggFunc.is_mergeable: Bool` says an aggregate *can* run as partials + merge,
but `AggFunc` does not carry the fold that does it — `AggFold` does, separately.
So holding an `AggFunc` you can learn the capability exists and still not invoke
it, and `FoldedAggregates` exists mainly to re-pair the two by index across
`List[AggFunc]` and `List[AggFold]` (parallel lists again). An
`Optional[AggFold]` on `AggFunc` would state the flag and the capability once —
though note this trades against the measured size win of keeping `AggFunc`
narrow, so it needs a gate run rather than a straight swap.

### 5f. Aggregate function names are unvalidated until plan build

`DynAgg(func, input, out_name)` takes the name as a bare positional `String` and
nothing checks it. `t.aggregate(total=("sumn", "v"))` builds fine and fails at
plan-build with `unknown aggregate function: sumn`. Acceptable for now (it is
still before execution), but the name is public API with no validation point and
no discoverable list of valid values — `resolve_agg`'s eight branches are the
only enumeration and they are not exported.

## 6. The lazy frontend cannot request parallel execution

`DynRelation.execute(ctx: ExecContext = ExecContext())` accepts a context, but
`ExecContext` is only reachable in the bindings as a registered type used by the
*eager* compute surface (which spells it `num_threads=`). I bound `execute()`
with the default rather than invent a third spelling. Result: **a lazy query
currently runs with the default context and there is no way to ask for more
threads.** Worth closing before alpha, ideally by settling on one spelling
(`num_threads=` everywhere) rather than exposing `ExecContext` twice.

## 7. Smaller notes

- `Schema` is registered with exactly one method (`__arrow_c_schema__`), so the
  Python side cannot read its own column names without importing pyarrow. Added
  `Plan.column_names()` in the binding to avoid that dependency on a core path;
  `Schema.names()` would be the better home.
- `DynRelation` has no `to_python_object()`, unlike `RecordBatch`/`Schema`/
  `Table`. Correct — `marrow.expr` should not depend on Python — but it means
  the allocation helper has to live in the binding. Fine as is; noting so nobody
  "fixes" it by adding a Python dependency to the expr layer.
- Accepting a plain `str` wherever a column reference is wanted (sort keys, join
  keys, select names) removed the lazy frontend's dependency on the expression
  bindings for everything except `filter` and computed projections. That was
  worth doing on its own merits, but it is also the reason this work could be
  tested end-to-end while the expression bindings were still unmerged.

---

## Recommendation on the `Table` naming collision

`marrow.Table` is already the eager, PyArrow-shaped table, and CLAUDE.md makes
PyArrow parity a project rule ("method names, signatures and defaults ... should
match, so PyArrow muscle memory carries over"). Taking that name for the lazy
type would break the rule for the sake of the newer feature.

**Implemented:** eager `marrow.Table` unchanged; lazy type is `marrow.LazyTable`,
reached through `marrow.read_parquet(path)` and `marrow.scan(batch)`.

**The class names are not where users will trip.** Almost nobody writes
`LazyTable(...)` — they call an entry point. The real asymmetry is:

```python
marrow.table(data)     # eager
marrow.scan(batch)     # lazy
marrow.read_parquet(p) # lazy
```

Two verbs that differ in evaluation strategy but not obviously in name. Two
options, in preference order:

1. **Rename `scan` to `memtable`** (ibis's name for exactly this) and keep
   `read_parquet`. Then the lazy entry points are `memtable`/`read_parquet` —
   both ibis spellings — and the eager ones are `table`/`record_batch` — both
   PyArrow spellings. Each namespace is internally consistent and a reader can
   tell which world they are in from the verb. This is my recommendation.
2. Put the lazy surface behind a submodule — `marrow.lazy.table(...)`,
   `marrow.lazy.read_parquet(...)`. Unambiguous, but noisier at every call site
   and it hides the headline feature one level down.

I would avoid `LazyFrame` (polars): it imports a *frame* vocabulary the rest of
this codebase does not use, where `Relation`/`RecordBatch`/`Table` are already
consistent.

Whatever is chosen, `read_parquet` should stay — pandas, polars and ibis all
agree on it, and it is the entry point most users will hit first.

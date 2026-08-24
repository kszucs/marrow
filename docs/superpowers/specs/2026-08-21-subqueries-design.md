# Subqueries — design

**Status:** design, not started. **Date:** 2026-08-21.

## Goal

Let a marrow plan *express* a subquery, so a query reads the way SQL and ibis
write it:

```mojo
rel.filter(is_in(col("dept", int64), dept_ids))          # dept IN (SELECT did FROM dept)
rel.filter(not_in(col("dept", int64), dept_ids))         # dept NOT IN (SELECT did FROM dept)
rel.filter(col("v", int64) > scalar_value(avg_v, int64)) # v > (SELECT avg(v) FROM t)
```

Today none of these can be written. `marrow/exprold/relations.mojo` has
`InMemoryTable / ParquetScan / Filter / Project / Limit / Sort / Aggregate /
Join`, and a `Join` already executes semi and anti correctly — but there is no
node, no expression and no builder that takes a *relation* as a predicate's
operand.

Exactly one of the four forms is a missing **capability**; the rest are missing
**spellings**. That distinction drives the whole plan below.

| SQL form | what it needs | status |
|---|---|---|
| `x IN (subquery)` | a semi join | **sugar** — executes correctly today, spelled as `.join(..., how=JOIN_SEMI)` |
| `EXISTS (subquery)` | a semi join | **sugar** |
| `NOT EXISTS (subquery)` | an anti join | **sugar** |
| `x NOT IN (subquery)` | a *null-aware* anti join | **missing capability** |
| `x > (SELECT agg(...))` | run a subplan to one scalar | **missing capability** |

`x NOT IN (S)` is **not** an anti join. If `S` yields any NULL the predicate is
never TRUE, so the query returns no rows at all; and a NULL `x` is excluded. An
anti join keeps both. `golden/cases/subquery_not_exists.mojo` already documents
the difference in prose — `eid 5` (NULL `dept`) is the row that separates them —
and this design is what turns that prose into a testable node.

## Non-goals

- **Not correlated subqueries.** See "Correlated subqueries are out of scope".
- **Not a SQL parser.** Both frontends stay programmatic, per
  `docs/backlog.md` §6 "Won't".
- **Not a subquery in a projection or in a compound predicate.** A subquery
  under `OR`/`NOT`, or in a `SELECT` list, needs a MARK join. Deferred; see
  "Why not a MARK join".
- **Not `ANY` / `ALL` with a non-equality comparator** (`x > ALL (…)`).
- **Not multi-column `NOT IN`.** Row-wise NULL semantics on a composite key are
  a separate problem; it raises a named error.

## Correlated subqueries are out of scope, and here is the cost

A correlated subquery references a column of the *enclosing* query inside the
subplan. Decorrelating one is a large feature in every engine that has it, and
it presupposes machinery marrow does not have:

- **A depth-aware name resolver.** DuckDB marks correlation while binding: a
  column that resolves in an *enclosing* binder becomes a
  `CorrelatedColumnInfo` with `depth > 0`
  (`src/planner/expression_binder.cpp:189-232`), and multi-level correlation is
  peeled one level at a time
  (`src/planner/binder/expression/bind_subquery_expression.cpp:102-111`).
  DataFusion does the same with `Expr::OuterReferenceColumn`
  (`datafusion/expr/src/expr.rs:413`), populated by pushing the outer schema on
  a stack during SQL planning
  (`datafusion/sql/src/expr/identifier.rs:92-103`).
  **Both mark correlation in the binder.** marrow has no binder — a plan is
  built by direct calls, and a column leaf is a name, not a resolved reference
  to a particular relation. There is nowhere for the mark to come from.
- **A dependent-join lowering.** DuckDB emits a `LogicalDependentJoin`
  placeholder and flattens the whole plan afterwards
  (`src/planner/subquery/flatten_dependent_join.cpp`, ~1100 lines), producing
  `DELIM_JOIN`/`DELIM_GET` pairs — the *magic set* `D` from Neumann & Kemper's
  *Unnesting Arbitrary Queries* — plus a `Deliminator` optimizer pass to remove
  the redundant ones (`src/optimizer/deliminator.cpp`), plus count-bug
  compensation (`rewrite_correlated_expressions.cpp:81-95`). DataFusion takes
  the cheaper classical route — predicate pull-up
  (`datafusion/optimizer/src/decorrelate.rs`, `PullUpCorrelatedExpr`) plus two
  rules — and still needs an `__always_true` sentinel column for the count bug
  (`decorrelate.rs:123-127`) and still refuses a long list of shapes
  (correlation under `Union`/`Sort`/`Limit`, non-equi correlation across an
  aggregate: `decorrelate.rs:147-172`, `:428-453`).

Cost estimate: **L**, comparable to the join or the aggregate layer, and it is
mostly *planner* code with no counterpart anywhere in marrow today. It stays
where `docs/backlog.md:912` already put it.

**But two of the four existing golden cases are written as correlated SQL**
(`subquery_exists.mojo`, `subquery_not_exists.mojo`: `EXISTS (SELECT 1 FROM
dept d WHERE d.did = e.dept)`), and they execute today as joins. The resolution
is to **spell the correlation explicitly** rather than infer it:

```mojo
rel.filter(exists(dept, left_on=col("dept", int64), right_on=col("did", int64)))
```

This is honest: the user supplies the join key that a decorrelator would have
derived. It covers the "correlation *is* the join condition" shape, which is
the overwhelming majority of real `EXISTS`/`IN` subqueries, and it needs no
decorrelation at all. Arbitrary correlated predicates remain unexpressible.

**Open question (naming).** Calling it `exists(...)` when the user hands over
the join key is arguably a lie — it *is* a semi join with a SQL-shaped name.
The alternative is to keep `semi`/`anti` spellings and reserve `exists` for a
future decorrelating planner. I lean toward `exists`/`not_exists` because the
golden case's SQL says `EXISTS` and the case body should echo it, but I do not
consider this settled.

## The surface

### One predicate family, four kinds

```mojo
struct SubqueryPredicate:          # in marrow/exprold/relations.mojo
    var kind: UInt8                # IN | NOT_IN | EXISTS | NOT_EXISTS
    var subplan: DynRelation
    var left_key: BoxedValue
    var right_key: BoxedValue
    def __invert__(self) -> Self   # IN <-> NOT_IN, EXISTS <-> NOT_EXISTS
```

Constructors, as free functions in `relations.mojo`:

```mojo
def is_in(var needle: BoxedValue, sub: DynRelation) raises -> SubqueryPredicate
def not_in(var needle: BoxedValue, sub: DynRelation) raises -> SubqueryPredicate
def exists(sub: DynRelation, *, var left_on: BoxedValue,
           var right_on: BoxedValue) raises -> SubqueryPredicate
def not_exists(sub: DynRelation, *, var left_on: BoxedValue,
               var right_on: BoxedValue) raises -> SubqueryPredicate
```

`is_in`/`not_in` require `sub` to have **exactly one column** and default
`right_key` to it — the same invariant ibis enforces in
`InSubquery.__init__` (`ibis/expr/operations/subqueries.py:74-78`,
`IntegrityError: "Relation passed to InSubquery() must have exactly one
column"`) and DuckDB in
`bind_subquery_expression.cpp:149-152` (`"Subquery returns %zu columns -
expected %d"`). So `is_in(x, sub)` is exactly `exists(sub, left_on=x,
right_on=<sub's only column>)`; one lowering serves all four.

### `filter` gains an overload

```mojo
def filter(self, var predicate: BoxedValue) raises -> DynRelation         # today
def filter(self, var predicate: SubqueryPredicate) raises -> DynRelation  # new
```

Overload resolution is static. `SubqueryPredicate` deliberately does **not**
conform to `Value`, so `BoxedValue`'s `@implicit` constructor cannot convert it
and there is no ambiguity.

### Both lanes, concretely

The AOT lane:

```mojo
var dept_ids = in_memory_table(dept_batch).select("did")
var q = in_memory_table(emp_batch)
    .filter(not_in(col("dept", int64), dept_ids))
    .sort([col("eid", int64)], [True])
```

The runtime lane — identical, because `col("dept")` erases into the same
`BoxedValue`:

```mojo
var q = in_memory_table(emp_batch).filter(not_in(col("dept"), dept_ids))
```

The Python binding, ibis-shaped:

```python
dept_ids = t_dept.select("did")
t_emp.filter(t_emp["dept"].isin(dept_ids))
t_emp.filter(~t_emp["dept"].isin(dept_ids))
t_emp.filter(mr.exists(t_dept, left_on="dept", right_on="did"))
```

`LazyTable.filter` branches on `isinstance(predicate, SubqueryPredicate)` in
pure Python and calls a second binding method `filter_subquery`; the Mojo
binding file stays minimal and strict, per CLAUDE.md's Python-API rule.

**A correction to the premise.** ibis's `isin` does **not** accept a `Table` —
its signature is `values: ArrayValue | Column | Iterable[Value]`
(`ibis/expr/types/generic.py:592`), and the dispatch at `:682-689` builds
`InSubquery(values.as_table(), needle=self)` only from a **Column**. Handing it
a Table falls into the `InValues` branch and dies in `Expr.__iter__`
(`ibis/expr/types/core.py:92-93`). The idiomatic ibis spelling is
`t.filter(t.x.isin(t2.y))`. marrow has no free-standing `Column` object bound
to a relation, so taking the one-column relation and validating its width is
the faithful adaptation — it is what `Column.as_table()` produces anyway.

### Why `is_in(x, sub)` and not `x.isin(sub)` in Mojo

The method spelling requires `values.mojo` to name `DynRelation`, because a
method's return type must name `SubqueryPredicate`, which holds a
`DynRelation`, which lives in `relations.mojo`. That is a new
`values -> relations` import edge. Mojo resolves such cycles inside a package
(CLAUDE.md: *"do not reorganize code or move types between files to avoid
them"*), so it is legal — but `relations.mojo` transitively imports the Parquet
writer, the IPC writer and the join kernel, and no existing binary-size gate
measures a fused expression tree *without* the relational layer, so the DCE
claim cannot be checked cheaply.

**Decision:** ship free functions in `relations.mojo` (zero new edges) in
Phase 1. The `.isin(sub)` method spelling is purely additive sugar and can be
added later behind a `pixi run binary_size` measurement.

Note the constraint is Mojo-only. Python does runtime dispatch, so
`t["dept"].isin(sub)` works there natively from day one — the ibis-shaped
spelling is available in the frontend where it matters most.

## The crux: how `Relation.filter` detects a subquery predicate

**It does not detect anything. The subquery predicate never enters the erasure
boundary.**

A predicate cannot turn its parent `Filter` into a `Join`, so `filter` must see
the subquery. There are two ways to arrange that, and the one that inspects
through erasure is the expensive one.

### The rejected alternative: ask `BoxedValue`

The precedent exists and is good: `Value.bound_column(schema) -> Int` is a
defaulted trait method returning `-1`, one trampoline field on `BoxedValue`,
and it is how the relational layer asks "are you a bare column?" without
reaching into a representation it should not know about
(`marrow/exprold/values.mojo:421`, `:608`). The analogue would be
`Value.subquery() -> Optional[...]`, defaulting to `None`.

It has one hard constraint and one cost.

**The constraint — and the CLAUDE.md limit it steers around.** The trampoline's
field type may **not** mention `DynRelation`. `DynRelation` already mentions
`BoxedValue` in `_virt_with_predicate`
(`marrow/exprold/relations.mojo:430-432`), so a `BoxedValue` field whose function
type mentioned `DynRelation` closes a mutual cycle and Mojo rejects it —
*"struct has recursive reference to itself"*. That failure has been hit three
times in this tree and is documented at each site:
`Relation.with_predicate` returns an erased `ArcPointer[NoneType]` rather than
an `Optional[DynRelation]` (`relations.mojo:130-140`),
`BoxedValue._resolve_names_fn` does the same (`values.mojo:518-533`), and
`EvalFn` "deliberately mentions no `Self`" for the same reason
(`dynamic.mojo:251-254`). The fix is the same each time: return
`Optional[ArcPointer[NoneType]]` and `rebind` it at the one call site that
knows the concrete type.

**The cost.** One more `def(...) thin` field on every `BoxedValue` value, and
one more trampoline instantiation per boxed node type — across every binary
that links `BoxedValue`, which is all of them. It is a *closed* cost (the
rebind resolves to one concrete type; no open dispatcher), but it is paid by
plans that contain no subquery at all, and the binding constraint is
"keep `marrow.expr` small-binary".

**Aside on a limit that looks violated but is not.** CLAUDE.md records that *"a
struct method does not override a trait default; the two become competing
overloads and every call reports `ambiguous call to 'x'`"*. That rule bit
`isnull`/`notnull`, whose conformer wanted a **different return type** (`Self`
vs the concrete `NullPredicate`). Identical-signature overrides work today and
are used throughout: `DynValue` overrides `Value`'s `name`, `render`,
`referenced_columns`, `bound_column` and `prune`
(`dynamic.mojo:592`, `:598`, `:608`, `:624`, `:647`), and `NumericColumn`
overrides `bound_column` (`values.mojo:790`). So a defaulted `subquery()` with
one signature everywhere would compile. The reason to reject it is size, not
legality.

### The decision: keep it out of the box

`SubqueryPredicate` is **its own type**, accepted only by `filter`. There is
nothing to detect because there is nothing erased.

This is the same principle CLAUDE.md already states for the value nodes —
*"the box is the erasure boundary; a node never needs an erased variant"*, the
rule under which `DynColumn`/`DynLiteral`/`DynCast` were each added and removed
— applied one level up: a thing that only one caller in one position ever
accepts does not need to travel through the box.

It is also, structurally, what **ibis** does. `Subquery` is a `Value` carrying
a `Relation`, and it buys self-containment by having
`Subquery.relations` return the **empty set**
(`ibis/expr/operations/subqueries.py:22-24`), so the embedded relation is
invisible to `_check_integrity` (`ibis/expr/operations/relations.py:111-117`).
ibis needs the node to be a `Value` because ibis supports subqueries in
arbitrary expression positions; it then has to lie about the node's parents to
keep the relational algebra sound. marrow does not support arbitrary positions
(see "Why not a MARK join"), so it does not need the node in the algebra, and
so it does not need the lie.

### The costs of this decision, stated plainly

- **A subquery predicate cannot be combined with `&` or `|`.** `BoolValue`'s
  operators take a `BoolValue`; `SubqueryPredicate` is not one, so
  `x.isin(sub) & (y > 3)` is a **compile error**, not a runtime one. A
  conjunction is spelled as two filters — `rel.filter(y > lit(3)).filter(is_in(x, sub))`
  — which is what ibis's variadic `t.filter(a, b)` means anyway. A
  *disjunction* is not expressible, and that is the deferred MARK-join case.
- **`~` still works**, via `SubqueryPredicate.__invert__`, so
  `~t["dept"].isin(sub)` reads as ibis writes it
  (`ibis/expr/types/generic.py:780`: `notin` is literally `~self.isin(values)`).

## The lowering

`Relation.filter(p: SubqueryPredicate)` builds a `Join`, not a `Filter`:

| kind | join | why it is exact |
|---|---|---|
| `IN`, `EXISTS` | `JOIN_SEMI` | below |
| `NOT EXISTS` | `JOIN_ANTI` | below |
| `NOT IN` | `JOIN_ANTI_NULL_AWARE` | "`NOT IN` semantics" |

The output schema is unchanged, because `JoinKind.emits_right_columns()` is
False for the existence filters (`marrow/kernels/join.mojo:179-187`) — a semi
join's schema is the left's schema, which is exactly `Filter`'s.

**Why a plain semi join is exactly SQL `IN` in a `WHERE` clause.** SQL `IN`
yields TRUE only when a match exists and UNKNOWN otherwise — whether the
unknown comes from a NULL on the left or a NULL in the set. A `WHERE` clause
keeps only TRUE. A semi join emits exactly the left rows that have a match, and
a NULL key matches nothing. The two agree on every input. That is why `IN` is
sugar and `NOT IN` is not: under negation, UNKNOWN and FALSE stop being
interchangeable.

**Beware the value-level `isin`, which is a different operation.**
`IsIn`/`IsInKernel` (`marrow/exprold/values.mojo:2243`,
`marrow/kernels/membership.mojo:20-25`) implement PyArrow's default
`null_matching_behavior="match"`: the output is never null, and a NULL input is
**true** when the value set contains a NULL. Arrow C++ enumerates four modes and
names the SQL one explicitly — `SetLookupOptions::INCONCLUSIVE`, *"null values
are regarded as unknown values, which is sql-compatible"*
(`arrow/cpp/src/arrow/compute/api_scalar.h:278-300`); the truth table is
implemented at `scalar_set_lookup.cc:435-476`. Under `MATCH`, a NULL left value
against a NULL-containing set is TRUE, where SQL says UNKNOWN. So
`isin(value_set: DynArray)` and `is_in(needle, subquery)` differ on exactly one
cell of the truth table and must not be conflated. polars has the same split
and gets `NOT IN` wrong because of it: its `is_in` drops haystack nulls
(`polars/crates/polars-ops/src/series/ops/is_in.rs:12-49`), and its SQL
frontend lowers `NOT IN` as `is_in(...).not()`
(`polars/crates/polars-sql/src/sql_expr.rs:1192-1193`), so polars answers
`3 NOT IN (1, 2, NULL)` as TRUE where SQL says NULL.

**Open question.** Should `IsInKernel` gain an `INCONCLUSIVE` mode, switch to
one, or stay PyArrow-compatible and be documented as a deliberate divergence? I
lean toward documenting for now — the kernel's contract is PyArrow's and
changing it is a behaviour change outside this feature's scope — but a user who
reaches for `col.isin([...])` expecting SQL will be wrong, and a golden case
cannot pin the divergence because golden expectations come from DuckDB. A
`marrow/kernels/tests/test_membership.mojo` case should pin it instead.

## `NOT IN` semantics

### Where it lives: in the join kernel

`marrow/kernels/join.mojo`, as a **null-aware anti join**. The relational layer
only selects the kind; it computes nothing about nulls.

This follows DataFusion, which added exactly this in
`feat: Add null-aware anti join support (#19635)` (commit `4c67d0208`, closing
issue #10583). It is a `null_aware: bool` on `LeftAnti`
(`datafusion/expr/src/logical_plan/plan.rs:4247`) carried down to a
specialized `HashJoinExec` (`physical-plan/src/joins/hash_join/exec.rs:769`);
the 3VL lives in the stream (`hash_join/stream.rs:743-780`, `:986-1020`).
Before #19635 a plain `LeftAnti` gave wrong answers.

### The three rules

For `left.key NOT IN (right.key)`, where marrow builds the **left** side into
the hash table and probes with the **right**
(`marrow/exprold/execution.mojo:959-987`):

1. **Probe side empty** → emit every left row, **including** rows with a NULL
   key. `NULL NOT IN ()` is TRUE. DataFusion carves this out explicitly
   (`hash_join/stream.rs:986-1020`: *"if probe side is empty, `NULL NOT IN
   (empty)` = TRUE, so NULL rows should be returned"*). **This is the rule most
   easily got wrong** — it is the only case where a NULL left key survives.
2. **Otherwise, probe side contains a NULL key** → emit **nothing**.
3. **Otherwise** → emit left rows with a **non-NULL** key and no match.

Rules 2 and 3 are the collapse of DuckDB's three-valued MARK column into the
one consumer marrow supports. DuckDB computes
(`src/execution/join_hashtable.cpp:1882-1931`): probe-key NULL ⇒ NULL; match ⇒
TRUE else FALSE; then *"if the right side contains NULL values, the result of
any FALSE becomes NULL"*, gated on a build-side `has_null` bool set during
build (`join_hashtable.cpp:634-637`). `NOT` maps NULL to NULL and a filter
keeps only TRUE, so only mark = FALSE survives — which is precisely rule 3.

### Why it is nearly free

`JoinProcessor._blocks_on_probe_side` already collects the whole probe side
before emitting for `ANTI` (`marrow/exprold/execution.mojo:936-967`), so all three
rules are decidable at `_emit_unmatched` time. Rule 1 is `right_rows == 0`,
which is already a parameter. Rule 2 is `null_count() > 0` on the single probe
key column — an O(1) read of a cached field, computed once in
`probe_serial`/`probe_parallel` and passed down. Rule 3 adds one validity check
per left row inside the existing `JOIN_ANTI` loop
(`marrow/kernels/join.mojo:805-812`).

### How the kind is spelled

A new `comptime JOIN_ANTI_NULL_AWARE = JoinKind(7)` rather than a `Bool`
alongside `strictness`. `JoinKind`'s own docstring records why the vocabulary
is a type and not a bare `UInt8`: *"`kind` and `strictness` were both `UInt8`.
Passing them swapped compiled silently"* (`join.mojo:150-155`). A second
boolean parameter next to `strictness` reproduces that shape.

**The trap this creates, and the required mitigation.** The same docstring
records that *"the column question had four answers"* and that MARK once made a
join build a `StructArray` declaring the right side's fields while carrying only
the left's. `emits_right_columns()` is currently written as
`self != JOIN_SEMI and self != JOIN_ANTI` (`join.mojo:179-187`) and would
answer **True** for a new anti-like kind — i.e. a corrupt result array with
nothing checking. So the same commit must introduce

```mojo
def is_semi_anti(self) -> Bool:
    return self == JOIN_SEMI or self == JOIN_ANTI or self == JOIN_ANTI_NULL_AWARE

def emits_right_columns(self) -> Bool:
    return not self.is_semi_anti()
```

which is the predicate polars puts on the type and which the docstring already
cites as the reference shape. `is_supported()`, `write_to()`, `parse()` and
`JoinProcessor._blocks_on_probe_side` all gain the kind in that commit or the
result is silently wrong.

### Restrictions

- **Single-column key only.** `not_in` with a composite key raises. DataFusion
  enforces the same at physical planning
  (`hash_join/exec.rs:422-436`: *"null_aware anti join only supports single
  column join key"*), and DuckDB throws
  `"Correlated IN/ANY/ALL with multiple columns not yet supported"`
  (`plan_subquery.cpp:392`).
- **No `is_supported()` shortcut on non-nullable keys.** DataFusion skips
  null-awareness when both key columns are `NOT NULL`
  (`decorrelate_predicate_subquery.rs:462-489`, `join_keys_may_be_null`). marrow
  should not do this in Phase 2 — `Field.nullable` is carried but not enforced
  anywhere, so trusting it would be trusting an unchecked declaration. Worth
  revisiting once nullability is validated.

## The scalar subquery

`WHERE v > (SELECT avg(v) FROM t)` — the subplan runs to completion, yields one
value, and that value is broadcast into the predicate.

### The mechanism: it is already built

marrow already has a leaf that is *"a scalar bound after the plan is built,
resolved in `state()` rather than `lane()`, so the lane splats a plain `Scalar`
byte-identical to a literal's and a parameter costs nothing per row"* —
`NumericParam`/`StringParam`/`TemporalParam` (`marrow/exprold/values.mojo:863-894`),
backed by `ParamCell`, *"a shared, mutable box for one late-bound scalar value"*
(`marrow/exprold/params.mojo:93-129`). The runtime lane has the same thing by name
(`DynValue.param`, `dynamic.mojo:961-966`).

**A scalar subquery is a `ParamCell` whose binder is a subplan instead of
`argv`.** No new expression node in either lane; no new fused node, no new
`EvalFn`.

This is also what a mature engine converged on. DataFusion's
`ScalarSubqueryExec` evaluates uncorrelated scalar subqueries **eagerly at
physical execution**, writing each result into a shared container that a
`ScalarSubqueryExpr` reads by index
(`datafusion/physical-plan/src/scalar_subquery.rs:57-77`,
`physical-expr/src/scalar_subquery.rs:32-45`), and it is the **default** since
PR #21240 (`datafusion/common/src/config.rs:1416`). The container-plus-index is
`ArcPointer[ParamCell]`; the pass-through node that *"exists only to populate
scalar subquery results as a side effect before those batches are produced"* is
the relation node below. The config doc for turning it off is the argument for
turning it on: the older join-based rewrite *"silently produces incorrect
results for multi-row subqueries and does not support scalar subqueries in
ORDER BY / JOIN ON / aggregate-function arguments"* (`config.rs:1402-1416`).

Note this is also why marrow does **not** need `JOIN_CROSS`. DuckDB attaches an
uncorrelated scalar subquery with `LogicalCrossProduct::Create` and carries a
standing `FIXME: should use something else besides cross product`
(`plan_subquery.cpp:180-183`); DataFusion uses `Left Join ... ON true`
(`scalar_subquery_to_join.rs:375-381`). Binding a cell needs neither.

### The node

```mojo
struct ScalarSubquery(Relation):    # in marrow/exprold/relations.mojo
    var input: DynRelation
    var subplan: DynRelation
    var cell: ArcPointer[ParamCell]
    var name: String
```

- `schema()` — `self.input.schema()`, unchanged.
- `to_processor(ctx)` — **runs the subplan, binds the cell, then returns
  `self.input.to_processor(ctx)`**. It contributes no `Processor` of its own,
  so it costs nothing per morsel.
- `with_predicate` / `with_projection` — **must forward to `input` and rebuild
  self**, exactly as `Filter.with_projection` does
  (`relations.mojo:1374-1385`). The trait defaults return `None`, which would
  silently sever predicate pushdown into a `ParquetScan` below — disabling
  row-group pruning for precisely the query shape (`WHERE ts > (SELECT …)`)
  that wants it most. This is the easiest thing to get wrong in the whole
  design.

**Ordering is correct by construction.** `execute()` is
`self.optimize().to_processor(ctx)` then `collect()`
(`relations.mojo:589`), and `prune` *"runs inside `to_processor`/`collect`,
after `parse_params` has bound it"* (`dynamic.mojo:673-677`). `ScalarSubquery`
sits above the `Filter`/`Scan`, so its `to_processor` binds the cell before any
descendant's `to_processor` reads it.

### Cardinality and null rules

| subplan yields | result | where |
|---|---|---|
| 0 rows | a typed NULL of the declared dtype | `to_processor` |
| exactly 1 row | that value | `to_processor` |
| > 1 rows | raise `"scalar subquery returned more than one row"` | `to_processor` |
| ≠ 1 column | raise, at **plan-build** time | `with_scalar` |

The `> 1` check is one `num_rows()` read, so it is unconditional. DuckDB makes
it optional behind `scalar_subquery_error_on_multiple_rows` (default true) and
skips it when the plan provably returns one row — an ungrouped aggregate,
`DUMMY_SCAN`, or `LIMIT 1` (`PlanReturnsExactlyOneRow`,
`plan_subquery.cpp:31-55`); DataFusion always checks
(`physical-plan/src/scalar_subquery.rs:287-320`). marrow should always check:
it is O(1), and marrow's keyless `Aggregate` is *not* statically single-row —
it returns `RecordBatch.empty` when the input produced no morsels
(`execution.mojo:839-840`), which is exactly the 0-row case above.

ibis, by contrast, does not check at all and says so:
*"If the table has more than one row an error will be raised by the backend"*
(`ibis/expr/types/relations.py:674-677`). marrow *is* the backend.

### The surface, and its weakest point

```mojo
var avg_v = ScalarSubquery.of(
    nums.aggregate([col("v", int64).mean().alias("m")]), "avg_v", float64)

var q = nums.with_scalar(avg_v)
            .filter(col("v", int64) > scalar_value(avg_v, float64))
```

`scalar_value(ref, dtype)` is an overload set mirroring `param`'s three
overloads (`builders.mojo:93`, `:128`, `:153`) and returning the same three
leaves; `scalar_value(ref)` with no dtype returns a runtime-lane `DynValue`
bound to the same cell by name.

**The handle is mentioned twice, and that is the honest cost of having no
compiler pass.** ibis gets one mention because `filter_wrap_reduction`
rewrites a bare reduction in a predicate into a `ScalarSubquery` automatically
(`ibis/expr/rewrites.py:259-277`), which is a *detection* pass over the
predicate tree — the thing this design deliberately refuses.

Alternatives considered and rejected:

- **A module-level pending registry drained by `execute()`**, mirroring
  `params.mojo`'s `_REGISTRY`. One mention, but it makes the binding
  process-global, and `params.mojo`'s own docstring records how carefully the
  drain/reset schedule had to be tuned to keep two plans in one process from
  resolving each other's names (`params.mojo:24-43`). It also weakens the
  standing property that a plan is *"a pure description, executed repeatedly
  and concurrently"* (`relations.mojo:585-588`).
- **Eager evaluation at build time** — `sub.execute()` and splice a literal.
  Semantically fine for an uncorrelated subquery over immutable sources, and a
  user can already do it today with no new code. But it breaks the AOT/compiled
  path (`execute_cli`, `benchmarks/binary_size/query_param.mojo`), where the
  plan is built before the data path is bound. Worth documenting as a
  workaround; not the design.

**Python gets the one-mention spelling for free.** Because `Column` in
`python/marrow/_expr_column.py` is a Python object, a `ScalarRef` used in a
comparison can record itself on the resulting `Column`, and `LazyTable.filter`
can drain that set and insert the `with_scalar` node itself — pure Python,
zero Mojo cost, and detection is legal there because Python already does
runtime type dispatch. The ergonomic gap is Mojo-only.

**Open question.** Is the two-mention Mojo surface acceptable, or is it worth
paying the `values -> relations` import edge (see "Why `is_in(x, sub)`…") to
get a one-mention spelling? I do not consider this settled.

## Why not a MARK join

`JOIN_MARK` and `JOIN_SINGLE` are already declared and unimplemented
(`marrow/kernels/join.mojo:286-291`), reserved with the comment *"generated by
the planner for subquery decorrelation"*. This design implements neither.

A MARK join emits one row per left row plus a boolean column. It is what you
need when the subquery's answer is consumed as a **value** rather than as the
whole predicate: a subquery under `OR`, under a general `NOT`, or in a
projection. DuckDB uses it for every `IN`/`ANY`
(`plan_subquery.cpp:192-204`) because its planner must handle all positions;
DataFusion uses `LeftMark` only for the disjunctive case
(`decorrelate_predicate_subquery.rs:291-321`).

Two reasons to leave it out:

1. **It is not needed for the scoped surface.** With `SubqueryPredicate`
   accepted only by `filter`, the answer is always consumed as the entire
   predicate, and semi / anti / null-aware-anti cover all four forms exactly.
   DataFusion reaches the same conclusion: `NOT IN` is *not* a mark join there
   (`decorrelate_predicate_subquery.rs:451`), and its `LeftMark` mark column is
   deliberately **non-nullable** — *"we currently do not implement the full null
   semantics for the mark join… the mark column will only be true… never null"*
   (`datafusion/common/src/join_type.rs:53-68`).
2. **It costs the most in exactly the place the gate watches.** A mark column
   makes `emits_right_columns()` a *third* answer, and it adds an output column
   to `_assemble` — code linked by every binary that joins at all, whether or
   not it uses MARK. Contrast the null-aware anti join, which is a branch inside
   an existing loop.

DuckDB's filter-pushdown pass is the cleanest statement of the boundary:
`Filter(mark)` becomes SEMI, and `Filter(NOT mark)` becomes ANTI **only when
every join condition is `IS NOT DISTINCT FROM`** — which is true for
`NOT EXISTS` and false for `NOT IN`
(`src/optimizer/pushdown/pushdown_mark_join.cpp:192-218`). In other words, the
industry's own optimizer reduces mark joins to exactly the three kinds this
design implements, in exactly the three cases this design implements them, and
keeps MARK only for the positions this design defers.

## Binary-size impact and how it is gated

The standing constraint is *"keep `marrow.expr` small-binary — preserve the
closed-erasure/DCE property (no open dispatchers, fused-only value boxes,
closed per-dtype kernels)"*, gated at `threshold_pct: 0.5` on five
`__text` baselines (`benchmarks/binary_size/baseline.json`).

**No open dispatcher is added anywhere.** Specifically:

| change | cost | why it is bounded |
|---|---|---|
| `SubqueryPredicate` + `filter` overload | ~0 on plans that do not use it | a plain struct and a second overload; both dead-code-eliminated when unreferenced. **No new field on `BoxedValue`, no new `Value` trait member.** |
| the four constructors | ~0 | free functions; lower to the existing `join()` builder and `Join` node, which already compile once |
| `JOIN_ANTI_NULL_AWARE` + `is_semi_anti()` | small, on `query_join` only | one constant, one predicate, one branch in `_emit_unmatched`; no new instantiation |
| `ScalarSubquery` relation node | one new node type's 8 `DynRelation` trampolines | trampolines are per-node-type and instantiated only where the node is constructed |
| `scalar_value` overloads | 0 | return the **existing** `NumericParam`/`StringParam`/`TemporalParam` leaves; no new fused node |

The one thing that would blow the budget — a `Value.subquery()` trampoline on
`BoxedValue`, paid by every binary — is the alternative this design rejects.

**The gate.** Add `benchmarks/binary_size/query_subquery.mojo` (a `NOT IN`
lowering plus a scalar subquery) and record it in `baseline.json`, in the style
of the existing programs. It is needed because the five current gates would
report **0.00%** for this whole feature: `query_join` links the join kernel but
nothing subquery-shaped, and none of them constructs a `ScalarSubquery` node.
That is exactly the gap `baseline.json`'s own `_comment` records for the cast
family, where *"a change that added +435,072 bytes measured 0.00% on every gate
the CI actually watches"*. `threshold_pct` stays 0.5.

Run `pixi run binary_size` at the end of every phase, and re-record only with a
written justification, per the existing convention.

## Phasing

Each phase is independently shippable and ends at a review gate. Nothing starts
until Phase 0 is reviewed.

**Phase 0 — corpus first, no engine code (S).**
Land the golden cases from "Testing" below, every one marked `-- xfail`, and
fix the four mislabelled ones' *prose*. `-- xfail` is **strict**
(`golden/runner.py:411-418`), so a case that starts passing turns red and
forces its marker's removal — which makes the corpus the phase tracker.
*Gate: the four `subquery_*` cases still pass as joins; every new case xfails
for the stated reason; spec reviewed.*

**Phase 1 — the surface and the sugar (M).**
`SubqueryPredicate`, the four constructors, the `filter` overload, the semi/anti
lowering, the Python `isin`/`exists` wrappers and the `filter_subquery` binding.
**This adds no capability, only spelling** — every case it turns green passes
today when written as an explicit join.
*Gate: `pixi run -e dev pytest golden marrow/exprold/tests`; `pixi run binary_size`
unchanged within noise; the `IN`/`EXISTS`/`NOT EXISTS` xfails removed.*

**Phase 2 — `NOT IN` (M).**
`JOIN_ANTI_NULL_AWARE`, `is_semi_anti()`, the three rules in
`_emit_unmatched`, the single-column restriction, the `JoinKind` round-trip
through the Python bindings, `not_in`/`__invert__`.
*Gate: the three `not_in` cases green; `marrow/kernels/tests/test_join.mojo`
extended with the three rules at kernel level; `pixi run binary_size`.*

**Phase 3 — the scalar subquery (M).**
`ScalarSubquery` node (including `with_predicate`/`with_projection`
forwarding), `with_scalar`, `scalar_value`, the cardinality rules, the Python
auto-binding, and the new `query_subquery` size gate.
*Gate: the two scalar cases green; a pushdown test asserting a predicate still
reaches a `ParquetScan` through a `ScalarSubquery` node; `pixi run binary_size`
with the new gate recorded.*

**Not scheduled.** MARK join and compound predicates; correlated
decorrelation; `ANY`/`ALL` with non-equality comparators; multi-column
`NOT IN`.

## Testing

Golden expectations come from **DuckDB** (`golden/runner.py:673-690`), so
writing the SQL in the docstring and regenerating produces the correct SQL
answer for free — including all the `NOT IN` NULL rules. That is the single
biggest reason to lead with the corpus.

### The four existing cases are mislabelled

All four build joins while claiming to test subqueries. Their SQL and their
expected blocks are correct and stay; only the plan bodies change.

| case | SQL is | today | becomes |
|---|---|---|---|
| `subquery_in` | uncorrelated `IN` | `JOIN_SEMI` | `filter(is_in(col("dept", int64), dept.select("did")))` |
| `subquery_in_string_keys` | uncorrelated `IN`, string key | `JOIN_SEMI` | `filter(is_in(col("region", string), regions.select("region")))` |
| `subquery_exists` | **correlated** `EXISTS` | `JOIN_SEMI` | `filter(exists(dept, left_on=…, right_on=…))` |
| `subquery_not_exists` | **correlated** `NOT EXISTS` | `JOIN_ANTI` | `filter(not_exists(dept, left_on=…, right_on=…))` |

The last two are worth flagging in review: their SQL is genuinely correlated,
and they become real subqueries only under the "spell the correlation" decision
above. If that decision is rejected, they stay joins and their prose should say
so instead of saying "subquery".

### New cases

Fixtures already contain everything needed. `emp.dept = [10, 20, 20, 99, NULL]`,
`dept.did = [10, 20, 30]` (no nulls), `sales.ref = [1, 2, 2, 3, NULL, 99]` (one
null), `basic.v = [1, 2, 3, 4, NULL, 6, 7]` (`golden/runner.py:74-215`).

| case | SQL | expected | pins |
|---|---|---|---|
| `subquery_not_in` | `… WHERE dept NOT IN (SELECT did FROM dept)` | `eid 4` **only** | rule 3 — the NULL-left exclusion. The one-row difference from `subquery_not_exists`, which gives 4 **and** 5. |
| `subquery_not_in_null_set` | `… WHERE dept NOT IN (SELECT ref FROM sales)` | **zero rows** | rule 2 — a NULL anywhere in the subquery poisons everything |
| `subquery_not_in_empty_set` | `… WHERE dept NOT IN (SELECT did FROM dept WHERE did < 0)` | **all 5 rows**, `eid 5` included | rule 1 — `NULL NOT IN ()` is TRUE |
| `subquery_in_null_set` | `… WHERE dept IN (SELECT ref FROM sales)` | `eid 4` | that a NULL in the set does **not** make a NULL left value match — the cell where the value-level `isin` disagrees |
| `subquery_scalar_uncorrelated` | `SELECT v FROM basic WHERE v > (SELECT avg(v) FROM basic) ORDER BY v` | `4, 6, 7` (avg = 23/6) | the scalar bind and the broadcast |
| `subquery_scalar_empty` | a scalar subquery over a subplan filtered to nothing | **zero rows** | 0 rows ⇒ NULL ⇒ predicate NULL ⇒ nothing survives |

`subquery_not_in_empty_set` is the case I would most expect an implementation
to fail, and the one whose existence I learned from DataFusion's carve-out
rather than from first principles.

### Beyond golden

- `marrow/kernels/tests/test_join.mojo` — the three null-aware rules at kernel
  level, so a failure localises to the kernel rather than to a whole plan.
- `marrow/kernels/tests/test_membership.mojo` — pin `IsInKernel`'s PyArrow
  `MATCH` behaviour as a **deliberate** divergence from SQL, with the reason.
  It cannot live in golden, because golden's oracle is DuckDB and DuckDB
  answers SQL.
- `marrow/exprold/tests/` — a pushdown case asserting a predicate still reaches a
  `ParquetScan` through a `ScalarSubquery`, and a parity case running the same
  subquery through both lanes.
- `python/marrow/tests/` — the ibis-shaped `t["x"].isin(sub)` / `~…` spellings
  and the Python auto-binding of a scalar handle.

## Risks

- **`emits_right_columns()` silently answering True for the new kind** produces
  a malformed `StructArray` with nothing checking — the failure `JoinKind`'s
  docstring already records once. Mitigated by introducing `is_semi_anti()` in
  the same commit, and by a kernel-level test on the output schema.
- **`ScalarSubquery` severing predicate pushdown** by inheriting the `None`
  defaults for `with_predicate`/`with_projection`. Silent: the query stays
  correct and gets slower. Mitigated by an explicit pushdown assertion in
  Phase 3's gate.
- **`ParamCell` sharing across concurrent executions.** `relations.mojo`
  advertises a plan as concurrently executable; a shared cell is written at
  `to_processor` time. For an *uncorrelated* subquery over immutable sources
  every execution binds the same value, so the race is benign — and it is
  pre-existing, since `param()` has the same shape. It becomes real if
  per-execution values are ever introduced. **Open question:** should the cell
  be per-execution rather than per-plan?
- **The runtime lane's `_LOOKUP` table is reset by `drain_params()`**
  (`params.mojo:24-43`). A subquery-registered name must be inserted at
  `to_processor` time — after any drain — not at plan-build time, or a second
  plan's drain strands it. **Open question:** is reusing `_LOOKUP` right, or
  does the subquery binding deserve its own table?
- **`docs/backlog.md:912` currently lists subqueries under "Won't"**, on the
  premise that they presuppose a SQL frontend. That premise holds for
  *decorrelation* and this design keeps it, but the entry needs rewording:
  uncorrelated subqueries need no frontend and no decorrelation.

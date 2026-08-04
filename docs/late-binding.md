# Late binding: prepared plans, `Env`, `Param`, and joins

> **⚠️ UNBUILT AND UNVERIFIED.** Nothing in this document is implemented. It is
> kept because the *problem* it addresses is real and recurring — compiling one
> plan and reusing it across data sources and parameter values — and because its
> `Env` foundation is a load-bearing change to every node in the AOT lane (five
> family traits, every node's `vectorwise`, the `BoxedValue` trampolines, and the
> processors). Do not start it without pricing that blast radius.
>
> The compiler findings that used to sit above this section have moved into
> CLAUDE.md (“Reflection, packs, and comptime aliases”); the erased-relations
> architecture that used to sit below it is now `docs/architecture.md`.

### Objective

A plan should be a **precompiled artifact you invoke many times**, reusable
across *both* different data sources and different scalar values — as long as the
schema shape (field names and dtypes) matches. This is the prepared-statement
model: build the query once, bind data and parameters at execution.

The plan's **type is the compile cache key**. Two invocations with the same schema
shape and the same parameter dtypes resolve to the same fully-specialized nested
type, so they share one monomorphized artifact; only the bound environment
differs.

Half of this is already true. A fused plan is data-free, and a column leaf
resolves its position *by name* against `batch.schema` at execute
(`values.mojo:575`, `:1479`) — so a plan already runs on any batch whose fields
match by name and dtype, regardless of column order or extra columns. The two
missing pieces are (1) binding **more than one** data source, and (2) binding
**scalar parameters** instead of baking constants into the plan.

### The generalization: thread an `Env`, not a `RecordBatch`

Both a named column and a bind parameter are the same thing — *a name resolved at
execute against the environment*. A column reads its array from a batch; a param
reads its scalar from a bindings map. So the value threaded down through the lane
methods stops being a bare `RecordBatch`:

```mojo
struct Env:
    var tables: Catalog    # name -> RecordBatch  (late-bound data sources)
    var params: Bindings   # name -> DynScalar    (late-bound scalar params)
```

(Distinct from `ExecContext`, the GPU/threading handle, and from
`values.Context`, the per-execute breaker-slot scratch. `Env` is the *data +
parameter* environment.) The single-source `execute(batch)` survives as a
one-line convenience wrapper (`execute(Env(Catalog(anon = batch)))`).

The key structural simplification: **only the join subtree is ever two-input.** A
`Join` executes its two children (each pulls its own source from the catalog) and
returns *one* joined batch; every node above the join is single-batch again over
that joined output. So the entire existing `col` / fused-predicate machinery
works unchanged above a join, and the multi-input concern is contained to the
join node itself.

### `Param[T]` — scalar late binding, the analog of a named column

`Param[T]` is to a scalar what `NumericColumn[T]` is to a column: the **dtype
lives in the type**, the **name is a runtime field**, and the **value is read from
the environment at execute** — never baked into the plan. The same plan object,
executed with different `Bindings`, yields different results with no rebuild and
no recompile.

```mojo
struct Param[T: NumericType](NumericValue):
    comptime OutType = Self.T
    comptime OutShape = 1
    var name: String

    @always_inline
    def vectorwise[
        W: Int
    ](self, env: Env, ctx: Context, mut slot: Int, idx: Int) -> SIMD[
        Self.OutType.native, W
    ]:
        # loop-invariant: resolve once, splat — see "resolution" below
        return env.params.get(self.name).as_primitive[Self.T]().splat[W]()
```

This contrasts with `NumericLiteral[T]` (`values.mojo:636`), which bakes a
constant into a runtime field — fine for genuinely fixed constants, but not
reusable across values without rebuilding the node. `lit()` and `param()` would
coexist: `lit` for constant folding, `param` for the late-bound case.

Parameters would get the same struct-reflection surface as tables — *dtype-tag
struct → typed placeholder nodes → resolved by name against the environment*:

```mojo
# polars-style leaf, dtype spelled explicitly
... .filter(o.amount > param("min_amount", float64))

# or the reflected handle, symmetric with Table[T]()
struct Args:
    var min_amount: Float64Type
    var region:     StringType

var p = Params[Args]()                 # mirror of Table[Orders]()
... .filter(o.amount > p.min_amount)   # p.min_amount : Param[Float64Type]
```

**Both handles are blocked by the same thing.** `Params[Args]()` needs the
parametric `reflect[T].field[name].T` alias that deferred `Table[T]` (§1), so it
cannot ship before that resolves.

### Resolution must stay loop-invariant

A param is a scalar constant for the whole run; a column's position is fixed for
the whole run. Neither may be resolved per SIMD lane — a `String`-keyed map lookup
inside the lane method would run once per lane. Column index resolution has the
same hazard today (`get_field_index` is called inside the lane body and the
compiler is assumed to hoist it, which is plausible for an integer field index and
far less certain for a `String`-keyed param lookup).

The safe design is a **per-execute resolve pass**: walk the tree once, turn each
column *name* into an index and each param *name* into its resolved scalar, then
run the fused loop over the resolved form. Whether this can stay implicit (trust
the hoist) or needs an explicit resolved-node representation is an **open
question** to settle when prototyping. Note the shipped code already has the
machinery for a resolve pass — `BoxedValue.resolve_names` (`relations.mojo:281`)
binds name references against a schema, keeping the node's type.

### Joins — the two-input node

A `Join[L: Relation, R: Relation, ...]` is the binary node that collapses two
sources into one batch. Two surfaces for the keys:

```mojo
# Pragmatic (near-term): argument position encodes side, zero new machinery.
# Mirrors the runtime join() and the hash_join kernel (positional left_on/right_on).
o.join(c, left_on = Tuple(o.cust_id), right_on = Tuple(c.cust_id), how = JOIN_INNER)

# Sugar (north star): needs an On[L, R] node (NOT the fused SIMD Eq — a join
# condition is a hash equijoin, not a per-row compare) with "left operand = left side".
o.join(c, on = o.cust_id == c.cust_id)
```

`left_on`/`right_on` needs nothing new — columns stay plain
`NumericColumn[T]("cust_id")`, the left key resolves against the left child's
output schema and the right against the right's. Composite keys are tuples of
equalities or two grouped tuples.

**Name collisions are the reason to invest in source-tagged columns.** After a
join, colliding names get suffixed (`cust_id` → `cust_id_right`). With a plain
`col("name", string)` in a post-join projection, the user must know and spell the
suffixed name — fragile. If columns were parameterized by their source struct —
`NumericColumn[T, Src]` — then `c.name` *knows* it came from the right side and
auto-resolves to the suffixed name. `Src` is a phantom parameter (like the name
today, it never touches the lane method), adding one instantiation axis whose
cardinality is the number of tables in the query. It also makes `on = a == b`
unambiguous without a positional convention and lets `join` verify at comptime
that a left key's `Src` really is the left table.

### Filter/select ordering over a join

The natural order is **JOIN → filter → select** (SQL/polars): the filter sees the
full joined schema and the projection picks from filtered rows, each resolving
against its *child's output*. This generalizes to N-way and is what the shipped
`DynRelation` verbs already do. Recorded here because the deleted typed slice
deliberately did the opposite — its `Filter`'s predicate evaluated against the
*source* batch, so a filter could reference a column absent from the projection.

### The reuse pattern (the payoff)

```mojo
# compile ONCE — no data, no scalar values in the type or the object
var query = (
    o.join(c, on = o.cust_id == c.cust_id)
     .filter(o.amount > param("min_amount", float64))
     .select(o.order_id, c.name, o.amount)
)

# reuse the SAME compiled plan across data sources AND parameter values
var jan = query.execute(Env(
    Catalog(orders = jan_orders, customers = customers),
    Bindings(min_amount = 100.0),
))
var feb = query.execute(Env(
    Catalog(orders = feb_orders, customers = customers),
    Bindings(min_amount = 250.0),
))
```

`query` is a single value of a single monomorphized type; `jan`/`feb` differ only
in the bound `Env`. This goes a step beyond DataFusion/substrait bound-vs-unbound
plans or polars `pl.lit`/`pl.col`, which do late binding but stay runtime-typed —
here the late binding would be statically typed and monomorphized.

### Type-safety of the binding — open question

`Bindings(min_amount = 100.0)` hands a `Param[Float64Type]` node a `DynScalar` at
execute; a dtype mismatch is a **runtime** error. A `Params[Args]()` handle would
close half the gap (the node's dtype is comptime-checked against the plan), but
the *provided value* is still checked at execute. Fully-static binding would make
`Bindings` itself a typed struct keyed to `Args`. The same tension applies to
`Catalog`: matching a bound batch's schema against the plan's expected shape is a
runtime check unless the catalog is typed. Runtime-checked bindings are an
acceptable first cut.

### Suggested build order

1. **`Env` foundation** — generalize the lane methods and `execute` from
   `RecordBatch` to `Env` (tables + params). Load-bearing; do first, and see the
   warning at the top of this section for its true blast radius. `Catalog` plus
   the single-source convenience wrapper.
2. **`Param[T]` + `param()` + `Bindings`** — scalar late binding. (`lit` already
   exists in both lanes: `values.mojo:2446` fused, `:2486` runtime.)
3. **`Join` binary node + `left_on`/`right_on`** — a working
   `join().filter().select()` on the `col()` surface, child-output semantics.
4. **Source-tagged columns + `Params[Args]()` + operator sugar** — the north star
   (`on = a == b`, `o.amount > p.min_amount`, auto-suffix resolution). Blocked on
   the `reflect` limitation in §1.

Prove the "one compiled plan, many executions" property with a reuse test plus a
binary-size comparison against the equivalent runtime-lane join, mirroring
`benchmarks/binary_size/`.

---

## Limitations of the AOT path (still true)

- **Operator-overload repetition.** `__add__`, `__gt__` and friends must be
  defined on each family trait rather than once, because a trait default cannot
  return a later-defined struct.
- **Dynamic SQL gets the runtime lane.** A plan parsed from a runtime string
  produces `DynValue` / `DynRelation` nodes regardless. AOT specialization
  requires the plan expressed as Mojo types before `mojo build`.
- **The bridge is one-way.** A fused node boxes into `BoxedValue`; the reverse is
  impossible, since comptime types are fixed before compilation finishes.
- **`VariadicPack` forwarding** and **trait comptime-alias access from external
  generics** — see CLAUDE.md, "Reflection, packs, and comptime aliases".
- **Nested and `Optional` fields, variable-length types** are not supported by
  `Schema.from_struct` — see `todo/reflect-schema-from-struct.md`.

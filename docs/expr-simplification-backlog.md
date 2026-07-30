# Simplification backlog — `marrow.expr`

Found while doing Step 3 (`docs/step3-expression-nodes.md`). Everything here was
measured mechanically — duplicate *method bodies* compared as text, callers
counted with grep — not eyeballed. Counts are as of `0bf560a`.

Nothing here is a bug. It is duplication and boilerplate that the Step 3 work
either exposed or added, recorded so it is not lost.

---

## Already fixed (listed so the pattern is visible)

| what | where | how it surfaced |
|---|---|---|
| `count_distinct`/`approx_count_distinct` defined **3×** | `NumericValue`, `StringValue`, `TemporalValue` | a struct conforming to two families fails to compile — `DynValue` was the first to try |
| `ConditionalBinaryKernel` + `CoalesceOp`/`NullifOp` — a second, parallel notion of "kernel" in the expression layer, wrapping free functions over kernels that already existed | `values.mojo` → moved to `kernels/conditional.mojo` | asked directly whether proper kernel structs existed |
| `coalesce`/`nullif` imported but unused after that move | `values.mojo` | grep for callers |
| `is_list_like()` excluded map while `dispatch_listlike` included it | `dtypes.mojo` | `rapidhash` had hand-written `is_list_like() or is_map()` as a workaround |

The recurring shape: **duplication across the family traits is not inert.** It is
invisible while every struct conforms to exactly one family, and becomes a
compile error the moment one conforms to two.

---

## Open — ranked by copies removed

### 1. `referenced_columns` — 29 identical copies

22 unary nodes are byte-identical:

```mojo
return self.a.referenced_columns()
```

and 7 binary nodes share the `_union_columns(self.l…, self.r…)` body. Every one
is mechanical from the node's *arity*, not its behaviour.

Same story, same nodes, for two more methods:

| method | copies | body |
|---|---:|---|
| `referenced_columns` (unary) | 22 | `self.a.referenced_columns()` |
| `prepare` (unary) | 8 | `self.a.prepare(batch, ctx)` |
| `validity` (unary passthrough) | 8 | `self.a.validity(batch)` |
| `referenced_columns` (binary) | 7 | `_union_columns(l, r)` |
| `prepare` (binary) | 5 | `l.prepare(); r.prepare()` |
| `validity` (binary intersect) | 3 | `Bitmap.intersect(l, r)` |

**Roughly 53 method bodies expressing two facts: "I have one child" and "I have
two children."**

Worth being careful here: the obvious fix is `UnaryNode`/`BinaryNode` traits
supplying defaults, but that is exactly what created the `materialize` catch-22
below — more defaulted methods on more traits is what makes multi-family
conformance fail. A safer shape is a plain helper (`_unary_columns(self.a)`)
that each node calls, trading 53 bodies for 53 *one-line* bodies with no new
trait defaults. Lower ceiling, no new coupling.

### 2. The 32 one-line `materialize` overrides

Added in `b56b886` and pure boilerplate:

```mojo
def materialize(self, batch: RecordBatch, mut ctx: Context) raises -> Datum:
    return self._numeric_fused(batch, ctx)
```

They exist because `NumericValue`, `BoolValue` and `StringValue` each *defaulted*
`materialize`, so a struct conforming to two inherits conflicting defaults and
the compiler demands a manual implementation — which then hits the
re-defaulted-method recursion `CLAUDE.md` documents.

This is the cost of `DynValue` conforming to several families, and it is paid
per node. It goes away only if the family traits stop defaulting shared method
names — the same tension as item 1, from the other side.

### 3. Column leaves duplicated 4×

`NumericColumn`, `StringColumn`, `TemporalColumn`, `ListColumn` each repeat
`referenced_columns`, `validity`, `name` and (3 of them) `materialize`
identically — 15 bodies for what is one concept, "a column resolved by name",
differing only in the family it reports to.

### 4. `count` is hoistable; `min`/`max` are not

`count` is byte-identical in `StringValue` and `TemporalValue`
(`AggExpr.of[CountAgg](self.copy())`) and can move to `Value` exactly as
`count_distinct` did.

`min`/`max` genuinely differ per family — `Min[Self]` vs
`StringMinMax[…]` vs `TemporalMinMax[…]` — so they cannot be hoisted. They are
the concrete blocker on `DynValue` conforming to `TemporalValue`, because the
erased answer has to be resolved from the *runtime* dtype. See Phase 4 in the
Step 3 plan.

### 5. Two entry points on the box

`DynValue.execute(batch) -> DynArray` and `Value.execute(batch) -> Datum` coexist.
Callers use the former (~45 sites); the trait supplies the latter. Not harmful —
`into_array` bridges them — but it is two names for one operation on a type that
now *is* a `Value`.

### 6. The interpreter's three parallel switches

`TagValue.eval` (41 arms), `prune` (9), `_op_name` (41), plus `write_to`'s own
tag branching — ~300 of `dynamic.mojo`'s 1,086 lines. Step 3 Phase 4 deletes
these outright rather than deduplicating them; listed here only so the count is
recorded alongside the rest.

Note `_op_name` is **not** duplication of the kernels' names: the vocabularies
deliberately differ (`sub` vs the kernel's `subtract`, `and` vs `and_`). It is a
display mapping, and it becomes a per-node `comptime` when the tags go.

---

## Non-findings, checked and dismissed

- **`_rank` / `_numeric_rank`** — cannot be one function (comptime type → `Int`
  vs runtime value → `Int`). The missing piece was enforcement, added as
  `test_numeric_rank_agrees_across_lanes` (`a7524a7`).
- **`coalesce`/`nullif`/`case_when` free functions** — zero callers outside
  `kernels/conditional.mojo` except `case_when`, used once by `CaseWhen._result`.
  The other two are now unimported. Deleting the free functions entirely is
  possible but they are the documented public spelling, so left alone.
- **`ConcatKernel.dispatch`** — added in Step 3, not redundant with `apply`:
  `apply` is typed, `dispatch` is the erased entry the runtime `+` needs.

# Simplification backlog — `marrow.expr`

Re-evaluated 2026-07-30 against the tree *after* Step 3 landed. Everything below
was measured mechanically — duplicate method bodies compared as text, callers
counted with grep — and each proposed fix was **tried**, not reasoned about. Two
of the four items in the first draft turned out not to be actionable at all, and
saying why is the point of this document.

Current: `values.mojo` is 3,153 lines with **93 duplicated method bodies**.

---

## Done

| what | result |
|---|---|
| **32 one-line `materialize` overrides** | **Removed.** The family traits default `materialize` again; only `DynValue` implements it. |
| `count_distinct`/`approx_count_distinct` defined 3× | Hoisted to `Value`. |
| `ConditionalBinaryKernel` + `CoalesceOp`/`NullifOp` — a second notion of "kernel" in the expression layer | Moved to `kernels/conditional.mojo`. |
| 14 repeated `comptime if Self.IsErased` blocks | Centralised into the three family drivers. |

The `materialize` one is the interesting one, because **it was not fixable when
first recorded**. The overrides existed because conforming `DynValue` to several
families made the defaults conflict, and implementing it manually then hit
`attempt to resolve a recursive reference to declaration 'DynValue.materialize'`.

Splitting `NumericOps` off `NumericValue` — done for an unrelated reason, to fix
a recursion that broke *external* builds — removed the cause. Re-testing an item
that had been ruled out is what found it. **+0 bytes on all eleven gates.**

---

## Not actionable — and why

### `referenced_columns` / `validity` / `prepare` — 69 duplicated bodies

The largest cluster by far, and there is no mechanism in Mojo to remove it:

| method | copies | body |
|---|---:|---|
| `referenced_columns` (unary) | 23 | `self.a.referenced_columns()` |
| `validity` (unary) | 9 | `self.a.validity(batch)` |
| `prepare` (unary) | 8 | `self.a.prepare(batch, ctx)` |
| `referenced_columns` (binary) | 7 | `_union_columns(l, r)` |
| `prepare` (binary) | 5 | `l.prepare(); r.prepare()` |
| `validity` (binary) | 3 | `Bitmap.intersect(l, r)` |
| column leaves | 14 | `_name`-based `__init__`/`name`/`validity`/`referenced_columns` |

The obvious fix is `UnaryNode`/`BinaryNode` traits supplying defaults. **Traits
cannot require `var` fields** — the compiler is explicit: `traits do not support
'var' fields; use 'comptime' to declare associated types`. A default body cannot
mention `self.a`, so it cannot be written.

A helper function does not help either: the bodies are *already* one-liners, so
`return _unary_columns(self.a)` trades one line for one line.

This is a language limitation, not a design flaw. **Accepted, not open.**

### Hoisting `count` to `Value`

Tried; it does not compile. The first draft said `count` was "byte-identical in
`StringValue` and `TemporalValue`" — true, but it compared only those two.
`NumericValue.count` returns `Count[Self]`, a `Reduction` node, not an
`AggExpr`. Hoisting the other two to `Value` makes a three-way conflict and
every node has to implement it manually — 30 new bodies to remove 1.

Same for `min`/`max`: three families, three genuinely different `Aggregation`s.

---

## Open

### `render` — 6 duplicated bodies

`String(Self.K.name, "(", self.l.render(), ", ", self.r.render(), ")")` ×4 and
the unary form ×2. Same field-access problem as above, so the same limit
applies — but only 6 bodies, so the payoff is small either way.

### Column leaves — 4 near-identical structs

`NumericColumn`, `StringColumn`, `TemporalColumn`, `ListColumn` differ only in
which family they report to; between them they repeat `__init__`, `name`,
`validity`, `referenced_columns` and `materialize`. A single generic leaf is
blocked by the same reason the families exist at all (`vectorwise` vs
`elementwise` have different signatures), but the *non-family* half — the four
`_name`-based methods — is 14 of the 93 duplicated bodies and is worth another
look if a mechanism ever appears.

---

## Non-findings, checked and dismissed

- **`_rank` / `_numeric_rank`** — cannot be one function (comptime type → `Int`
  vs runtime value → `Int`). Enforcement was the gap; `test_numeric_rank_agrees_across_lanes` closed it.
- **`_op_name`** — gone with the interpreter. It was never duplication of the
  kernels' names: the vocabularies deliberately differ (`sub` vs `subtract`).
- **`ConcatKernel.dispatch`** — not redundant with `apply`; `apply` is typed,
  `dispatch` is the erased entry the runtime `+` needs.

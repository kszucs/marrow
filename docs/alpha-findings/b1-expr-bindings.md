# B1 — Binding `marrow.expr` to Python: findings

Written while adding `python/bindings/expressions.mojo` and
`python/marrow/_expr_column.py` on `alpha` (`fb31b2d`). Everything below was
observed on that tree, not inferred.

---

## 1. Two hard blockers found while binding

### 1.1 `add_type[T]` rejects any struct holding a function pointer

`PythonModuleBuilder.add_type[T]` installs a default `tp_repr` that calls
`repr(value)`. In this Mojo version `repr` is *not* keyed on a `Representable`
trait — there is no such trait; `std/format/repr.mojo` is

```mojo
def repr[T: Writable](value: T) -> String:
    var string = String()
    value.write_repr_to(string)
    return string^
```

and `Writable.write_repr_to`'s **default** body is `_reflection_write_to`,
which walks every field and asserts `conforms_to(FieldType, Writable)`. So:

```
format/__init__.mojo:287: constraint failed: Could not derive Writable for
DynValue - member field `_eval_fn` does not implement Writable
```

`DynValue._eval_fn` is the `EvalFn` pointer that the whole size argument in
`dynamic.mojo` rests on, and `DynAgg` fails identically through its `input`
field. **Neither type can be handed to `add_type` as it stands.**

Worked around here by two one-field boxes (`Expr`, `Agg`) in the binding
module that override `write_repr_to`. The real fix is two lines in
`marrow/expr/dynamic.mojo`:

```mojo
def write_repr_to(self, mut writer: Some[Writer]):
    writer.write("DynValue(", self.render(), ")")
```

on `DynValue` and `DynAgg`. That deletes both boxes and the `unwrap` /
`wrap_expr` seam other binding modules now have to know about. It was out of
scope for this task (`marrow/expr/**` is owned by another agent this wave) but
it should be done before the boxes calcify — they are pure overhead and they
add a downcast hop to every one of ~55 bound methods.

Note this is a *general* trap, not an expression-layer one: any marrow struct
that grows a function-pointer field becomes un-bindable, silently, at the next
`add_type` call. `AggFunc._grouped_fn` and all three of `AggFold`'s fields are
in the same position — if the plan bindings ever need to expose an `AggFunc`,
they will hit this too.

### 1.2 `def_method` does not fill CPython slots

`def_method[f]("__str__")` puts `__str__` in the type's `tp_dict`; it does not
set `tp_str`. Measured on the built extension:

```
>>> e = libmarrow.expr_column('a')
>>> e.__str__()          # the bound method
'a'
>>> str(e)               # the slot — falls through to the derived tp_repr
'<marrow.Expr: a>'
```

So the whole question of "can `__eq__` on a bound type return an `Expr`?" is
moot: **operator dunders registered through `def_method` never fire.** They
would sit in `tp_dict` as dead weight. Every operator has to live in pure
Python, which is what CLAUDE.md already mandates for a different reason.

Two consequences worth writing down:

- `python/bindings/scalars.mojo` binds `__str__`, `__repr__` and `__bool__`
  this way, and `python/marrow/__init__.py`'s `Scalar.__str__` does
  `str(self._binding)` — which therefore returns the *derived* repr, not the
  bound `__str__`. That is very likely a live cosmetic bug in `Scalar` and
  `Array`; I did not chase it because it is outside this task, but somebody
  should. Same shape in `arrays.mojo`.
- The binding layer should stop pretending: name the methods `render` /
  `to_string` and let Python own `__str__`. I added `render` to both `Expr`
  and `Agg` for exactly this reason and the Python wrappers use it.

---

## 2. The aggregate cluster — five types where two would do

`AggFunction` → `Aggregation` → {`AggFunc`, `AggFold`, `FoldedAggregates`},
plus `AggExpr` (fused lane) and `DynAgg` (runtime lane). Binding it forced me
to read all six, and the shape is worse than the docstrings claim.

### 2.1 `DynAgg` is a three-field struct that is pure duplication of `AggExpr`

```mojo
struct DynAgg(Copyable, Movable, Writable):
    var func: String
    var input: DynValue
    var out_name: String
```

and `AggExpr` (values.mojo:2159) has *exactly* these three as
`_func` / `_unresolved` / `out_name`, plus `input: BoxedValue` and
`_of: Optional[...]`. `AggExpr.__init__(var agg: DynAgg)` is a five-line
field-for-field copy that also duplicates the "empty alias means use the
function name" rule — which `DynAgg` itself does *not* apply, so
`DynAgg.out_name` is `""` where `AggExpr.out_name` is `"sum"` for the same
aggregate. I had to reimplement that rule a third time in the binding
(`_agg_name`) to give Python an honest `name()`.

**`DynAgg` earns nothing.** It is not an erasure boundary (it holds a concrete
`DynValue`), it resolves nothing, and it is not what the plan holds — the plan
holds `AggExpr`. It exists solely because `DynValue.sum()` needs a return type
and `AggExpr` lives in `values.mojo`. Since `values` and `dynamic` already
form an acknowledged cycle (see the package docstring), that is not a real
constraint: `DynValue.sum()` could return `AggExpr` directly and `DynAgg`
could be deleted, along with the copy constructor and the third copy of the
naming rule.

### 2.2 `AggExpr` is two types wearing one struct

```mojo
var input: BoxedValue
var _unresolved: Optional[DynValue]
var _func: String
var _of: Optional[def(DynType) thin raises -> AggFunc]
```

`_of` is set **iff** `_unresolved`/`_func` are not, and `resolve()` /
`input_for()` are both `if` on that. It is a two-variant sum type spelled as
four fields and two `Optional`s, which means every reader has to reconstruct
the invariant ("fused ⇒ `_of`; dynamic ⇒ `_func` + `_unresolved`") from the
constructors. `write_to` even picks between `_func` and `out_name` to decide
what to print. A `Variant[Fused, Named]` — or just resolving the dynamic case
eagerly, since `resolve(in_dtype)` is called with a dtype anyway — would
remove all four `if`s.

### 2.3 `AggFunc` / `AggFold` / `FoldedAggregates`: the split is real, the
naming is not

The split *is* justified and the docstrings carry the measurement (+3.2 MB /
+24 % if `AggFold` folds into `AggFunc`; +1.2× if a plan carries the fold).
That part is good engineering and should stay.

What is wrong is that the three names say nothing about the split:

- `AggFunc` — "one aggregate, grouped-only, plan-side"
- `AggFold` — "the same aggregate's whole/partials/merge, driver-side"
- `FoldedAggregates` — "N of both, driver-side"

Nothing in `AggFunc` says "grouped only". Nothing in `AggFold` says it is the
*other half of the same aggregate*. And `FoldedAggregates` is the only one of
the three a `GroupBy` driver ever names, while the plan side holds a bare
`List[AggFunc]` with no type at all — so the two halves of the same concept
have wildly asymmetric packaging. `GroupedAgg` / `MergeableAgg` /
`AggregateSet`, or a comment block naming the two audiences, would cost
nothing.

### 2.4 The name→type dispatch is written twice

`resolve_agg` (aggregates.mojo:191) is an eight-arm `if name == X.name` ladder.
`AggFunc.__init__(name, value_dtype)` calls it. `FoldedAggregates.append(name,
value_dtype)` calls it. Fine. But `DynValue.aggregate(func)` accepts **any**
string with no validation — `col("a").aggregate("nonesuch")` builds happily and
fails much later, when the plan resolves it, with `unknown aggregate function:
nonesuch`. The Python binding inherits that: I have a test asserting the bad
name survives, because that is the actual behaviour. For an alpha aimed at
Python users this is the wrong place to fail; `resolve_agg`'s arm list should
be reachable as a `is_known_aggregate(name)` predicate so the frontend can
raise at build time.

### 2.5 `Value.count_distinct` has no runtime-lane twin

`values.mojo:460` defaults `count_distinct` / `approx_count_distinct` on
`Value`, and `DynValue` conforms to `Value` — so those two exist on `DynValue`
but return an `AggExpr` while `sum`/`mean`/… return a `DynAgg`. Two aggregate
return types on one struct, differing by which of the two lanes happened to
define the method. I did not bind them for that reason; the binding would have
needed a third box. Worth unifying when `DynAgg` goes.

---

## 3. `DynValue`'s fluent surface

### 3.1 The good part

The `EvalFn`-per-operation design does what its docstring claims. Binding it
was mechanical: 40 of the ~55 methods fall into exactly three shapes
(`Expr→Expr`, `(Expr,Expr)→Expr`, `Expr→Agg`) and became three comptime
helpers. That regularity is a direct consequence of the tag/eval split, and it
is the main reason this task was a day and not a week.

### 3.2 Naming inconsistencies that leak straight into the Python API

- `isin` (no underscore) vs `day_of_week`, `date_trunc`, `day_of_year`. PyArrow
  spells it `is_in`. Python's `Column.isin` matches pandas, which is probably
  the right answer, but the Mojo side should pick one convention.
- `length` is a string kernel, but there is also `array_length` in
  `kernels/nested.mojo` with no `DynValue` method. A Python user reaching for
  list length finds nothing.
- `ln` (not `log`) while the kernel is `LogKernel` and `LogKernel.name` is what
  ends up in `render()`. So `col("x").ln()` renders as `log(x)`. Minor, but the
  rendered plan does not round-trip to the API that built it.
- `like`/`ilike` take a `String` payload while `startswith`/`endswith`/
  `contains` take a `DynValue` operand. Both are defensible; the asymmetry is
  invisible from the method names and I had to read the bodies to bind them
  correctly.

### 3.3 `literal` has no Python-facing constructor

`DynScalar` is `ConvertibleToPython` but **not** `ConvertibleFromPython`
(`marrow/scalars.mojo:577`). There is no Python-value → `DynScalar` path
anywhere in the tree. `lit(3)` in Python therefore has to go
`[3] → marrow.array → DynArray → arr[0] → DynScalar → DynValue.literal`, which
allocates a one-element Arrow array per literal. It works and it reuses the
one well-tested inference path, but it is silly for a scalar. `DynScalar`
should gain a `__init__(out self, *, py: PythonObject)` mirroring
`DynArray`'s.

### 3.4 `is_null` / `is_valid` / `is_nan` / `fill_null` are missing

Confirmed absent on `DynValue` on this branch. `dynamic.mojo`'s own module
docstring says `col("x").isnull()` "hands back a fused node" — that sentence
describes a method that does not exist; `NullPredicate` is only reachable from
`values.mojo`. A Python user cannot write a null check at all today, which is
a hole in a query API. Another agent is adding these; the binding and the
Python wrapper both carry a marked `TODO(alpha)` at the right spot.

---

## 4. Smaller things noticed in passing

- `marrow.timestamp(unit, tz)` cannot be called as `timestamp(unit)` from
  Python: `dtypes.mojo:119` gives `tz` a Mojo default, but Mojo defaults do not
  become Python defaults, so `ma.timestamp("s")` raises
  `TypeError: <mojo function>() missing 1 required positional argument`. Same
  class of bug as the `record_batch(data, schema, names)` signature. These are
  exactly the "no optional args in the binding, sugar in Python" rule being
  half-applied — the binding has the optional arg *and* the Python side does
  not re-expose it.
- `marrow.array([...])` cannot build a `timestamp` column (`unsupported type:
  timestamp[s]`) and cannot consume `datetime.datetime` (`cannot include value
  of type: datetime`). The tests here build timestamps by casting int64, which
  is a workaround a user should not need.
- `Scalar.as_py` raises `as_py: unsupported dtype timestamp[s]`, so
  `Array.to_pylist()` on a timestamp column fails. That makes temporal results
  unreadable from Python even when the kernels are correct — `date_trunc`
  computes fine and cannot be printed.
- `DynValue.render()` renders literals as the bare word `literal`, so
  `col("a") > 5` and `col("a") > 500` render identically. Fine for a tag, bad
  as the user-visible `repr` of a predicate, which is what it now is. The
  payload is right there.

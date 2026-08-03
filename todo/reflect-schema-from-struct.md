# Prototype: derive an Arrow `Schema` from a Mojo struct via `reflect[T]`

## Motivation

Every `Schema`/`Field` list in marrow's tests and examples is written by hand
today:

```mojo
var schema = Schema(
    Field("id", dt.int64, nullable=False),
    Field("name", dt.string),
    Field("score", dt.float64),
)
```

That's fine for a handful of columns, but it's a second source of truth next
to any Mojo struct that actually models the row (e.g. a benchmark's record
type, or a future struct-of-arrays builder helper) — the two drift
independently. Mojo 1.0.0b3 added `reflect[T].field_at[idx]`, the by-index
counterpart to `reflect[T].field[name]`, which makes it practical to walk a
struct's fields at compile time. This is a spike to see whether
`Schema.from_struct[T]()` is worth building.

## The real reflection API (confirmed against `~/Workspace/modular`)

`std.reflection.reflect.mojo` gives us, for a struct type `T`:

- `reflect[T].field_count() -> Int`
- `reflect[T].field_names() -> InlineArray[StaticString, N]`
- `reflect[T].field_at[idx].T` — the field's concrete Mojo type, usable in
  type position (works even when `T` is itself a generic parameter, unlike
  `.field["name"]` which needs a concrete `T`)
- `reflect[T].is_struct() -> Bool`
- `reflect[T].base_name() -> StaticString`

None of this exists in marrow today — `grep -rn "reflect\[" marrow/ python/`
comes back empty. This would be the first use of compile-time reflection in
the codebase.

## Sketch (untested — this is the thing to prototype, not a finished design)

```mojo
trait ArrowMappable:
    """Marker for Mojo types with a known Arrow DataType mapping."""
    @staticmethod
    def arrow_type() -> AnyDataType: ...

# One overload per native type marrow already has a DataType constant for:
def arrow_type_of[T: AnyType]() -> AnyDataType:
    comptime if T == Int32:
        return dt.int32
    elif T == Int64:
        return dt.int64
    elif T == Float32:
        return dt.float32
    elif T == Float64:
        return dt.float64
    elif T == String:
        return dt.string
    elif T == Bool:
        return dt.bool_
    elif conforms_to(T, ArrowMappable):
        return T.arrow_type()  # nested struct opts in explicitly
    else:
        comptime assert False, "no Arrow mapping for this field type"


@staticmethod
def from_struct[T: AnyType]() -> Schema:
    comptime assert reflect[T].is_struct(), "from_struct[T] requires a struct"
    var fields = List[Field]()
    comptime r = reflect[T]
    comptime for i in range(r.field_count()):
        comptime FieldT = r.field_at[i].T
        fields.append(Field(String(r.field_names()[i]), arrow_type_of[FieldT]()))
    return Schema(fields^)
```

## Open questions to resolve during the spike (not before)

1. **Nullability.** A Mojo field has no notion of "nullable" — every field
   is present. The obvious mapping is `Optional[T]` field → nullable
   `Field`, plain `T` → `nullable=False`. Needs `arrow_type_of` to unwrap
   `Optional[T]` before the type-equality chain, and the reflected
   `FieldT` at that point is `Optional[T]`, not `T` — check whether
   `reflect[Optional[T]]` even reports as a distinct type usable for this,
   or whether unwrapping has to happen before calling `reflect` at all.
2. **Nested structs.** The sketch above requires an opt-in `ArrowMappable`
   conformance rather than recursing through `reflect` automatically —
   simpler to prototype first, but a fully-automatic recursive version
   (walk nested structs' fields transitively) is the more useful end state
   if the manual-opt-in version proves the concept works.
3. **Variable-length types.** `List[T]` → `list_(...)`, `String` → `string`
   vs `large_string`, fixed-size arrays (`InlineArray[T, N]`) →
   `fixed_size_list_(...)`. Each needs its own `arrow_type_of` branch and
   there's no obvious default for e.g. `List[T]` where `T` isn't itself
   mappable.
4. **Unsupported fields should fail at compile time, not silently drop.**
   The `comptime assert False` branch in the sketch is the right shape —
   confirm the error message actually points at the offending field name,
   not just "no mapping," since a bad message here would be worse than no
   feature at all.
5. **Where does this live?** Probably a new `Schema.from_struct[T]()`
   static method in `marrow/schema.mojo`, gated behind whatever module the
   `ArrowMappable` trait ends up in — needs a decision once the mapping
   table's shape is clearer.

## Definition of done for the spike (not the feature)

- One real struct (pick something from `marrow/tests/` or a benchmark
  record type) round-trips through `Schema.from_struct[T]()` and produces a
  `Schema` equal to its hand-written counterpart.
- A clear answer, even if "no," on whether nested structs and `Optional`
  fields are in scope for a first version.
- If the spike works: a short design note (in `docs/`, matching the
  existing design-doc convention) before writing the real implementation —
  this is exactly the kind of API-shape decision that benefits from being
  designed in the open first.

## Downstream

This is the reflection foundation for `docs/late-binding.md` — the
fully-monomorphized `Project[*Es]`/`Filter[Pred]` layer needs `field_at[i].T`
(confirmed usable in generic type position) to type each column and
`schema_of[T: Table]()` to produce leaf/output schemas. Worth building first
because it's independently useful and de-risks the reflection mechanics.

## Status

Not started. No existing call site depends on this — purely additive if it
pans out.

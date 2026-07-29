# Decimal Type and Unified Primitive Array Design

## Motivation

The current codebase has two parallel array structs for fixed-width physical storage:

- `PrimitiveArray[T: PrimitiveType]` — numeric and boolean types
- `TemporalArray[T: TemporalType]` — date, time, duration, timestamp types

`TemporalArray` was introduced because temporal types carry runtime logical information
(time unit, timezone) that isn't captured by the Mojo type parameter alone. Adding
decimal support would require a third struct (`DecimalArray`) for the same reason
(runtime precision and scale). This is redundant — all three structs hold the same
fields: `dtype`, `length`, `nulls`, `offset`, `bitmap`, `buffer`.

The refactor unifies them into a single `PrimitiveArray[T: PrimitiveType]` parameterised
on a richer trait hierarchy, and uses sub-trait bounds to keep arithmetic kernels
restricted to numeric types only.

## Trait Hierarchy

```
DataType
└── PrimitiveType(DataType)          comptime native: DType
    ├── NumericType(PrimitiveType)   + TrivialRegisterPassable, Defaultable
    ├── TemporalType(PrimitiveType)
    └── DecimalType(PrimitiveType)
```

### `PrimitiveType`

The root trait for all fixed-width, buffer-backed types. Provides a single compile-time
requirement: `comptime native: DType` — the physical storage type used for buffer reads
and SIMD operations. Does **not** require `TrivialRegisterPassable` or `Defaultable`,
because some sub-types (e.g. temporal types with timezone strings, decimal types with
runtime precision/scale) cannot satisfy those constraints.

```mojo
trait PrimitiveType(DataType):
    comptime native: DType
```

### `NumericType`

Integers, unsigned integers, floats, and bool. These are zero-sized marker structs with
no runtime state, so they can satisfy the stricter register-passable constraint. This is
the bound used by existing arithmetic kernels.

```mojo
trait NumericType(PrimitiveType, TrivialRegisterPassable, Defaultable):
    pass
```

Concrete types: `Int8Type`, `Int16Type`, `Int32Type`, `Int64Type`, `UInt8Type`,
`UInt16Type`, `UInt32Type`, `UInt64Type`, `Float16Type`, `Float32Type`, `Float64Type`,
`BoolType`.

### `TemporalType`

Date, time, duration, and timestamp types. The physical storage is int32 or int64, but
the logical type carries a time unit and (for timestamps) an optional timezone string —
both runtime values stored in `array.dtype`.

```mojo
trait TemporalType(PrimitiveType):
    pass
```

Concrete types: `Date32Type`, `Date64Type`, `Time32Type`, `Time64Type`, `DurationType`,
`TimestampType`.

### `DecimalType`

Fixed-point decimal types backed by int32, int64, int128, or int256. The logical type
carries precision and scale — runtime values stored in `array.dtype`.

```mojo
trait DecimalType(PrimitiveType):
    pass
```

Concrete types: `Decimal32Type`, `Decimal64Type`, `Decimal128Type`, `Decimal256Type`.

## Unified `PrimitiveArray[T: PrimitiveType]`

A single array struct replaces both `PrimitiveArray` and `TemporalArray`:

```mojo
struct PrimitiveArray[T: PrimitiveType]:
    var dtype: DynType          # full logical type at runtime
    var length: Int
    var nulls: Int
    var offset: Int
    var bitmap: Optional[Bitmap[mut=False]]
    var buffer: Buffer[mut=False]
```

The `dtype` field is the key addition:

- For `NumericType` arrays it is always derivable from `T` (e.g. `Int32Type` →
  `DynType(Int32Type())`), so it is slightly redundant but keeps the struct
  uniform.
- For `TemporalType` arrays it carries the time unit and timezone.
- For `DecimalType` arrays it carries precision and scale.

`T.native` gives the physical `DType` for all buffer reads and SIMD operations,
regardless of which sub-trait `T` belongs to.

### Type aliases

```mojo
# Numeric (unchanged names, now all PrimitiveArray)
comptime Int8Array    = PrimitiveArray[Int8Type]
comptime Int32Array   = PrimitiveArray[Int32Type]
# ... etc.

# Temporal (drop TemporalArray, same names)
comptime Date32Array    = PrimitiveArray[Date32Type]
comptime TimestampArray = PrimitiveArray[TimestampType]
# ... etc.

# Decimal (new)
comptime Decimal32Array  = PrimitiveArray[Decimal32Type]
comptime Decimal64Array  = PrimitiveArray[Decimal64Type]
comptime Decimal128Array = PrimitiveArray[Decimal128Type]
comptime Decimal256Array = PrimitiveArray[Decimal256Type]
```

## Effect on Arithmetic Kernels

### Existing numeric kernels — minimal change

Change the trait bound from `PrimitiveType` to `NumericType`. Everything else is
identical.

```mojo
# Before
def _binary[T: PrimitiveType, func: ...](
    left: PrimitiveArray[T], right: PrimitiveArray[T], ...
) raises -> PrimitiveArray[T]:

# After
def _binary[T: NumericType, func: ...](
    left: PrimitiveArray[T], right: PrimitiveArray[T], ...
) raises -> PrimitiveArray[T]:
```

Decimal and temporal arrays are excluded at compile time by the `NumericType` bound.
No runtime check needed.

### Decimal arithmetic — custom overloads

Decimal addition is fundamentally different from numeric addition:

1. **Scale alignment** — the lower-scale operand must be multiplied by `10^(scale_diff)`
   before adding. This is a runtime value, so it cannot be expressed as a compile-time
   SIMD `func` parameter.
2. **Result type changes** — the output precision and scale are derived from the inputs:
   ```
   result_scale     = max(s1, s2)
   result_precision = max(p1-s1, p2-s2) + result_scale + 1
   ```
   The result carries a different `dtype` than either input.
3. **Overflow promotion** — Decimal128 arithmetic requires 256-bit intermediates to
   prevent overflow during scale alignment. There are no 128-bit SIMD lanes, so the
   `vectorize`/`elementwise` path does not apply. The kernel uses a scalar element-wise
   loop instead.

Decimal kernels take a `DecimalType` bound and read precision/scale from `array.dtype`:

```mojo
def add[T: DecimalType](
    left: PrimitiveArray[T],
    right: PrimitiveArray[T],
) raises -> PrimitiveArray[T]:
    var left_type  = left.dtype.as_decimal()
    var right_type = right.dtype.as_decimal()
    var s1 = left_type.scale()
    var s2 = right_type.scale()
    var result_scale     = max(s1, s2)
    var result_precision = max(left_type.precision()  - s1,
                               right_type.precision() - s2) + result_scale + 1
    var result_dtype = DynType(decimal128(result_precision, result_scale))
    # scalar loop: promote each pair to wider type, align, add, check overflow
    ...
    return PrimitiveArray[T](dtype=result_dtype^, ...)
```

Multiplication, division, and other decimal operations follow the same pattern with
their own result-type formulas per the SQL standard.

## Physical Storage of Decimal Values

| Type | `native` DType | Bytes | Max precision |
|---|---|---|---|
| `Decimal32Type` | `DType.int32` | 4 | 9 |
| `Decimal64Type` | `DType.int64` | 8 | 18 |
| `Decimal128Type` | `DType.int128` | 16 | 38 |
| `Decimal256Type` | `DType.int256` | 32 | 76 |

Values are stored as two's complement scaled integers: the logical value is
`stored_integer * 10^(-scale)`. Negative scale is valid (e.g. scale=-2 means the unit
is hundreds).

For Decimal256, Mojo has no native `DType.int256`, so a custom `Int256` struct
is needed (matching Arrow Rust's `i256 { low: UInt128, high: Int128 }`).

## Migration Notes

- `TemporalArray[T]` is removed. Rename to `PrimitiveArray[T]` throughout. Type aliases
  (`TimestampArray` etc.) preserve the public names.
- `DynArray.VariantType` gains decimal entries; `as_decimal32()`, `as_decimal64()`,
  `as_decimal128()`, `as_decimal256()` shorthand accessors are added alongside the
  existing temporal shorthands.
- `binary_array_dispatch` gains a decimal branch that routes to decimal-specific
  overloads rather than forwarding to the generic `add[T]` numeric overload.
- `TrivialRegisterPassable` and `Defaultable` move from `PrimitiveType` to `NumericType`.
  Any site that had `T: PrimitiveType` and relied on those constraints should be updated
  to `T: NumericType`.

## Prior Art

This design mirrors Arrow Rust's approach:

- `DataType::Decimal128(precision, scale)` carries runtime logical info in the enum
  variant, while `Decimal128Type {}` is a zero-sized marker implementing
  `ArrowPrimitiveType` with `type Native = i128`.
- `Decimal128Array = PrimitiveArray<Decimal128Type>` — a type alias, not a new struct.
- Arithmetic kernels in `arrow-arith` use separate `decimal::*` modules that operate
  on the runtime precision/scale rather than reusing the numeric kernel infrastructure.

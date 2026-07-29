# Dynamic Dispatch Design for Marrow

## Problem

Mojo lacks runtime polymorphism (no virtual methods, no trait objects). Marrow needs
type-erased containers (`DynArray`, `DynBuilder`, `AnyConverter`) that dispatch to
concrete typed implementations at runtime — e.g. holding `PrimitiveArray[int32]`,
`ListArray`, and `StructArray` in a single `List[DynArray]`.

## Mechanism: Function-Pointer Trampolines

Each type-erased container (`Any*`) stores:

1. **`ArcPointer[NoneType]`** — the typed value on the heap, erased to `NoneType`
2. **Function pointers** — one per virtual method slot, capturing the concrete type
3. **A drop trampoline** — for correct typed destruction

```
┌─────────────────────────────────────┐
│ DynArray                            │
│                                     │
│  _data: ArcPointer[NoneType] ──────────► heap: ArcInner<PrimitiveArray[int64]>
│  _virt_length:    fn(ptr) -> Int    │         { refcount, value }
│  _virt_null_count: fn(ptr) -> Int   │
│  _virt_is_valid:  fn(ptr,Int)->Bool │
│  _virt_drop:      fn(var ptr)       │
└─────────────────────────────────────┘
```

**Trampolines** are `@staticmethod` functions parameterized on `T: Array` that
`rebind` the erased pointer back to the concrete type and forward the call:

```mojo
@staticmethod
fn _tramp_length[T: Array](ptr: ArcPointer[NoneType]) -> Int:
    return rebind[ArcPointer[T]](ptr)[].length()
```

At construction time, the concrete type `T` is captured into the function pointers:

```mojo
@implicit
fn __init__[T: Array](out self, var value: T):
    var ptr = ArcPointer(value^)
    self._data = rebind[ArcPointer[NoneType]](ptr^)
    self._virt_length = Self._tramp_length[T]
    # ...
```

### Copying

Copies are O(1) ref-count bumps via `ArcPointer.copy()`. The function pointers
are copied verbatim — they encode the concrete type, not the instance.

### Destruction

A dedicated `_virt_drop` trampoline rebinds and drops as the correct type:

```mojo
@staticmethod
fn _tramp_drop[T: Array](var ptr: ArcPointer[NoneType]):
    var typed = rebind[ArcPointer[T]](ptr^)
    _ = typed^  # drop with correct destructor
```

`__del__` uses `deinit self` (not `var self`) to move `_data` out for the drop
trampoline.

### Downcasting

`downcast[T]()` returns `ArcPointer[T]` with a bumped refcount. The caller
must ensure the type matches (no runtime check). The returned handle keeps
data alive independently of the `DynArray`:

```mojo
fn downcast[T: Array](self) -> ArcPointer[T]:
    return rebind[ArcPointer[T]](self._data.copy())
```

## Virtual Method Surfaces

Based on analysis of the Arrow C++ codebase (`libarrow`), these are the dynamic
methods each type-erased container needs.

### DynArray

Arrow's C++ `Array` has only a virtual destructor. All other methods operate on
the flat `ArrayData` struct and dispatch via the Visitor pattern externally. We
adopt a similar minimal surface:

| Method | Signature | Notes |
|--------|-----------|-------|
| `length` | `(self) -> Int` | Array length |
| `null_count` | `(self) -> Int` | Cached null count |
| `is_valid` | `(self, Int) -> Bool` | Validity bitmap check |
| `type` | `(self) -> DataType` | Runtime type introspection |
| `n_children` | `(self) -> Int` | Number of child arrays |
| `child` | `(self, Int) -> DynArray` | Access child by index |
| `n_buffers` | `(self) -> Int` | Number of buffers |
| `slice` | `(self, Int, Int) -> DynArray` | Zero-copy slice |

The trait that typed arrays implement:

```mojo
trait Array(Movable, ImplicitlyDestructible):
    fn length(self) -> Int: ...
    fn null_count(self) -> Int: ...
    fn is_valid(self, index: Int) -> Bool: ...
    fn type(self) -> DataType: ...
    fn n_children(self) -> Int: ...
    fn child(self, index: Int) -> DynArray: ...
    fn n_buffers(self) -> Int: ...
    fn slice(self, offset: Int, length: Int) -> DynArray: ...
```

Typed arrays: `PrimitiveArray[T: DataType]`, `ListArray`, `StructArray`,
`StringArray`, `FixedSizeListArray`.

### DynBuilder

Arrow's `ArrayBuilder` has 6 pure virtual methods — the richest virtual surface
of the three hierarchies. Builders are inherently polymorphic because composite
builders (struct, list) hold child builders whose types are determined at runtime.

| Method | Signature | Notes |
|--------|-----------|-------|
| `append_null` | `(mut self)` | Append a null |
| `append_nulls` | `(mut self, Int)` | Append n nulls |
| `append_empty_value` | `(mut self)` | Append zero-initialized non-null |
| `append_empty_values` | `(mut self, Int)` | Append n empty values |
| `finish` | `(mut self) -> DynArray` | Produce the array |
| `type` | `(self) -> DataType` | Builder's target type |
| `length` | `(self) -> Int` | Current length |
| `null_count` | `(self) -> Int` | Current null count |
| `resize` | `(mut self, Int)` | Resize capacity |
| `reset` | `(mut self)` | Reset to empty |
| `append_scalar` | `(mut self, Scalar)` | Type-erased value append |
| `append_array_slice` | `(mut self, DynArray, Int, Int)` | Append from existing array |

The trait:

```mojo
trait Builder(Movable, ImplicitlyDestructible):
    fn append_null(mut self) raises: ...
    fn append_nulls(mut self, n: Int) raises: ...
    fn append_empty_value(mut self) raises: ...
    fn append_empty_values(mut self, n: Int) raises: ...
    fn finish(mut self) raises -> DynArray: ...
    fn type(self) -> DataType: ...
    fn length(self) -> Int: ...
    fn null_count(self) -> Int: ...
    fn resize(mut self, capacity: Int) raises: ...
    fn reset(mut self): ...
```

Typed builders: `PrimitiveBuilder[T: DataType]`, `ListBuilder`, `StructBuilder`,
`StringBuilder`, `FixedSizeListBuilder`.

### AnyConverter (Python → Arrow)

Arrow's `Converter` base class handles Python object → Arrow array conversion.
The virtual surface is small because most type-specific logic lives in the
`Append` override:

| Method | Signature | Notes |
|--------|-----------|-------|
| `append` | `(mut self, PythonObject) raises` | Convert + append one value |
| `extend` | `(mut self, PythonObject, Int) raises` | Convert a sequence |
| `finish` | `(mut self) raises -> DynArray` | Produce the array |
| `type` | `(self) -> DataType` | Target type |
| `reserve` | `(mut self, Int) raises` | Pre-allocate |
| `reset` | `(mut self)` | Reset to empty |

The trait:

```mojo
trait Converter(Movable, ImplicitlyDestructible):
    fn append(mut self, value: PythonObject) raises: ...
    fn extend(mut self, values: PythonObject, size: Int) raises: ...
    fn finish(mut self) raises -> DynArray: ...
    fn type(self) -> DataType: ...
    fn reserve(mut self, capacity: Int) raises: ...
    fn reset(mut self): ...
```

Typed converters: `PrimitiveConverter[T: DataType]`, `ListConverter`,
`StructConverter`, `StringConverter`.

Arrow uses `MakeConverter(DataType) -> Converter` as a factory that switches on
the type code. We would use the same pattern via `DataTypeVisitor`.

## Type Relationships

```
trait Array ──────────────► DynArray (type-erased, fn-ptr vtable)
  │  PrimitiveArray[T]         │  .downcast[T]() -> ArcPointer[T]
  │  ListArray                 │  .length(), .null_count(), .is_valid()
  │  StructArray               │  .type(), .child(), .slice()
  │  StringArray               │
  │  FixedSizeListArray        │

trait Builder ────────────► DynBuilder (type-erased, fn-ptr vtable)
  │  PrimitiveBuilder[T]       │  .downcast[T]() -> ArcPointer[T]
  │  ListBuilder               │  .append_null(), .finish(), .type()
  │  StructBuilder             │  .append_scalar(), .append_array_slice()
  │  StringBuilder             │
  │  FixedSizeListBuilder      │

trait Converter ──────────► AnyConverter (type-erased, fn-ptr vtable)
  │  PrimitiveConverter[T]     │  .downcast[T]() -> ArcPointer[T]
  │  ListConverter             │  .append(), .extend(), .finish()
  │  StructConverter           │
  │  StringConverter           │
```

All three `Any*` types follow the same structural pattern:
- `ArcPointer[NoneType]` for heap storage
- Function pointers for each virtual method
- `@implicit __init__[T: Trait]` for type erasure
- `__copyinit__` with `.copy()` for O(1) ref-count bumps
- `__del__(deinit self)` with typed drop trampoline
- `downcast[T]() -> ArcPointer[T]` for recovery

## Design Decisions

**Why function pointers over Variant?**
Variant requires enumerating all types upfront and grows linearly with type count.
Function pointers are open — any type conforming to the trait can be erased without
modifying the container. This is critical for `PrimitiveArray[T]` where `T` can be
any `DataType`.

**Why `ArcPointer[NoneType]` + `rebind`?**
Mojo's `ArcPointer[T]` has uniform layout regardless of `T` (it stores a pointer
to a heap block). `rebind` reinterprets the bits without copying or allocating.
This gives us type erasure with zero overhead beyond the function pointer call.

**Why `deinit self` in `__del__`?**
Mojo requires `deinit` (not `var`) to move fields out of self during destruction.
`var self` causes "field destroyed out of the middle of a value" errors.

**Why `ImplicitlyCopyable` on `DynArray` but not on the `Array` trait?**
`DynArray` needs `ImplicitlyCopyable` to be stored in `List[DynArray]`. The trait
doesn't require it because concrete types like `ListArray` contain `List` fields
which aren't `ImplicitlyCopyable`. The `DynArray` wrapper adds copyability through
`ArcPointer` ref-counting.

**Why a generic `downcast[T]` instead of `as_primitive[T]`, `as_list()`, etc.?**
`DynArray` is defined before the concrete types (it only depends on the `Array`
trait). Adding methods that return concrete types would create circular references.
A generic `downcast[T: Array]` avoids this by using the trait bound.

## Prototype Status

The current prototype in `marrow/dyn.mojo` implements `DynArray` with 3 virtual
methods (`length`, `null_count`, `is_valid`) plus typed destruction. Tests cover
all array types, heterogeneous lists, nested structures, copy semantics, refcount
safety, and upcast/downcast round-trips.

Next steps:
- Expand `Array` trait with `type()`, `child()`, `slice()`
- Implement `DynBuilder` with the same pattern
- Implement `AnyConverter` for Python interop
- Replace the existing `Array`/`Builder` types in `marrow/arrays.mojo` and
  `marrow/builders.mojo` with this pattern

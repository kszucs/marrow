# marrow.expr.ibis — hooking to kernels & enabling fusion

## Where we are

`marrow.expr.ibis` is a pure **type architecture**: value families are traits
(`NumericValue` / `BoolValue` / `StringValue` / `ListValue`), operations are node
structs whose family + `OutType` encode Arrow promotion, and kernels are name
markers. Nothing executes yet.

`marrow.expr.values` (the older layer) already proves *how* to execute + fuse
numeric/bool trees, and `marrow.kernels.*` already supply the compute:

- `BinaryKernel.core[T, W](a, b) -> SIMD[T, W]` — the per-lane SIMD functor
- `BinaryKernel.apply[…](arrays) -> array` — the eager, whole-array kernel

So "hook to kernels + enable fusion" = give ibis nodes an `execute` verb, let the
lane-computable ones fuse through the existing `core`, and route everything to the
existing kernels — **promotion stays in the value hierarchy, compute lives in the
kernel** (the two are already cleanly split).

## The taxonomy that drives the struct architecture

| Bucket | Ops | Mechanism |
|---|---|---|
| **lane** (fixed→fixed) | `+ - * / %`, `neg`/`abs`/`ceil`/…, compares, `& | ^ ~`, cast, `isnull` | per-lane `core[W]`; **fuses** |
| **reduction** (N→1) | `sum`/`mean`/`min`/`max` | consume a lane's `execute`, emit a **scalar** |
| **materialize→lane leaf** (var-len in, fixed out) | string `==`/`startswith`/`contains`, `array_contains`, `array_length` | eager kernel **once** → typed array → re-enters fusion as a lane leaf |
| **var-len out** | `upper`/`lower`/`reverse` | `StringValue`; `execute → StringArray`; not a lane |

The bucket is *which trait/branch a node is*, never a runtime tag.

## Struct architecture

### 1. `Value.execute` — the uniform verb (leverages `ArrayType`)

```mojo
trait Value(...):
    comptime OutType: DataType
    def execute(self, batch: RecordBatch) raises -> Self.OutType.ArrayType
```

`Self.OutType.ArrayType` is exactly the dtype→array associated type we just added,
so `execute` returns the concrete companion array (`PrimitiveArray[T]` / `BoolArray`
/ `StringArray` / `ListArray`) with no type erasure.

### 2. Lane sub-families carry `core[W]` + a fusing `execute` default

```mojo
trait NumericLane(NumericValue):
    comptime NativeType: DType                                   # OutType.native
    def core[W: Int](self, batch, idx) -> SIMD[Self.NativeType, W]  # fusion primitive
    def execute(self, batch) -> PrimitiveArray[Self.OutType]:       # default: vectorize core
        ... one pass over core[W] ...

trait BoolLane(BoolValue):
    def core[W: Int](self, batch, idx) -> SIMD[DType.bool, W]
    def execute(self, batch) -> BoolArray: ...                       # default: vectorize, bit-pack
```

A node that is a lane requires **lane children** and is itself a lane, so the
compiler inlines the whole `core` chain into one loop — zero intermediates.
`NumericValue`/`BoolValue` remain the *family* (what composes); the `…Lane`
refinement is *what fuses*. Reductions and var-len-output nodes stay plain
`NumericValue`/`StringValue` (they `execute` but have no `core`).

### 3. Fusable nodes are parameterized by the **real** compute kernels

The node keeps promotion (`OutType`); the kernel supplies `core`:

```mojo
struct NumericBinary[K: BinaryKernel, L: NumericLane, R: NumericLane](NumericLane):
    comptime OutType   = highest_precedence[L, R]        # promotion — value layer
    comptime NativeType = Self.OutType.native
    def core[W](self, batch, idx) -> SIMD[Self.NativeType, W]:
        var l = self.left.core[W](batch, idx).cast[Self.NativeType]()
        var r = self.right.core[W](batch, idx).cast[Self.NativeType]()
        return Self.K.core[Self.NativeType, W](l, r)     # compute — kernel

comptime Add = NumericBinary[kernels.AddKernel, _, _]    # real AddKernel (has core)
```

The ibis marker kernels collapse into the real kernels for every lane op — the
real kernel already *is* `name + core`, and promotion was never on it. Ops with
no lane kernel (string, list, reductions) keep a marker + a boundary `execute`.

### 4. Leaves

- `Column[T]` lane `core[W]` = `batch.column(name).as_<T>().values().load[W](idx)`.
- `Literal[T]` lane `core[W]` = broadcast its `T.ScalarType` value.

### 5. Boundaries (materialize once → lane leaf)

`Equal(s, t)` over strings, `startswith`, `array_contains`, `array_length` are
`BoolLane`/`NumericLane` boundary nodes: their `execute`/`prepare` runs the eager
kernel **once** into a typed array, cached in an `ArcPointer`; their `core[W]`
then just `load[W]`s that cache — so a mixed expr like
`startswith(s, "x") & (a > b)` runs the boundary once and one fused pass.
Reductions (`sum`) `execute` the child lane then reduce to a typed scalar.

## Phasing

1. **Numeric lane fusion** (this pass): `NumericLane`, `Column`/`Literal` numeric
   leaves, `NumericBinary`/`FloatBinary`/`NumericUnary`/`FloatUnary` → real kernels;
   `(a + b) * c` fuses and `.execute(batch)` matches PyArrow.
2. **Bool lanes**: `BoolLane`, compares + `& | ^ ~` over numeric; `(a > b) & (b < c)`.
3. **Boundaries**: string `==`/`startswith`/`contains`, `array_*`, materialize-cache.
4. **Reductions**: `sum`/`mean`/`min`/`max` → typed scalar.
5. **Var-len out**: `upper`/`lower`/`reverse` → `StringArray`.

Keep the small-binary property (closed per-dtype kernels, no open dispatch) —
gate on `benchmarks/binary_size/`.

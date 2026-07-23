"""Comptime-typed expression system with fused execution — the foundation of
`marrow.expr`.

Value families are traits, operations are node structs, and each node statically
conforms to the family of its *output*. The numeric family additionally
*executes*: it is hooked to the real `marrow.kernels` (which supply the `core[T,W]`
SIMD functors) and fuses lane-computable chains into a single vectorized pass.

Layers:
  * `Value.execute(batch) -> Self.ArrayType` — the uniform verb (abstract on
    `Value` so `AnyValue` can call it on any boxed node). `ArrayType` is a *direct*
    associated member (each node fixes it to its concrete result array), a single
    projection that always reduces — so a node returns / a parent consumes the
    concrete array with neither a `rebind` nor a reducer helper. (Spelling it as
    the dtype's `OutType.ArrayType` would be a double projection Mojo won't reduce
    at the call site.) Each family *refines* `execute`'s return to its concrete
    array — `NumericValue` to `PrimitiveArray[Self.OutType]`, `StringValue` to
    `BinaryLikeArray[Self.OutType]`, `ListValue` to `ListLikeArray[Self.OutType]`,
    `BoolValue` to `BoolArray` — so a child's `execute()` yields a fully typed
    array at the call site and consumers pass it straight to the typed kernels with
    no `rebind` anywhere. `AnyValue` boxes any node and `.execute(batch)`s it to an
    `AnyArray` (`.to_any()`).
  * `NumericValue` **is** the numeric lane: it refines `OutType` to `NumericType`,
    adds the `core[W]` SIMD primitive, and its `execute` vectorizes `core` across
    the whole tree — composite nodes call the kernel's `core` on their children's
    `core`, so the compiler inlines the entire chain (zero intermediate arrays).
  * Promotion lives in the value nodes (`OutType`); compute lives in the kernel.
    Every op node is parameterized by a real `marrow.kernels` kernel — arithmetic /
    compare / boolean / aggregate / string / list (`length`, `contains`) are all
    implemented. No kernels are defined here.
  * `StringValue` **executes** by materializing: leaves (`StringColumn`,
    `StringConst`) resolve/broadcast to a `StringArray`, unary ops (`upper`,
    `lower`, `strip`, `reverse`, `capitalize`, …) apply a `StringMapKernel`, and
    predicates (`startswith`, `endswith`, `contains`) apply a
    `StringPredicateKernel` → `BoolValue`. Variable-width UTF-8 has no
    fixed lane, so string ops do not fuse (unlike the numeric lane); `length` is
    the exception — byte length is `offsets[i+1]-offsets[i]`, which `LengthKernel`
    vectorizes internally.
  * `BoolValue` **executes**: numeric comparisons (`<`, `>`, `==`, …) fuse
    (`NumericCompare` has a `core[W]` bool lane, bit-packed in one pass); boolean
    logic (`&`, `|`, `^`, `~`) materializes its `BoolValue` children and combines
    the masks with the boolean kernels (`BoolBinary` and/or/xor, `BoolUnary` not);
    and the unary predicates `is_null`/`not_null` (any family) and `is_nan`/`is_inf`
    (floating) materialize the operand and apply a `UnaryPredicateKernel`
    (`Predicate`);
    string `==`/`!=` materialize and compare element-wise (`StringPredicate`);
    `any`/`all` fold a bool column to a length-1 result (`BoolReduce`); and list
    `contains` scans each sublist for the search element (`ListContains`).
  * `ListValue` executes: `ListColumn` resolves the list column from the batch,
    `length()` counts elements per list (`ArrayLengthKernel`, offset subtraction)
    → an int32 boundary, and `contains(elem)` scans each sublist for membership
    (`ArrayContainsKernel`) → a `BoolValue`. Cross-family numeric-producing
    boundaries (reductions — `sum`/`product`/`mean`/`min`/`max` via one `Reduce`
    node, plus the family-agnostic `count` via `Count`) are non-lane `Value`
    nodes: they materialize the operand (the numeric lane fuses up to it), fold it
    through the real `marrow.kernels.aggregate` kernels, and return the scalar as a
    length-1 result array rather than fusing. `Reduce`'s output dtype is the
    kernel's own `AccType[A.OutType]` — each aggregate is the single source of
    truth for its result type.

`AnyValue` boxes either a comptime node (`[V: Value]`) or the runtime `DynValue`
interpreter (dedicated constructor) and exposes `execute` / `name` / `prune` /
`write_to`. Pruning is plumbed through `Value.prune` (conservative "unknown"
default; only `DynValue` overrides it with the real min/max rule) — the old
per-node comptime pruning is parked (a commented reference at the bottom).

Dedicated per-family leaves (`NumericColumn` / `StringColumn` / `ListColumn`,
`NumericLiteral` / `StringConst`) keep `core`/`NativeType` unconditional and the
hierarchy sharp; `col` / `lit` overload by dtype family.
"""

from std.sys import bit_width_of, size_of
from std.sys.info import simd_byte_width
from std.builtin.rebind import downcast
from std.builtin.simd import Scalar
from std.utils.index import IndexList
from std.algorithm.backend.vectorize import vectorize
from std.reflection import reflect

from .. import dtypes as dt
from ..dtypes import (
    DataType,
    NumericType,
    IntegerType,
    FloatingType,
    BoolType,
    StringLikeType,
    ListLikeType,
    DType,
)
from std.memory import ArcPointer

from ..scalars import PrimitiveScalar, StringScalar, Int64Scalar
from ..buffers import Buffer, Bitmap
from ..views import apply
from ..arrays import (
    Array,
    PrimitiveArray,
    BinaryLikeArray,
    ListLikeArray,
    BoolArray,
    AnyArray,
)
from ..builders import BinaryLikeBuilder, BoolBuilder
from ..tabular import RecordBatch
from .pruning import PruneStats, PruneBound
from .dynamic import DynValue
from ..kernels.helpers import Kernel
from ..kernels.arithmetic import (
    BinaryKernel,
    UnaryKernel,
    AddKernel,
    SubKernel,
    MulKernel,
    DivKernel,
    ModKernel,
    PowKernel,
    NegKernel,
    AbsKernel,
    CeilKernel,
    FloorKernel,
    RoundKernel,
    SignKernel,
    TruncKernel,
    SqrtKernel,
    ExpKernel,
    Exp2Kernel,
    LogKernel,
    Log2Kernel,
    Log10Kernel,
    Log1pKernel,
    SinKernel,
    CosKernel,
)
from ..kernels.compare import (
    BinaryCompareKernel,
    EqKernel,
    NeKernel,
    LtKernel,
    LeKernel,
    GtKernel,
    GeKernel,
)
from ..kernels.boolean import (
    BoolBinaryKernel,
    BoolUnaryKernel,
    UnaryPredicateKernel,
    AndKernel,
    OrKernel,
    NotKernel,
    XorKernel,
    IsNullKernel,
    NotNullKernel,
    IsNanKernel,
    IsInfKernel,
)
from ..kernels.aggregate import (
    AggKernel,
    SumKernel,
    MeanKernel,
    MinKernel,
    MaxKernel,
    CountKernel,
    ProductKernel,
    BoolReduceKernel,
    AnyKernel,
    AllKernel,
)
from ..kernels.string import (
    StringMapKernel,
    StringPredicateKernel,
    StartsWithKernel,
    EndsWithKernel,
    ContainsKernel,
    StringEqKernel,
    StringNeKernel,
    LengthKernel,
    UpperKernel,
    LowerKernel,
    ReverseKernel,
    StripKernel,
    LStripKernel,
    RStripKernel,
    CapitalizeKernel,
)
from ..kernels.nested import ArrayLengthKernel, ArrayContainsKernel


# ---------------------------------------------------------------------------
# Promotion rules — reusable parametric comptime aliases (rlz-style)
# ---------------------------------------------------------------------------


def _rank[T: DataType]() -> Int:
    comptime if conforms_to(T, NumericType):
        comptime N = downcast[T, NumericType]()
        return bit_width_of[N.native]() + (
            1000 if N.native.is_floating_point() else 0
        )
    else:
        return 0


comptime highest_precedence[L: NumericValue, R: NumericValue] = L.OutType if (
    _rank[L.OutType]() >= _rank[R.OutType]()
) else R.OutType
"""Output dtype is the wider operand — `Add(int32, int64) → int64`. Bound on
`NumericValue` so the result is a `NumericType` (has `.native`)."""


# ---------------------------------------------------------------------------
# Value — base trait; `execute` is the uniform verb
# ---------------------------------------------------------------------------


trait Value(Copyable, ImplicitlyDeletable, Movable, Writable):
    """Every expression node. `execute` returns the dtype's companion array.
    Copies are explicit (`.copy()`); nodes are not `ImplicitlyCopyable`."""

    comptime OutType: DataType

    # The node's concrete result array. A *direct* member (not `OutType.ArrayType`)
    # so `execute`'s return is a single associated-type projection that always
    # reduces — a node returns / a parent consumes the concrete array with no
    # `rebind` or reducer helper. Each node declares it from its own type params
    # (spelling it via `Self.OutType`, a sibling associated type, would make the
    # default self-referential and Mojo rejects that as a recursive reference).
    comptime ArrayType: Array

    # Abstract — the numeric family fuses (vectorized), every other concrete node
    # materializes through its real kernel. Declared here (not only per-family) so
    # `AnyValue`'s trampoline can `.execute(batch).to_any()` on any `V: Value`.
    def execute(self, batch: RecordBatch) raises -> Self.ArrayType:
        ...

    def name(self) -> String:
        return String()

    def prune(self, stats: PruneStats) raises -> PruneBound:
        """Evaluate this node against per-column statistics for pruning. The
        conservative default returns unknown bounds / maybe-true; the runtime
        `DynValue` interpreter overrides it with the real min/max rule (see
        `marrow.expr.pruning`). Comptime-node-specific pruning is not yet ported
        (the old per-node `prune` methods are kept as a commented reference at the
        bottom of this module), so every comptime node currently inherits this
        conservative default — a caller only ever skips data it has proven cannot
        match."""
        return PruneBound.unknown()

    def isnull(self) -> IsNull[Self]:
        """Null predicate — any value in any family yields a `BoolValue`."""
        return IsNull(self.copy())

    def notnull(self) -> NotNull[Self]:
        """Non-null predicate — any value in any family yields a `BoolValue`."""
        return NotNull(self.copy())

    def count(self) -> Count[Self]:
        """Count of valid (non-null) values — a reduction available in every
        family (it reads only the validity bitmap)."""
        return Count(self.copy())


trait NumericValue(Value):
    """The numeric lane: refines `OutType` to `NumericType`, carries the `core[W]`
    SIMD fusion primitive + a fusing `execute`, and the arithmetic/comparison
    operator surface. Arithmetic nodes hook to the real kernels."""

    comptime OutType: NumericType
    comptime NativeType: DType

    @always_inline
    def core[
        W: Int
    ](self, batch: RecordBatch, idx: Int) -> SIMD[Self.NativeType, W]:
        ...

    # Abstract, NOT a shared default: re-defaulting the base `Value.execute` (which
    # returns `Self.ArrayType`) in this sub-trait recurses when a node's `ArrayType`
    # transitively references a `NumericValue` child (the compiler loops elaborating
    # the child's own `execute`). Each numeric node overrides `execute` with a
    # one-liner delegating to `_fused` — a *differently named* method default, which
    # does not trip the recursion.
    def execute(
        self, batch: RecordBatch
    ) raises -> PrimitiveArray[Self.OutType]:
        ...

    def _fused(self, batch: RecordBatch) raises -> PrimitiveArray[Self.OutType]:
        """The shared fused body: fill a buffer with `core` in one pass (no
        intermediate arrays) through `views.apply`'s producer overload — the same
        CPU serial/parallel dispatch the kernels use. Returns the concrete
        `PrimitiveArray[Self.OutType]` directly (the family refines `execute` to
        that type, so every numeric node's `ArrayType` is `PrimitiveArray[Self.OutType]`
        and no rebind is needed)."""
        comptime native = Self.NativeType
        var length = batch.num_rows()
        var buf = Buffer.alloc_uninit[native](length)

        @parameter
        @always_inline
        def producer[W: Int](i: Int) -> SIMD[native, W]:
            return self.core[W](batch, i)

        apply[native, producer](buf.view[native](0, length))
        return PrimitiveArray[Self.OutType](
            dtype=Self.OutType(),
            length=length,
            nulls=0,
            offset=0,
            bitmap=None,
            buffer=buf.to_immutable(),
        )

    # --- arithmetic (fusable, real kernels) --------------------------------

    def __add__[Rhs: NumericValue](self, o: Rhs) -> Add[Self, Rhs]:
        return Add(self.copy(), o.copy())

    def __sub__[Rhs: NumericValue](self, o: Rhs) -> Sub[Self, Rhs]:
        return Sub(self.copy(), o.copy())

    def __mul__[Rhs: NumericValue](self, o: Rhs) -> Mul[Self, Rhs]:
        return Mul(self.copy(), o.copy())

    def __mod__[Rhs: NumericValue](self, o: Rhs) -> Mod[Self, Rhs]:
        return Mod(self.copy(), o.copy())

    def __truediv__[Rhs: NumericValue](self, o: Rhs) -> Div[Self, Rhs]:
        return Div(self.copy(), o.copy())

    def __pow__[Rhs: NumericValue](self, o: Rhs) -> Pow[Self, Rhs]:
        return Pow(self.copy(), o.copy())

    def __neg__(self) -> Neg[Self]:
        return Neg(self.copy())

    def abs(self) -> Abs[Self]:
        return Abs(self.copy())

    def ceil(self) -> Ceil[Self]:
        return Ceil(self.copy())

    def floor(self) -> Floor[Self]:
        return Floor(self.copy())

    def round(self) -> Round[Self]:
        return Round(self.copy())

    def sign(self) -> Sign[Self]:
        return Sign(self.copy())

    def trunc(self) -> Trunc[Self]:
        return Trunc(self.copy())

    # transcendental unary -> float64 (fused via the real kernels)
    def sqrt(self) -> Sqrt[Self]:
        return Sqrt(self.copy())

    def exp(self) -> Exp[Self]:
        return Exp(self.copy())

    def exp2(self) -> Exp2[Self]:
        return Exp2(self.copy())

    def ln(self) -> Ln[Self]:
        return Ln(self.copy())

    def log2(self) -> Log2[Self]:
        return Log2(self.copy())

    def log10(self) -> Log10[Self]:
        return Log10(self.copy())

    def log1p(self) -> Log1p[Self]:
        return Log1p(self.copy())

    def sin(self) -> Sin[Self]:
        return Sin(self.copy())

    def cos(self) -> Cos[Self]:
        return Cos(self.copy())

    # numeric -> bool predicates (type-only until bool execution is wired)
    def isnan(self) -> IsNan[Self]:
        return IsNan(self.copy())

    def isinf(self) -> IsInf[Self]:
        return IsInf(self.copy())

    # --- cast (fused, numeric -> numeric) ----------------------------------

    def cast[Target: NumericType](self, target: Target) -> Cast[Target, Self]:
        """Cast to another numeric dtype (`col.cast(int64)`). Fuses into the
        numeric lane; `target` is only for dtype inference."""
        return Cast[Target, Self](self.copy())

    # --- reductions (N -> 1, boundary; non-lane `Value` result nodes) -------

    def sum(self) -> Sum[Self]:
        return Sum(self.copy())

    def product(self) -> Product[Self]:
        return Product(self.copy())

    def mean(self) -> Mean[Self]:
        return Mean(self.copy())

    def min(self) -> Min[Self]:
        return Min(self.copy())

    def max(self) -> Max[Self]:
        return Max(self.copy())

    # --- comparisons (-> BoolValue) ----------------------------------------

    def __lt__[Rhs: NumericValue](self, o: Rhs) -> Less[Self, Rhs]:
        return Less(self.copy(), o.copy())

    def __le__[Rhs: NumericValue](self, o: Rhs) -> LessEqual[Self, Rhs]:
        return LessEqual(self.copy(), o.copy())

    def __gt__[Rhs: NumericValue](self, o: Rhs) -> Greater[Self, Rhs]:
        return Greater(self.copy(), o.copy())

    def __ge__[Rhs: NumericValue](self, o: Rhs) -> GreaterEqual[Self, Rhs]:
        return GreaterEqual(self.copy(), o.copy())

    def __eq__[Rhs: NumericValue](self, o: Rhs) -> Equal[Self, Rhs]:
        return Equal(self.copy(), o.copy())

    def __ne__[Rhs: NumericValue](self, o: Rhs) -> NotEqual[Self, Rhs]:
        return NotEqual(self.copy(), o.copy())


trait BoolValue(Value):
    """Boolean-typed nodes: logical operator surface. Every `BoolValue` outputs a
    `BoolArray`, so `execute` is refined to that concrete type — a `BoolValue`
    child's `execute()` resolves to `BoolArray` at the call site, so composite bool
    nodes compose their children with no `rebind`. Each bool node also declares
    `comptime ArrayType = BoolArray` (a family default there does not satisfy the
    base `Value` requirement)."""

    def execute(self, batch: RecordBatch) raises -> BoolArray:
        ...

    def __and__[Rhs: BoolValue](self, o: Rhs) -> And[Self, Rhs]:
        return And(self.copy(), o.copy())

    def __or__[Rhs: BoolValue](self, o: Rhs) -> Or[Self, Rhs]:
        return Or(self.copy(), o.copy())

    def __xor__[Rhs: BoolValue](self, o: Rhs) -> Xor[Self, Rhs]:
        return Xor(self.copy(), o.copy())

    def __invert__(self) -> Not[Self]:
        return Not(self.copy())

    def any(self) -> Any[Self]:
        """True if any valid element is True (`any`) — a length-1 reduction."""
        return Any(self.copy())

    def all(self) -> All[Self]:
        """True if all valid elements are True (`all`) — a length-1 reduction.
        """
        return All(self.copy())


trait StringValue(Value):
    """String-typed nodes. Cross-family methods follow the *result*: `length()`
    yields a numeric boundary, `startswith()` a `BoolValue`. `OutType` refines to
    `StringLikeType` so `execute` can rebuild the typed string array from the
    erased kernel result."""

    comptime OutType: StringLikeType

    # Refined to the concrete `BinaryLikeArray[Self.OutType]` (a type application of
    # the sibling `OutType`, not a `.ArrayType` projection — so it reduces at every
    # call site, exactly like `BoolValue.execute -> BoolArray`). A `StringValue`
    # child's `execute()` therefore yields the concrete string array directly, so
    # consumers (`Length`, `StringUnary`, `StringPredicate`) call the typed kernels
    # with no `rebind`. Each string node still declares
    # `comptime ArrayType = BinaryLikeArray[Self.OutType]` to satisfy `Value`.
    def execute(
        self, batch: RecordBatch
    ) raises -> BinaryLikeArray[Self.OutType]:
        ...

    def length(self) -> Length[Self]:
        return Length(self.copy())

    def upper(self) -> Upper[Self]:
        return Upper(self.copy())

    def lower(self) -> Lower[Self]:
        return Lower(self.copy())

    def reverse(self) -> Reverse[Self]:
        return Reverse(self.copy())

    def strip(self) -> Strip[Self]:
        return Strip(self.copy())

    def lstrip(self) -> LStrip[Self]:
        return LStrip(self.copy())

    def rstrip(self) -> RStrip[Self]:
        return RStrip(self.copy())

    def capitalize(self) -> Capitalize[Self]:
        return Capitalize(self.copy())

    def startswith[Rhs: StringValue](self, o: Rhs) -> StartsWith[Self, Rhs]:
        return StartsWith(self.copy(), o.copy())

    def endswith[Rhs: StringValue](self, o: Rhs) -> EndsWith[Self, Rhs]:
        return EndsWith(self.copy(), o.copy())

    def contains[Rhs: StringValue](self, o: Rhs) -> Contains[Self, Rhs]:
        return Contains(self.copy(), o.copy())

    def __eq__[Rhs: StringValue](self, o: Rhs) -> StringEqual[Self, Rhs]:
        return StringEqual(self.copy(), o.copy())

    def __ne__[Rhs: StringValue](self, o: Rhs) -> StringNotEqual[Self, Rhs]:
        return StringNotEqual(self.copy(), o.copy())


trait ListValue(Value):
    """List-typed nodes (nested family). `length()` yields a numeric boundary,
    `contains()` a `BoolValue`. `OutType` refines to a list dtype so `execute` can
    rebuild the typed list array from the erased child (`ListLikeType` is not a
    `DataType` subtrait, so the bound is the intersection)."""

    comptime OutType: DataType & ListLikeType

    # Refined to the concrete `ListLikeArray[Self.OutType]` (same reasoning as
    # `StringValue.execute`): a `ListValue` child's `execute()` yields the concrete
    # list array directly, so `ArrayLength` / `ListContains` consume it with no
    # `rebind`. Each list node declares `comptime ArrayType = ListLikeArray[Self.OutType]`.
    def execute(self, batch: RecordBatch) raises -> ListLikeArray[Self.OutType]:
        ...

    def length(self) -> ArrayLength[Self]:
        return ArrayLength(self.copy())

    def contains[E: NumericValue](self, elem: E) -> ArrayContains[Self, E]:
        """Element-wise membership: `elem[i] ∈ list[i]` → a `BoolValue` (a literal
        element broadcasts). Numeric element types only."""
        return ArrayContains(self.copy(), elem.copy())


# ---------------------------------------------------------------------------
# Numeric lane nodes — real kernels + fused `core`
# ---------------------------------------------------------------------------


@fieldwise_init
struct NumericBinary[K: BinaryKernel, L: NumericValue, R: NumericValue](
    NumericValue
):
    """Arithmetic binary widening to the higher-precedence operand — `Add`, `Sub`,
    `Mul`, `Mod`. Operands cast to the promoted `NativeType`, then `K.core`."""

    comptime OutType = highest_precedence[Self.L, Self.R]

    comptime ArrayType = PrimitiveArray[Self.OutType]
    comptime NativeType = Self.OutType.native

    var left: Self.L
    var right: Self.R

    @always_inline
    def core[
        W: Int
    ](self, batch: RecordBatch, idx: Int) -> SIMD[Self.NativeType, W]:
        var l = self.left.core[W](batch, idx).cast[Self.NativeType]()
        var r = self.right.core[W](batch, idx).cast[Self.NativeType]()
        return Self.K.core[Self.NativeType, W](l, r)

    def execute(self, batch: RecordBatch) raises -> Self.ArrayType:
        return self._fused(batch)

    def write_to[W: Writer](self, mut writer: W):
        writer.write(Self.K.name, "(", self.left, ", ", self.right, ")")


@fieldwise_init
struct FloatBinary[K: BinaryKernel, L: NumericValue, R: NumericValue](
    NumericValue
):
    """Binary op whose result is always float64 — `Div`, `Pow`."""

    comptime OutType = dt.Float64Type

    comptime ArrayType = PrimitiveArray[Self.OutType]
    comptime NativeType = DType.float64

    var left: Self.L
    var right: Self.R

    @always_inline
    def core[
        W: Int
    ](self, batch: RecordBatch, idx: Int) -> SIMD[Self.NativeType, W]:
        var l = self.left.core[W](batch, idx).cast[Self.NativeType]()
        var r = self.right.core[W](batch, idx).cast[Self.NativeType]()
        return Self.K.core[Self.NativeType, W](l, r)

    def execute(self, batch: RecordBatch) raises -> Self.ArrayType:
        return self._fused(batch)

    def write_to[W: Writer](self, mut writer: W):
        writer.write(Self.K.name, "(", self.left, ", ", self.right, ")")


@fieldwise_init
struct NumericUnary[K: UnaryKernel, A: NumericValue](NumericValue):
    """Unary numeric op preserving the operand dtype — `Neg`, `Abs`, `Ceil`, ….
    """

    comptime OutType = Self.A.OutType

    comptime ArrayType = PrimitiveArray[Self.OutType]
    comptime NativeType = Self.A.NativeType

    var arg: Self.A

    @always_inline
    def core[
        W: Int
    ](self, batch: RecordBatch, idx: Int) -> SIMD[Self.NativeType, W]:
        return Self.K.core[Self.NativeType, W](self.arg.core[W](batch, idx))

    def execute(self, batch: RecordBatch) raises -> Self.ArrayType:
        return self._fused(batch)

    def write_to[W: Writer](self, mut writer: W):
        writer.write(Self.K.name, "(", self.arg, ")")


@fieldwise_init
struct FloatUnary[K: UnaryKernel, A: NumericValue](NumericValue):
    """Unary op whose result is always float64 — `sqrt`, `exp`, `ln`."""

    comptime OutType = dt.Float64Type

    comptime ArrayType = PrimitiveArray[Self.OutType]
    comptime NativeType = DType.float64

    var arg: Self.A

    @always_inline
    def core[
        W: Int
    ](self, batch: RecordBatch, idx: Int) -> SIMD[Self.NativeType, W]:
        var a = self.arg.core[W](batch, idx).cast[Self.NativeType]()
        return Self.K.core[Self.NativeType, W](a)

    def execute(self, batch: RecordBatch) raises -> Self.ArrayType:
        return self._fused(batch)

    def write_to[W: Writer](self, mut writer: W):
        writer.write(Self.K.name, "(", self.arg, ")")


@fieldwise_init
struct Cast[To: NumericType, A: NumericValue](NumericValue):
    """Fused numeric → numeric cast — reinterprets the operand's SIMD lane at the
    target dtype (`SIMD.cast`, truncating like the unchecked cast kernel), so it
    composes into the numeric lane like any other op (`col.cast(int64) + other`
    stays a single vectorized pass)."""

    comptime OutType = Self.To

    comptime ArrayType = PrimitiveArray[Self.OutType]
    comptime NativeType = Self.To.native

    var arg: Self.A

    @always_inline
    def core[
        W: Int
    ](self, batch: RecordBatch, idx: Int) -> SIMD[Self.NativeType, W]:
        return self.arg.core[W](batch, idx).cast[Self.NativeType]()

    def execute(self, batch: RecordBatch) raises -> Self.ArrayType:
        return self._fused(batch)

    def write_to[W: Writer](self, mut writer: W):
        writer.write("cast(", self.arg, ")")


# ---------------------------------------------------------------------------
# Boundary nodes — numeric-producing but non-lane (`Value`); materialize (future)
# ---------------------------------------------------------------------------


@fieldwise_init
struct StringLength[A: StringValue](Value):
    """String byte length → int32 boundary (`length()`). Not a numeric lane (its
    operand is a materialized string, not a fixed-width column), so it evaluates
    the child and calls `LengthKernel.apply` on the typed string array directly —
    no type erasure. `LengthKernel` vectorizes the offset subtraction internally.
    """

    comptime OutType = dt.Int32Type

    comptime ArrayType = PrimitiveArray[dt.Int32Type]
    var arg: Self.A

    def execute(self, batch: RecordBatch) raises -> Self.ArrayType:
        return LengthKernel.apply(self.arg.execute(batch))


@fieldwise_init
struct Counting[K: Kernel, A: ListValue](Value):
    """Unary op whose result is int32 — list `length()` (element count). A
    boundary: its operand is a materialized list, not a fixed-width column, so it
    evaluates the child and calls `ArrayLengthKernel.apply` on the typed list
    array directly (offset subtraction, vectorized internally)."""

    comptime OutType = dt.Int32Type

    comptime ArrayType = PrimitiveArray[dt.Int32Type]
    var arg: Self.A

    def execute(self, batch: RecordBatch) raises -> Self.ArrayType:
        return ArrayLengthKernel.apply(self.arg.execute(batch))


@fieldwise_init
struct Reduce[K: AggKernel, A: NumericValue](Value):
    """Whole-array numeric reduction — `sum`/`product` (widen to int64/float64),
    `mean` (float64), and `min`/`max` (operand dtype preserved). The output dtype
    is the kernel's own accumulator algebra, `K.AccType[A.OutType]`, so each
    kernel is the single source of truth for its result type (no separate
    per-node dtype rule).

    A materialization boundary: the numeric lane fuses up to the operand (which is
    computed in full), then folds it to a length-1 result array. `Count` is a
    separate node because it is family-agnostic (any input dtype → int64)."""

    comptime OutType = Self.K.AccType[Self.A.OutType]

    comptime ArrayType = PrimitiveArray[Self.K.AccType[Self.A.OutType]]
    var arg: Self.A

    def execute(self, batch: RecordBatch) raises -> Self.ArrayType:
        # The operand `execute()`s to a concrete `PrimitiveArray[A.OutType]`, so the
        # typed `K.reduce[V]` folds it to a `PrimitiveScalar[K.AccType[A.OutType]]`
        # with no erasure/downcast; broadcast the scalar to a length-1 result.
        return Self.K.reduce(self.arg.execute(batch)).repeat(1)

    def write_to[W: Writer](self, mut writer: W):
        writer.write(Self.K.name, "(", self.arg, ")")


@fieldwise_init
struct Count[A: Value](Value):
    """Whole-array valid (non-null) count — `count()`, available on any family.
    Result is a length-1 int64 array. Always `CountKernel` (family-agnostic), so
    unlike `Reduce` it carries no kernel parameter."""

    comptime OutType = dt.Int64Type

    comptime ArrayType = PrimitiveArray[dt.Int64Type]
    var arg: Self.A

    def execute(self, batch: RecordBatch) raises -> Self.ArrayType:
        # Valid count is validity metadata (`len - null_count`) available on every
        # `Array`, so `Count` reads it off the typed operand array directly — no
        # erasure, no dispatch, no downcast — and broadcasts to a length-1 result.
        var arr = self.arg.execute(batch)
        return Int64Scalar(Int64(len(arr) - arr.null_count())).repeat(1)

    def write_to[W: Writer](self, mut writer: W):
        writer.write(CountKernel.name, "(", self.arg, ")")


# ---------------------------------------------------------------------------
# Type-only nodes — bool / string families (execution is future work)
# ---------------------------------------------------------------------------


@fieldwise_init
struct NumericCompare[K: BinaryCompareKernel, L: NumericValue, R: NumericValue](
    BoolValue
):
    """Fused numeric comparison → a bit-packed `BoolArray` in one vectorized
    pass (no intermediate operand arrays). Operands cast to the left's native;
    `K.core` yields the SIMD bool lane, which `execute` bit-packs directly. As a
    `BoolValue` it composes with `&`/`|`/`~` into the logical surface."""

    comptime OutType = dt.BoolType
    comptime ArrayType = BoolArray
    comptime NativeType = Self.L.NativeType
    var left: Self.L
    var right: Self.R

    @always_inline
    def core[W: Int](self, batch: RecordBatch, idx: Int) -> SIMD[DType.bool, W]:
        var l = self.left.core[W](batch, idx)
        var r = self.right.core[W](batch, idx).cast[Self.NativeType]()
        return Self.K.core[Self.NativeType, W](l, r)

    def execute(self, batch: RecordBatch) raises -> Self.ArrayType:
        comptime native = Self.NativeType
        comptime width = simd_byte_width() // size_of[Scalar[native]]()
        var length = batch.num_rows()
        var bm = Bitmap.alloc_uninit(length)

        @parameter
        @always_inline
        def fill[W: Int, rank: Int, alignment: Int = 1](idx: IndexList[rank]):
            var i = idx[0]
            bm.view().store[W](i, self.core[W](batch, i))

        @always_inline
        def lane[W: Int](i: Int):
            fill[W, rank=1](IndexList[1](i))

        vectorize[width](length, lane)
        return BoolArray(
            length=length,
            nulls=0,
            offset=0,
            bitmap=None,
            buffer=bm.to_immutable(),
        )

    def write_to[W: Writer](self, mut writer: W):
        writer.write(Self.K.name, "(", self.left, ", ", self.right, ")")


@fieldwise_init
struct BoolBinary[K: BoolBinaryKernel, L: BoolValue, R: BoolValue](BoolValue):
    """Binary bool → bool logic (`and`/`or`/`xor`) over two `BoolValue` children.
    Each child materializes to a `BoolArray` (they may be heterogeneous predicates
    — a fused numeric compare, a string predicate, …), then `K.apply` combines the
    two bit-packed masks with 64-bit word ops."""

    comptime OutType = dt.BoolType
    comptime ArrayType = BoolArray
    var left: Self.L
    var right: Self.R

    def execute(self, batch: RecordBatch) raises -> Self.ArrayType:
        return Self.K.apply(
            self.left.execute(batch),
            self.right.execute(batch),
        )

    def write_to[W: Writer](self, mut writer: W):
        writer.write(Self.K.name, "(", self.left, ", ", self.right, ")")


@fieldwise_init
struct BoolUnary[K: BoolUnaryKernel, A: BoolValue](BoolValue):
    """Unary bool → bool op over a `BoolValue` child — currently negation
    (`not_`). Materializes the child mask and applies `K` (a `BoolUnaryKernel`).
    """

    comptime OutType = dt.BoolType
    comptime ArrayType = BoolArray
    var arg: Self.A

    def execute(self, batch: RecordBatch) raises -> Self.ArrayType:
        return Self.K.apply(self.arg.execute(batch))

    def write_to[W: Writer](self, mut writer: W):
        writer.write(Self.K.name, "(", self.arg, ")")


@fieldwise_init
struct BoolReduce[K: BoolReduceKernel, A: BoolValue](BoolValue):
    """`any()` / `all()` over a boolean column → a length-1 bool result. The
    aggregate is chosen by the kernel type param `K` (`AnyKernel`/`AllKernel`).
    Materializes the child mask, then folds it with the optimized bitmap
    reduction in `kernels.aggregate`."""

    comptime OutType = dt.BoolType
    comptime ArrayType = BoolArray
    var arg: Self.A

    def execute(self, batch: RecordBatch) raises -> Self.ArrayType:
        var builder = BoolBuilder(1)
        builder.append(Self.K.reduce(self.arg.execute(batch)))
        return builder.finish()

    def write_to[W: Writer](self, mut writer: W):
        writer.write(Self.K.name, "(", self.arg, ")")


@fieldwise_init
struct ListContains[L: ListValue, E: NumericValue](BoolValue):
    """Element-wise list membership → bool: `elem[i] ∈ list[i]`. Both operands
    materialize (a literal element broadcasts to every row), then
    `ArrayContainsKernel` scans each sublist. Null list rows propagate to null.
    """

    comptime OutType = dt.BoolType
    comptime ArrayType = BoolArray
    var arg: Self.L
    var elem: Self.E

    def execute(self, batch: RecordBatch) raises -> Self.ArrayType:
        # Both children `execute()` to concrete typed arrays (ListValue refines to
        # `ListLikeArray[L.OutType]`, NumericValue to `PrimitiveArray[E.OutType]`),
        # consumed directly by the typed kernel `apply` — no erase, no rebind.
        return ArrayContainsKernel.apply(
            self.arg.execute(batch),
            self.elem.execute(batch),
        )

    def write_to[W: Writer](self, mut writer: W):
        writer.write("array_contains(", self.arg, ", ", self.elem, ")")


@fieldwise_init
struct Predicate[K: UnaryPredicateKernel, A: Value](BoolValue):
    """Unary predicate `any family -> bool` — `is_null`/`not_null` (read validity)
    and `is_nan`/`is_inf` (floating values). Unlike `BoolUnary` (bool -> bool) the
    operand is any `Value`; materializes it and applies `K` (which uses the shared
    `views` helpers under the hood)."""

    comptime OutType = dt.BoolType
    comptime ArrayType = BoolArray
    var arg: Self.A

    def execute(self, batch: RecordBatch) raises -> Self.ArrayType:
        return Self.K.apply(self.arg.execute(batch).to_any())

    def write_to[W: Writer](self, mut writer: W):
        writer.write(Self.K.name, "(", self.arg, ")")


@fieldwise_init
struct StringUnary[K: StringMapKernel, A: StringValue](StringValue):
    comptime OutType = Self.A.OutType
    comptime ArrayType = BinaryLikeArray[Self.A.OutType]
    var arg: Self.A

    def execute(self, batch: RecordBatch) raises -> Self.ArrayType:
        # The child's `execute()` yields `BinaryLikeArray[A.OutType]` (StringValue
        # refines the return type), which `K.apply` consumes directly — no rebind.
        return Self.K.apply(self.arg.execute(batch))

    def write_to[W: Writer](self, mut writer: W):
        writer.write(Self.K.name, "(", self.arg, ")")


@fieldwise_init
struct StringPredicate[
    K: StringPredicateKernel, L: StringValue, R: StringValue
](BoolValue):
    """Binary string predicate producing a bool column — `startswith`,
    `endswith`, `contains`. Both operands materialize; the kernel compares
    element-wise (a constant pattern broadcasts through `StringConst`)."""

    comptime OutType = dt.BoolType
    comptime ArrayType = BoolArray
    var left: Self.L
    var right: Self.R

    def execute(self, batch: RecordBatch) raises -> Self.ArrayType:
        # Both operands' `execute()` yield concrete `BinaryLikeArray` (StringValue
        # refines the return type), consumed directly by the predicate kernel.
        return Self.K.apply(
            self.left.execute(batch),
            self.right.execute(batch),
        )

    def write_to[W: Writer](self, mut writer: W):
        writer.write(Self.K.name, "(", self.left, ", ", self.right, ")")


comptime Add = NumericBinary[AddKernel, _, _]
comptime Sub = NumericBinary[SubKernel, _, _]
comptime Mul = NumericBinary[MulKernel, _, _]
comptime Mod = NumericBinary[ModKernel, _, _]
comptime Div = FloatBinary[DivKernel, _, _]
comptime Pow = FloatBinary[PowKernel, _, _]
comptime Neg = NumericUnary[NegKernel, _]
comptime Abs = NumericUnary[AbsKernel, _]
comptime Ceil = NumericUnary[CeilKernel, _]
comptime Floor = NumericUnary[FloorKernel, _]
comptime Round = NumericUnary[RoundKernel, _]
comptime Sign = NumericUnary[SignKernel, _]
comptime Trunc = NumericUnary[TruncKernel, _]
comptime Sqrt = FloatUnary[SqrtKernel, _]
comptime Exp = FloatUnary[ExpKernel, _]
comptime Exp2 = FloatUnary[Exp2Kernel, _]
comptime Ln = FloatUnary[LogKernel, _]
comptime Log2 = FloatUnary[Log2Kernel, _]
comptime Log10 = FloatUnary[Log10Kernel, _]
comptime Log1p = FloatUnary[Log1pKernel, _]
comptime Sin = FloatUnary[SinKernel, _]
comptime Cos = FloatUnary[CosKernel, _]

comptime Sum = Reduce[SumKernel, _]
comptime Product = Reduce[ProductKernel, _]
comptime Mean = Reduce[MeanKernel, _]
comptime Min = Reduce[MinKernel, _]
comptime Max = Reduce[MaxKernel, _]

comptime Less = NumericCompare[LtKernel, _, _]
comptime LessEqual = NumericCompare[LeKernel, _, _]
comptime Greater = NumericCompare[GtKernel, _, _]
comptime GreaterEqual = NumericCompare[GeKernel, _, _]
comptime Equal = NumericCompare[EqKernel, _, _]
comptime NotEqual = NumericCompare[NeKernel, _, _]
comptime StartsWith = StringPredicate[StartsWithKernel, _, _]
comptime EndsWith = StringPredicate[EndsWithKernel, _, _]
comptime Contains = StringPredicate[ContainsKernel, _, _]
comptime StringEqual = StringPredicate[StringEqKernel, _, _]
comptime StringNotEqual = StringPredicate[StringNeKernel, _, _]

comptime And = BoolBinary[AndKernel, _, _]
comptime Or = BoolBinary[OrKernel, _, _]
comptime Xor = BoolBinary[XorKernel, _, _]
comptime Not = BoolUnary[NotKernel, _]
comptime Any = BoolReduce[AnyKernel, _]
comptime All = BoolReduce[AllKernel, _]
comptime IsNull = Predicate[IsNullKernel, _]
comptime NotNull = Predicate[NotNullKernel, _]
comptime IsNan = Predicate[IsNanKernel, _]
comptime IsInf = Predicate[IsInfKernel, _]

comptime Length = StringLength[_]
comptime Upper = StringUnary[UpperKernel, _]
comptime Lower = StringUnary[LowerKernel, _]
comptime Reverse = StringUnary[ReverseKernel, _]
comptime Strip = StringUnary[StripKernel, _]
comptime LStrip = StringUnary[LStripKernel, _]
comptime RStrip = StringUnary[RStripKernel, _]
comptime Capitalize = StringUnary[CapitalizeKernel, _]

comptime ArrayLength = Counting[ArrayLengthKernel, _]
comptime ArrayContains = ListContains[_, _]


# ---------------------------------------------------------------------------
# Leaves — dedicated per-family (single-family → unconditional core/NativeType)
# ---------------------------------------------------------------------------


struct NumericColumn[T: NumericType](NumericValue):
    """A named numeric column, resolved by name against `batch.schema` per pass.
    """

    comptime OutType = Self.T

    comptime ArrayType = PrimitiveArray[Self.T]
    comptime NativeType = Self.T.native

    var _name: String

    def __init__(out self, var name: String):
        self._name = name^

    @always_inline
    def core[
        W: Int
    ](self, batch: RecordBatch, idx: Int) -> SIMD[Self.NativeType, W]:
        return (
            batch.columns[batch.schema.get_field_index(self._name)]
            .as_primitive[Self.T]()
            .values()
            .load[W](idx)
        )

    def execute(self, batch: RecordBatch) raises -> Self.ArrayType:
        # Return the resolved column as-is, preserving its validity bitmap. The
        # fused-lane default (vectorize `core` into a fresh buffer) would drop
        # nulls; a leaf column needs no recompute anyway. Composite numeric nodes
        # still fuse through `core`, so this only affects standalone execution
        # (e.g. a column that is a reduction / predicate operand).
        return (
            batch.columns[batch.schema.get_field_index(self._name)]
            .as_primitive[Self.T]()
            .copy()
        )

    def name(self) -> String:
        return self._name.copy()

    def write_to[W: Writer](self, mut writer: W):
        writer.write("col(", self._name, ")")


struct NumericLiteral[T: NumericType](NumericValue):
    """A numeric constant, broadcast into every lane."""

    comptime OutType = Self.T

    comptime ArrayType = PrimitiveArray[Self.OutType]
    comptime NativeType = Self.T.native

    var _value: Scalar[Self.NativeType]

    def __init__(out self, value: Int):
        self._value = Scalar[Self.NativeType](value)

    @always_inline
    def core[
        W: Int
    ](self, batch: RecordBatch, idx: Int) -> SIMD[Self.NativeType, W]:
        return SIMD[Self.NativeType, W](self._value)

    def execute(self, batch: RecordBatch) raises -> Self.ArrayType:
        return self._fused(batch)

    def write_to[W: Writer](self, mut writer: W):
        writer.write("lit(", self._value, ")")


struct StringColumn[T: StringLikeType](StringValue):
    """A named string column (type architecture; execution is future work)."""

    comptime OutType = Self.T

    comptime ArrayType = BinaryLikeArray[Self.T]

    var _name: String

    def __init__(out self, var name: String):
        self._name = name^

    def name(self) -> String:
        return self._name.copy()

    def execute(self, batch: RecordBatch) raises -> Self.ArrayType:
        return (
            batch.columns[batch.schema.get_field_index(self._name)]
            .as_binary_like[Self.T]()
            .copy()
        )

    def write_to[W: Writer](self, mut writer: W):
        writer.write("col(", self._name, ")")


struct StringConst[T: StringLikeType](StringValue):
    """A string constant leaf holding a `StringScalar`."""

    comptime OutType = Self.T

    comptime ArrayType = BinaryLikeArray[Self.T]

    var _value: StringScalar

    def __init__(out self, var value: String):
        self._value = StringScalar(value^)

    def execute(self, batch: RecordBatch) raises -> Self.ArrayType:
        # Broadcast the constant to a full-length array (one row per batch row).
        var n = batch.num_rows()
        var builder = BinaryLikeBuilder[Self.T](capacity=n)
        var value = self._value.to_string()
        for _ in range(n):
            builder.append(value)
        return builder.finish()

    def write_to[W: Writer](self, mut writer: W):
        writer.write("lit(", self._value, ")")


struct ListColumn[T: DataType & ListLikeType](ListValue):
    """A named list column, resolved by name against `batch.schema` per pass."""

    comptime OutType = Self.T

    comptime ArrayType = ListLikeArray[Self.T]

    var _name: String

    def __init__(out self, var name: String):
        self._name = name^

    def name(self) -> String:
        return self._name.copy()

    def execute(self, batch: RecordBatch) raises -> Self.ArrayType:
        return (
            batch.columns[batch.schema.get_field_index(self._name)]
            .as_list_like[Self.T]()
            .copy()
        )

    def write_to[W: Writer](self, mut writer: W):
        writer.write("col(", self._name, ")")


# ---------------------------------------------------------------------------
# col / lit — overload by dtype family
# ---------------------------------------------------------------------------


def col[T: NumericType](var name: String, dtype: T) -> NumericColumn[T]:
    """Reference a numeric column — `col("a", int64)`."""
    return NumericColumn[T](name^)


def col[T: StringLikeType](var name: String, dtype: T) -> StringColumn[T]:
    """Reference a string column — `col("s", string)`."""
    return StringColumn[T](name^)


def col[
    T: DataType & ListLikeType
](var name: String, dtype: T) -> ListColumn[T]:
    """Reference a list column — `col("l", list_(int64))`."""
    return ListColumn[T](name^)


def lit[T: NumericType](value: Int, dtype: T) -> NumericLiteral[T]:
    """A numeric constant — `lit(2, int64)`."""
    return NumericLiteral[T](value)


def lit[T: StringLikeType](var value: String, dtype: T) -> StringConst[T]:
    """A string constant — `lit("x", string)`."""
    return StringConst[T](value^)


# ---------------------------------------------------------------------------
# Table[T] — column-access handle over a schema struct
# ---------------------------------------------------------------------------


struct Table[T: AnyType](Copyable, Movable):
    """Column-access handle over a plain schema struct — `Table[Orders]()`.

    `T` is any struct of plain dtype-tag fields (`var a: Int64Type`). `t.a`
    reflects field `a`'s dtype on `T` at compile time (`reflect[T].field[name].T`)
    to pick the column leaf; the position is resolved by name at execution. A
    companion handle is required because `T`'s own fields shadow
    `__getattr_param__`; `T` is never instantiated (only reflected). Overloads
    route numeric/string/list fields to the matching typed column via a `where`
    clause the constraint solver can prove."""

    comptime _dtype[name: StringLiteral] = reflect[Self.T].field[name].T

    def __init__(out self):
        pass

    @always_inline
    def __getattr_param__[
        name: StringLiteral
    ](self) -> NumericColumn[Self._dtype[name]] where conforms_to(
        Self._dtype[name], NumericType
    ):
        return NumericColumn[Self._dtype[name]](String(name))

    @always_inline
    def __getattr_param__[
        name: StringLiteral
    ](self) -> StringColumn[Self._dtype[name]] where conforms_to(
        Self._dtype[name], StringLikeType
    ):
        return StringColumn[Self._dtype[name]](String(name))


# ---------------------------------------------------------------------------
# AnyValue — type-erased handle: box any `Value`, then `.execute(batch)` it
# ---------------------------------------------------------------------------


struct AnyValue(Copyable, Movable, Writable):
    """Type-erased handle over any expression node — the one box that lets
    runtime-typed / heterogeneous code hold a value regardless of its family and
    still `.execute(batch)` it. Erasure is via per-boxed-type trampolines that
    `rebind` the node back and forward to its methods; every typed result array
    converts to `AnyArray` via `.to_any()`.

    Two boxing paths, so a program that only boxes fused comptime nodes never
    links the runtime interpreter (it is dead-code-eliminated and the binary
    stays small):
      * `__init__[V: Value]` — box a comptime node (column / fused expression /
        boundary). `execute` returns the typed `Self.OutType.ArrayType`, erased
        via `.to_any()`; `prune` inherits the conservative default.
      * `__init__(DynValue)` — box the runtime interpreter (`marrow.expr.dynamic`),
        whose `execute` is already `AnyArray` and whose `prune` carries the real
        min/max rule the relational layer uses for row-group / page skipping."""

    var _boxed: ArcPointer[NoneType]
    var _execute: def(ArcPointer[NoneType], RecordBatch) thin raises -> AnyArray
    var _name_fn: def(ArcPointer[NoneType]) thin -> String
    var _write_fn: def(ArcPointer[NoneType]) thin -> String
    var _prune_fn: def(
        ArcPointer[NoneType], PruneStats
    ) thin raises -> PruneBound

    # --- comptime-node trampolines (generic over any `V: Value`) -----------

    @staticmethod
    def _execute_tramp[
        V: Value
    ](ptr: ArcPointer[NoneType], batch: RecordBatch) raises -> AnyArray:
        return rebind[ArcPointer[V]](ptr)[].execute(batch).to_any()

    @staticmethod
    def _name_tramp[V: Value](ptr: ArcPointer[NoneType]) -> String:
        return rebind[ArcPointer[V]](ptr)[].name()

    @staticmethod
    def _write_tramp[V: Value](ptr: ArcPointer[NoneType]) -> String:
        var s = String()
        rebind[ArcPointer[V]](ptr)[].write_to(s)
        return s^

    @staticmethod
    def _prune_tramp[
        V: Value
    ](ptr: ArcPointer[NoneType], stats: PruneStats) raises -> PruneBound:
        return rebind[ArcPointer[V]](ptr)[].prune(stats)

    @implicit
    def __init__[V: Value](out self, value: V):
        """Box any comptime `Value` — a column, a fused numeric expression, a
        boundary."""
        var ptr = ArcPointer[V](value.copy())
        self._boxed = rebind[ArcPointer[NoneType]](ptr^)
        self._execute = Self._execute_tramp[V]
        self._name_fn = Self._name_tramp[V]
        self._write_fn = Self._write_tramp[V]
        self._prune_fn = Self._prune_tramp[V]

    # --- runtime interpreter trampolines (concrete `DynValue`) -------------

    @staticmethod
    def _execute_tramp_dyn(
        ptr: ArcPointer[NoneType], batch: RecordBatch
    ) raises -> AnyArray:
        return rebind[ArcPointer[DynValue]](ptr)[].execute(batch)

    @staticmethod
    def _name_tramp_dyn(ptr: ArcPointer[NoneType]) -> String:
        return rebind[ArcPointer[DynValue]](ptr)[].name()

    @staticmethod
    def _write_tramp_dyn(ptr: ArcPointer[NoneType]) -> String:
        var s = String()
        rebind[ArcPointer[DynValue]](ptr)[].write_to(s)
        return s^

    @staticmethod
    def _prune_tramp_dyn(
        ptr: ArcPointer[NoneType], stats: PruneStats
    ) raises -> PruneBound:
        return rebind[ArcPointer[DynValue]](ptr)[].prune(stats)

    @implicit
    def __init__(out self, var value: DynValue):
        """Box the runtime interpreter node so runtime-built plans (Python
        bindings, dynamic relations) flow through the same handle. Linking this
        overload is what pulls in the interpreter; a fused-only program never
        instantiates it and stays small."""
        var ptr = ArcPointer[DynValue](value^)
        self._boxed = rebind[ArcPointer[NoneType]](ptr^)
        self._execute = Self._execute_tramp_dyn
        self._name_fn = Self._name_tramp_dyn
        self._write_fn = Self._write_tramp_dyn
        self._prune_fn = Self._prune_tramp_dyn

    def execute(self, batch: RecordBatch) raises -> AnyArray:
        """Evaluate the boxed node against `batch`, erased to `AnyArray`."""
        return self._execute(self._boxed, batch)

    def name(self) -> String:
        return self._name_fn(self._boxed)

    def prune(self, stats: PruneStats) raises -> PruneBound:
        """Evaluate the boxed predicate against per-column statistics for
        row-group / page skipping (see `marrow.expr.pruning`). Comptime nodes
        return the conservative default; a boxed `DynValue` returns the real
        min/max rule."""
        return self._prune_fn(self._boxed, stats)

    def write_to[W: Writer](self, mut writer: W):
        writer.write(self._write_fn(self._boxed))


# ===========================================================================
# Comptime-node pruning — PARKED (copied from the previous `values.mojo`;
# not yet ported to the fused nodes above). Row-group / page skipping still
# works through the runtime `DynValue.prune` path; these per-node overrides
# would let the *comptime* predicates skip too. Re-enable by adding a `prune`
# override to the matching node (NumericLiteral, NumericColumn, BoolBinary
# comparisons) using the `PruneBound` min/max rules.
# ---------------------------------------------------------------------------
#
# NumericLiteral.prune:
#     var s = AnyScalar(PrimitiveScalar[Self.T](self._value))
#     return PruneBound.interval(s.copy(), s.copy())
#
# NumericColumn.prune:
#     var iv = stats.by_name(self._name)
#     return PruneBound.interval(iv[0].copy(), iv[1].copy())
#
# Less(left, right).prune:
#     return PruneBound.boolean(
#         self.left.prune(stats).maybe_lt(self.right.prune(stats)))
# Greater -> maybe_gt ; Equal -> maybe_eq ; (Le/Ge analogous)
#
# And/Or.prune: combine children's `maybe_true` with `and` / `or`.
# ===========================================================================

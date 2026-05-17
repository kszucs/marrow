"""Prototype: typed expression tree with kernel fusion.

Two separate layers:

- **Static layer** (`Expr` / `NumericExpr` traits, `Column` / `Negate` / `Add`
  structs): each node implements `exec_core[W](idx) -> SIMD[OutType.native, W]`.
  `execute()` drives a single fused `vectorize` loop — the compiler inlines the
  full tree, producing zero intermediate arrays.

- **Runtime layer** (`RuntimeExpr`): lazy tree built at construction time,
  evaluated on `.execute()` via the existing typed/`AnyArray` kernel overloads
  in `marrow/kernels/arithmetic.mojo`. Does NOT implement `Expr`; no static
  SIMD type guarantees. This is the Python-facing path.
"""

from std.algorithm.backend.vectorize import vectorize
from std.sys import size_of
from std.sys.info import simd_byte_width
from std.sys.intrinsics import _type_is_eq

import marrow.dtypes as dt
from marrow.arrays import AnyArray, BoolArray, PrimitiveArray
from marrow.buffers import Bitmap, Buffer
from marrow.builders import arange
from marrow.dtypes import Int32Type
from marrow.kernels.arithmetic import add, neg, AddKernel, NegKernel, SubKernel, MulKernel
from marrow.kernels.boolean import AndKernel, OrKernel, NotKernel
from marrow.kernels.compare import EqKernel, NeKernel, LtKernel, LeKernel, GtKernel, GeKernel
from marrow.tabular import RecordBatch
from marrow.views import BufferView
from std.memory import OwnedPointer
from std.utils.index import IndexList


# ===========================================================================
# Static layer — kernel fusion via comptime exec_core
# ===========================================================================


trait Expr(Copyable, Movable, Writable, ImplicitlyDestructible):
    """Base for statically-typed expression nodes.

    OutType is NumericType (not the broader PrimitiveType) so that
    exec_core can return SIMD[OutType.native, W] and the executor can
    construct a PrimitiveArray[OutType] via OutType() (Defaultable).
    """

    comptime OutType: dt.NumericType

    @always_inline
    def exec_core[W: Int](self, idx: Int) -> SIMD[Self.OutType.native, W]:
        ...

    def bind(mut self, batch: RecordBatch) raises:
        pass


trait NumericExpr(Expr):
    """Marker: OutType is numeric. Inherits exec_core from Expr, so
    T: NumericExpr implies T.exec_core[W](idx) is callable.
    Operator overloads build the expression tree without executing it.
    """

    def __neg__(self) -> Negate[Self]:
        return Negate(self.copy())

    def negate(self) -> Negate[Self]:
        return Negate(self.copy())

    def __add__[RHS: NumericExpr](self, rhs: RHS) -> Add[Self, RHS]:
        return Add(self.copy(), rhs.copy())

    def __sub__[RHS: NumericExpr](self, rhs: RHS) -> Sub[Self, RHS]:
        return Sub(self.copy(), rhs.copy())

    def __mul__[RHS: NumericExpr](self, rhs: RHS) -> Mul[Self, RHS]:
        return Mul(self.copy(), rhs.copy())

    def __eq__[RHS: NumericExpr](self, rhs: RHS) -> EqExpr[Self, RHS]:
        return EqExpr(self.copy(), rhs.copy())

    def __ne__[RHS: NumericExpr](self, rhs: RHS) -> NeExpr[Self, RHS]:
        return NeExpr(self.copy(), rhs.copy())

    def __lt__[RHS: NumericExpr](self, rhs: RHS) -> LtExpr[Self, RHS]:
        return LtExpr(self.copy(), rhs.copy())

    def __le__[RHS: NumericExpr](self, rhs: RHS) -> LeExpr[Self, RHS]:
        return LeExpr(self.copy(), rhs.copy())

    def __gt__[RHS: NumericExpr](self, rhs: RHS) -> GtExpr[Self, RHS]:
        return GtExpr(self.copy(), rhs.copy())

    def __ge__[RHS: NumericExpr](self, rhs: RHS) -> GeExpr[Self, RHS]:
        return GeExpr(self.copy(), rhs.copy())


trait BoolExpr(Copyable, Movable, Writable, ImplicitlyDestructible):
    """Base for bool-output expression nodes (comparisons).

    InNative carries the numeric dtype of the leaves so execute() can choose
    the correct SIMD width.  exec_core[W](idx) returns W booleans that
    execute() bit-packs into a BoolArray via BitmapView.store[W].
    """

    comptime InNative: DType

    @always_inline
    def exec_core[W: Int](self, idx: Int) -> SIMD[DType.bool, W]: ...

    def bind(mut self, batch: RecordBatch) raises:
        pass

    def __and__[RHS: BoolExpr](self, rhs: RHS) -> AndExpr[Self, RHS]:
        return AndExpr(self.copy(), rhs.copy())

    def __or__[RHS: BoolExpr](self, rhs: RHS) -> OrExpr[Self, RHS]:
        return OrExpr(self.copy(), rhs.copy())

    def __invert__(self) -> NotExpr[Self]:
        return NotExpr(self.copy())


struct Column[T: dt.NumericType](Expr, NumericExpr):
    comptime OutType = Self.T

    var arr: PrimitiveArray[Self.T]

    def __init__(out self, var arr: PrimitiveArray[Self.T]):
        self.arr = arr^

    def __init__(out self, *, copy: Self):
        self.arr = copy.arr.copy()

    @always_inline
    def exec_core[W: Int](self, idx: Int) -> SIMD[Self.T.native, W]:
        return self.arr.values().load[W](idx)

    def write_to[W: Writer](self, mut writer: W):
        writer.write("Col[", len(self.arr), "]")


struct ColumnRef[name: StringLiteral, T: dt.NumericType](Expr, NumericExpr):
    """Named column placeholder resolved from a RecordBatch at execute time.

    `name` and `T` are compile-time constants, so each distinct (name, T) pair
    is a unique type — full AOT specialization and DCE are preserved.
    """

    comptime OutType = Self.T

    var _arr: Optional[PrimitiveArray[Self.T]]

    def __init__(out self):
        self._arr = None

    def __init__(out self, *, copy: Self):
        if copy._arr:
            self._arr = copy._arr.unsafe_value().copy()
        else:
            self._arr = None

    def bind(mut self, batch: RecordBatch) raises:
        self._arr = batch.column(Self.name).as_primitive[Self.T]().copy()

    @always_inline
    def exec_core[W: Int](self, idx: Int) -> SIMD[Self.T.native, W]:
        return self._arr.unsafe_value().values().load[W](idx)

    def col_name(self) -> String:
        """Column name."""
        return Self.name

    def col_dtype(self) -> Self.T:
        """Runtime DataType instance for this column."""
        return Self.T()

    def where[Pred: BoolExpr](self, pred: Pred) -> FilterPipeline[Self.name, Self.T, Pred]:
        """Build a reusable filter pipeline: self is the data column, pred is the filter.

        Usage::

            col['data'](dt.int32).where(col['a'](dt.int32) + col['b'](dt.int32) > col['c'](dt.int32))
        """
        return FilterPipeline[Self.name, Self.T, Pred](pred.copy())

    def write_to[W: Writer](self, mut writer: W):
        writer.write("col['", Self.name, "', ", Self.T(), "]")


@always_inline
def lit[T: dt.NumericType](value: Scalar[T.native], dtype: T) -> Literal[T]:
    """Create a scalar constant for use in AOT expression trees.

    Usage: ``lit(Int32(5), dt.int32)`` or ``lit[dt.int32](5)``.
    Returns a ``Literal[T]`` broadcast to all SIMD lanes.
    """
    return Literal[T](value)


@always_inline
def lit[T: dt.NumericType](dtype: T, value: Int) -> Literal[T]:
    """Create a scalar constant from an ``Int`` literal.

    Usage: ``lit(dt.int32, 5)`` — more ergonomic when the type is the lead arg.
    """
    return Literal[T](Scalar[T.native](value))


@always_inline
def col[name: StringLiteral, T: dt.NumericType](dtype: T) -> ColumnRef[name, T]:
    """Create a named column placeholder for use in AOT expression trees.

    Usage: ``col['price'](dt.float32)``
    The return type is ``ColumnRef['price', Float32Type]`` — a unique
    compile-time type that preserves full AOT specialization.
    """
    return ColumnRef[name, T]()


struct Negate[T: NumericExpr](NumericExpr):
    comptime OutType = Self.T.OutType

    var arg: Self.T

    def __init__(out self, var arg: Self.T):
        self.arg = arg^

    def __init__(out self, *, copy: Self):
        self.arg = copy.arg.copy()

    def bind(mut self, batch: RecordBatch) raises:
        self.arg.bind(batch)

    @always_inline
    def exec_core[W: Int](self, idx: Int) -> SIMD[Self.T.OutType.native, W]:
        return NegKernel.core[Self.T.OutType.native, W](self.arg.exec_core[W](idx))

    def write_to[W: Writer](self, mut writer: W):
        writer.write("Negate(", self.arg, ")")


struct Add[L: NumericExpr, R: NumericExpr](NumericExpr):
    comptime OutType = Self.L.OutType

    var left: Self.L
    var right: Self.R

    def __init__(out self, var left: Self.L, var right: Self.R):
        comptime assert _type_is_eq[
            Self.L.OutType, Self.R.OutType
        ](), "Add requires matching output types"
        self.left = left^
        self.right = right^

    def __init__(out self, *, copy: Self):
        self.left = copy.left.copy()
        self.right = copy.right.copy()

    def bind(mut self, batch: RecordBatch) raises:
        self.left.bind(batch)
        self.right.bind(batch)

    @always_inline
    def exec_core[W: Int](self, idx: Int) -> SIMD[Self.L.OutType.native, W]:
        var l = self.left.exec_core[W](idx)
        var r = self.right.exec_core[W](idx).cast[Self.L.OutType.native]()
        return AddKernel.core[Self.L.OutType.native, W](l, r)

    def write_to[W: Writer](self, mut writer: W):
        writer.write("Add(", self.left, ", ", self.right, ")")


struct Literal[T: dt.NumericType](NumericExpr):
    """Scalar constant broadcast to all SIMD lanes."""

    comptime OutType = Self.T

    var value: Scalar[Self.T.native]

    def __init__(out self, value: Scalar[Self.T.native]):
        self.value = value

    def __init__(out self, *, copy: Self):
        self.value = copy.value

    @always_inline
    def exec_core[W: Int](self, idx: Int) -> SIMD[Self.T.native, W]:
        return SIMD[Self.T.native, W](self.value)

    def write_to[W: Writer](self, mut writer: W):
        writer.write("Lit[", self.value, "]")


struct Sub[L: NumericExpr, R: NumericExpr](NumericExpr):
    comptime OutType = Self.L.OutType

    var left: Self.L
    var right: Self.R

    def __init__(out self, var left: Self.L, var right: Self.R):
        comptime assert _type_is_eq[
            Self.L.OutType, Self.R.OutType
        ](), "Sub requires matching output types"
        self.left = left^
        self.right = right^

    def __init__(out self, *, copy: Self):
        self.left = copy.left.copy()
        self.right = copy.right.copy()

    def bind(mut self, batch: RecordBatch) raises:
        self.left.bind(batch)
        self.right.bind(batch)

    @always_inline
    def exec_core[W: Int](self, idx: Int) -> SIMD[Self.L.OutType.native, W]:
        var l = self.left.exec_core[W](idx)
        var r = self.right.exec_core[W](idx).cast[Self.L.OutType.native]()
        return SubKernel.core[Self.L.OutType.native, W](l, r)

    def write_to[W: Writer](self, mut writer: W):
        writer.write("Sub(", self.left, ", ", self.right, ")")


struct Mul[L: NumericExpr, R: NumericExpr](NumericExpr):
    comptime OutType = Self.L.OutType

    var left: Self.L
    var right: Self.R

    def __init__(out self, var left: Self.L, var right: Self.R):
        comptime assert _type_is_eq[
            Self.L.OutType, Self.R.OutType
        ](), "Mul requires matching output types"
        self.left = left^
        self.right = right^

    def __init__(out self, *, copy: Self):
        self.left = copy.left.copy()
        self.right = copy.right.copy()

    def bind(mut self, batch: RecordBatch) raises:
        self.left.bind(batch)
        self.right.bind(batch)

    @always_inline
    def exec_core[W: Int](self, idx: Int) -> SIMD[Self.L.OutType.native, W]:
        var l = self.left.exec_core[W](idx)
        var r = self.right.exec_core[W](idx).cast[Self.L.OutType.native]()
        return MulKernel.core[Self.L.OutType.native, W](l, r)

    def write_to[W: Writer](self, mut writer: W):
        writer.write("Mul(", self.left, ", ", self.right, ")")


# ---------------------------------------------------------------------------
# Comparison expression nodes — BoolExpr
# ---------------------------------------------------------------------------


struct EqExpr[L: NumericExpr, R: NumericExpr](BoolExpr):
    comptime InNative = Self.L.OutType.native

    var left: Self.L
    var right: Self.R

    def __init__(out self, var left: Self.L, var right: Self.R):
        comptime assert _type_is_eq[
            Self.L.OutType, Self.R.OutType
        ](), "EqExpr requires matching output types"
        self.left = left^
        self.right = right^

    def __init__(out self, *, copy: Self):
        self.left = copy.left.copy()
        self.right = copy.right.copy()

    def bind(mut self, batch: RecordBatch) raises:
        self.left.bind(batch)
        self.right.bind(batch)

    @always_inline
    def exec_core[W: Int](self, idx: Int) -> SIMD[DType.bool, W]:
        return EqKernel.core[Self.L.OutType.native, W](
            self.left.exec_core[W](idx),
            self.right.exec_core[W](idx).cast[Self.L.OutType.native](),
        )

    def write_to[W: Writer](self, mut writer: W):
        writer.write("(", self.left, " == ", self.right, ")")


struct NeExpr[L: NumericExpr, R: NumericExpr](BoolExpr):
    comptime InNative = Self.L.OutType.native

    var left: Self.L
    var right: Self.R

    def __init__(out self, var left: Self.L, var right: Self.R):
        comptime assert _type_is_eq[
            Self.L.OutType, Self.R.OutType
        ](), "NeExpr requires matching output types"
        self.left = left^
        self.right = right^

    def __init__(out self, *, copy: Self):
        self.left = copy.left.copy()
        self.right = copy.right.copy()

    def bind(mut self, batch: RecordBatch) raises:
        self.left.bind(batch)
        self.right.bind(batch)

    @always_inline
    def exec_core[W: Int](self, idx: Int) -> SIMD[DType.bool, W]:
        return NeKernel.core[Self.L.OutType.native, W](
            self.left.exec_core[W](idx),
            self.right.exec_core[W](idx).cast[Self.L.OutType.native](),
        )

    def write_to[W: Writer](self, mut writer: W):
        writer.write("(", self.left, " != ", self.right, ")")


struct LtExpr[L: NumericExpr, R: NumericExpr](BoolExpr):
    comptime InNative = Self.L.OutType.native

    var left: Self.L
    var right: Self.R

    def __init__(out self, var left: Self.L, var right: Self.R):
        comptime assert _type_is_eq[
            Self.L.OutType, Self.R.OutType
        ](), "LtExpr requires matching output types"
        self.left = left^
        self.right = right^

    def __init__(out self, *, copy: Self):
        self.left = copy.left.copy()
        self.right = copy.right.copy()

    def bind(mut self, batch: RecordBatch) raises:
        self.left.bind(batch)
        self.right.bind(batch)

    @always_inline
    def exec_core[W: Int](self, idx: Int) -> SIMD[DType.bool, W]:
        return LtKernel.core[Self.L.OutType.native, W](
            self.left.exec_core[W](idx),
            self.right.exec_core[W](idx).cast[Self.L.OutType.native](),
        )

    def write_to[W: Writer](self, mut writer: W):
        writer.write("(", self.left, " < ", self.right, ")")


struct LeExpr[L: NumericExpr, R: NumericExpr](BoolExpr):
    comptime InNative = Self.L.OutType.native

    var left: Self.L
    var right: Self.R

    def __init__(out self, var left: Self.L, var right: Self.R):
        comptime assert _type_is_eq[
            Self.L.OutType, Self.R.OutType
        ](), "LeExpr requires matching output types"
        self.left = left^
        self.right = right^

    def __init__(out self, *, copy: Self):
        self.left = copy.left.copy()
        self.right = copy.right.copy()

    def bind(mut self, batch: RecordBatch) raises:
        self.left.bind(batch)
        self.right.bind(batch)

    @always_inline
    def exec_core[W: Int](self, idx: Int) -> SIMD[DType.bool, W]:
        return LeKernel.core[Self.L.OutType.native, W](
            self.left.exec_core[W](idx),
            self.right.exec_core[W](idx).cast[Self.L.OutType.native](),
        )

    def write_to[W: Writer](self, mut writer: W):
        writer.write("(", self.left, " <= ", self.right, ")")


struct GtExpr[L: NumericExpr, R: NumericExpr](BoolExpr):
    comptime InNative = Self.L.OutType.native

    var left: Self.L
    var right: Self.R

    def __init__(out self, var left: Self.L, var right: Self.R):
        comptime assert _type_is_eq[
            Self.L.OutType, Self.R.OutType
        ](), "GtExpr requires matching output types"
        self.left = left^
        self.right = right^

    def __init__(out self, *, copy: Self):
        self.left = copy.left.copy()
        self.right = copy.right.copy()

    def bind(mut self, batch: RecordBatch) raises:
        self.left.bind(batch)
        self.right.bind(batch)

    @always_inline
    def exec_core[W: Int](self, idx: Int) -> SIMD[DType.bool, W]:
        return GtKernel.core[Self.L.OutType.native, W](
            self.left.exec_core[W](idx),
            self.right.exec_core[W](idx).cast[Self.L.OutType.native](),
        )

    def write_to[W: Writer](self, mut writer: W):
        writer.write("(", self.left, " > ", self.right, ")")


struct GeExpr[L: NumericExpr, R: NumericExpr](BoolExpr):
    comptime InNative = Self.L.OutType.native

    var left: Self.L
    var right: Self.R

    def __init__(out self, var left: Self.L, var right: Self.R):
        comptime assert _type_is_eq[
            Self.L.OutType, Self.R.OutType
        ](), "GeExpr requires matching output types"
        self.left = left^
        self.right = right^

    def __init__(out self, *, copy: Self):
        self.left = copy.left.copy()
        self.right = copy.right.copy()

    def bind(mut self, batch: RecordBatch) raises:
        self.left.bind(batch)
        self.right.bind(batch)

    @always_inline
    def exec_core[W: Int](self, idx: Int) -> SIMD[DType.bool, W]:
        return GeKernel.core[Self.L.OutType.native, W](
            self.left.exec_core[W](idx),
            self.right.exec_core[W](idx).cast[Self.L.OutType.native](),
        )

    def write_to[W: Writer](self, mut writer: W):
        writer.write("(", self.left, " >= ", self.right, ")")


# ---------------------------------------------------------------------------
# Boolean combinator nodes — BoolExpr × BoolExpr → BoolExpr
# ---------------------------------------------------------------------------


struct AndExpr[L: BoolExpr, R: BoolExpr](BoolExpr):
    comptime InNative = Self.L.InNative

    var left: Self.L
    var right: Self.R

    def __init__(out self, var left: Self.L, var right: Self.R):
        self.left = left^
        self.right = right^

    def __init__(out self, *, copy: Self):
        self.left = copy.left.copy()
        self.right = copy.right.copy()

    def bind(mut self, batch: RecordBatch) raises:
        self.left.bind(batch)
        self.right.bind(batch)

    @always_inline
    def exec_core[W: Int](self, idx: Int) -> SIMD[DType.bool, W]:
        return AndKernel.core[W](
            self.left.exec_core[W](idx),
            self.right.exec_core[W](idx),
        )

    def write_to[W: Writer](self, mut writer: W):
        writer.write("(", self.left, " AND ", self.right, ")")


struct OrExpr[L: BoolExpr, R: BoolExpr](BoolExpr):
    comptime InNative = Self.L.InNative

    var left: Self.L
    var right: Self.R

    def __init__(out self, var left: Self.L, var right: Self.R):
        self.left = left^
        self.right = right^

    def __init__(out self, *, copy: Self):
        self.left = copy.left.copy()
        self.right = copy.right.copy()

    def bind(mut self, batch: RecordBatch) raises:
        self.left.bind(batch)
        self.right.bind(batch)

    @always_inline
    def exec_core[W: Int](self, idx: Int) -> SIMD[DType.bool, W]:
        return OrKernel.core[W](
            self.left.exec_core[W](idx),
            self.right.exec_core[W](idx),
        )

    def write_to[W: Writer](self, mut writer: W):
        writer.write("(", self.left, " OR ", self.right, ")")


struct NotExpr[E: BoolExpr](BoolExpr):
    comptime InNative = Self.E.InNative

    var expr: Self.E

    def __init__(out self, var expr: Self.E):
        self.expr = expr^

    def __init__(out self, *, copy: Self):
        self.expr = copy.expr.copy()

    def bind(mut self, batch: RecordBatch) raises:
        self.expr.bind(batch)

    @always_inline
    def exec_core[W: Int](self, idx: Int) -> SIMD[DType.bool, W]:
        return NotKernel.core[W](self.expr.exec_core[W](idx))

    def write_to[W: Writer](self, mut writer: W):
        writer.write("NOT(", self.expr, ")")


# ---------------------------------------------------------------------------
# FilterExpr — fused predicate evaluation + scatter-filter
# ---------------------------------------------------------------------------


struct FilterExpr[T: dt.NumericType, Pred: BoolExpr](
    Copyable, Movable, Writable, ImplicitlyDestructible
):
    """Compile-time expression: single-pass fused predicate evaluation + filter.

    execute() runs one loop:
      - For each SIMD block: evaluate Pred → SIMD[DType.bool, W], then
        compressed_store the matching data elements directly.
      - No BoolArray allocated, no bit-pack/unpack round-trip.

    Compare to dispatch, which materialises one intermediate numeric array per
    arithmetic node in the predicate, then a BoolArray, before the filter runs.
    """

    var data: PrimitiveArray[Self.T]
    var pred: Self.Pred

    def __init__(out self, var data: PrimitiveArray[Self.T], var pred: Self.Pred):
        self.data = data^
        self.pred = pred^

    def __init__(out self, *, copy: Self):
        self.data = copy.data.copy()
        self.pred = copy.pred.copy()

    def write_to[W: Writer](self, mut writer: W):
        writer.write("Filter(data[", len(self.data), "] WHERE ", self.pred, ")")


struct FilterPipeline[
    data_name: StringLiteral,
    T: dt.NumericType,
    Pred: BoolExpr,
](Copyable, Movable, Writable, ImplicitlyDestructible):
    """Reusable filter pipeline: define once, execute against many RecordBatches.

    Per call: binds ColumnRef nodes in pred to the batch columns (O(cols)
    ref-count bumps), then runs the single-pass fused filter loop.
    """

    var _pred: Self.Pred

    def __init__(out self, var pred: Self.Pred):
        self._pred = pred^

    def __init__(out self, *, copy: Self):
        self._pred = copy._pred.copy()

    def __call__(
        mut self, batch: RecordBatch
    ) raises -> PrimitiveArray[Self.T]:
        self._pred.bind(batch)
        var data = batch.column(Self.data_name).as_primitive[Self.T]().copy()
        return execute(FilterExpr(data^, self._pred.copy()), batch.num_rows())

    def execute(
        var self, batch: RecordBatch
    ) raises -> PrimitiveArray[Self.T]:
        """Single-call execution for use in chained expressions.

        Unlike ``__call__``, this consumes the pipeline, allowing it to be
        called directly on a temporary returned by ``.where()``:

        .. code-block:: mojo

            col['data'](dt.int32).where(col['a'](dt.int32) + col['b'](dt.int32) > col['c'](dt.int32)).execute(batch)
        """
        self._pred.bind(batch)
        var data = batch.column(Self.data_name).as_primitive[Self.T]().copy()
        return execute(FilterExpr(data^, self._pred.copy()), batch.num_rows())

    def write_to[W: Writer](self, mut writer: W):
        writer.write("FilterPipeline['", Self.data_name, "'] WHERE ", self._pred)


@always_inline
def filter_pipeline[data_col: StringLiteral, T: dt.NumericType, Pred: BoolExpr](
    var pred: Pred,
    dtype: T,
) -> FilterPipeline[data_col, T, Pred]:
    """Create a reusable filter pipeline bound to named RecordBatch columns.

    Usage::

        var p = filter_pipeline['output'](
            GtExpr(Add(col['a'](dt.int32), col['b'](dt.int32)), col['c'](dt.int32)),
            dt.int32,
        )
        var result1 = p(batch1)
        var result2 = p(batch2)
    """
    return FilterPipeline[data_col, T, Pred](pred^)


struct Pipeline[E: NumericExpr](Copyable, Movable, Writable, ImplicitlyDestructible):
    """Reusable numeric expression pipeline over named RecordBatch columns."""

    var _expr: Self.E

    def __init__(out self, var expr: Self.E):
        self._expr = expr^

    def __init__(out self, *, copy: Self):
        self._expr = copy._expr.copy()

    def __call__(
        mut self, batch: RecordBatch
    ) raises -> PrimitiveArray[Self.E.OutType]:
        self._expr.bind(batch)
        return execute(self._expr, batch.num_rows())

    def write_to[W: Writer](self, mut writer: W):
        writer.write("Pipeline(", self._expr, ")")


def _vectorize_dispatch[
    native: DType,
    cpu_width: Int,
    process: def[W: Int, rank: Int, alignment: Int = 1](
        IndexList[rank]
    ) capturing -> None,
](length: Int):
    """Run process over [0, length) using vectorize. Mirrors _apply_dispatch."""

    @always_inline
    def lane[W: Int](i: Int):
        process[W, rank=1](IndexList[1](i))

    vectorize[cpu_width](length, lane)


def execute[E: NumericExpr](expr: E, length: Int) raises -> PrimitiveArray[E.OutType]:
    """Drive a single fused vectorize loop over the expression tree.

    Uses the same @parameter-closure pattern as views.mojo's apply() so that
    the mutable view capture is safe and buf.to_immutable() is valid afterward.
    """
    comptime native = E.OutType.native
    comptime width = simd_byte_width() // size_of[Scalar[native]]()
    var buf = Buffer.alloc_uninit[native](length)
    # view captured inside @parameter fill — suppress false-positive unused warning
    var view = buf.view[native](0, length)
    _ = view

    @parameter
    @always_inline
    def fill[W: Int, rank: Int, alignment: Int = 1](idx: IndexList[rank]) -> None:
        var i = idx[0]
        view.store[W](i, expr.exec_core[W](i))

    _vectorize_dispatch[native, width, fill](length)
    return PrimitiveArray[E.OutType](
        dtype=E.OutType(),
        length=length,
        nulls=0,
        offset=0,
        bitmap=None,
        buffer=buf.to_immutable(),
    )


def execute[B: BoolExpr](expr: B, length: Int) raises -> BoolArray:
    """Drive a single fused vectorize loop over a boolean expression tree.

    Computes W booleans per iteration and bit-packs them into a Bitmap via
    BitmapView.store[W].  cpu_width is derived from B.InNative so the SIMD
    width matches the numeric leaves of the expression tree.
    """
    comptime width = max(8, simd_byte_width() // size_of[Scalar[B.InNative]]())
    var bm = Bitmap.alloc_uninit(length)
    var view = bm.view()
    _ = view

    @parameter
    @always_inline
    def fill[W: Int, rank: Int, alignment: Int = 1](idx: IndexList[rank]) -> None:
        var i = idx[0]
        view.store[W](i, expr.exec_core[W](i))

    _vectorize_dispatch[B.InNative, width, fill](length)
    return BoolArray(
        length=length,
        nulls=0,
        offset=0,
        bitmap=None,
        buffer=bm.to_immutable(),
    )


@always_inline
def _pack_bools[W: Int](mask: SIMD[DType.bool, W]) -> UInt64:
    """Pack W boolean SIMD lanes into the low W bits of a UInt64."""
    var result: UInt64 = 0
    comptime for b in range(W):
        result |= UInt64(Int(mask[b])) << UInt64(b)
    return result


def execute[
    T: dt.NumericType,
    Pred: BoolExpr,
](expr: FilterExpr[T, Pred], length: Int) raises -> PrimitiveArray[T]:
    """Single-pass fused filter: predicate eval and scatter-select in one loop.

    Evaluates the predicate tree for 64 elements at a time, assembles the
    results into a UInt64 selection word in registers (no BoolArray, no bitmap
    write/read round-trip), then uses the adaptive compressed_store (sparse CTZ
    or dense byte-chunked branchless) for the scatter step.
    """
    comptime native = T.native
    comptime width = max(8, simd_byte_width() // size_of[Scalar[Pred.InNative]]())

    var src = expr.data.values()
    var out_buf = Buffer.alloc_uninit[native](length)
    var out_view = out_buf.view[native](0, length)
    var out_pos = 0

    var i = 0
    while i + 64 <= length:
        var word: UInt64 = 0
        comptime for k in range(64 // width):
            word |= _pack_bools[width](
                expr.pred.exec_core[width](i + k * width)
            ) << UInt64(k * width)
        out_pos += out_view.slice(out_pos).compressed_store(src.slice(i), word)
        i += 64

    while i < length:
        if expr.pred.exec_core[1](i)[0]:
            out_view.unsafe_set(out_pos, src.unsafe_get(i))
            out_pos += 1
        i += 1

    return PrimitiveArray[T](
        dtype=T(),
        length=out_pos,
        nulls=0,
        offset=0,
        bitmap=None,
        buffer=out_buf.to_immutable(),
    )


# ===========================================================================
# Runtime layer — lazy tree, NOT implementing Expr
# ===========================================================================

# sizeof[RuntimeExpr] = sizeof(OwnedPointer) = pointer width, so
# List[RuntimeExpr] inside _NodeContent has a known element size and the
# recursive size dependency resolves to a finite constant.

comptime _RT_COLUMN = 0
comptime _RT_NEGATE = 1
comptime _RT_ADD = 2


struct _NodeContent(Copyable, Movable, ImplicitlyDestructible):
    var _kind: Int
    var _name: String
    var _args: List[RuntimeExpr]

    def __init__(
        out self,
        kind: Int,
        name: String,
        var args: List[RuntimeExpr],
    ):
        self._kind = kind
        self._name = name
        self._args = args^

    def __init__(out self, *, copy: Self):
        self._kind = copy._kind
        self._name = copy._name
        self._args = List[RuntimeExpr]()
        for i in range(len(copy._args)):
            self._args.append(RuntimeExpr(copy=copy._args[i]))


struct RuntimeExpr(Copyable, Movable, ImplicitlyDestructible):
    """Lazy runtime expression tree. Call execute() to materialise.

    sizeof[RuntimeExpr] = sizeof(OwnedPointer) = pointer width, allowing
    List[RuntimeExpr] children in _NodeContent without infinite size recursion.
    Does NOT implement Expr — no static SIMD type guarantees.
    """

    var _node: OwnedPointer[_NodeContent]

    def __init__(out self, var node: _NodeContent):
        self._node = OwnedPointer[_NodeContent](node^)

    def __init__(out self, *, copy: Self):
        var inner = _NodeContent(copy=copy._node[])
        self._node = OwnedPointer[_NodeContent](inner^)

    @staticmethod
    def column(name: String) -> Self:
        return Self(_NodeContent(_RT_COLUMN, name, List[RuntimeExpr]()))

    def negate(self) raises -> Self:
        var args = List[RuntimeExpr]()
        args.append(self.copy())
        return Self(_NodeContent(_RT_NEGATE, "", args^))

    def add(self, other: Self) raises -> Self:
        var args = List[RuntimeExpr]()
        args.append(self.copy())
        args.append(other.copy())
        return Self(_NodeContent(_RT_ADD, "", args^))

    def execute(self, data: Dict[String, AnyArray]) raises -> AnyArray:
        if self._node[]._kind == _RT_COLUMN:
            return data[self._node[]._name].copy()
        elif self._node[]._kind == _RT_NEGATE:
            return neg(self._node[]._args[0].execute(data))
        else:
            return add(
                self._node[]._args[0].execute(data),
                self._node[]._args[1].execute(data),
            )


def main() raises:
    var a = arange[Int32Type](1, 9)    # [1, 2, 3, 4, 5, 6, 7, 8]
    var b = arange[Int32Type](10, 18)  # [10, 11, 12, 13, 14, 15, 16, 17]
    var c = arange[Int32Type](1, 9)    # [1, 2, 3, 4, 5, 6, 7, 8]

    # -(a) + b — numeric fusion (existing)
    var num_expr = Add(Negate(Column(a.copy())), Column(b.copy()))
    print(num_expr)                    # Add(Negate(Col[8]), Col[8])
    print(execute(num_expr, 8))        # [9, 9, 9, 9, 9, 9, 9, 9]

    # (a + b) == (c + 1) — fused: 3 ops, 1 pass, 0 intermediate arrays
    var eq_expr = EqExpr(
        Add(Column(a.copy()), Column(b.copy())),
        Add(Column(c.copy()), Literal[Int32Type](1)),
    )
    print(eq_expr)
    print(execute(eq_expr, 8))

    # a > 0 AND b < 15 — compound predicate, 1 fused pass
    var and_expr = AndExpr(
        GtExpr(Column(a.copy()), Literal[Int32Type](0)),
        LtExpr(Column(b.copy()), Literal[Int32Type](15)),
    )
    print(and_expr)                    # (Col[8] > Lit[0]) AND (Col[8] < Lit[15])
    print(execute(and_expr, 8))        # [true, true, true, true, true, false, false, false]

    # NOT (a > 5)
    var not_expr = NotExpr(GtExpr(Column(a.copy()), Literal[Int32Type](5)))
    print(not_expr)                    # NOT((Col[8] > Lit[5]))
    print(execute(not_expr, 8))        # [true, true, true, true, true, false, false, false]

    # filter(a, WHERE a + b > 15) — fused predicate + filter
    var filter_expr = FilterExpr(
        a.copy(),
        GtExpr(Add(Column(a.copy()), Column(b.copy())), Literal[Int32Type](15)),
    )
    print(filter_expr)
    print(execute(filter_expr, 8))     # elements of a where a+b > 15

    # Runtime path (unchanged)
    var rt_expr = RuntimeExpr.column("a").negate().add(RuntimeExpr.column("b"))
    var data = Dict[String, AnyArray]()
    data["a"] = a^
    data["b"] = b^
    print(rt_expr.execute(data))

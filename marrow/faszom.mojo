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
from marrow.kernels.arithmetic import (
    add,
    neg,
    AddKernel,
    NegKernel,
    SubKernel,
    MulKernel,
)
from marrow.kernels.boolean import AndKernel, OrKernel, NotKernel
from marrow.kernels.compare import (
    EqKernel,
    NeKernel,
    LtKernel,
    LeKernel,
    GtKernel,
    GeKernel,
)
from marrow.kernels.execution import ExecutionContext
from marrow.kernels.filter import filter as filter_kernel
from marrow.tabular import RecordBatch
from marrow.views import BufferView
from std.memory import OwnedPointer
from std.utils.index import IndexList
from std.reflection import reflect



# ===========================================================================
# Static layer — kernel fusion via comptime exec_core
# ===========================================================================


trait Expr(Copyable, ImplicitlyDestructible, Movable, Writable):
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

    def execute(self, length: Int) raises -> PrimitiveArray[Self.OutType]:
        comptime native = Self.OutType.native
        comptime width = simd_byte_width() // size_of[Scalar[native]]()
        var buf = Buffer.alloc_uninit[native](length)
        var view = buf.view[native](0, length)
        _ = view

        @parameter
        @always_inline
        def fill[
            W: Int, rank: Int, alignment: Int = 1
        ](idx: IndexList[rank]) -> None:
            var i = idx[0]
            view.store[W](i, self.exec_core[W](i))

        _vectorize_dispatch[native, width, fill](length)
        return PrimitiveArray[Self.OutType](
            dtype=Self.OutType(),
            length=length,
            nulls=0,
            offset=0,
            bitmap=None,
            buffer=buf.to_immutable(),
        )


trait BoolExpr(Copyable, ImplicitlyDestructible, Movable, Writable):
    """Base for bool-output expression nodes (comparisons).

    InNative carries the numeric dtype of the leaves so execute() can choose
    the correct SIMD width.  exec_core[W](idx) returns W booleans that
    execute() bit-packs into a BoolArray via BitmapView.store[W].
    """

    comptime InNative: DType

    @always_inline
    def exec_core[W: Int](self, idx: Int) -> SIMD[DType.bool, W]:
        ...

    def bind(mut self, batch: RecordBatch) raises:
        pass

    def __and__[RHS: BoolExpr](self, rhs: RHS) -> AndExpr[Self, RHS]:
        return AndExpr(self.copy(), rhs.copy())

    def __or__[RHS: BoolExpr](self, rhs: RHS) -> OrExpr[Self, RHS]:
        return OrExpr(self.copy(), rhs.copy())

    def __invert__(self) -> NotExpr[Self]:
        return NotExpr(self.copy())

    def execute(self, length: Int) raises -> BoolArray:
        comptime width = max(
            8, simd_byte_width() // size_of[Scalar[Self.InNative]]()
        )
        var bm = Bitmap.alloc_uninit(length)
        var view = bm.view()
        _ = view

        @parameter
        @always_inline
        def fill[
            W: Int, rank: Int, alignment: Int = 1
        ](idx: IndexList[rank]) -> None:
            var i = idx[0]
            view.store[W](i, self.exec_core[W](i))

        _vectorize_dispatch[Self.InNative, width, fill](length)
        return BoolArray(
            length=length,
            nulls=0,
            offset=0,
            bitmap=None,
            buffer=bm.to_immutable(),
        )


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


struct ColumnRef[name: StaticString, T: dt.NumericType](Expr, NumericExpr):
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
def col[name: StaticString, T: dt.NumericType](dtype: T) -> ColumnRef[name, T]:
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
        return NegKernel.core[Self.T.OutType.native, W](
            self.arg.exec_core[W](idx)
        )

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
# Relation — base trait for compile-time relational nodes
# ---------------------------------------------------------------------------


trait Relation(Copyable, ImplicitlyDestructible, Movable, Writable):
    """Base for compile-time relational nodes that consume a RecordBatch."""

    def execute(self, batch: RecordBatch) raises -> RecordBatch:
        ...


# ---------------------------------------------------------------------------
# FilterRel — multi-column relational filter returning RecordBatch
# ---------------------------------------------------------------------------


struct FilterRel[
    Pred: BoolExpr,
    *Fields: FieldDescriptor,
](Relation):
    """Relational filter: evaluate Pred, apply selection mask to all batch columns.

    *Fields defines the schema made available to the predicate (via ColumnRef
    bindings). All columns of the input RecordBatch are filtered and returned.

    Usage::

        var t = Schema[Field['a', Int32Type], Field['b', Int32Type], Field['data', Int32Type]]()
        var result: RecordBatch = t.filter(t.a + t.b > lit(int32, 5)).execute(batch)
        var col_data = result.column("data")
    """

    var _pred: Self.Pred

    def __init__(out self, var pred: Self.Pred):
        self._pred = pred^

    def __init__(out self, *, copy: Self):
        self._pred = copy._pred.copy()

    def execute(self, batch: RecordBatch) raises -> RecordBatch:
        """Execute the filter, binding a fresh copy of the predicate each call.

        Works on both rvalues (``t.filter(pred).execute(batch)``) and stored
        vars (``rel.execute(batch1); rel.execute(batch2)``).
        """
        var pred = self._pred.copy()
        pred.bind(batch)
        var sel: AnyArray = pred.execute(batch.num_rows())
        var out_cols = List[AnyArray]()
        for i in range(batch.num_columns()):
            out_cols.append(filter_kernel(batch.column(i), sel.copy()))
        return RecordBatch(schema=batch.schema, columns=out_cols^)

    def write_to[W: Writer](self, mut writer: W):
        writer.write("FilterRel WHERE ", self._pred)


# ---------------------------------------------------------------------------
# Schema — Field type tags + __getattr__ for t.col_name syntax
# ---------------------------------------------------------------------------


trait FieldDescriptor:
    """Compile-time trait for schema field descriptors."""

    comptime dtype: dt.NumericType

    @staticmethod
    def _name_matches[other: StaticString]() -> Bool:
        ...


struct Field[name: StaticString, T: dt.NumericType](
    Copyable, FieldDescriptor, Movable
):
    """Compile-time schema field descriptor. Carries no runtime data.

    Pass as a type parameter or as a value with dtype inference::

        Schema[Field['price', Float32Type], Field['qty', Int32Type]]()  # explicit
        Schema(Field['price'](float32), Field['qty'](int32))             # inferred T
    """

    comptime dtype = Self.T

    def __init__(out self, _dtype: Self.T):
        """Construct with dtype value; T is inferred, name must be in brackets.

        Enables ``Field['price'](float32)`` → ``Field['price', Float32Type]``.
        """
        pass

    @staticmethod
    def _name_matches[other: StaticString]() -> Bool:
        return Self.name == other


@always_inline
def field[name: StaticString, T: dt.NumericType](dtype: T) -> Field[name, T]:
    """Convenience shorthand for ``Field["name"](dtype)``.

    Usage: ``field["a"](int32)`` — same as ``Field["a"](int32)`` but shorter.
    """
    return Field[name](dtype)


def _schema_find_idx[
    name: StaticString,
    *Fields: FieldDescriptor,
    start: Int = 0,
]() -> Int:
    """Compile-time recursion: return the index of the field named ``name``."""
    comptime if start >= Fields.size:
        comptime assert False, "Schema has no field named '" + name + "'"
        return -1
    comptime if Fields[start]._name_matches[name]():
        return start
    return _schema_find_idx[name, *Fields, start=start + 1]()


struct Schema[*Fields: FieldDescriptor](Copyable, Movable):
    """Schema parameterised by ``Field[name, T]`` type tags; no runtime fields.

    Every attribute access (``t.col_name``) goes through ``__getattr_param__``,
    which returns a ``ColumnRef[name, T]`` for the matching field.

    Usage::

        var t = Schema[Field['a', Int32Type], Field['data', Int32Type]]()
        var t = Schema(Field("a", int32), Field("data", int32))  # inferred
        var result = t.data.where(t.a + t.b > lit(int32, 5)).execute(batch)
    """

    def __init__(out self):
        pass

    def __init__(out self, *fields: *Self.Fields):
        """Infer *Fields from Field value arguments.

        Enables ``Schema(Field("a", int32), Field("b", float32))`` syntax.
        """
        pass

    def __init__(out self, *, copy: Self):
        pass

    @always_inline
    def __getattr_param__[
        name: StaticString,
        idx: Int = _schema_find_idx[name, *Self.Fields](),
    ](self) -> ColumnRef[name, Self.Fields[idx].dtype]:
        """Return ``ColumnRef[name, T]`` for the matching field in the schema.
        """
        return ColumnRef[name, Self.Fields[idx].dtype]()

    def filter[
        Pred: BoolExpr
    ](self, pred: Pred) -> FilterRel[Pred, *Self.Fields]:
        """Build a FilterRel over all fields in this schema.

        Usage::

            var t = Schema[Field['a', Int32Type], Field['data', Int32Type]]()
            var result: RecordBatch = t.filter(t.a > lit(int32, 5)).execute(batch)
        """
        return FilterRel[Pred, *Self.Fields](pred.copy())


struct Fiszem[N: StaticString, T: AnyType]():
    """Compile-time schema field descriptor. Carries no runtime data.

    Pass as a type parameter or as a value with dtype inference::

        Schema[Field['price', Float32Type], Field['qty', Int32Type]]()  # explicit
        Schema(Field['price'](float32), Field['qty'](int32))             # inferred T
    """

    def __init__(out self):
        pass


struct Pina[T: AnyType](Copyable, Movable):
    """Schema parameterised by ``Field[name, T]`` type tags; no runtime fields.

    Every attribute access (``t.col_name``) goes through ``__getattr_param__``,
    which returns a ``ColumnRef[name, T]`` for the matching field.

    Usage::

        var t = Schema[Field['a', Int32Type], Field['data', Int32Type]]()
        var t = Schema(Field("a", int32), Field("data", int32))  # inferred
        var result = t.data.where(t.a + t.b > lit(int32, 5)).execute(batch)
    """

    def __init__(out self):
        pass

    def __init__(out self, *, copy: Self):
        pass

    @always_inline
    def __getattr_param__[
        name: StringLiteral,
        typ: AnyType = reflect[Self.T].field_type[name].T,
    ](self) -> Fiszem[name, typ]:
        """Return ``ColumnRef[name, T]`` for the matching field in the schema.
        """
        return Fiszem[name, typ]()


@always_inline
def table[*Fields: FieldDescriptor](*fields: *Fields) -> Schema[*Fields]:
    """Convenience alias for Schema().

    Usage: ``table(field["a"](int32), field["b"](int32))``
    """
    return Schema[*Fields](copy=Schema[*Fields]())




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


# ===========================================================================
# Runtime layer — lazy tree, NOT implementing Expr
# ===========================================================================

# sizeof[RuntimeExpr] = sizeof(OwnedPointer) = pointer width, so
# List[RuntimeExpr] inside _NodeContent has a known element size and the
# recursive size dependency resolves to a finite constant.

comptime _RT_COLUMN = 0
comptime _RT_NEGATE = 1
comptime _RT_ADD = 2


struct _NodeContent(Copyable, ImplicitlyDestructible, Movable):
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


struct RuntimeExpr(Copyable, ImplicitlyDestructible, Movable):
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
    var a = arange[Int32Type](1, 9)  # [1, 2, 3, 4, 5, 6, 7, 8]
    var b = arange[Int32Type](10, 18)  # [10, 11, 12, 13, 14, 15, 16, 17]
    var c = arange[Int32Type](1, 9)  # [1, 2, 3, 4, 5, 6, 7, 8]

    # -(a) + b — numeric fusion (existing)
    var num_expr = Add(Negate(Column(a.copy())), Column(b.copy()))
    print(num_expr)  # Add(Negate(Col[8]), Col[8])
    print(execute(num_expr, 8))  # [9, 9, 9, 9, 9, 9, 9, 9]

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
    print(and_expr)  # (Col[8] > Lit[0]) AND (Col[8] < Lit[15])
    print(
        execute(and_expr, 8)
    )  # [true, true, true, true, true, false, false, false]

    # NOT (a > 5)
    var not_expr = NotExpr(GtExpr(Column(a.copy()), Literal[Int32Type](5)))
    print(not_expr)  # NOT((Col[8] > Lit[5]))
    print(
        execute(not_expr, 8)
    )  # [true, true, true, true, true, false, false, false]

    # Schema-based FilterRel — multi-column relational filter (print-only demo)
    var t = Schema[
        Field["a", Int32Type], Field["b", Int32Type], Field["data", Int32Type]
    ]()
    var filter_rel = t.filter(
        GtExpr(
            Add(col["a"](Int32Type()), col["b"](Int32Type())),
            Literal[Int32Type](15),
        )
    )
    print(
        filter_rel
    )  # FilterRel WHERE (col['a', ...] + col['b', ...] > Lit[15])

    # Runtime path (unchanged)
    var rt_expr = RuntimeExpr.column("a").negate().add(RuntimeExpr.column("b"))
    var data = Dict[String, AnyArray]()
    data["a"] = a^
    data["b"] = b^
    print(rt_expr.execute(data))

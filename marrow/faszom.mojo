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
from marrow.kernels.compare import EqKernel, NeKernel, LtKernel, LeKernel, GtKernel, GeKernel
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


trait NumericExpr(Expr):
    """Marker: OutType is numeric. Inherits exec_core from Expr, so
    T: NumericExpr implies T.exec_core[W](idx) is callable.
    Default negate() avoids per-struct repetition.
    """

    def negate(self) raises -> Negate[Self]:
        return Negate(self.copy())


trait BoolExpr(Copyable, Movable, Writable, ImplicitlyDestructible):
    """Base for bool-output expression nodes (comparisons).

    InNative carries the numeric dtype of the leaves so execute() can choose
    the correct SIMD width.  exec_core[W](idx) returns W booleans that
    execute() bit-packs into a BoolArray via BitmapView.store[W].
    """

    comptime InNative: DType

    @always_inline
    def exec_core[W: Int](self, idx: Int) -> SIMD[DType.bool, W]: ...


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


struct Negate[T: NumericExpr](NumericExpr):
    comptime OutType = Self.T.OutType

    var arg: Self.T

    def __init__(out self, var arg: Self.T):
        self.arg = arg^

    def __init__(out self, *, copy: Self):
        self.arg = copy.arg.copy()

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

    @always_inline
    def exec_core[W: Int](self, idx: Int) -> SIMD[DType.bool, W]:
        return GeKernel.core[Self.L.OutType.native, W](
            self.left.exec_core[W](idx),
            self.right.exec_core[W](idx).cast[Self.L.OutType.native](),
        )

    def write_to[W: Writer](self, mut writer: W):
        writer.write("(", self.left, " >= ", self.right, ")")


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
    print(eq_expr)                     # ((Col[8] + Col[8]) == (Col[8] + Lit[1]))
    print(execute(eq_expr, 8))         # [false, false, ..., false] (11!=2, 13!=3, ...)

    # a > 0 — single comparison pass
    var gt_expr = GtExpr(Column(a.copy()), Literal[Int32Type](0))
    print(gt_expr)                     # (Col[8] > Lit[0])
    print(execute(gt_expr, 8))         # [true, true, true, true, true, true, true, true]

    # Runtime path (unchanged)
    var rt_expr = RuntimeExpr.column("a").negate().add(RuntimeExpr.column("b"))
    var data = Dict[String, AnyArray]()
    data["a"] = a^
    data["b"] = b^
    print(rt_expr.execute(data))

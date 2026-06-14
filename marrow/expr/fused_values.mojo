"""Comptime-fused expression nodes for the marrow expression system.

Mirrors the static layer from ``faszom.mojo`` — each node is a generic struct
where type parameters encode the expression tree structure, enabling the
compiler to inline the full ``core[W]()`` call chain into a single fused
vectorize loop with zero intermediate arrays.

Expression nodes
----------------
``FusedColumn[T]`` — typed column reference (resolves from RecordBatch at core time)
``FusedAdd[L, R]`` — fused binary add (generic over any two NumericTypedValue children)

Traits
------
``Value`` — base trait for all expression nodes (kind, dtype, inputs, write_to)
``TypedValue`` — extends Value with comptime fusion support
``NumericTypedValue`` — numeric nodes (core[W](batch, idx), execute(batch))

Usage
-----
    var col_a = FusedColumn[Int64Type](0)
    var col_b = FusedColumn[Int64Type](1)
    var expr = FusedAdd(col_a, col_b)
    var proc = FusedProcessor(expr)
    var result = proc.eval(batch)  # single fused pass, zero intermediates
"""

from std.algorithm.backend.vectorize import vectorize
from std.builtin.simd import Scalar
from std.memory import ArcPointer
from std.sys import size_of
from std.sys.info import simd_byte_width
from std.utils.index import IndexList

import marrow.dtypes as dt
from marrow.arrays import AnyArray, PrimitiveArray
from marrow.buffers import Buffer
from marrow.dtypes import AnyDataType, DType, NumericType
from marrow.expr.values import Expr
from marrow.tabular import RecordBatch


# ---------------------------------------------------------------------------
# Value trait — base for all expression nodes
# ---------------------------------------------------------------------------


trait Value(ImplicitlyDestructible, Movable):
    """Interface for immutable scalar expression nodes.

    This trait is implemented by both the type-erased expressions in
    ``values.mojo`` and the comptime-fused expressions in this module.
    """

    def kind(self) -> UInt8:
        """Return the node-kind constant."""
        ...

    def dtype(self) -> Optional[AnyDataType]:
        """Return the output data type, or None if not yet inferred."""
        ...

    def inputs(self) -> List[Expr]:
        """Return child expressions (empty for leaf nodes)."""
        ...

    def write_to[W: Writer](self, mut writer: W):
        """Format this node for display (children formatted recursively)."""
        ...


# ---------------------------------------------------------------------------
# TypedValue trait — base for comptime-fused expression nodes
# ---------------------------------------------------------------------------


trait TypedValue(Value, Copyable, ImplicitlyCopyable, ImplicitlyDestructible, Movable, Writable):
    """Base trait for statically-typed (comptime-fused) expression nodes.

    Unlike the type-erased ``Value`` trait in ``values.mojo``, nodes that
    implement ``TypedValue`` are generic structs where type parameters encode
    the full expression tree — enabling the compiler to inline ``core[W]()``
    across the entire tree into a single fused vectorize loop.

    ``core[W](batch, idx)`` — compute one SIMD lane, receiving the batch so
        leaf nodes can resolve column data on demand.
    ``execute(batch)``      — run the fused vectorize loop, return a PrimitiveArray.
    ``write_to(writer)``    — format this node for display.
    """

    def write_to[W: Writer](self, mut writer: W):
        """Format this node for display."""
        ...


# ---------------------------------------------------------------------------
# NumericTypedValue trait — numeric fused expression nodes
# ---------------------------------------------------------------------------


trait NumericTypedValue(TypedValue):
    """Trait for numeric fused expression nodes.

    Provides ``core[W](batch, idx)`` (SIMD lane computation with batch) and
    ``execute(batch)`` (fused vectorize loop).  Both ``FusedColumn[T]`` and
    ``FusedAdd[L, R]`` implement this trait.

    ``comptime OutType`` — the output ``NumericType``.
    ``comptime NativeType`` — the Mojo scalar type for SIMD operations.
    """

    comptime OutType: NumericType
    """Output numeric data type (inferred from children for composite nodes)."""

    comptime NativeType: DType
    """Native Mojo scalar type for SIMD operations."""

    @always_inline
    def core[W: Int](self, batch: RecordBatch, idx: Int) -> SIMD[Self.NativeType, W]:
        """Compute one SIMD lane at the given index, resolving data from batch."""
        ...

    def execute(self, batch: RecordBatch) raises -> PrimitiveArray[Self.OutType]:
        """Run the fused vectorize loop and return the result array."""
        ...

    # Default Value trait implementations

    def kind(self) -> UInt8:
        """Return the node-kind constant."""
        return 0  # Fused nodes don't have a specific kind constant

    def dtype(self) -> Optional[AnyDataType]:
        """Return the output data type."""
        return Optional[AnyDataType](Self.OutType())

    def inputs(self) -> List[Expr]:
        """Return child expressions (empty for leaf nodes)."""
        return List[Expr]()

    def write_to[W: Writer](self, mut writer: W):
        """Format this node for display."""
        ...

    def to_expr(self) -> Expr:
        """Box this comptime-fused expression into a runtime Expr.

        The resulting ``Expr`` carries the fused expression in its ``_fused``
        slot, so ``dtype()`` and ``write_to()`` delegate to the comptime-fused
        implementation.
        """
        var ptr = ArcPointer[Self](self.copy())
        var result = Expr(
            tag=FUSED,
            args=List[Expr](),
            kind_data=0,
            value=None,
            name=String(),
        )
        result._fused = rebind[ArcPointer[NoneType]](ptr^)
        result._virt_fused_dtype = _fused_dtype_tramp[Self]
        result._virt_fused_write = _fused_write_tramp[Self]
        return result^


# ---------------------------------------------------------------------------
# Trampoline helpers for boxing fused expressions into Expr
# ---------------------------------------------------------------------------


def _fused_dtype_tramp[T: NumericTypedValue](
    ptr: ArcPointer[NoneType],
) -> Optional[AnyDataType]:
    """Thin trampoline: delegate dtype() to a concrete NumericTypedValue."""
    var typed = rebind[ArcPointer[T]](ptr)
    return typed[].dtype()


def _fused_write_tramp[T: NumericTypedValue](
    ptr: ArcPointer[NoneType],
) -> String:
    """Thin trampoline: delegate write_to() to a concrete NumericTypedValue."""
    var typed = rebind[ArcPointer[T]](ptr)
    var s = String()
    typed[].write_to(s)
    return s^


# ---------------------------------------------------------------------------
# FusedColumn — typed column reference
# ---------------------------------------------------------------------------


struct FusedColumn[T: dt.NumericType](NumericTypedValue):
    """Typed column reference that resolves from a RecordBatch at core time.

    ``index``  — positional index into the batch's column list.

    The column data is resolved on demand from the batch passed to ``core()``,
    so no separate bind phase is needed.

    Usage::

        var col = FusedColumn[Int64Type](0)
        var result = col.execute(batch)  # resolves batch.columns[0] internally
    """

    comptime OutType = Self.T
    comptime NativeType = Self.T.native

    var index: Int

    def __init__(out self, index: Int):
        self.index = index

    def __init__(out self, *, copy: Self):
        self.index = copy.index

    @always_inline
    def core[W: Int](self, batch: RecordBatch, idx: Int) -> SIMD[Self.NativeType, W]:
        return batch.columns[self.index].as_primitive[Self.T]().values().load[W](idx)

    def execute(self, batch: RecordBatch) raises -> PrimitiveArray[Self.OutType]:
        comptime native = Self.NativeType
        comptime width = simd_byte_width() // size_of[Scalar[native]]()
        var length = batch.num_rows()
        var buf = Buffer.alloc_uninit[native](length)

        @parameter
        @always_inline
        def fill[W: Int, rank: Int, alignment: Int = 1](
            idx: IndexList[rank],
        ) -> None:
            var i = idx[0]
            var val = self.core[W](batch, i)
            var v = buf.view[native](i, length)
            v.store[W](0, val)

        _vectorize_dispatch[native, width, fill](length)
        return PrimitiveArray[Self.OutType](
            dtype=Self.OutType(),
            length=length,
            nulls=0,
            offset=0,
            bitmap=None,
            buffer=buf.to_immutable(),
        )

    def write_to[W: Writer](self, mut writer: W):
        writer.write(t"FusedCol[{self.index}]")


# ---------------------------------------------------------------------------
# FusedAdd — comptime-fused binary add
# ---------------------------------------------------------------------------


struct FusedAdd[L: NumericTypedValue, R: NumericTypedValue](NumericTypedValue):
    """Fused binary add: evaluates left + right in a single vectorized pass.

    Generic over any two ``NumericTypedValue`` children, so the compiler can
    inline the full ``core[W]()`` call chain.  Mirrors ``Add[L, R]`` from
    ``faszom.mojo`` exactly.

    ``comptime OutType`` and ``comptime NativeType`` are inherited from the
    left child (both operands must have the same dtype).

    Usage::

        var col_a = FusedColumn[Int64Type](0)
        var col_b = FusedColumn[Int64Type](1)
        var expr = FusedAdd(col_a, col_b)
        var result = expr.execute(batch)  # single fused pass
    """

    comptime OutType = Self.L.OutType
    comptime NativeType = Self.L.NativeType

    var left: Self.L
    var right: Self.R

    def __init__(out self, var left: Self.L, var right: Self.R):
        self.left = left^
        self.right = right^

    def __init__(out self, *, copy: Self):
        self.left = copy.left.copy()
        self.right = copy.right.copy()

    @always_inline
    def core[W: Int](self, batch: RecordBatch, idx: Int) -> SIMD[Self.NativeType, W]:
        var l = self.left.core[W](batch, idx)
        var r = self.right.core[W](batch, idx).cast[Self.NativeType]()
        return l + r

    def execute(self, batch: RecordBatch) raises -> PrimitiveArray[Self.OutType]:
        comptime native = Self.NativeType
        comptime width = simd_byte_width() // size_of[Scalar[native]]()
        var length = batch.num_rows()
        var buf = Buffer.alloc_uninit[native](length)

        @parameter
        @always_inline
        def fill[W: Int, rank: Int, alignment: Int = 1](
            idx: IndexList[rank],
        ) -> None:
            var i = idx[0]
            var val = self.core[W](batch, i)
            var v = buf.view[native](i, length)
            v.store[W](0, val)

        _vectorize_dispatch[native, width, fill](length)
        return PrimitiveArray[Self.OutType](
            dtype=Self.OutType(),
            length=length,
            nulls=0,
            offset=0,
            bitmap=None,
            buffer=buf.to_immutable(),
        )

    def write_to[W: Writer](self, mut writer: W):
        writer.write(t"FusedAdd(")
        self.left.write_to(writer)
        writer.write(t", ")
        self.right.write_to(writer)
        writer.write(t")")


# ---------------------------------------------------------------------------
# FusedSub — comptime-fused binary subtract
# ---------------------------------------------------------------------------


struct FusedSub[L: NumericTypedValue, R: NumericTypedValue](NumericTypedValue):
    """Fused binary subtract: evaluates left - right in a single vectorized pass.

    Mirrors ``Sub[L, R]`` from ``faszom.mojo``.
    """

    comptime OutType = Self.L.OutType
    comptime NativeType = Self.L.NativeType

    var left: Self.L
    var right: Self.R

    def __init__(out self, var left: Self.L, var right: Self.R):
        self.left = left^
        self.right = right^

    def __init__(out self, *, copy: Self):
        self.left = copy.left.copy()
        self.right = copy.right.copy()

    @always_inline
    def core[W: Int](self, batch: RecordBatch, idx: Int) -> SIMD[Self.NativeType, W]:
        var l = self.left.core[W](batch, idx)
        var r = self.right.core[W](batch, idx).cast[Self.NativeType]()
        return l - r

    def execute(self, batch: RecordBatch) raises -> PrimitiveArray[Self.OutType]:
        comptime native = Self.NativeType
        comptime width = simd_byte_width() // size_of[Scalar[native]]()
        var length = batch.num_rows()
        var buf = Buffer.alloc_uninit[native](length)

        @parameter
        @always_inline
        def fill[W: Int, rank: Int, alignment: Int = 1](
            idx: IndexList[rank],
        ) -> None:
            var i = idx[0]
            var val = self.core[W](batch, i)
            var v = buf.view[native](i, length)
            v.store[W](0, val)

        _vectorize_dispatch[native, width, fill](length)
        return PrimitiveArray[Self.OutType](
            dtype=Self.OutType(),
            length=length,
            nulls=0,
            offset=0,
            bitmap=None,
            buffer=buf.to_immutable(),
        )

    def write_to[W: Writer](self, mut writer: W):
        writer.write(t"FusedSub(")
        self.left.write_to(writer)
        writer.write(t", ")
        self.right.write_to(writer)
        writer.write(t")")


# ---------------------------------------------------------------------------
# Vectorize dispatch helper (mirrors faszom.mojo)
# ---------------------------------------------------------------------------


def _vectorize_dispatch[
    native: DType,
    cpu_width: Int,
    process: def[W: Int, rank: Int, alignment: Int = 1](
        IndexList[rank]
    ) capturing -> None,
](length: Int):
    """Run process over [0, length) using vectorize."""

    @always_inline
    def lane[W: Int](i: Int):
        process[W, rank=1](IndexList[1](i))

    vectorize[cpu_width](length, lane)

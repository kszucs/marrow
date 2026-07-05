"""Comptime-typed expression nodes for the marrow expression system.

This is the **default** expression layer: each node is a generic struct
where type parameters encode the expression tree structure, enabling the
compiler to inline the full ``core[W]()`` call chain into a single fused
vectorize loop with zero intermediate arrays.

Expression nodes
----------------
``Column[T]`` — typed column reference (resolves from RecordBatch at core time)
``Add[L, R]`` — fused binary add (generic over any two NumericValue children)
``Sub[L, R]`` — fused binary subtract

Traits
------
``Value`` — base trait for every expression node (dtype, write_to;
    Copyable/Writable/etc.), shared with the type-erased ``Expr`` in
    ``runtime.mojo``.
``NumericValue`` — numeric nodes (core[W](batch, idx), execute(batch))

Bridging to the runtime layer
-----------------------------
The ``Expr(value)`` constructor (see ``runtime.mojo``) boxes a comptime node
into a runtime ``Expr``, so it can flow through APIs that build/execute plans
without knowing the concrete comptime type (e.g. the Python bindings).  The
boxed ``Expr`` fully delegates ``dtype()``, ``write_to()``, and ``eval()``
back to the concrete node via trampolines, so a fused subtree keeps its
single-pass execution even when driven through the type-erased path.

Usage
-----
    var col_a = Column[Int64Type](0)
    var col_b = Column[Int64Type](1)
    var expr = Add(col_a, col_b)
    var result = expr.execute(batch)  # single fused pass, zero intermediates
"""

from std.algorithm.backend.vectorize import vectorize
from std.builtin.simd import Scalar
from std.sys import size_of
from std.sys.info import simd_byte_width
from std.utils.index import IndexList

import marrow.dtypes as dt
from marrow.arrays import AnyArray, PrimitiveArray
from marrow.buffers import Buffer
from marrow.dtypes import AnyDataType, DType, NumericType
from marrow.tabular import RecordBatch


# ---------------------------------------------------------------------------
# Value trait — base for all expression nodes (shared with runtime.mojo)
# ---------------------------------------------------------------------------


trait Value(
    Copyable,
    ImplicitlyCopyable,
    ImplicitlyDestructible,
    Movable,
    Writable,
):
    """Interface every expression node must implement.

    This is the single canonical trait implemented by both the comptime-typed
    expressions in this module and the type-erased ``Expr`` in ``runtime.mojo``.
    """

    def dtype(self) -> Optional[AnyDataType]:
        """Return the output data type, or None if not yet inferred."""
        ...

    def write_to[W: Writer](self, mut writer: W):
        """Format this node for display (children formatted recursively)."""
        ...


# ---------------------------------------------------------------------------
# NumericValue trait — numeric comptime expression nodes
# ---------------------------------------------------------------------------


trait NumericValue(Value):
    """Trait for numeric comptime-typed expression nodes.

    Nodes implementing this trait are generic structs where type parameters
    encode the full expression tree — enabling the compiler to inline
    ``core[W]()`` across the entire tree into a single fused vectorize loop.

    Provides ``core[W](batch, idx)`` (SIMD lane computation with batch) and
    ``execute(batch)`` (fused vectorize loop).  Both ``Column[T]`` and
    ``Add[L, R]`` implement this trait.

    ``comptime OutType`` — the output ``NumericType``.
    ``comptime NativeType`` — the Mojo scalar type for SIMD operations.
    """

    comptime OutType: NumericType
    """Output numeric data type (inferred from children for composite nodes)."""

    comptime NativeType: DType
    """Native Mojo scalar type for SIMD operations."""

    @always_inline
    def core[
        W: Int
    ](self, batch: RecordBatch, idx: Int) -> SIMD[Self.NativeType, W]:
        """Compute one SIMD lane at the given index, resolving data from batch.
        """
        ...

    def execute(
        self, batch: RecordBatch
    ) raises -> PrimitiveArray[Self.OutType]:
        """Run the fused vectorize loop and return the result array."""
        ...

    # Default Value trait implementation

    def dtype(self) -> Optional[AnyDataType]:
        """Return the output data type."""
        return Optional[AnyDataType](Self.OutType())

    def write_to[W: Writer](self, mut writer: W):
        """Format this node for display."""
        ...


# ---------------------------------------------------------------------------
# Column — typed column reference
# ---------------------------------------------------------------------------


struct Column[T: dt.NumericType](NumericValue):
    """Typed column reference that resolves from a RecordBatch at core time.

    ``index``  — positional index into the batch's column list.

    The column data is resolved on demand from the batch passed to ``core()``,
    so no separate bind phase is needed.

    Usage::

        var col = Column[Int64Type](0)
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
    def core[
        W: Int
    ](self, batch: RecordBatch, idx: Int) -> SIMD[Self.NativeType, W]:
        return (
            batch.columns[self.index]
            .as_primitive[Self.T]()
            .values()
            .load[W](idx)
        )

    def execute(
        self, batch: RecordBatch
    ) raises -> PrimitiveArray[Self.OutType]:
        comptime native = Self.NativeType
        comptime width = simd_byte_width() // size_of[Scalar[native]]()
        var length = batch.num_rows()
        var buf = Buffer.alloc_uninit[native](length)

        @parameter
        @always_inline
        def fill[
            W: Int, rank: Int, alignment: Int = 1
        ](idx: IndexList[rank],) -> None:
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
        writer.write(t"Col[{self.index}]")


# ---------------------------------------------------------------------------
# Add — comptime-typed binary add
# ---------------------------------------------------------------------------


struct Add[L: NumericValue, R: NumericValue](NumericValue):
    """Fused binary add: evaluates left + right in a single vectorized pass.

    Generic over any two ``NumericValue`` children, so the compiler can
    inline the full ``core[W]()`` call chain.

    ``comptime OutType`` and ``comptime NativeType`` are inherited from the
    left child (both operands must have the same dtype).

    Usage::

        var col_a = Column[Int64Type](0)
        var col_b = Column[Int64Type](1)
        var expr = Add(col_a, col_b)
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
    def core[
        W: Int
    ](self, batch: RecordBatch, idx: Int) -> SIMD[Self.NativeType, W]:
        var l = self.left.core[W](batch, idx)
        var r = self.right.core[W](batch, idx).cast[Self.NativeType]()
        return l + r

    def execute(
        self, batch: RecordBatch
    ) raises -> PrimitiveArray[Self.OutType]:
        comptime native = Self.NativeType
        comptime width = simd_byte_width() // size_of[Scalar[native]]()
        var length = batch.num_rows()
        var buf = Buffer.alloc_uninit[native](length)

        @parameter
        @always_inline
        def fill[
            W: Int, rank: Int, alignment: Int = 1
        ](idx: IndexList[rank],) -> None:
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
        writer.write(t"Add(")
        self.left.write_to(writer)
        writer.write(t", ")
        self.right.write_to(writer)
        writer.write(t")")


# ---------------------------------------------------------------------------
# Sub — comptime-typed binary subtract
# ---------------------------------------------------------------------------


struct Sub[L: NumericValue, R: NumericValue](NumericValue):
    """Fused binary subtract: evaluates left - right in a single vectorized pass.
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
    def core[
        W: Int
    ](self, batch: RecordBatch, idx: Int) -> SIMD[Self.NativeType, W]:
        var l = self.left.core[W](batch, idx)
        var r = self.right.core[W](batch, idx).cast[Self.NativeType]()
        return l - r

    def execute(
        self, batch: RecordBatch
    ) raises -> PrimitiveArray[Self.OutType]:
        comptime native = Self.NativeType
        comptime width = simd_byte_width() // size_of[Scalar[native]]()
        var length = batch.num_rows()
        var buf = Buffer.alloc_uninit[native](length)

        @parameter
        @always_inline
        def fill[
            W: Int, rank: Int, alignment: Int = 1
        ](idx: IndexList[rank],) -> None:
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
        writer.write(t"Sub(")
        self.left.write_to(writer)
        writer.write(t", ")
        self.right.write_to(writer)
        writer.write(t")")


# ---------------------------------------------------------------------------
# Vectorize dispatch helper
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

"""Comptime-typed expression nodes and traits — the foundation of
``marrow.aot``.

Each node is a generic struct where type parameters encode the expression
tree structure, enabling the compiler to inline the full ``core[W]()`` call
chain into a single fused vectorize loop with zero intermediate arrays.
Columns are referenced by position (``Column[T](index)``); see
``marrow.aot.table`` for the named, reflection-based, ``Table``-schema
variant used by ``Project``/``Filter``.

Expression nodes
----------------
``Column[T]`` — typed column reference (resolves from RecordBatch at core time)
``Add[L, R]`` — fused binary add (generic over any two NumericValue children)
``Sub[L, R]`` — fused binary subtract
``Lt[L, R]``, ``Gt[L, R]``, ``Eq[L, R]`` — fused binary comparisons; result is
    a bit-packed BoolArray, not a PrimitiveArray (see ``BoolValue`` below)
``StringColumn`` — typed string column reference
``Length[S]`` — fused string byte length

Traits
------
``Value`` — base trait for every expression node (dtype, write_to;
    Copyable/Writable/etc.), shared with the type-erased ``Expr`` in
    ``marrow.dyn.expr`` — that module imports these traits directly
    from here (``aot`` has no dependency on ``dyn``; the reverse does).
``NumericValue`` — numeric nodes (core[W](batch, idx), execute(batch))
``BoolValue`` — boolean/predicate nodes (core[W] returns SIMD[bool, W];
    execute(batch) bit-packs directly into a BoolArray)

Bridging to the runtime layer
-----------------------------
The ``Expr(value)`` constructor (see ``marrow.dyn.expr``) boxes a
comptime node from this module into a runtime ``Expr``, so it can flow
through APIs that build/execute plans without knowing the concrete comptime
type (e.g. the Python bindings). The boxed ``Expr`` fully delegates
``dtype()``, ``write_to()``, and ``eval()`` back to the concrete node via
trampolines, so a fused subtree keeps its single-pass execution even when
driven through the type-erased path.

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
from marrow.arrays import AnyArray, BoolArray, PrimitiveArray, StringArray
from marrow.buffers import Bitmap, Buffer
from marrow.dtypes import AnyDataType, DType, NumericType
from marrow.tabular import RecordBatch


# ---------------------------------------------------------------------------
# Value trait — base for all expression nodes (shared with runtime.mojo)
# ---------------------------------------------------------------------------


trait Value(
    Copyable,
    ImplicitlyCopyable,
    ImplicitlyDeletable,
    Movable,
    Writable,
):
    """Interface every expression node must implement.

    This is the single canonical trait implemented by both the comptime-typed
    expressions in this module and the type-erased ``Expr`` in ``runtime.mojo``.

    ``dtype()`` is declared by the ``NumericValue`` / ``StringValue`` sub-traits
    (which supply a default implementation); ``write_to()`` comes from the
    inherited ``Writable`` requirement. Declaring them here too would make calls
    on a generic ``Value`` ambiguous between the base and sub-trait candidates.
    """

    pass


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
        """Run the fused vectorize loop over ``core[W]()`` and return the
        result array.  Shared by every ``NumericValue`` node — only
        ``core[W]()`` (and ``comptime OutType``/``NativeType``) differ per
        node, so this default implementation covers all of them.
        """
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

    # Default Value trait implementation (write_to comes from Writable)

    def dtype(self) -> Optional[AnyDataType]:
        """Return the output data type."""
        return Optional[AnyDataType](Self.OutType())


# ---------------------------------------------------------------------------
# Column — typed column reference
# ---------------------------------------------------------------------------


@fieldwise_init
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

    def write_to[W: Writer](self, mut writer: W):
        writer.write(t"Col[{self.index}]")


# ---------------------------------------------------------------------------
# Add — comptime-typed binary add
# ---------------------------------------------------------------------------


@fieldwise_init
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

    @always_inline
    def core[
        W: Int
    ](self, batch: RecordBatch, idx: Int) -> SIMD[Self.NativeType, W]:
        var l = self.left.core[W](batch, idx)
        var r = self.right.core[W](batch, idx).cast[Self.NativeType]()
        return l + r

    def write_to[W: Writer](self, mut writer: W):
        writer.write(t"Add(")
        self.left.write_to(writer)
        writer.write(t", ")
        self.right.write_to(writer)
        writer.write(t")")


# ---------------------------------------------------------------------------
# Sub — comptime-typed binary subtract
# ---------------------------------------------------------------------------


@fieldwise_init
struct Sub[L: NumericValue, R: NumericValue](NumericValue):
    """Fused binary subtract: evaluates left - right in a single vectorized pass.
    """

    comptime OutType = Self.L.OutType
    comptime NativeType = Self.L.NativeType

    var left: Self.L
    var right: Self.R

    @always_inline
    def core[
        W: Int
    ](self, batch: RecordBatch, idx: Int) -> SIMD[Self.NativeType, W]:
        var l = self.left.core[W](batch, idx)
        var r = self.right.core[W](batch, idx).cast[Self.NativeType]()
        return l - r

    def write_to[W: Writer](self, mut writer: W):
        writer.write(t"Sub(")
        self.left.write_to(writer)
        writer.write(t", ")
        self.right.write_to(writer)
        writer.write(t")")


# ---------------------------------------------------------------------------
# BoolValue trait — boolean comptime expression nodes (comparisons/predicates)
# ---------------------------------------------------------------------------


trait BoolValue(Value):
    """Trait for boolean comptime-typed expression nodes (predicates).

    Mirrors ``NumericValue`` but the result is a bit-packed ``BoolArray``
    instead of a ``PrimitiveArray[OutType]``: ``core[W]()`` returns a
    ``SIMD[DType.bool, W]`` mask, and ``execute()`` bit-packs that mask
    directly into a ``Bitmap`` in the same single fused vectorize loop — no
    intermediate array for either comparison operand, no separate
    array-of-bools-to-BoolArray conversion.

    ``comptime NativeType`` is the operands' native Mojo scalar type (drives
    vectorize width); the output dtype is always ``bool_``, so there is no
    ``OutType`` here.
    """

    comptime NativeType: DType
    """Native Mojo scalar type of the compared operands (both sides must
    match, like NumericValue's Add/Sub)."""

    @always_inline
    def core[
        W: Int
    ](self, batch: RecordBatch, idx: Int) -> SIMD[DType.bool, W]:
        """Compute one SIMD lane of the predicate at the given index."""
        ...

    def execute(self, batch: RecordBatch) raises -> BoolArray:
        """Run the fused vectorize loop over ``core[W]()``, bit-packing each
        lane directly into the result bitmap.
        """
        comptime native = Self.NativeType
        comptime width = simd_byte_width() // size_of[Scalar[native]]()
        var length = batch.num_rows()
        var bm = Bitmap.alloc_uninit(length)

        @parameter
        @always_inline
        def fill[
            W: Int, rank: Int, alignment: Int = 1
        ](idx: IndexList[rank],) -> None:
            var i = idx[0]
            var val = self.core[W](batch, i)
            var view = bm.view()
            view.store[W](i, val)

        _vectorize_dispatch[native, width, fill](length)
        return BoolArray(
            length=length,
            nulls=0,
            offset=0,
            bitmap=None,
            buffer=bm.to_immutable(),
        )

    # Default Value trait implementation (write_to comes from Writable)

    def dtype(self) -> Optional[AnyDataType]:
        """Return the output data type (always bool)."""
        return Optional[AnyDataType](dt.bool_)


# ---------------------------------------------------------------------------
# Lt — comptime-typed binary less-than
# ---------------------------------------------------------------------------


@fieldwise_init
struct Lt[L: NumericValue, R: NumericValue](BoolValue):
    """Fused binary less-than: evaluates left < right in a single vectorized
    pass, producing a bit-packed BoolArray with zero intermediate arrays.
    """

    comptime NativeType = Self.L.NativeType

    var left: Self.L
    var right: Self.R

    @always_inline
    def core[
        W: Int
    ](self, batch: RecordBatch, idx: Int) -> SIMD[DType.bool, W]:
        var l = self.left.core[W](batch, idx)
        var r = self.right.core[W](batch, idx).cast[Self.NativeType]()
        return l.lt(r)

    def write_to[W: Writer](self, mut writer: W):
        writer.write(t"Lt(")
        self.left.write_to(writer)
        writer.write(t", ")
        self.right.write_to(writer)
        writer.write(t")")


# ---------------------------------------------------------------------------
# Gt — comptime-typed binary greater-than
# ---------------------------------------------------------------------------


@fieldwise_init
struct Gt[L: NumericValue, R: NumericValue](BoolValue):
    """Fused binary greater-than: evaluates left > right in a single
    vectorized pass, producing a bit-packed BoolArray with zero intermediate
    arrays.
    """

    comptime NativeType = Self.L.NativeType

    var left: Self.L
    var right: Self.R

    @always_inline
    def core[
        W: Int
    ](self, batch: RecordBatch, idx: Int) -> SIMD[DType.bool, W]:
        var l = self.left.core[W](batch, idx)
        var r = self.right.core[W](batch, idx).cast[Self.NativeType]()
        return l.gt(r)

    def write_to[W: Writer](self, mut writer: W):
        writer.write(t"Gt(")
        self.left.write_to(writer)
        writer.write(t", ")
        self.right.write_to(writer)
        writer.write(t")")


# ---------------------------------------------------------------------------
# Eq — comptime-typed binary equality
# ---------------------------------------------------------------------------


@fieldwise_init
struct Eq[L: NumericValue, R: NumericValue](BoolValue):
    """Fused binary equality: evaluates left == right in a single vectorized
    pass, producing a bit-packed BoolArray with zero intermediate arrays.
    """

    comptime NativeType = Self.L.NativeType

    var left: Self.L
    var right: Self.R

    @always_inline
    def core[
        W: Int
    ](self, batch: RecordBatch, idx: Int) -> SIMD[DType.bool, W]:
        var l = self.left.core[W](batch, idx)
        var r = self.right.core[W](batch, idx).cast[Self.NativeType]()
        return l.eq(r)

    def write_to[W: Writer](self, mut writer: W):
        writer.write(t"Eq(")
        self.left.write_to(writer)
        writer.write(t", ")
        self.right.write_to(writer)
        writer.write(t")")


# ---------------------------------------------------------------------------
# StringValue trait — string comptime expression nodes
# ---------------------------------------------------------------------------


trait StringValue(Value):
    """Trait for string comptime-typed expression nodes.

    Mirrors ``NumericValue`` but for the variable-length string
    representation: rather than a per-lane SIMD ``core[W]()``, nodes
    implementing this trait resolve directly to a ``StringArray``.

    Provides ``resolve(batch)`` (fetch the node's string data from a batch)
    and ``execute(batch)`` (top-level entry point, equal to ``resolve`` for
    leaf nodes).  ``StringColumn`` implements this trait.
    """

    def resolve(self, batch: RecordBatch) -> StringArray:
        """Resolve this node's string data from *batch*."""
        ...

    def execute(self, batch: RecordBatch) raises -> StringArray:
        """Run this node and return the resulting string array."""
        ...

    # Default Value trait implementation (write_to comes from Writable)

    def dtype(self) -> Optional[AnyDataType]:
        """Return the output data type."""
        return Optional[AnyDataType](dt.string)


# ---------------------------------------------------------------------------
# StringColumn — typed string column reference
# ---------------------------------------------------------------------------


@fieldwise_init
struct StringColumn(StringValue):
    """Typed string column reference that resolves from a RecordBatch.

    ``index``  — positional index into the batch's column list.

    Usage::

        var col = StringColumn(0)
        var result = col.execute(batch)  # resolves batch.columns[0] internally
    """

    var index: Int

    def resolve(self, batch: RecordBatch) -> StringArray:
        return batch.columns[self.index].as_string().copy()

    def execute(self, batch: RecordBatch) raises -> StringArray:
        return self.resolve(batch)

    def write_to[W: Writer](self, mut writer: W):
        writer.write(t"StrCol[{self.index}]")


# ---------------------------------------------------------------------------
# Length — comptime-typed string length
# ---------------------------------------------------------------------------


@fieldwise_init
struct Length[S: StringValue](NumericValue):
    """Fused string length: per-element byte length of a string column.

    Wraps any ``StringValue`` child (e.g. ``StringColumn``) and produces a
    ``UInt32Array`` of byte lengths, matching
    ``marrow.kernels.string.string_lengths``.  Implements ``NumericValue``
    (rather than duplicating a whole-array kernel call) so it composes with
    other numeric nodes through the same single fused vectorize loop —
    ``core[W]`` vectorizes the length computation by loading ``W+1``
    contiguous string offsets and subtracting the shifted-by-one lanes.

    Usage::

        var expr = Length(StringColumn(0))
        var result = expr.execute(batch)  # single fused pass
    """

    comptime OutType = dt.UInt32Type
    comptime NativeType = Self.OutType.native

    var child: Self.S

    @always_inline
    def core[
        W: Int
    ](self, batch: RecordBatch, idx: Int) -> SIMD[Self.NativeType, W]:
        var arr = self.child.resolve(batch)
        var off = arr.offsets.view[Self.NativeType](arr.offset + idx, W + 1)
        var starts = off.load[W](0)
        var ends = off.load[W](1)
        return ends - starts

    def write_to[W: Writer](self, mut writer: W):
        writer.write(t"Length(")
        self.child.write_to(writer)
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

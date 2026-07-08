"""Comptime-typed expression nodes and traits — the foundation of
``marrow.expr``.

Each node is a generic struct where type parameters encode the expression
tree structure, enabling the compiler to inline the full ``core[W]()`` call
chain into a single fused vectorize loop with zero intermediate arrays.
The named leaf column nodes (``NumericColumn[T]`` / ``StringColumn``) live at the
bottom of this module — they resolve by name against the batch schema and are
built by ``Table[Tbl]()`` / ``col(name, dtype)``; the fused algebra
(``Add``/``Greater``/``Length``…) composes over them.

Expression nodes
----------------
``Add[L, R]`` — fused binary add (generic over any two NumericValue children)
``Sub[L, R]`` — fused binary subtract
``Less[L, R]``, ``Greater[L, R]``, ``Equal[L, R]`` — fused binary comparisons; result is
    a bit-packed BoolArray, not a PrimitiveArray (see ``BoolValue`` below)
``Length[S]`` — fused string byte length

Traits
------
``Value`` — base trait for every expression node (dtype, write_to;
    Copyable/Writable/etc.), shared with the type-erased ``DynValue`` in
    ``marrow.expr.dynamic`` — that module imports these traits directly
    from here (``aot`` has no dependency on ``dyn``; the reverse does).
``NumericValue`` — numeric nodes (core[W](batch, idx), execute(batch))
``BoolValue`` — boolean/predicate nodes (core[W] returns SIMD[bool, W];
    execute(batch) bit-packs directly into a BoolArray)

Bridging to the runtime layer
-----------------------------
The ``DynValue(value)`` constructor (see ``marrow.expr.dynamic``) boxes a
comptime node from this module into a runtime ``DynValue``, so it can flow
through APIs that build/execute plans without knowing the concrete comptime
type (e.g. the Python bindings). The boxed ``DynValue`` fully delegates
``dtype()``, ``write_to()``, and ``eval()`` back to the concrete node via
trampolines, so a fused subtree keeps its single-pass execution even when
driven through the type-erased path.

Usage
-----
    var expr = Add(col("a", int64), col("b", int64))
    var result = expr.execute(batch)  # single fused pass, zero intermediates
"""

from std.algorithm.backend.vectorize import vectorize
from std.builtin.simd import Scalar
from std.sys import size_of
from std.sys.info import simd_byte_width
from std.utils.index import IndexList
from std.memory import ArcPointer
from std.reflection import reflect

from .. import dtypes as dt
from ..arrays import AnyArray, BoolArray, PrimitiveArray, StringArray
from ..buffers import Bitmap, Buffer
from ..dtypes import AnyDataType, DType, NumericType
from ..tabular import RecordBatch


# ---------------------------------------------------------------------------
# Value trait — base for all expression nodes (shared with dynamic.mojo)
# ---------------------------------------------------------------------------


trait Value(
    Copyable,
    ImplicitlyCopyable,
    ImplicitlyDeletable,
    Movable,
    Writable,
):
    """Interface every expression node must implement — and the one interface
    ``AnyValue`` erases behind.

    Implemented by the comptime-typed expressions in this module and the
    type-erased ``DynValue`` in ``dynamic.mojo``. Two members make a value
    boxable into ``AnyValue``:

    - ``to_array(batch)`` — evaluate against a batch and erase to ``AnyArray``.
      A bare requirement here; the ``NumericValue`` / ``BoolValue`` /
      ``StringValue`` sub-traits each supply a default that runs their fused
      ``execute()`` and calls ``.to_any()``.
    - ``field_name()`` — the output column name, empty for anonymous computed
      values (the default below). The named column leaves override it. It is a
      method rather than a ``comptime name`` alias because a ``StringLiteral``
      type parameter only converts to ``String`` reliably from inside the
      concrete node's own method body (see ``docs/aot-relations-design.md``).

    ``dtype()`` is declared by the sub-traits (which supply a default), not here;
    ``write_to()`` comes from the inherited ``Writable`` requirement. Declaring
    those here too would make calls on a generic ``Value`` ambiguous between the
    base and sub-trait candidates.
    """

    def to_array(self, batch: RecordBatch) raises -> AnyArray:
        ...

    def field_name(self) -> String:
        return String()


# ---------------------------------------------------------------------------
# NumericValue trait — numeric comptime expression nodes
# ---------------------------------------------------------------------------


trait NumericValue(Value):
    """Trait for numeric comptime-typed expression nodes.

    Nodes implementing this trait are generic structs where type parameters
    encode the full expression tree — enabling the compiler to inline
    ``core[W]()`` across the entire tree into a single fused vectorize loop.

    Provides ``core[W](batch, idx)`` (SIMD lane computation with batch) and
    ``execute(batch)`` (fused vectorize loop).  Both the named ``NumericColumn``
    leaf (below) and ``Add[L, R]`` implement this trait.

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

    def to_array(self, batch: RecordBatch) raises -> AnyArray:
        """Materialise the fused result and erase to ``AnyArray``."""
        return self.execute(batch).to_any()

    def dtype(self) -> AnyDataType:
        """Return the output data type (always known at compile time)."""
        return AnyDataType(Self.OutType())


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

        var col_a = NumericColumn[Int64Type](0)
        var col_b = NumericColumn[Int64Type](1)
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
    def core[W: Int](self, batch: RecordBatch, idx: Int) -> SIMD[DType.bool, W]:
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

    def dtype(self) -> AnyDataType:
        """Return the output data type (always bool)."""
        return AnyDataType(dt.bool_)

    # to_array materialises the predicate's BoolArray; field_name defaults to
    # anonymous (Value's default).

    def to_array(self, batch: RecordBatch) raises -> AnyArray:
        return self.execute(batch).to_any()


# ---------------------------------------------------------------------------
# Less — comptime-typed binary less-than
# ---------------------------------------------------------------------------


@fieldwise_init
struct Less[L: NumericValue, R: NumericValue](BoolValue):
    """Fused binary less-than: evaluates left < right in a single vectorized
    pass, producing a bit-packed BoolArray with zero intermediate arrays.
    """

    comptime NativeType = Self.L.NativeType

    var left: Self.L
    var right: Self.R

    @always_inline
    def core[W: Int](self, batch: RecordBatch, idx: Int) -> SIMD[DType.bool, W]:
        var l = self.left.core[W](batch, idx)
        var r = self.right.core[W](batch, idx).cast[Self.NativeType]()
        return l.lt(r)

    def write_to[W: Writer](self, mut writer: W):
        writer.write(t"Less(")
        self.left.write_to(writer)
        writer.write(t", ")
        self.right.write_to(writer)
        writer.write(t")")


# ---------------------------------------------------------------------------
# Greater — comptime-typed binary greater-than
# ---------------------------------------------------------------------------


@fieldwise_init
struct Greater[L: NumericValue, R: NumericValue](BoolValue):
    """Fused binary greater-than: evaluates left > right in a single
    vectorized pass, producing a bit-packed BoolArray with zero intermediate
    arrays.
    """

    comptime NativeType = Self.L.NativeType

    var left: Self.L
    var right: Self.R

    @always_inline
    def core[W: Int](self, batch: RecordBatch, idx: Int) -> SIMD[DType.bool, W]:
        var l = self.left.core[W](batch, idx)
        var r = self.right.core[W](batch, idx).cast[Self.NativeType]()
        return l.gt(r)

    def write_to[W: Writer](self, mut writer: W):
        writer.write(t"Greater(")
        self.left.write_to(writer)
        writer.write(t", ")
        self.right.write_to(writer)
        writer.write(t")")


# ---------------------------------------------------------------------------
# Equal — comptime-typed binary equality
# ---------------------------------------------------------------------------


@fieldwise_init
struct Equal[L: NumericValue, R: NumericValue](BoolValue):
    """Fused binary equality: evaluates left == right in a single vectorized
    pass, producing a bit-packed BoolArray with zero intermediate arrays.
    """

    comptime NativeType = Self.L.NativeType

    var left: Self.L
    var right: Self.R

    @always_inline
    def core[W: Int](self, batch: RecordBatch, idx: Int) -> SIMD[DType.bool, W]:
        var l = self.left.core[W](batch, idx)
        var r = self.right.core[W](batch, idx).cast[Self.NativeType]()
        return l.eq(r)

    def write_to[W: Writer](self, mut writer: W):
        writer.write(t"Equal(")
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

    def to_array(self, batch: RecordBatch) raises -> AnyArray:
        """Materialise the string result and erase to ``AnyArray``."""
        return self.execute(batch).to_any()

    def dtype(self) -> AnyDataType:
        """Return the output data type (always string)."""
        return AnyDataType(dt.string)


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


# ---------------------------------------------------------------------------
# col — polars-style column factory, resolved by name against the batch
# ---------------------------------------------------------------------------


def col[T: dt.NumericType](var name: String, dtype: T) -> NumericColumn[T]:
    """Reference a numeric column by name — ``col("a", int64)``.

    A schema-struct-free, polars-style leaf: the dtype is given explicitly (the
    AOT layer needs it as a type parameter to stay fused), and the position is
    resolved by name against the batch schema at execution. Overloaded on the
    dtype's trait, so ``col("a", int64)`` returns a numeric column and
    ``col("s", string)`` a string one — both the named leaves defined below
    (``NumericColumn[T]`` / ``StringColumn``), so they compose with
    ``Add``/``Greater`` and drop into ``Project``/``Filter``.

    Usage::

        var plan = Project(Tuple(col("a", int64), col("name", string)))
        var plan = ... .filter(Greater(col("a", int64), col("b", int64)))
    """
    return NumericColumn[T](name^)


def col[T: dt.StringLikeType](var name: String, dtype: T) -> StringColumn:
    """Reference a string column by name — ``col("name", string)``. See the
    numeric overload above."""
    return StringColumn(name^)


# ===========================================================================
# AnyValue — the universal, type-erased value box
# ===========================================================================
#
# Wraps any ``Value`` node behind a thin trampoline exposing only
# ``to_array(batch)`` + ``field_name()`` (no tag, no ``eval()`` switch): the
# named column leaves and fused expressions/predicates *and* the runtime
# ``DynValue`` interpreter. The fused-vs-interpreted choice is which node you box
# — boxing a ``DynValue`` links the interpreter; a program that only boxes fused
# nodes leaves it dead-code-eliminated and stays ~250 KB. The relational layer
# holds ``List[AnyValue]``.


def _value_to_array_tramp[
    V: Value
](ptr: ArcPointer[NoneType], batch: RecordBatch) raises -> AnyArray:
    return rebind[ArcPointer[V]](ptr)[].to_array(batch)


def _value_name_tramp[V: Value](ptr: ArcPointer[NoneType]) -> String:
    return rebind[ArcPointer[V]](ptr)[].field_name()


def _write_tramp[V: Value](ptr: ArcPointer[NoneType]) -> String:
    """One write trampoline for every boxed type (all conform to ``Value``,
    which is ``Writable``)."""
    var s = String()
    rebind[ArcPointer[V]](ptr)[].write_to(s)
    return s^


struct AnyValue(Copyable, Movable, Writable):
    """Type-erased handle over a single value node — the one value box the
    relational layer holds. No tag and no ``eval()`` switch of its own, so
    boxing a fused node never links the interpreter."""

    var _boxed: ArcPointer[NoneType]
    var _to_array: def(
        ArcPointer[NoneType], RecordBatch
    ) thin raises -> AnyArray
    var _field_name: def(ArcPointer[NoneType]) thin -> String
    var _write_to_str: def(ArcPointer[NoneType]) thin -> String

    @implicit
    def __init__[V: Value](out self, value: V):
        """Box any ``Value`` — a named column, a fused expression/predicate, or a
        ``DynValue`` interpreter. The interpreter's code links only when a
        ``DynValue`` is boxed here; otherwise it is dead-code-eliminated and the
        fused-only path stays tiny."""
        var ptr = ArcPointer[V](value.copy())
        self._boxed = rebind[ArcPointer[NoneType]](ptr^)
        self._to_array = _value_to_array_tramp[V]
        self._field_name = _value_name_tramp[V]
        self._write_to_str = _write_tramp[V]

    def to_array(self, batch: RecordBatch) raises -> AnyArray:
        return self._to_array(self._boxed, batch)

    def field_name(self) -> String:
        return self._field_name(self._boxed)

    def write_to[W: Writer](self, mut writer: W):
        writer.write(self._write_to_str(self._boxed))


# ===========================================================================
# Name-resolved column handles — Table[T]() / col() produce these fused leaves
# ===========================================================================


struct NumericColumn[T: dt.NumericType](NumericValue):
    """Named typed numeric column reference — carries only its ``name`` (runtime
    field); the type parameter is just the dtype that drives the SIMD ``core``.
    The position is resolved by name against ``batch.schema`` at execution. Built
    by ``Table[Tbl]()`` and ``col(name, dtype)``, never directly.

    ``to_array`` comes from ``NumericValue``'s default; only ``field_name`` is
    overridden here (columns are named, unlike anonymous computed values)."""

    comptime OutType = Self.T
    comptime NativeType = Self.T.native

    var name: String

    def __init__(out self, var name: String):
        self.name = name^

    @always_inline
    def core[
        W: Int
    ](self, batch: RecordBatch, idx: Int) -> SIMD[Self.NativeType, W]:
        return (
            batch.columns[batch.schema.get_field_index(self.name)]
            .as_primitive[Self.T]()
            .values()
            .load[W](idx)
        )

    def field_name(self) -> String:
        return self.name.copy()

    def write_to[W: Writer](self, mut writer: W):
        writer.write("Col[", self.name, "]")


struct StringColumn(StringValue):
    """Named typed string column reference — the string counterpart of
    ``NumericColumn[T]`` (one type across all string columns; position resolved
    by name). ``to_array`` comes from ``StringValue``'s default."""

    var name: String

    def __init__(out self, var name: String):
        self.name = name^

    def resolve(self, batch: RecordBatch) -> StringArray:
        return (
            batch.columns[batch.schema.get_field_index(self.name)]
            .as_string()
            .copy()
        )

    def execute(self, batch: RecordBatch) raises -> StringArray:
        return self.resolve(batch)

    def field_name(self) -> String:
        return self.name.copy()

    def write_to[W: Writer](self, mut writer: W):
        writer.write("StrCol[", self.name, "]")


struct Table[T: AnyType](Copyable, Movable):
    """Column-access handle over a plain schema struct — ``Table[Orders]()``.

    ``T`` is any struct of plain dtype-tag fields (``var a: Int64Type``).
    ``t.a`` reflects field ``a``'s dtype on ``T`` at compile time
    (``reflect[T].field[name].T``) to pick the column type; the position is
    resolved by name at execution. A companion handle is required because
    ``T``'s own fields shadow ``__getattr_param__``; ``T`` is never instantiated
    (only reflected). The two overloads route numeric/string fields to
    ``NumericColumn``/``StringColumn`` via a ``where`` clause the constraint
    solver can prove (the reflection query folds to a builtin KGEN attribute).
    """

    comptime _dtype[name: StringLiteral] = reflect[Self.T].field[name].T

    def __init__(out self):
        pass

    @always_inline
    def __getattr_param__[
        name: StringLiteral
    ](self) -> NumericColumn[Self._dtype[name]] where conforms_to(
        Self._dtype[name], dt.NumericType
    ):
        return NumericColumn[Self._dtype[name]](String(name))

    @always_inline
    def __getattr_param__[
        name: StringLiteral
    ](self) -> StringColumn where conforms_to(
        Self._dtype[name], dt.StringLikeType
    ):
        return StringColumn(String(name))

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
``FusedBinary[K, L, R]`` — fused binary numeric op, generic over the kernel ``K``
    (which supplies the per-lane ``core`` functor) and its two NumericValue
    children. ``Add`` / ``Sub`` are ``comptime`` aliases binding ``K``.
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
from ..scalars import AnyScalar, PrimitiveScalar
from ..tabular import RecordBatch
from ..kernels.cast import NumericCast, NumToBool, BoolToNum
from ..kernels.arithmetic import (
    BinaryKernel,
    AddKernel,
    SubKernel,
    MulKernel,
    DivKernel,
    MinKernel,
    MaxKernel,
)
from .pruning import PruneStats, PruneBound


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
    - ``name()`` — the output column name, empty for anonymous computed
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

    def name(self) -> String:
        return String()

    def prune(self, stats: PruneStats) raises -> PruneBound:
        """Evaluate this node against per-column statistics for pruning. The
        conservative default returns unknown bounds / maybe-true; nodes that can
        do better (columns, literals, comparisons) override it. Shared by the
        fused static nodes here and the runtime ``DynValue`` interpreter, so a
        predicate of either kind can be evaluated against a row-group or page
        index (see ``marrow.expr.pruning``)."""
        return PruneBound.unknown()


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

    def cast[Target: dt.NumericType](self, dtype: Target) -> Cast[Self, Target]:
        """Wrap this node in a fused numeric cast to ``dtype`` —
        ``col("a", int32).cast(int64)``. The result is a ``NumericValue`` so it
        composes with ``Add``/``Less``/… in the same fused pass."""
        return Cast[Self, Target](self.copy(), dtype)

    def cast(self, dtype: dt.BoolType) -> NumToBoolValue[Self]:
        """Wrap this node in a fused ``x != 0`` cast to bool —
        ``col("a", int32).cast(bool_)``. The result is a ``BoolValue`` so it
        composes with the predicate nodes in the same fused pass. Selected over
        the numeric overload above by the concrete ``BoolType`` argument."""
        return NumToBoolValue[Self](self.copy())


# ---------------------------------------------------------------------------
# FusedBinary — one fused numeric binary node, generic over the kernel
# ---------------------------------------------------------------------------


@fieldwise_init
struct FusedBinary[K: BinaryKernel, L: NumericValue, R: NumericValue](
    NumericValue
):
    """Fused binary numeric op: ``K.core(left, right)`` evaluated in a single
    vectorized pass, zero intermediate arrays.

    Generic over *both* the kernel ``K`` (which supplies the per-lane functor
    ``core``) and the two ``NumericValue`` children, so the compiler inlines the
    whole ``core`` chain. The eager kernel (``K.apply``) and this fused node share
    the one ``K.core`` definition — the functor lives only in the kernel.

    ``OutType`` / ``NativeType`` are inherited from the left child (both operands
    carry the same dtype; the right lane is cast to match).

    Usage::

        var expr = Add(col("a", int64), col("b", int64))  # = FusedBinary[AddKernel]
        var result = expr.execute(batch)                  # single fused pass
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
        return Self.K.core[Self.NativeType, W](l, r)

    def write_to[W: Writer](self, mut writer: W):
        writer.write(Self.K.name, "(")
        self.left.write_to(writer)
        writer.write(t", ")
        self.right.write_to(writer)
        writer.write(t")")


comptime Add = FusedBinary[AddKernel, _, _]
comptime Sub = FusedBinary[SubKernel, _, _]
comptime Mul = FusedBinary[MulKernel, _, _]
comptime Div = FusedBinary[DivKernel, _, _]
comptime Min = FusedBinary[MinKernel, _, _]
comptime Max = FusedBinary[MaxKernel, _, _]


# ---------------------------------------------------------------------------
# Literal — a broadcast numeric constant leaf
# ---------------------------------------------------------------------------


struct Literal[T: dt.NumericType](NumericValue):
    """A numeric constant broadcast across every lane — ``lit(2, int64)``.

    A ``NumericValue`` leaf whose ``core[W]`` splats the stored scalar, so a
    constant composes into a fused pass exactly like a column (``x * 2`` fuses to
    one loop with no materialized ``2`` array)."""

    comptime OutType = Self.T
    comptime NativeType = Self.T.native

    var value: Scalar[Self.NativeType]

    def __init__(out self, value: Scalar[Self.NativeType], dtype: Self.T):
        """``dtype`` only pins the ``T`` parameter, like ``col(name, dtype)``.
        """
        self.value = value

    @always_inline
    def core[
        W: Int
    ](self, batch: RecordBatch, idx: Int) -> SIMD[Self.NativeType, W]:
        return SIMD[Self.NativeType, W](self.value)

    def prune(self, stats: PruneStats) raises -> PruneBound:
        var s = AnyScalar(PrimitiveScalar[Self.T](self.value))
        return PruneBound.interval(s.copy(), s.copy())

    def write_to[W: Writer](self, mut writer: W):
        writer.write("Lit[", self.value, "]")


def lit[T: dt.NumericType](value: Int, dtype: T) -> Literal[T]:
    """Build a broadcast numeric constant — ``lit(2, int64)``.

    ``value`` is a plain ``Int`` (so ``T`` resolves cleanly from ``dtype``) and is
    converted to the column's native scalar type.
    """
    return Literal[T](Scalar[T.native](value), dtype)


# ---------------------------------------------------------------------------
# Cast — comptime-typed numeric cast
# ---------------------------------------------------------------------------


struct Cast[C: NumericValue, To: dt.NumericType](NumericValue):
    """Fused numeric cast: reinterprets a numeric child as ``To`` inside the
    single fused vectorize loop — one ``pop.cast`` per lane, zero intermediate
    arrays.

    ``OutType``/``NativeType`` come from ``To`` (not the child), mirroring
    ``Length``. Uses the raw truncating/wrapping ``SIMD.cast`` semantics (the
    unsafe fast path); the eager ``marrow.kernels.cast`` offers safe checking.
    Composes with ``Add``/``Less``/… so ``Cast(Add(a, b))`` collapses to one
    pass.

    Usage::

        var expr = Cast(Add(col("a", int32), col("b", int32)), int64)
        var result = expr.execute(batch)  # single fused pass
    """

    comptime OutType = Self.To
    comptime NativeType = Self.To.native

    var child: Self.C

    def __init__(out self, var child: Self.C, dtype: Self.To):
        """The ``dtype`` value is only used to infer the ``To`` type parameter
        (like ``col(name, dtype)``)."""
        self.child = child^

    @always_inline
    def core[
        W: Int
    ](self, batch: RecordBatch, idx: Int) -> SIMD[Self.NativeType, W]:
        # Reuse the eager kernel's lane functor so the fused and eager numeric
        # casts share a single conversion definition.
        return NumericCast.core[Self.C.NativeType, Self.NativeType, W](
            self.child.core[W](batch, idx)
        )

    def write_to[W: Writer](self, mut writer: W):
        writer.write(t"Cast(")
        self.child.write_to(writer)
        writer.write(t", ", AnyDataType(Self.OutType()), t")")


# ---------------------------------------------------------------------------
# NumToBoolValue — fused numeric → bool cast
# ---------------------------------------------------------------------------


@fieldwise_init
struct NumToBoolValue[C: NumericValue](BoolValue):
    """Fused numeric → bool cast: ``x != 0`` per lane, bit-packed straight into
    the fused ``BoolArray`` in one vectorized pass — no intermediate array.

    ``NativeType`` is the child's numeric native (it drives the vectorize
    width); the output dtype is always ``bool_``. Reuses ``NumToBool.core`` so
    the fused and eager numeric→bool casts share one conversion definition.

    Usage::

        var pred = Cast(col("a", int32), bool_)  # a != 0
    """

    comptime NativeType = Self.C.NativeType

    var child: Self.C

    @always_inline
    def core[W: Int](self, batch: RecordBatch, idx: Int) -> SIMD[DType.bool, W]:
        return NumToBool.core[Self.C.NativeType, W](
            self.child.core[W](batch, idx)
        )

    def write_to[W: Writer](self, mut writer: W):
        writer.write(t"Cast(")
        self.child.write_to(writer)
        writer.write(t", bool)")


# ---------------------------------------------------------------------------
# BoolToNumValue — fused bool → numeric cast
# ---------------------------------------------------------------------------


struct BoolToNumValue[C: BoolValue, To: dt.NumericType](NumericValue):
    """Fused bool → numeric cast: ``True→1, False→0`` per lane, written straight
    into the fused output buffer in one vectorized pass — no intermediate array.

    ``OutType``/``NativeType`` come from ``To`` (the vectorize width is driven by
    the output native). Reuses ``BoolToNum.core`` so the fused and eager
    bool→numeric casts share one conversion definition.

    Usage::

        var expr = (col("a", int32) < col("b", int32)).cast(int8)  # 0/1
    """

    comptime OutType = Self.To
    comptime NativeType = Self.To.native

    var child: Self.C

    def __init__(out self, var child: Self.C, dtype: Self.To):
        """The ``dtype`` value is only used to infer the ``To`` type parameter
        (like ``col(name, dtype)``)."""
        self.child = child^

    @always_inline
    def core[
        W: Int
    ](self, batch: RecordBatch, idx: Int) -> SIMD[Self.NativeType, W]:
        return BoolToNum.core[Self.NativeType, W](
            self.child.core[W](batch, idx)
        )

    def write_to[W: Writer](self, mut writer: W):
        writer.write(t"Cast(")
        self.child.write_to(writer)
        writer.write(t", ", AnyDataType(Self.OutType()), t")")


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

    # to_array materialises the predicate's BoolArray; name defaults to
    # anonymous (Value's default).

    def to_array(self, batch: RecordBatch) raises -> AnyArray:
        return self.execute(batch).to_any()

    def cast[
        Target: dt.NumericType
    ](self, dtype: Target) -> BoolToNumValue[Self, Target]:
        """Wrap this predicate in a fused ``True→1, False→0`` cast to numeric
        ``dtype`` — ``(col("a", int32) < col("b", int32)).cast(int8)``. The
        result is a ``NumericValue`` so it composes with ``Add``/``Less``/… in
        the same fused pass."""
        return BoolToNumValue[Self, Target](self.copy(), dtype)


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

    def prune(self, stats: PruneStats) raises -> PruneBound:
        return PruneBound.boolean(
            self.left.prune(stats).maybe_lt(self.right.prune(stats))
        )

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

    def prune(self, stats: PruneStats) raises -> PruneBound:
        return PruneBound.boolean(
            self.left.prune(stats).maybe_gt(self.right.prune(stats))
        )

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

    def prune(self, stats: PruneStats) raises -> PruneBound:
        return PruneBound.boolean(
            self.left.prune(stats).maybe_eq(self.right.prune(stats))
        )

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
# ``to_array(batch)`` + ``name()`` (no tag, no ``eval()`` switch): the
# named column leaves and fused expressions/predicates *and* the runtime
# ``DynValue`` interpreter. The fused-vs-interpreted choice is which node you box
# — boxing a ``DynValue`` links the interpreter; a program that only boxes fused
# nodes leaves it dead-code-eliminated and stays ~250 KB. The relational layer
# holds ``List[AnyValue]``.


struct AnyValue(Copyable, Movable, Writable):
    """Type-erased handle over a single value node — the one value box the
    relational layer holds. No tag and no ``eval()`` switch of its own, so
    boxing a fused node never links the interpreter."""

    var _boxed: ArcPointer[NoneType]
    var _to_array: def(
        ArcPointer[NoneType], RecordBatch
    ) thin raises -> AnyArray
    var _name_fn: def(ArcPointer[NoneType]) thin -> String
    var _write_to_str: def(ArcPointer[NoneType]) thin -> String
    var _prune_fn: def(
        ArcPointer[NoneType], PruneStats
    ) thin raises -> PruneBound

    # Per-boxed-type trampolines: one instantiation per concrete ``V`` recovers
    # the erased node via ``rebind`` and forwards to its ``Value`` method.
    @staticmethod
    def _to_array_tramp[
        V: Value
    ](ptr: ArcPointer[NoneType], batch: RecordBatch) raises -> AnyArray:
        return rebind[ArcPointer[V]](ptr)[].to_array(batch)

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
        """Box any ``Value`` — a named column, a fused expression/predicate, or a
        ``DynValue`` interpreter. The interpreter's code links only when a
        ``DynValue`` is boxed here; otherwise it is dead-code-eliminated and the
        fused-only path stays tiny."""
        var ptr = ArcPointer[V](value.copy())
        self._boxed = rebind[ArcPointer[NoneType]](ptr^)
        self._to_array = Self._to_array_tramp[V]
        self._name_fn = Self._name_tramp[V]
        self._write_to_str = Self._write_tramp[V]
        self._prune_fn = Self._prune_tramp[V]

    def to_array(self, batch: RecordBatch) raises -> AnyArray:
        return self._to_array(self._boxed, batch)

    def name(self) -> String:
        return self._name_fn(self._boxed)

    def prune(self, stats: PruneStats) raises -> PruneBound:
        """Evaluate the boxed predicate against per-column statistics, returning
        whether it could be true (see ``marrow.expr.pruning``). Works for both a
        fused static predicate and a ``DynValue`` — the trampoline forwards to
        whichever node is boxed."""
        return self._prune_fn(self._boxed, stats)

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

    ``to_array`` comes from ``NumericValue``'s default; only ``name`` is
    overridden here (columns are named, unlike anonymous computed values)."""

    comptime OutType = Self.T
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

    def name(self) -> String:
        return self._name.copy()

    def prune(self, stats: PruneStats) raises -> PruneBound:
        var iv = stats.by_name(self._name)
        return PruneBound.interval(iv[0].copy(), iv[1].copy())

    def write_to[W: Writer](self, mut writer: W):
        writer.write("Col[", self._name, "]")


struct StringColumn(StringValue):
    """Named typed string column reference — the string counterpart of
    ``NumericColumn[T]`` (one type across all string columns; position resolved
    by name). ``to_array`` comes from ``StringValue``'s default."""

    var _name: String

    def __init__(out self, var name: String):
        self._name = name^

    def resolve(self, batch: RecordBatch) -> StringArray:
        return (
            batch.columns[batch.schema.get_field_index(self._name)]
            .as_string()
            .copy()
        )

    def execute(self, batch: RecordBatch) raises -> StringArray:
        return self.resolve(batch)

    def name(self) -> String:
        return self._name.copy()

    def write_to[W: Writer](self, mut writer: W):
        writer.write("StrCol[", self._name, "]")


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

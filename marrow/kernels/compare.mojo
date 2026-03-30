"""Element-wise comparison kernels.

Each kernel compares two ``PrimitiveArray[T]`` values element-wise and returns
a ``PrimitiveArray[bool_]`` following the Arrow boolean layout.

Null propagation: if either input has a null at position ``i``, the output is
null at ``i`` (validity = ``bitmap_and(left.bitmap, right.bitmap)``).  Data
bits for null positions are set to the comparison result of the underlying
values (undefined per Arrow spec, but branch-free for performance).

Available kernels
-----------------
* ``equal``          — left[i] == right[i]
* ``not_equal``      — left[i] != right[i]
* ``less``           — left[i] <  right[i]
* ``less_equal``     — left[i] <= right[i]
* ``greater``        — left[i] >  right[i]
* ``greater_equal``  — left[i] >= right[i]

Each has a typed overload ``def[T: DataType](PrimitiveArray[T], PrimitiveArray[T])``
and a runtime-typed overload ``def(AnyArray, AnyArray)`` that dispatches via
``binary_array_dispatch``.
"""

from std.algorithm.functional import elementwise
from std.sys import size_of
from std.sys.info import simd_byte_width, simd_width_of
from std.utils.index import IndexList
from std.gpu.host import DeviceContext, get_gpu_target

from ..arrays import (
    BoolArray,
    PrimitiveArray,
    StringArray,
    AnyArray,
    StructArray,
)
from ..buffers import Buffer
from ..views import BitmapView, BufferView
from ..dtypes import DataType, bool_ as bool_dt
from ..buffers import Bitmap
from . import bitmap_and, bool_array_dispatch


# ---------------------------------------------------------------------------
# Elementwise compare + bit-pack — pointers as params for GPU DevicePassable
# ---------------------------------------------------------------------------


def _elementwise_cmp_pack[
    T: DataType,
    func: def[W: Int](SIMD[T.native, W], SIMD[T.native, W]) -> SIMD[
        DType.bool, W
    ],
    out_origin: Origin[mut=True],
](
    output: BufferView[DType.uint8, out_origin],
    lhs: BufferView[T.native, _],
    rhs: BufferView[T.native, _],
    length: Int,
    ctx: Optional[DeviceContext] = None,
) raises:
    """Compare elements and bit-pack via elementwise.

    Views are function parameters (not closure captures) so they transfer
    correctly to GPU via DevicePassable.
    Safe to load beyond length: buffers are 64-byte aligned and padded.
    """

    # TODO: use std.memory.unsafe.pack_bits instead of manual packing!
    @parameter
    @always_inline
    def process[
        W: Int, rank: Int, alignment: Int = 1
    ](idx: IndexList[rank]) -> None:
        var i = idx[0]
        # Always compare 8 elements and pack one output byte.  When W >= 8
        # we produce W // 8 bytes; for the scalar tail (W < 8) we load 8
        # from the byte-aligned base instead (safe: 64-byte padded buffers,
        # and 8 × sizeof(T) ≤ 64 for all primitive types).
        comptime assert 8 * size_of[Scalar[T.native]]() <= 64
        comptime shifts = SIMD[DType.uint8, 8](0, 1, 2, 3, 4, 5, 6, 7)
        var base = (i // 8) * 8
        comptime packs = (W + 7) // 8
        comptime for k in range(packs):
            var off = base + k * 8
            var cmp = func[8](lhs.load[8](off), rhs.load[8](off))
            output.store[1](
                off // 8, (cmp.cast[DType.uint8]() << shifts).reduce_or()
            )

    if ctx:
        comptime if has_accelerator_support[T.native]():
            comptime gpu_width = simd_width_of[
                T.native, target=get_gpu_target()
            ]()
            elementwise[process, gpu_width, target="gpu"](length, ctx.value())
        else:
            raise Error("_elementwise_cmp_pack: type not supported on GPU")
    else:
        comptime cpu_width = simd_byte_width() // size_of[Scalar[T.native]]()
        elementwise[process, cpu_width, target="cpu", use_blocking_impl=True](
            length
        )


# ---------------------------------------------------------------------------
# Generic comparison kernel — single-pass compare + bit-pack
# ---------------------------------------------------------------------------


def _binary_cmp[
    T: DataType,
    func: def[W: Int](SIMD[T.native, W], SIMD[T.native, W]) -> SIMD[
        DType.bool, W
    ],
    name: StringLiteral = "",
](
    left: PrimitiveArray[T],
    right: PrimitiveArray[T],
    ctx: Optional[DeviceContext] = None,
) raises -> BoolArray:
    """Binary comparison kernel — single-pass compare + bit-pack (CPU and GPU).
    """
    if len(left) != len(right):
        raise Error(
            t"{name} arrays must have the same length, got {len(left)} and"
            t" {len(right)}"
        )

    comptime native = T.native
    var length = len(left)
    var bm = bitmap_and(left.bitmap, right.bitmap)

    if ctx:
        var out_buf = Buffer.alloc_device[DType.bool](ctx.value(), length)
        _elementwise_cmp_pack[T, func](
            out_buf.device_view[DType.uint8](),
            left.buffer.device_view[native](left.offset),
            right.buffer.device_view[native](right.offset),
            length,
            ctx,
        )
        var result_buf = out_buf.to_immutable().to_cpu(ctx.value())
        return BoolArray(
            length=length,
            nulls=length - bm.value().view().count_set_bits() if bm else 0,
            offset=0,
            bitmap=bm,
            buffer=Bitmap[mut=False](result_buf, length=length),
        )
    else:
        var out_buf = Buffer.alloc_zeroed[DType.bool](length)
        _elementwise_cmp_pack[T, func](
            out_buf.view[DType.uint8](),
            left.buffer.view[native](left.offset),
            right.buffer.view[native](right.offset),
            length,
        )
        var result_buf = out_buf.to_immutable()
        return BoolArray(
            length=length,
            nulls=length - bm.value().view().count_set_bits() if bm else 0,
            offset=0,
            bitmap=bm,
            buffer=Bitmap[mut=False](result_buf, length=length),
        )


# ---------------------------------------------------------------------------
# SIMD predicates — def[T: DType, W: Int](SIMD[T, W], SIMD[T, W]) -> SIMD[bool, W]
# ---------------------------------------------------------------------------


def _eq[T: DType, W: Int](a: SIMD[T, W], b: SIMD[T, W]) -> SIMD[DType.bool, W]:
    return a.eq(b)


def _ne[T: DType, W: Int](a: SIMD[T, W], b: SIMD[T, W]) -> SIMD[DType.bool, W]:
    return a.ne(b)


def _lt[T: DType, W: Int](a: SIMD[T, W], b: SIMD[T, W]) -> SIMD[DType.bool, W]:
    return a.lt(b)


def _le[T: DType, W: Int](a: SIMD[T, W], b: SIMD[T, W]) -> SIMD[DType.bool, W]:
    return a.le(b)


def _gt[T: DType, W: Int](a: SIMD[T, W], b: SIMD[T, W]) -> SIMD[DType.bool, W]:
    return a.gt(b)


def _ge[T: DType, W: Int](a: SIMD[T, W], b: SIMD[T, W]) -> SIMD[DType.bool, W]:
    return a.ge(b)


# ---------------------------------------------------------------------------
# Typed public API
# ---------------------------------------------------------------------------


def equal[
    T: DataType
](
    left: PrimitiveArray[T],
    right: PrimitiveArray[T],
    ctx: Optional[DeviceContext] = None,
) raises -> BoolArray:
    """Element-wise equality: result[i] = left[i] == right[i]."""
    return _binary_cmp[T, _eq[T.native, _], "equal"](left, right, ctx)


def not_equal[
    T: DataType
](
    left: PrimitiveArray[T],
    right: PrimitiveArray[T],
    ctx: Optional[DeviceContext] = None,
) raises -> BoolArray:
    """Element-wise inequality: result[i] = left[i] != right[i]."""
    return _binary_cmp[T, _ne[T.native, _], "not_equal"](left, right, ctx)


def less[
    T: DataType
](
    left: PrimitiveArray[T],
    right: PrimitiveArray[T],
    ctx: Optional[DeviceContext] = None,
) raises -> BoolArray:
    """Element-wise less-than: result[i] = left[i] < right[i]."""
    return _binary_cmp[T, _lt[T.native, _], "less"](left, right, ctx)


def less_equal[
    T: DataType
](
    left: PrimitiveArray[T],
    right: PrimitiveArray[T],
    ctx: Optional[DeviceContext] = None,
) raises -> BoolArray:
    """Element-wise less-or-equal: result[i] = left[i] <= right[i]."""
    return _binary_cmp[T, _le[T.native, _], "less_equal"](left, right, ctx)


def greater[
    T: DataType
](
    left: PrimitiveArray[T],
    right: PrimitiveArray[T],
    ctx: Optional[DeviceContext] = None,
) raises -> BoolArray:
    """Element-wise greater-than: result[i] = left[i] > right[i]."""
    return _binary_cmp[T, _gt[T.native, _], "greater"](left, right, ctx)


def greater_equal[
    T: DataType
](
    left: PrimitiveArray[T],
    right: PrimitiveArray[T],
    ctx: Optional[DeviceContext] = None,
) raises -> BoolArray:
    """Element-wise greater-or-equal: result[i] = left[i] >= right[i]."""
    return _binary_cmp[T, _ge[T.native, _], "greater_equal"](left, right, ctx)


# ---------------------------------------------------------------------------
# String overloads
# ---------------------------------------------------------------------------


def equal(left: StringArray, right: StringArray) raises -> BoolArray:
    """Element-wise string equality."""
    var n = len(left)
    if len(right) != n:
        raise Error("equal: string arrays must have the same length")
    var bm = bitmap_and(left.bitmap, right.bitmap)
    var bm_builder = Bitmap.alloc_zeroed(n)
    for i in range(n):
        var eq = String(left.unsafe_get(UInt(i))) == String(
            right.unsafe_get(UInt(i))
        )
        if eq:
            bm_builder.set(i)
    return BoolArray(
        length=n,
        nulls=n - bm.value().view().count_set_bits() if bm else 0,
        offset=0,
        bitmap=bm,
        buffer=bm_builder.to_immutable(),
    )


# ---------------------------------------------------------------------------
# Runtime-typed overloads
# ---------------------------------------------------------------------------


def equal(left: AnyArray, right: AnyArray) raises -> AnyArray:
    """Runtime-typed equal."""
    if left.dtype().is_string():
        return equal(left.as_string(), right.as_string()).to_any()
    return bool_array_dispatch["equal", equal[_]](left, right)


def equal(left: StructArray, right: StructArray) raises -> BoolArray:
    """Element-wise struct equality: all corresponding columns must match.

    Returns a boolean array where element ``i`` is True iff
    ``left[i] == right[i]`` across every child column.
    """
    from .boolean import and_

    var n_keys = len(left.children)
    var mask = (
        equal(left.children[0].copy(), right.children[0].copy())
        .as_bool()
        .copy()
    )
    for k in range(1, n_keys):
        mask = and_(
            mask,
            equal(left.children[k].copy(), right.children[k].copy())
            .as_bool()
            .copy(),
        )
    return mask^


def not_equal(left: AnyArray, right: AnyArray) raises -> AnyArray:
    """Runtime-typed not_equal."""
    return bool_array_dispatch["not_equal", not_equal[_]](left, right)


def less(left: AnyArray, right: AnyArray) raises -> AnyArray:
    """Runtime-typed less."""
    return bool_array_dispatch["less", less[_]](left, right)


def less_equal(left: AnyArray, right: AnyArray) raises -> AnyArray:
    """Runtime-typed less_equal."""
    return bool_array_dispatch["less_equal", less_equal[_]](left, right)


def greater(left: AnyArray, right: AnyArray) raises -> AnyArray:
    """Runtime-typed greater."""
    return bool_array_dispatch["greater", greater[_]](left, right)


def greater_equal(left: AnyArray, right: AnyArray) raises -> AnyArray:
    """Runtime-typed greater_equal."""
    return bool_array_dispatch["greater_equal", greater_equal[_]](left, right)

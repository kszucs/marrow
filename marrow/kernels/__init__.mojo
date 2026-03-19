"""Core arity helpers for compute kernels.

These generic loop functions handle buffer allocation, null propagation,
and iteration. Specific kernels (add, sum, etc.) are thin wrappers.

Null propagation model
----------------------
Null handling is separated from data computation:
1. `bitmap_and` computes the output validity bitmap upfront (bitwise AND of inputs).
2. The data kernel (elementwise, simd, or gpu) runs on ALL elements without
   per-element null checks. Null slots in the output have undefined data values
   but are correctly marked invalid by the precomputed bitmap.

This matches Arrow C++ and Arrow Rust's design and enables SIMD vectorization
of the data loop since there are no conditional branches.

Kernel tiers for binary array → array operations
-------------------------------------------------
  - `binary_simd` — SIMD-width-parameterized function. Since `Scalar[T]` is
    `SIMD[T, 1]`, all scalar functions satisfy this signature and the tail
    loop uses `func[1]` as the scalar fallback. This is the only CPU tier.
  - `binary_gpu` — GPU kernel launch via DeviceContext; not all kernels need this.

For reductions (array → scalar):
  - Numeric reductions (sum, product, min, max) use ``std.algorithm``
    reduction primitives with bitmap-aware ``input_fn`` callbacks.
  - Boolean reductions (any_, all_) use bitmap AND + popcount.

NOTE: `reduce_gpu` is intentionally absent. Reductions are low arithmetic
intensity (≤1 FLOP/element for sum), and GPU tree-reduction logic is complex.
CPU SIMD wins here — the same guidance as element-wise add. Only consider a
GPU reduction when data is already device-resident and a round-trip is
prohibitively expensive.

TODO: Add scalar operand support — broadcasting a single scalar value to match
      an array's length. Currently all kernels require two equal-length array
      operands.
TODO: binary_gpu does not propagate null bitmaps; GPU-side bitmap_and is not
      yet implemented. All GPU kernel outputs are marked all-valid.
"""

import std.math as math
from std.algorithm import vectorize
from std.sys import size_of, has_accelerator
from std.sys.info import simd_byte_width
from std.gpu import global_idx
from std.gpu.host import DeviceContext

from marrow.arrays import PrimitiveArray, Array
from marrow.buffers import Buffer, BufferBuilder
from marrow.bitmap import Bitmap, BitmapBuilder
from marrow.builders import PrimitiveBuilder
from marrow.dtypes import DataType, bool_ as bool_dt, numeric_dtypes


# ---------------------------------------------------------------------------
# Null bitmap kernel
# ---------------------------------------------------------------------------


def bitmap_and(
    a: Optional[Bitmap], b: Optional[Bitmap]
) raises -> Optional[Bitmap]:
    """Compute the output validity bitmap as the bitwise AND of two input bitmaps.

    Output bit i is True iff both a[i] and b[i] are True (valid).
    None represents an all-valid bitmap.

    Args:
        a: First input bitmap (None = all valid).
        b: Second input bitmap (None = all valid).

    Returns:
        None if both are all-valid; otherwise the AND of the two bitmaps.
    """
    if not a and not b:
        return None
    if not a:
        return b
    if not b:
        return a
    return a.value() & b.value()


# ---------------------------------------------------------------------------
# Unary arity helper
# ---------------------------------------------------------------------------


def unary[
    InT: DataType,
    OutT: DataType,
    func: def(Scalar[InT.native]) -> Scalar[OutT.native],
](array: PrimitiveArray[InT]) raises -> PrimitiveArray[OutT]:
    """Apply a scalar function element-wise to produce a new array.

    Null propagation: if input element is null, output element is null.

    Parameters:
        InT: Input array DataType.
        OutT: Output array DataType.
        func: The element-wise transformation function.

    Args:
        array: The input array.

    Returns:
        A new PrimitiveArray with the function applied to each valid element.
    """
    var length = len(array)
    var result = PrimitiveBuilder[OutT](length)
    for i in range(length):
        if array.is_valid(i):
            result._buffer.unsafe_set[OutT.native](i, func(array.unsafe_get(i)))
    result._length = length
    return result.finish_typed()


def unary_simd[
    T: DataType,
    func: def[W: Int](SIMD[T.native, W]) -> SIMD[T.native, W],
](array: PrimitiveArray[T]) -> PrimitiveArray[T]:
    """SIMD-vectorized unary kernel.

    Computes the output validity bitmap upfront (copy of input bitmap), then
    applies func element-wise using full SIMD vectors followed by a scalar tail.
    Null slots in the output have undefined data values but are correctly
    marked invalid by the copied bitmap.

    Parameters:
        T: Element DataType (same for input and output).
        func: The element-wise unary function parameterized by SIMD width W.
              Must accept and return SIMD[T.native, W] for any W >= 1.

    Args:
        array: The input array.

    Returns:
        A new PrimitiveArray with func applied element-wise using SIMD.
    """
    comptime native = T.native
    comptime width = simd_byte_width() // size_of[native]()
    var length = len(array)

    var bm = array.bitmap
    var buf = BufferBuilder.alloc[native](length)
    var arr_off = array.offset

    ref arr_buf = array.buffer

    def process[W: Int](i: Int) unified {mut buf, read arr_buf, read arr_off}:
        buf.simd_store[native, W](
            i,
            func[W](arr_buf.simd_load[native, W](arr_off + i)),
        )

    vectorize[width, unroll_factor=4](length, process)

    return PrimitiveArray[T](
        length=length,
        nulls=length - bm.value().count_set_bits() if bm else 0,
        offset=0,
        bitmap=bm,
        buffer=buf.finish(),
    )


# ---------------------------------------------------------------------------
# Binary arity helpers
# ---------------------------------------------------------------------------


def binary_not_null[
    T: DataType,
    func: def[W: Int](SIMD[T.native, W], SIMD[T.native, W]) -> SIMD[
        T.native, W
    ],
    name: StringLiteral = "",
](
    left: PrimitiveArray[T],
    right: PrimitiveArray[T],
) raises -> PrimitiveArray[
    T
]:
    """Binary kernel that only calls func on non-null element pairs (like Arrow's ScalarBinaryNotNull).

    Computes the output validity bitmap upfront via `bitmap_and`, then iterates
    element-by-element, calling func only where both inputs are valid. Null output
    positions are left zero-initialized. This avoids undefined behavior (e.g. SIGFPE
    from integer division by zero) when null slots hold a zero value.

    Parameters:
        T: Element DataType (same for both inputs and output).
        func: The element-wise binary function. Called with W=1 (scalar).
        name: Operation name used in length-mismatch error messages.

    Args:
        left: The left input array.
        right: The right input array.

    Returns:
        A new PrimitiveArray with func applied to each valid element pair.
    """
    if len(left) != len(right):
        raise Error(
            t"{name} arrays must have the same length, got {len(left)} and"
            t" {len(right)}"
        )

    var length = len(left)
    comptime native = T.native

    var bm = bitmap_and(left.bitmap, right.bitmap)
    var buf = BufferBuilder.alloc[native](length)

    for i in range(length):
        if left.is_valid(i) and right.is_valid(i):
            buf.unsafe_set[native](
                i,
                func[1](
                    left.buffer.unsafe_get[native](left.offset + i),
                    right.buffer.unsafe_get[native](right.offset + i),
                ),
            )

    return PrimitiveArray[T](
        length=length,
        nulls=length - bm.value().count_set_bits() if bm else 0,
        offset=0,
        bitmap=bm,
        buffer=buf.finish(),
    )


def binary_simd[
    T: DataType,
    OutT: DataType,
    func: def[W: Int](SIMD[T.native, W], SIMD[T.native, W]) -> SIMD[
        OutT.native, W
    ],
    name: StringLiteral = "",
](
    left: PrimitiveArray[T],
    right: PrimitiveArray[T],
) raises -> PrimitiveArray[
    OutT
]:
    """SIMD-vectorized binary kernel with independent input and output types.

    Computes the output validity bitmap upfront via ``bitmap_and``, then applies
    ``func`` element-wise using full SIMD vectors followed by a scalar tail.
    Null slots in the output have undefined data values but are correctly
    marked invalid by the precomputed bitmap.

    When ``OutT`` is ``bool_`` the output is bit-packed via ``BitmapBuilder``;
    otherwise a regular ``BufferBuilder`` is used.  Both paths share the same
    SIMD load loop — the ``comptime if`` eliminates the dead branch at compile
    time.

    Parameters:
        T: Input DataType.
        OutT: Output DataType.  Pass ``T`` for same-type arithmetic; pass
              ``bool_`` for comparison kernels.
        func: Binary function parameterized by SIMD width W.  Because
              ``Scalar[T] == SIMD[T, 1]``, the same function covers both the
              SIMD main loop and the scalar tail.
        name: Operation name used in length-mismatch error messages.

    Args:
        left: The left input array.
        right: The right input array.

    Returns:
        A new ``PrimitiveArray[OutT]`` with ``func`` applied element-wise.
    """
    if len(left) != len(right):
        raise Error(
            t"{name} arrays must have the same length, got {len(left)} and"
            t" {len(right)}"
        )

    var length = len(left)
    comptime native = T.native
    comptime out_native = OutT.native
    comptime width = simd_byte_width() // size_of[native]()

    var bm = bitmap_and(left.bitmap, right.bitmap)
    ref lb = left.buffer
    ref rb = right.buffer

    var l_off = left.offset
    var r_off = right.offset

    comptime if out_native == DType.bool:
        var data_bm = BitmapBuilder.alloc(length)

        def process_bool[W: Int](i: Int) unified {
            mut data_bm, read lb, read rb, read l_off, read r_off
        }:
            var result = func[W](
                lb.simd_load[native, W](l_off + i),
                rb.simd_load[native, W](r_off + i),
            )
            for j in range(W):
                data_bm.set_bit(i + j, result[j].__bool__())

        vectorize[width](length, process_bool)
        var nulls = 0
        if bm:
            nulls = length - bm.value().count_set_bits()
        return PrimitiveArray[OutT](
            length=length,
            nulls=nulls,
            offset=0,
            bitmap=bm,
            buffer=data_bm.finish(length)._buffer,
        )
    else:
        var buf = BufferBuilder.alloc[out_native](length)

        def process[W: Int](i: Int) unified {
            mut buf, read lb, read rb, read l_off, read r_off
        }:
            buf.simd_store[out_native, W](
                i,
                func[W](
                    lb.simd_load[native, W](l_off + i),
                    rb.simd_load[native, W](r_off + i),
                ),
            )

        vectorize[width, unroll_factor=4](length, process)
        return PrimitiveArray[OutT](
            length=length,
            nulls=length - bm.value().count_set_bits() if bm else 0,
            offset=0,
            bitmap=bm,
            buffer=buf.finish(),
        )


# ---------------------------------------------------------------------------
# GPU binary arity helpers
# ---------------------------------------------------------------------------


def binary_gpu_kernel[
    dtype: DType,
    func: def[W: Int](SIMD[dtype, W], SIMD[dtype, W]) -> SIMD[dtype, W],
](
    lhs: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    rhs: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    result: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    length: Int,
):
    """Generic GPU kernel: applies a binary SIMD function element-wise (W=1 per thread).
    """
    var tid = global_idx.x
    if tid < UInt(length):
        result[tid] = func[1](lhs[tid], rhs[tid])


def binary_gpu[
    T: DataType,
    func: def[W: Int](SIMD[T.native, W], SIMD[T.native, W]) -> SIMD[
        T.native, W
    ],
    name: StringLiteral = "",
](
    left: PrimitiveArray[T],
    right: PrimitiveArray[T],
    ctx: DeviceContext,
) raises -> PrimitiveArray[T]:
    """GPU orchestrator: launches binary_gpu_kernel and returns a device array.

    Parameters:
        T: Element DataType.
        func: SIMD binary function (same signature as binary_simd). Called with W=1
              per GPU thread since `Scalar[T] = SIMD[T, 1]`.
        name: Operation name used in length-mismatch error messages.

    Args:
        left: Left operand (device-resident).
        right: Right operand (device-resident).
        ctx: GPU device context.

    Returns:
        A new device-resident PrimitiveArray with func applied element-wise.
    """
    comptime if not has_accelerator():
        raise Error(name, ": no GPU accelerator available on this system")
    if len(left) != len(right):
        raise Error(
            t"{name}: arrays must have the same length, ",
            t"got {len(left)} and {len(right)}",
        )
    var length = len(left)
    comptime native = T.native
    comptime BLOCK_SIZE = 256

    var lhs_dev = left.buffer.device_buffer().create_sub_buffer[native](
        0, length
    )
    var rhs_dev = right.buffer.device_buffer().create_sub_buffer[native](
        0, length
    )

    var out_dev = ctx.enqueue_create_buffer[native](length)

    var num_blocks = math.ceildiv(length, BLOCK_SIZE)
    comptime kernel = binary_gpu_kernel[native, func]
    ctx.enqueue_function_experimental[kernel](
        lhs_dev,
        rhs_dev,
        out_dev,
        length,
        grid_dim=num_blocks,
        block_dim=BLOCK_SIZE,
    )

    var device_bytes = length * size_of[native]()
    var buf = Buffer.from_device(
        out_dev.create_sub_buffer[DType.uint8](0, device_bytes), device_bytes
    )
    return PrimitiveArray[T](
        length=length,
        nulls=0,
        offset=0,
        bitmap=None,  # nulls=0, no bitmap needed
        buffer=buf^,
    )


# ---------------------------------------------------------------------------
# Runtime-typed dispatch helper
# ---------------------------------------------------------------------------


def binary_array_dispatch[
    name: StringLiteral,
    func: def[T: DataType](
        PrimitiveArray[T], PrimitiveArray[T], Optional[DeviceContext]
    ) raises -> PrimitiveArray[T],
](
    left: Array,
    right: Array,
    ctx: Optional[DeviceContext] = None,
) raises -> Array:
    """Runtime-typed binary dispatch: checks dtype match, loops over numeric types.

    Parameters:
        name: Operation name used in error messages.
        func: The typed binary kernel to dispatch to.

    Args:
        left: Left operand (runtime-typed Array).
        right: Right operand (runtime-typed Array).
        ctx: GPU device context, forwarded to the typed kernel.

    Returns:
        A new Array with the element-wise result.
    """
    if left.dtype != right.dtype:
        raise Error(t"{name}: dtype mismatch: {left.dtype} vs {right.dtype}")

    comptime for dtype in numeric_dtypes:
        if left.dtype == dtype:
            return Array(
                func[dtype](
                    PrimitiveArray[dtype](data=left),
                    PrimitiveArray[dtype](data=right),
                    ctx,
                )
            )
    raise Error(t"{name}: unsupported dtype {left.dtype}")


def binary_array_dispatch[
    name: StringLiteral,
    OutT: DataType,
    func: def[T: DataType](
        PrimitiveArray[T], PrimitiveArray[T]
    ) raises -> PrimitiveArray[OutT],
](left: Array, right: Array,) raises -> Array:
    """Runtime-typed binary dispatch with a fixed output type (e.g. comparisons).

    Parameters:
        name: Operation name used in error messages.
        OutT: Output DataType (e.g. ``bool_`` for comparisons).
        func: The typed binary kernel to dispatch to.

    Args:
        left: Left operand (runtime-typed Array).
        right: Right operand (runtime-typed Array).

    Returns:
        A new Array wrapping ``PrimitiveArray[OutT]`` with the result.
    """
    if left.dtype != right.dtype:
        raise Error(t"{name}: dtype mismatch: {left.dtype} vs {right.dtype}")

    comptime for dtype in numeric_dtypes:
        if left.dtype == dtype:
            return Array(
                func[dtype](
                    PrimitiveArray[dtype](data=left),
                    PrimitiveArray[dtype](data=right),
                )
            )
    raise Error(t"{name}: unsupported dtype {left.dtype}")

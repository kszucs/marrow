"""Batch cosine similarity kernel — CPU SIMD and GPU specializations."""

import std.math as math
from std.sys import size_of, has_accelerator
from std.sys.info import simd_byte_width
from std.gpu import global_idx
from std.gpu.host import DeviceBuffer, DeviceContext

from ..arrays import PrimitiveArray, FixedSizeListArray
from ..buffers import Buffer
from ..builders import PrimitiveBuilder
from ..dtypes import DataType


# ---------------------------------------------------------------------------
# CPU SIMD helper
# ---------------------------------------------------------------------------


# TODO: these kernels should be rewritten to align with the rest of the kernels
def _cosine_similarity_no_nulls[
    T: DataType
](
    vectors: FixedSizeListArray,
    query: PrimitiveArray[T],
    n_vectors: Int,
    dim: Int,
) raises -> PrimitiveArray[T]:
    """SIMD-vectorized batch cosine similarity.

    For each of n_vectors vectors of dimension dim, computes:
        score[i] = dot(v_i, q) / (||v_i|| * ||q||)
    """
    comptime native = T.native
    comptime width = simd_byte_width() // size_of[native]()

    var result = PrimitiveBuilder[T](n_vectors)

    # Flat values pointer from the child array
    var child_data = vectors.values.to_data()
    var vp = child_data.buffers[0].view[native](child_data.offset).unsafe_ptr()
    var qp = query.buffer.view[native](query.offset).unsafe_ptr()

    # Pre-compute query norm
    var norm_q = Scalar[native](0)
    var j = 0
    while j + width <= dim:
        var q = (qp + j).load[width=width]()
        norm_q += (q * q).reduce_add()
        j += width
    while j < dim:
        norm_q += qp[j] * qp[j]
        j += 1
    var query_norm = math.sqrt(norm_q)

    # Compute similarity for each vector
    for i in range(n_vectors):
        var offset = (vectors.offset + i) * dim
        var dot = Scalar[native](0)
        var norm_v = Scalar[native](0)

        j = 0
        while j + width <= dim:
            var v = (vp + offset + j).load[width=width]()
            var q = (qp + j).load[width=width]()
            dot += (v * q).reduce_add()
            norm_v += (v * v).reduce_add()
            j += width
        while j < dim:
            dot += vp[offset + j] * qp[j]
            norm_v += vp[offset + j] * vp[offset + j]
            j += 1

        var denom = math.sqrt(norm_v) * query_norm
        if denom > 0:
            result.unsafe_set(i, dot / denom)
        else:
            result.unsafe_set(i, Scalar[native](0))

    result.set_length(n_vectors)
    return result.finish()


# ---------------------------------------------------------------------------
# GPU kernel
# ---------------------------------------------------------------------------


def _cosine_similarity_gpu_kernel[
    dtype: DType
](
    vectors: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    query: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    result: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    n_vectors: Int,
    dim: Int,
):
    """GPU kernel: each thread computes one vector's cosine similarity."""
    var tid = Int(global_idx.x)
    if tid < n_vectors:
        var offset = tid * dim
        var dot = Scalar[dtype](0)
        var norm_v = Scalar[dtype](0)
        var norm_q = Scalar[dtype](0)
        for j in range(dim):
            var v = vectors[offset + j]
            var q = query[j]
            dot += v * q
            norm_v += v * v
            norm_q += q * q
        var denom = math.sqrt(norm_v) * math.sqrt(norm_q)
        if denom > 0:
            result[tid] = dot / denom
        else:
            result[tid] = Scalar[dtype](0)


def _cosine_similarity_gpu[
    T: DataType
](
    vectors: FixedSizeListArray,
    query: PrimitiveArray[T],
    n_vectors: Int,
    dim: Int,
    ctx: DeviceContext,
) raises -> PrimitiveArray[T]:
    """GPU-accelerated batch cosine similarity on device-resident data."""
    comptime native = T.native
    comptime BLOCK_SIZE = 256

    var n_values = n_vectors * dim

    var child_data = vectors.values.to_data()
    var vec_dev = (
        child_data.buffers[0]
        .devicebuffer()
        .create_sub_buffer[native](0, n_values)
    )
    var query_dev = query.buffer.devicebuffer().create_sub_buffer[native](
        0, dim
    )

    var out_dev = ctx.enqueue_create_buffer[native](n_vectors)

    var num_blocks = math.ceildiv(n_vectors, BLOCK_SIZE)
    comptime kernel = _cosine_similarity_gpu_kernel[native]
    ctx.enqueue_function_experimental[kernel](
        vec_dev,
        query_dev,
        out_dev,
        n_vectors,
        dim,
        grid_dim=num_blocks,
        block_dim=BLOCK_SIZE,
    )

    var device_bytes = n_vectors * size_of[native]()
    var buf = Buffer.from_device(
        out_dev.create_sub_buffer[DType.uint8](0, device_bytes), device_bytes
    )
    return PrimitiveArray[T](
        length=n_vectors,
        nulls=0,
        offset=0,
        bitmap=None,
        buffer=buf^,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def cosine_similarity[
    T: DataType
](
    vectors: FixedSizeListArray,
    query: PrimitiveArray[T],
    ctx: Optional[DeviceContext] = None,
) raises -> PrimitiveArray[T]:
    """Batch cosine similarity: N vectors vs one query → N scores.

    Args:
        vectors: FixedSizeListArray of N vectors, each of dimension D.
        query: PrimitiveArray[T] of D elements (the query vector).
        ctx: GPU device context. If provided, runs on GPU; otherwise uses CPU SIMD.

    Returns:
        PrimitiveArray[T] of N cosine similarity scores in [-1, 1].
        When ctx is provided, result is device-resident; call `.to_cpu(ctx)` to read on CPU.
    """
    var dim = vectors.dtype.size
    var n_vectors = len(vectors)

    if len(query) != dim:
        raise Error(
            t"cosine_similarity: query length {len(query)} != vector dim {dim}"
        )

    if ctx:
        comptime if has_accelerator():
            return _cosine_similarity_gpu[T](
                vectors, query, n_vectors, dim, ctx.value()
            )
        else:
            raise Error(
                "cosine_similarity: no GPU accelerator available on this system"
            )
    return _cosine_similarity_no_nulls[T](vectors, query, n_vectors, dim)

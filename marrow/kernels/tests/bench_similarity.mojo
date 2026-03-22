"""CPU vs GPU cosine similarity benchmarks.

Run with: pixi run bench_similarity
"""

from std.sys import has_accelerator
from std.time import perf_counter_ns
from std.benchmark import keep
from std.gpu.host import DeviceContext

from marrow.arrays import AnyArray, PrimitiveArray, FixedSizeListArray
from marrow.builders import AnyBuilder, PrimitiveBuilder, FixedSizeListBuilder
from marrow.dtypes import float32
from marrow.kernels.similarity import cosine_similarity


def _make_vectors(n_vectors: Int, dim: Int) raises -> FixedSizeListArray:
    """Build N random-ish vectors of given dimension."""
    var total = n_vectors * dim
    var b = PrimitiveBuilder[float32](total)
    for i in range(total):
        # Simple deterministic "pseudo-random" values
        b.unsafe_append(
            Scalar[float32.native](Float64((i * 7 + 13) % 1000) / 1000.0 - 0.5)
        )
    var builder = FixedSizeListBuilder(AnyBuilder(b^), list_size=dim)
    for _ in range(n_vectors):
        builder.append_valid()
    return builder.finish_typed()


def _make_query(dim: Int) raises -> PrimitiveArray[float32]:
    """Build a query vector of given dimension."""
    var b = PrimitiveBuilder[float32](dim)
    for i in range(dim):
        b.unsafe_append(
            Scalar[float32.native](Float64((i * 11 + 17) % 1000) / 1000.0 - 0.5)
        )
    return b.finish_typed()


def _bench_cpu(
    vectors: FixedSizeListArray,
    query: PrimitiveArray[float32],
    iters: Int,
) raises -> Float64:
    """CPU SIMD cosine similarity, mean microseconds per iteration."""
    # Warmup
    for _ in range(3):
        var r = cosine_similarity[float32](vectors, query)
        keep(len(r))

    var start = perf_counter_ns()
    for _ in range(iters):
        var r = cosine_similarity[float32](vectors, query)
        keep(len(r))
    return Float64(perf_counter_ns() - start) / Float64(iters) / 1000.0


def _bench_gpu(
    vectors: FixedSizeListArray,
    query: PrimitiveArray[float32],
    iters: Int,
    ctx: DeviceContext,
) raises -> Float64:
    """GPU cosine similarity (includes upload), mean us/iter."""
    # Warmup
    for _ in range(3):
        var r = cosine_similarity[float32](
            vectors.to_device(ctx), query.to_device(ctx), ctx
        )
        keep(len(r))
    ctx.synchronize()

    var start = perf_counter_ns()
    for _ in range(iters):
        var r = cosine_similarity[float32](
            vectors.to_device(ctx), query.to_device(ctx), ctx
        )
        keep(len(r))
    ctx.synchronize()
    return Float64(perf_counter_ns() - start) / Float64(iters) / 1000.0


def _bench_gpu_preloaded(
    vectors: FixedSizeListArray,
    query: PrimitiveArray[float32],
    iters: Int,
    ctx: DeviceContext,
) raises -> Float64:
    """GPU cosine similarity with pre-uploaded data, mean us/iter."""
    var dev_vectors = vectors.to_device(ctx)
    var dev_query = query.to_device(ctx)
    ctx.synchronize()

    # Warmup
    for _ in range(3):
        var r = cosine_similarity[float32](dev_vectors, dev_query, ctx)
        keep(len(r))
    ctx.synchronize()

    var start = perf_counter_ns()
    for _ in range(iters):
        var r = cosine_similarity[float32](dev_vectors, dev_query, ctx)
        keep(len(r))
    ctx.synchronize()
    return Float64(perf_counter_ns() - start) / Float64(iters) / 1000.0


def _print_row(kernel: String, n: Int, dim: Int, us: Float64):
    var n_str = String(n)
    var dim_str = String(dim)
    # TODO: use format specifiers once supported
    var pad1 = " " * (12 - kernel.byte_length())
    var pad2 = " " * (10 - n_str.byte_length())
    var pad3 = " " * (6 - dim_str.byte_length())
    print(kernel, pad1, n_str, pad2, dim_str, pad3, us, " us", sep="")


def main() raises:
    if not has_accelerator():
        print("No GPU accelerator found, running CPU-only benchmarks.")

    print("=== Cosine Similarity Benchmarks ===\n")
    print("kernel      N         dim   mean (us/iter)")
    print("------      -         ---   --------------")

    comptime n_values = (1_000, 10_000, 100_000, 500_000, 1_000_000)
    comptime dims = (128, 384, 768)
    comptime iters_per_n = (100, 50, 10, 5, 3)

    # CPU benchmarks
    comptime for ni in range(5):
        comptime for di in range(3):
            var vectors = _make_vectors(n_values[ni], dims[di])
            var query = _make_query(dims[di])
            _print_row(
                "cpu",
                n_values[ni],
                dims[di],
                _bench_cpu(vectors, query, iters_per_n[ni]),
            )

    # GPU benchmarks
    if has_accelerator():
        var ctx = DeviceContext()
        comptime for ni in range(5):
            comptime for di in range(3):
                var vectors = _make_vectors(n_values[ni], dims[di])
                var query = _make_query(dims[di])
                _print_row(
                    "gpu",
                    n_values[ni],
                    dims[di],
                    _bench_gpu(vectors, query, iters_per_n[ni], ctx),
                )

        # GPU with pre-loaded data (no upload overhead)
        comptime for ni in range(5):
            comptime for di in range(3):
                var vectors = _make_vectors(n_values[ni], dims[di])
                var query = _make_query(dims[di])
                _print_row(
                    "gpu-preload",
                    n_values[ni],
                    dims[di],
                    _bench_gpu_preloaded(vectors, query, iters_per_n[ni], ctx),
                )

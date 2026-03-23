# Modular Kernels for Analytical / Columnar Workloads

Survey of kernel implementations in `max/kernels/src/` relevant to Apache Arrow-style analytical compute (columnar data). Checked as of March 2026.

## Support Matrix

| Kernel | Source file(s) | CPU | GPU | Vectorized (SIMD) |
|---|---|---|---|---|
| **Sort / Argsort** | `nn/argsort.mojo` | ✓ | ✓ (NVIDIA+AMD, warp) | ✓ `elementwise` + SIMD |
| **ArgMax / ArgMin** | `nn/argmaxmin.mojo`, `nn/argmaxmin_gpu.mojo` | ✓ (parallel, SIMD) | ✓ (separate GPU file) | ✓ SIMD |
| **Top-K** | `nn/topk.mojo` | ✓ (quicksort based) | ✓ (warp-level sampling) | ✓ |
| **Gather / Scatter** | `nn/gather_scatter.mojo` | ✓ | ✓ (DeviceContext dispatch) | ✓ SIMD index normalize |
| **Gather ND / Scatter ND** | `nn/gather_scatter.mojo` | ✓ | ✓ | ✓ |
| **Slice** | `nn/slice.mojo` | ✓ | ✓ (DeviceContext) | ✓ (parallel memcpy) |
| **Concat** | `nn/concat.mojo` | ✓ | ✓ (`elementwise` target="gpu") | ✓ |
| **Pad** | `nn/pad.mojo`, `nn/pad_gpu.mojo` | ✓ | ✓ (dedicated GPU kernel) | ✓ |
| **Repeat / Interleave** | `nn/repeat_interleave.mojo` | ✓ | ✗ | ✓ `elementwise` SIMD |
| **Broadcast** | `nn/broadcast.mojo` | ✓ | ✗ | ✗ (tiled memcpy) |
| **Prefix Sum (CumSum)** | `nn/cumsum.mojo` | ✓ | ✗ | ✗ (scalar, multi-axis) |
| **Range / Arange** | `nn/arange.mojo` | ✓ | ✗ | ✓ SIMD `iota` |
| **Index Tensor** | `nn/index_tensor.mojo` | ✓ | ✓ (DeviceContext) | ✓ |
| **Non-Zero Filter** | `nn/arg_nonzero.mojo` | ✓ | ✗ | ✗ (scalar scan) |
| **Softmax / Log-Softmax** | `nn/softmax.mojo` | ✓ | ✓ (fused online, warp) | ✓ SIMD |
| **Normalization** (L1/L2/RMS/LayerNorm) | `nn/normalization.mojo` | ✓ (parallel, vectorize) | ✓ (warp + block reduce) | ✓ SIMD |
| **Pooling** (Max/Avg, sliding window) | `nn/pool.mojo` | ✓ | ✓ (`stencil_gpu`) | ✓ SIMD |
| **Non-Max Suppression** | `nn/nms.mojo` | ✓ | ✗ | ✓ SIMD[2] pairs |
| **Transpose** | `linalg/transpose.mojo` | ✓ | ✗ | ✓ SIMD 4×4 tiles |
| **MatMul / GEMM** | `linalg/matmul/` | ✓ (NEON/AVX/VNNI) | ✓ (SM80/90/100, AMD RDNA, tensor cores) | ✓ |
| **Batched MatMul (BMM)** | `linalg/bmm.mojo` | ✓ | ✓ (A100+, AMD, TMA) | ✓ |
| **GEMV** | `linalg/gemv.mojo` | ✓ | ✓ (NVIDIA+AMD warp ops) | ✓ SIMD |
| **FP8 Quantization / Cast** | `linalg/fp8_quantization.mojo` | ✓ | ✓ (H100/B200, SM10x) | ✓ |
| **FP4 Quantization** | `linalg/fp4_quantization.mojo` | ✓ | ✓ | ✓ |
| **QR Factorization** | `linalg/qr_factorization.mojo` | ✓ | ✗ | ✗ (Householder, scalar) |

## Notes

- All source files are under `max/kernels/src/` in the `modular` repo.
- **GPU targets**: NVIDIA SM80 (Ampere/A100), SM90 (Hopper/H100), SM100 (Blackwell/B200), and AMD RDNA. Dispatch via `DeviceContext` and `elementwise[..., target="gpu"]`.
- **CPU vectorization**: uses Mojo's `vectorize`/`elementwise` with `simd_width_of[dtype]()` — auto-selects AVX-512/AVX2 on x86, NEON/SVE on ARM.
- CPU parallelism via `sync_parallelize` / `parallelize_over_rows`.

## Gaps vs. Apache Arrow Compute

The following common Arrow compute kernels are **absent** (this is a DL inference codebase):

- Hash-based joins / group-by aggregation
- Dictionary encoding / decoding
- String / binary operations (split, match, cast to/from string)
- Date/time arithmetic
- General element-wise comparison and boolean filter (only partial via `arg_nonzero`)
- Compressed bitmap / run-length encoding operations

## Most Relevant for Columnar Workloads

In priority order for an Arrow compute replacement:

1. `argsort` — sort indices over a column (CPU+GPU, SIMD)
2. `argmax` / `argmin` — reduction over a column (CPU+GPU, SIMD)
3. `gather` / `scatter` — indexed read/write into columns (CPU+GPU, SIMD)
4. `slice` — column range selection (CPU+GPU, parallel)
5. `concat` — column concatenation (CPU+GPU, SIMD)
6. `cumsum` — prefix sum / running total (CPU only, scalar)
7. `arg_nonzero` — filter / boolean selection (CPU only, scalar — needs vectorization)
8. `normalization` — aggregate statistics (CPU+GPU, SIMD)
9. `pooling` — windowed aggregation (CPU+GPU, SIMD)
10. `transpose` — row↔column layout switch (CPU only, 4×4 SIMD tiles)

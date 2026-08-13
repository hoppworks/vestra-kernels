# Public API contract

The public API is intentionally narrow: primitive F32 or Q8_0 buffers plus
explicit dimensions. The crate provides generic GEMM, attention, LayerNorm,
RoPE, softmax/exp, Conv2D, transposed convolution, prepared Winograd paths,
bilinear resize, and ISA dispatch. The DA3-BASE specialization is shape-gated
and returns `false` when a caller must use the generic fallback.

`Kernels::detect` chooses AVX-512 only when the required features are present,
then AVX2 where supported, otherwise the scalar reference route. Callers must
not silently assume that an optimized path ran. Each oracle documents whether
it requires bit equality or a bounded F32 tolerance; changing an accumulation
order requires updating that evidence before enabling the path.

The API never accepts an engine type or a GGUF reader type. Kernel-owned Q8_0
blocks are a storage value only; model-byte parsing remains the engine's job.

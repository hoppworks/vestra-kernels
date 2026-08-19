# Benchmarking Vestra Kernels

Kernel microbenchmarks are diagnosis tools, not product performance claims.
They use the real DA3-BASE shapes where available: 865 tokens, 768 embedding
channels, 12 heads, and 64-dimensional head vectors for the 504×336 workload.

```bash
cargo test --lib
cargo test --tests
cargo bench --bench gemm_bench
cargo run --release --example current_shape_bench
# Requires an external OpenMP BLIS installation and matching linker flags:
cargo run --release --features blis-experiment --example blis_shape_bench
```

Run target-specific experiments on the AMD Ryzen 9 9950X with
`-C target-cpu=znver5`, a fixed 16-thread budget, no competing load, and a
recorded compiler and binary hash. The kernel result is admissible only when
the engine subsequently confirms both numerical parity and end-to-end timing
under the unchanged CPU-F32 protocol.

The current split baseline is documented by Vestra Engine. No microbenchmark
may be presented as a replacement for that study.

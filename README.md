# Vestra Kernels

Vestra Kernels is the independently benchmarked low-level compute layer for
Vestra Engine. It owns fixed-shape CPU and CUDA kernels; model semantics,
scheduling, preprocessing, and reconstruction remain outside this repository.

## Current code paths

- Shape-gated AVX-512 projection and online-attention kernels for the
  DA3-BASE 504×336 workload, with explicit generic fallbacks
- Prepared F(2, 3×3) and experimental F(4, 3×3) Winograd filters used by
  the depth head
- Fused high-resolution resize/output-convolution and non-overlapping
  transposed-convolution paths
- Scalar, Faer, AVX2, AVX-512, and Q8_0 numerical oracles and dispatch tests
- An optional `cuda` module with device-resident tensors, cuBLAS GEMM, online
  attention, normalization, GELU, patch/token layouts, and DA3 Q/K RoPE

The CUDA module is an implemented experimental backend, not a claim that the
complete Vestra Engine inference graph is GPU-qualified. Engine adoption still
requires the same parity and end-to-end gates as every CPU kernel. The BLIS
bridge and its standalone benchmark are likewise explicit experiments rather
than default runtime dependencies.

The public surface accepts primitive buffers, explicit dimensions, and
kernel-owned prepared values. It deliberately does not import model, GGUF,
CLI, or reconstruction types.

## Qualification gate

A kernel may enter the production path only when it:

1. has an isolated numerical oracle;
2. preserves the declared model and precision workload;
3. wins an alternating same-binary benchmark on the target hardware;
4. passes the four-image C++ F32 parity corpus; and
5. wins the randomized end-to-end study.

The authoritative CPU benchmark artifacts live in Vestra Engine. The current
qualified same-machine study used an AMD Ryzen 9 9950X with 16 benchmark
threads at 504×336.

## Validation

```bash
cargo fmt --all -- --check
cargo clippy --all-targets -- -D warnings
cargo clippy --lib --tests --features cuda -- -D warnings
cargo test --all-targets
cargo test --lib --tests --features cuda
```

The repository pins Rust 1.93.0 with rustfmt and Clippy, and CI runs the same
commands on Linux. AVX-512 execution tests require compatible x86-64 hardware;
unsupported hosts validate the fallback contract instead. CUDA compilation
does not require a locally installed toolkit because the driver is loaded
dynamically, while execution still requires a compatible NVIDIA device.

Local unit tests alone are not sufficient evidence for a performance claim.
The external BLIS feasibility benchmark additionally requires an OpenMP BLIS
installation and is enabled explicitly with `--features blis-experiment`.

See [API.md](API.md) for the stable boundary and
[BENCHMARKING.md](BENCHMARKING.md) for the qualification protocol.

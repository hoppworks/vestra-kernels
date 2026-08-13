# Vestra Kernels

Vestra Kernels is the independently benchmarked low-level compute layer for
Vestra Engine. It owns fixed-shape CPU and CUDA kernels; model semantics,
scheduling, preprocessing, and reconstruction remain outside this repository.

## Current production paths

- AVX-512 projection and attention kernels for the DA3-BASE 504×336 workload
- Optional BLIS bridges for qualified linear shapes
- Prepared F(2, 3×3) Winograd products used by the depth head
- Fused high-resolution resize and output convolution support

CUDA is part of the Vestra architecture but is not implemented in this initial
repository snapshot.

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
cargo test --lib
```

AVX-512 tests require compatible x86-64 hardware. Local unit tests alone are
not sufficient evidence for a performance claim.

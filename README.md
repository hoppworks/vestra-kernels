# da3-kernels

`da3-kernels` is the narrow CPU-kernel companion to the Rust DA3-BASE engine.
It owns only fixed-shape kernels whose numerical behaviour and whole-model
benefit were measured on the target workload. The engine owns DA3 semantics,
model loading, scheduling and parity orchestration; this repository owns the
specialized CPU inner loops.

## Contract

A kernel is eligible for the production path only when all of these hold:

1. it executes the same DA3-BASE F32 work at 504×336;
2. it has an isolated correctness oracle;
3. it wins a same-binary, alternating benchmark on the Ryzen 9 workhorse;
4. the engine passes the four-image C++-F32 fidelity gate; and
5. the randomized ten-trial end-to-end study confirms the gain.

No quantized result, changed resize policy, altered thread budget or best
single sample is a direct F32 performance claim.

## Current accepted paths

- Fixed DA3 projection shapes use native AVX-512 kernels and, in the explicit
  workhorse build, selected BLIS SGEMM bridges.
- Prepared F(2,3×3) Winograd kernels serve the DPT head. The engine fuses the
  final resize with the output convolution.
- Flash attention uses an AVX-512 eight-query × two-32-output-panel kernel.
  Complete eight-query tiles use a dedicated persistent-K variant; it is
  bit-identical to the generic QT8 path while avoiding diagnostic fallback
  scratch.

The qualified DA3 CPU-F32 study with these paths measured Rust at 165.751 ms
and C++/ggml at 238.647 ms on the Ryzen 9 9950X, 16 threads. That is 1.44×
throughput, or 44.0% faster Rust execution. Full protocol, raw trials, parity
results and the decision ledger live with the engine repository in its
CPU-F32 status guide.

## Development rules

- Keep every candidate opt-in until the full gate passes.
- Preserve F32 operation order unless the end-to-end parity gate explicitly
  validates the change.
- Keep an old path available only when it is useful as a controlled A/B arm;
  remove rejected experiments rather than accumulating runtime switches.
- Do not claim a generic Rust or AVX-512 advantage from these locked-shape
  measurements.

## Validation

```bash
cargo test --lib
```

The AVX-512-specific tests run on supported x86-64 hardware. A qualifying
change additionally requires the target-hardware oracle and the engine-level
benchmark protocol; local unit tests alone are not enough.

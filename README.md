# da3-kernels

Small, independently versioned CPU kernels for Depth Anything 3.

The repository deliberately owns only kernels with a measurable DA3 benefit.
Every candidate must be compared against the existing Rust fallback using the
same F32 model, input, resolution and thread budget. Integration into
`depth-anything-rs` happens only after the four-image F32 parity contract and
the locked Workhorse benchmark pass.

Initial target: AVX-512 F32 matrix multiplication for the four repeated
DA3-BASE transformer projection shapes at 504×336. No kernel is enabled yet.

## Experiment log

| Candidate | Workhorse A/B result | Decision |
|---|---:|---|
| 4×96 output-channel projection microkernel | 337.251 ms vs Faer 326.949 ms | Rejected; it reread weight tiles per token group. |
| 4-query × 64-key F32 flash-attention prototype | 378.414 ms end-to-end smoke | Rejected from the main runtime; it still spends too much time in scalar exponentiation and temporary accumulation. |

The flash candidate remains here for measurement work only. It is not a
dependency of the production runtime. A replacement must first beat the
existing attention implementation in an isolated A/B before it is imported.

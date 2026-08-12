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
| 64-query × 64-key packed revision | 376.761 ms end-to-end smoke | Rejected; higher tile reuse did not beat the established per-query path. |
| GGML-style 4×64 small GEMMs + packed Flash tiles | 299.535 ms vs 325.157 ms fallback (same binary, 1 warm-up + median of 5) | Accepted into `depth-anything-rs` after four-image F32 parity. |

The accepted Flash kernel is an explicit local dependency of the production
runtime. Future variants remain measurement-only until they beat it in an
isolated A/B and pass the full F32 parity gate.

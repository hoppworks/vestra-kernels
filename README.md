# da3-kernels

Small, independently versioned CPU kernels for Depth Anything 3.

The repository deliberately owns only kernels with a measurable DA3 benefit.
Every candidate must be compared against the existing Rust fallback using the
same F32 model, input, resolution and thread budget. Integration into
`depth-anything-rs` happens only after the four-image F32 parity contract and
the locked Workhorse benchmark pass.

Initial target: AVX-512 F32 matrix multiplication for the four repeated
DA3-BASE transformer projection shapes at 504×336. No kernel is enabled yet.

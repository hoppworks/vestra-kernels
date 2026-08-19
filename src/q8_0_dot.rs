//! q8_0 quantization + block-dot GEMM. This is the "most expensive hand-built
//! component" of the kernel set (Task 9): a Rust port of ggml's
//! `ggml_vec_dot_q8_0_q8_0` (AVX-512/VNNI `_mm512_dpbusd_epi32`-based).
//! Activations are quantized per 32-element block to int8
//! (`quantize_row_q8_0`), then the block-dot against q8_0-quantized weights
//! is accumulated in int32 and scaled by `d_a * d_b` per block.
//!
//! `scalar::gemm_q8_0` (in `crate::scalar`) is the oracle used directly on
//! non-x86_64 hosts via `Kernels::gemm_q8_0`. The AVX-512/VNNI path
//! (`simd_avx512::gemm_q8_0_avx512`) must match that reference under the
//! qualification protocol in `BENCHMARKING.md`.

use crate::{BlockQ8_0, QK8_0};

/// Quantize one row of `QK8_0`-aligned f32 activations into q8_0 blocks.
///
/// Per 32-element block: `amax = max|x|`, `d = amax/127` (stored as f16),
/// `qs[j] = round(x[j]/d)` clamped to `i8` range. Matches ggml's
/// `quantize_row_q8_0` reference quantization (symmetric, no zero-point).
///
/// # Panics
/// Panics (via `assert_eq!`) if `x.len() != out.len() * QK8_0`.
pub fn quantize_row_q8_0(x: &[f32], out: &mut [BlockQ8_0]) {
    assert_eq!(
        x.len(),
        out.len() * QK8_0,
        "x.len() must be out.len()*QK8_0"
    );

    for (bi, block) in out.iter_mut().enumerate() {
        let src = &x[bi * QK8_0..(bi + 1) * QK8_0];
        let amax = src.iter().fold(0f32, |acc, &v| acc.max(v.abs()));
        let d = amax / 127.0;
        let inv_d = if d != 0.0 { 1.0 / d } else { 0.0 };

        block.d = half::f16::from_f32(d);
        for (j, &v) in src.iter().enumerate() {
            let q = (v * inv_d).round();
            block.qs[j] = q.clamp(-127.0, 127.0) as i8;
        }
    }
}

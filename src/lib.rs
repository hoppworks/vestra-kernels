//! Vestra's CPU-kernel boundary.
//!
//! This crate accepts only primitive slices, explicit shapes, and small
//! kernel-owned value types. It deliberately contains no GGUF reader, model
//! configuration, CLI, or engine types.

mod specialized;

pub mod attention;
pub mod conv;
pub mod gemm;
pub mod q8_0_dot;
pub mod resample;
pub mod rope;
pub mod scalar;

mod dispatch;
#[cfg(target_arch = "x86_64")]
mod simd_avx512;

pub use attention::{attention, attention_naive};
pub use conv::{conv_transpose2d, conv2d, conv2d_naive};
pub use dispatch::{Isa, Kernels};
pub use q8_0_dot::quantize_row_q8_0;
pub use resample::{bilinear_resize, bilinear_resize_align_corners, bilinear_resize_naive};
pub use rope::rope2d;

/// Logical block width of GGML's Q8_0 representation.
pub const QK8_0: usize = 32;

/// Kernel-owned Q8_0 storage. The GGUF reader converts raw model bytes into
/// its own model representation; kernels only need this layout-neutral value.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BlockQ8_0 {
    pub d: half::f16,
    pub qs: [i8; QK8_0],
}

/// Expands Q8_0 blocks into F32 values for kernel tests and callers that need
/// an explicit fallback. This is not a GGUF-reading API.
pub fn dequantize_q8_0(blocks: &[BlockQ8_0], out: &mut [f32]) {
    assert_eq!(out.len(), blocks.len() * QK8_0);
    for (block, output) in blocks.iter().zip(out.chunks_exact_mut(QK8_0)) {
        let scale = block.d.to_f32();
        for (dst, &value) in output.iter_mut().zip(block.qs.iter()) {
            *dst = scale * value as f32;
        }
    }
}

pub fn qkv_f32_da3_base(
    input: &[f32],
    weight: &[f32],
    bias: &[f32],
    q: &mut [f32],
    k: &mut [f32],
    v: &mut [f32],
) -> bool {
    crate::specialized::qkv_f32_da3_base(input, weight, bias, q, k, v)
}

pub fn linear_bias_scale_f32_da3_base(
    m: usize,
    n: usize,
    k: usize,
    a: &[f32],
    b: &[f32],
    bias: &[f32],
    scale: &[f32],
    c: &mut [f32],
) -> bool {
    crate::specialized::linear_bias_scale_f32_da3_base(m, n, k, a, b, bias, scale, c)
}

#[allow(clippy::too_many_arguments)]
pub fn qk_norm_rope_f32_da3_base(
    q: &mut [f32],
    k: &mut [f32],
    q_gamma: &[f32],
    q_beta: &[f32],
    k_gamma: &[f32],
    k_beta: &[f32],
    positions_yx: &[i64],
    frequency: f32,
    epsilon: f32,
) -> bool {
    crate::specialized::qk_norm_rope_f32_da3_base(
        q,
        k,
        q_gamma,
        q_beta,
        k_gamma,
        k_beta,
        positions_yx,
        frequency,
        epsilon,
    )
}
#[cfg(feature = "cuda")]
pub mod cuda;

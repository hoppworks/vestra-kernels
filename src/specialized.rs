//! Independently versioned CPU kernels for the fixed DA3 inference workload.
//!
//! A kernel is admitted only after it is faster than the caller's fallback on
//! the target shape and passes the caller's end-to-end F32 parity gate.

#[cfg(da3_blis)]
use std::sync::Once;
use std::sync::{
    Arc, OnceLock,
    atomic::{AtomicU64, Ordering},
};
use std::time::Instant;

/// The DA3-BASE token count for a 504×336 input (36×24 patches plus special
/// tokens).  This is intentionally explicit: specialised kernels must never
/// pretend to support arbitrary matrix shapes.
pub const DA3_BASE_TOKENS_504X336: usize = 865;
const DA3_BASE_ROPE_SPECIAL_TOKENS: usize = 1;

static DA3_BASE_LOCAL_ROPE_ROTATIONS: OnceLock<Arc<[f32]>> = OnceLock::new();
static DA3_BASE_GLOBAL_ROPE_ROTATIONS: OnceLock<Arc<[f32]>> = OnceLock::new();

// The BLIS experiment is compiled only by the dedicated feasibility build
// (`RUSTFLAGS='--cfg da3_blis …'`). It never alters ordinary DA3 binaries.
#[cfg(da3_blis)]
#[link(name = "bliso")]
unsafe extern "C" {
    fn bli_thread_set_num_threads(n_threads: i64);
    fn sgemm_(
        trans_a: *const u8,
        trans_b: *const u8,
        m: *const i32,
        n: *const i32,
        k: *const i32,
        alpha: *const f32,
        a: *const f32,
        lda: *const i32,
        b: *const f32,
        ldb: *const i32,
        beta: *const f32,
        c: *mut f32,
        ldc: *const i32,
    );
}

#[cfg(da3_blis)]
fn blis_sgemm_row_major(m: usize, n: usize, k: usize, a: &[f32], b: &[f32], c: &mut [f32]) -> bool {
    static INITIALIZED: Once = Once::new();
    INITIALIZED.call_once(|| unsafe { bli_thread_set_num_threads(16) });
    let (m_i32, n_i32, k_i32) = (m as i32, n as i32, k as i32);
    let (alpha, beta, no_transpose) = (1.0f32, 0.0f32, b'N');
    // Row-major A[m,k] B[k,n] C[m,n] is exactly the same storage as the
    // column-major Cᵀ[n,m] = Bᵀ[n,k] × Aᵀ[k,m]. This avoids a runtime
    // transpose and lets BLIS consume immutable GGUF weights directly.
    unsafe {
        sgemm_(
            &no_transpose,
            &no_transpose,
            &n_i32,
            &m_i32,
            &k_i32,
            &alpha,
            b.as_ptr(),
            &n_i32,
            a.as_ptr(),
            &k_i32,
            &beta,
            c.as_mut_ptr(),
            &n_i32,
        );
    }
    true
}

/// Optional BLIS route for serial, row-major head projections. This is only
/// compiled by the experimental `da3_blis` build and is deliberately not a
/// generic default GEMM backend: callers must still explicitly select it.
#[cfg(da3_blis)]
pub fn blis_gemm_f32(m: usize, n: usize, k: usize, a: &[f32], b: &[f32], c: &mut [f32]) -> bool {
    a.len() == m * k
        && b.len() == k * n
        && c.len() == m * n
        && blis_sgemm_row_major(m, n, k, a, b, c)
}

#[cfg(not(da3_blis))]
pub fn blis_gemm_f32(_: usize, _: usize, _: usize, _: &[f32], _: &[f32], _: &mut [f32]) -> bool {
    false
}

#[cfg(all(test, da3_blis))]
mod blis_tests {
    use super::*;

    #[test]
    fn row_major_view_matches_naive_product() {
        let (m, n, k) = (3usize, 5usize, 7usize);
        let a = (0..m * k)
            .map(|i| i as f32 * 0.03125 - 0.25)
            .collect::<Vec<_>>();
        let b = (0..k * n)
            .map(|i| i as f32 * -0.0625 + 0.5)
            .collect::<Vec<_>>();
        let mut actual = vec![0.0; m * n];
        assert!(blis_sgemm_row_major(m, n, k, &a, &b, &mut actual));
        for row in 0..m {
            for col in 0..n {
                let expected = (0..k)
                    .map(|inner| a[row * k + inner] * b[inner * n + col])
                    .sum::<f32>();
                assert!((actual[row * n + col] - expected).abs() <= 1e-5);
            }
        }
    }
}

/// Model-owned IOHW filter packed as contiguous input-channel vectors for a
/// non-overlapping transposed convolution.  Preparing it once keeps the
/// runtime path free of a repeated weight-layout conversion.
#[derive(Clone)]
pub struct NonoverlapTransposeF32 {
    packed: Arc<[f32]>,
    input_channels: usize,
    output_channels: usize,
    kernel_h: usize,
    kernel_w: usize,
}

/// Converts an immutable IOHW transposed-convolution filter into the layout
/// consumed by [`nonoverlap_transpose_f32`].
pub fn prepare_nonoverlap_transpose_f32(
    weight: &[f32],
    input_channels: usize,
    output_channels: usize,
    kernel_h: usize,
    kernel_w: usize,
) -> NonoverlapTransposeF32 {
    assert_eq!(
        weight.len(),
        input_channels * output_channels * kernel_h * kernel_w
    );
    let mut packed = vec![0.0; weight.len()];
    for output in 0..output_channels {
        for ky in 0..kernel_h {
            for kx in 0..kernel_w {
                for input in 0..input_channels {
                    packed[((output * kernel_h + ky) * kernel_w + kx) * input_channels + input] =
                        weight
                            [((input * output_channels + output) * kernel_h + ky) * kernel_w + kx];
                }
            }
        }
    }
    NonoverlapTransposeF32 {
        packed: packed.into(),
        input_channels,
        output_channels,
        kernel_h,
        kernel_w,
    }
}

/// Executes DA3's non-overlapping `kernel == stride` transposed convolution
/// from a prepacked, model-owned filter. Returns `false` when the CPU or
/// tensor geometry is unsupported, leaving the caller's fallback available.
pub fn nonoverlap_transpose_f32(
    input: &[f32],
    input_h: usize,
    input_w: usize,
    filter: &NonoverlapTransposeF32,
    bias: Option<&[f32]>,
    out: &mut [f32],
) -> bool {
    if std::env::var_os("DA3_KERNELS_DISABLE_TRANSPOSE").is_some()
        || filter.input_channels % 16 != 0
        || input.len() != filter.input_channels * input_h * input_w
        || bias.is_some_and(|values| values.len() != filter.output_channels)
        || out.len()
            != filter.output_channels * input_h * filter.kernel_h * input_w * filter.kernel_w
    {
        return false;
    }
    #[cfg(target_arch = "x86_64")]
    if std::is_x86_feature_detected!("avx512f") && std::is_x86_feature_detected!("fma") {
        // SAFETY: all vector-width, lengths, and non-overlap geometry were
        // validated above; every output plane is disjoint.
        unsafe { nonoverlap_transpose_avx512(input, input_h, input_w, filter, bias, out) };
        return true;
    }
    false
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,fma")]
unsafe fn nonoverlap_transpose_avx512(
    input: &[f32],
    input_h: usize,
    input_w: usize,
    filter: &NonoverlapTransposeF32,
    bias: Option<&[f32]>,
    out: &mut [f32],
) {
    use core::arch::x86_64::*;
    use rayon::prelude::*;
    let input_channels = filter.input_channels;
    let output_h = input_h * filter.kernel_h;
    let output_w = input_w * filter.kernel_w;
    let mut pixels = vec![0.0f32; input_h * input_w * input_channels];
    for iy in 0..input_h {
        for ix in 0..input_w {
            let dst = &mut pixels
                [(iy * input_w + ix) * input_channels..(iy * input_w + ix + 1) * input_channels];
            for input_channel in 0..input_channels {
                dst[input_channel] = input[(input_channel * input_h + iy) * input_w + ix];
            }
        }
    }
    out.par_chunks_mut(output_h * output_w)
        .enumerate()
        .for_each(|(output_channel, plane)| {
            let bias_value = bias.map_or(0.0, |values| values[output_channel]);
            for iy in 0..input_h {
                for ix in 0..input_w {
                    let pixel = &pixels[(iy * input_w + ix) * input_channels
                        ..(iy * input_w + ix + 1) * input_channels];
                    for ky in 0..filter.kernel_h {
                        for kx in 0..filter.kernel_w {
                            let weights = &filter.packed[((output_channel * filter.kernel_h + ky)
                                * filter.kernel_w
                                + kx)
                                * input_channels
                                ..((output_channel * filter.kernel_h + ky) * filter.kernel_w
                                    + kx
                                    + 1)
                                    * input_channels];
                            let mut lanes = _mm512_setzero_ps();
                            for input_channel in (0..input_channels).step_by(16) {
                                let x =
                                    unsafe { _mm512_loadu_ps(pixel.as_ptr().add(input_channel)) };
                                let w =
                                    unsafe { _mm512_loadu_ps(weights.as_ptr().add(input_channel)) };
                                lanes = _mm512_fmadd_ps(x, w, lanes);
                            }
                            let mut partial = [0.0f32; 16];
                            unsafe { _mm512_storeu_ps(partial.as_mut_ptr(), lanes) };
                            plane[(iy * filter.kernel_h + ky) * output_w
                                + ix * filter.kernel_w
                                + kx] = partial.iter().sum::<f32>() + bias_value;
                        }
                    }
                }
            }
        });
}

/// Fuses DA3-BASE Q/K LayerNorm and 2D RoPE. The caller uses this only for
/// the late blocks where both operations are enabled. Each row keeps the
/// scalar LayerNorm reduction order; fusion merely avoids materializing the
/// normalized Q and K tensors before immediately rotating them.
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
    const HEADS: usize = 12;
    const DIM: usize = 64;
    if std::env::var_os("DA3_KERNELS_DISABLE_QK_NORM_ROPE").is_some()
        || q.len() != HEADS * DA3_BASE_TOKENS_504X336 * DIM
        || k.len() != q.len()
        || q_gamma.len() != DIM
        || q_beta.len() != DIM
        || k_gamma.len() != DIM
        || k_beta.len() != DIM
        || positions_yx.len() != DA3_BASE_TOKENS_504X336 * 2
    {
        return false;
    }

    if let Some(rotations) = da3_base_cached_rope_rotations(positions_yx, frequency) {
        rayon::join(
            || normalize_and_rotate(q, q_gamma, q_beta, epsilon, rotations),
            || normalize_and_rotate(k, k_gamma, k_beta, epsilon, rotations),
        );
    } else {
        let rotations = rope_rotations(positions_yx, frequency);
        rayon::join(
            || normalize_and_rotate(q, q_gamma, q_beta, epsilon, &rotations),
            || normalize_and_rotate(k, k_gamma, k_beta, epsilon, &rotations),
        );
    }
    true
}

#[derive(Clone, Copy)]
enum Da3BaseRopeLayout {
    Local,
    Global,
}

/// Returns a process-lifetime immutable rotation table only for the two exact
/// position patterns emitted by DA3-BASE at 504×336: its CLS token is `(0,0)`,
/// followed by the row-major 24×36 patch grid. Any other equal-length layout,
/// frequency, or altered special-token coordinate uses the generic calculation
/// instead.
fn da3_base_cached_rope_rotations(positions_yx: &[i64], frequency: f32) -> Option<&'static [f32]> {
    if frequency.to_bits() != 100.0f32.to_bits() {
        return None;
    }
    let layout = da3_base_rope_layout(positions_yx)?;
    let cache = match layout {
        Da3BaseRopeLayout::Local => &DA3_BASE_LOCAL_ROPE_ROTATIONS,
        Da3BaseRopeLayout::Global => &DA3_BASE_GLOBAL_ROPE_ROTATIONS,
    };
    Some(
        cache
            .get_or_init(|| Arc::from(rope_rotations(&da3_base_rope_positions(layout), 100.0)))
            .as_ref(),
    )
}

fn da3_base_rope_layout(positions_yx: &[i64]) -> Option<Da3BaseRopeLayout> {
    if positions_yx.len() != DA3_BASE_TOKENS_504X336 * 2
        || !positions_yx[..DA3_BASE_ROPE_SPECIAL_TOKENS * 2]
            .chunks_exact(2)
            .all(|position| position == [0, 0])
    {
        return None;
    }
    let global = positions_yx[DA3_BASE_ROPE_SPECIAL_TOKENS * 2..]
        .chunks_exact(2)
        .all(|position| position == [1, 1]);
    if global {
        return Some(Da3BaseRopeLayout::Global);
    }
    for token in DA3_BASE_ROPE_SPECIAL_TOKENS..DA3_BASE_TOKENS_504X336 {
        let patch = token - DA3_BASE_ROPE_SPECIAL_TOKENS;
        if positions_yx[2 * token] != (patch / 36 + 1) as i64
            || positions_yx[2 * token + 1] != (patch % 36 + 1) as i64
        {
            return None;
        }
    }
    Some(Da3BaseRopeLayout::Local)
}

fn da3_base_rope_positions(layout: Da3BaseRopeLayout) -> Vec<i64> {
    let mut positions = vec![0i64; DA3_BASE_TOKENS_504X336 * 2];
    match layout {
        Da3BaseRopeLayout::Global => {
            for position in positions[DA3_BASE_ROPE_SPECIAL_TOKENS * 2..].chunks_exact_mut(2) {
                position.copy_from_slice(&[1, 1]);
            }
        }
        Da3BaseRopeLayout::Local => {
            for token in DA3_BASE_ROPE_SPECIAL_TOKENS..DA3_BASE_TOKENS_504X336 {
                let patch = token - DA3_BASE_ROPE_SPECIAL_TOKENS;
                positions[2 * token] = (patch / 36 + 1) as i64;
                positions[2 * token + 1] = (patch % 36 + 1) as i64;
            }
        }
    }
    positions
}

fn rope_rotations(positions_yx: &[i64], frequency: f32) -> Vec<f32> {
    const QUARTER: usize = 16;
    const HALF_DIM: usize = 32;
    let inv_freq: Vec<f32> = (0..QUARTER)
        .map(|i| frequency.powf(-2.0 * i as f32 / HALF_DIM as f32))
        .collect();
    let mut rotations = vec![0.0; DA3_BASE_TOKENS_504X336 * 2 * QUARTER * 2];
    for token in 0..DA3_BASE_TOKENS_504X336 {
        for (axis, position) in [positions_yx[2 * token], positions_yx[2 * token + 1]]
            .into_iter()
            .enumerate()
        {
            for (i, freq) in inv_freq.iter().enumerate() {
                let (sin, cos) = (position as f32 * freq).sin_cos();
                let offset = (token * 2 * QUARTER + axis * QUARTER + i) * 2;
                rotations[offset] = sin;
                rotations[offset + 1] = cos;
            }
        }
    }
    rotations
}

fn normalize_and_rotate(
    values: &mut [f32],
    gamma: &[f32],
    beta: &[f32],
    epsilon: f32,
    rotations: &[f32],
) {
    use rayon::prelude::*;
    const DIM: usize = 64;
    const QUARTER: usize = 16;
    values
        .par_chunks_mut(DIM)
        .enumerate()
        .for_each(|(row, data)| {
            let mean = data.iter().sum::<f32>() / DIM as f32;
            let variance = data
                .iter()
                .map(|value| {
                    let delta = value - mean;
                    delta * delta
                })
                .sum::<f32>()
                / DIM as f32;
            let inverse = 1.0 / (variance + epsilon).sqrt();
            for dim in 0..DIM {
                data[dim] = (data[dim] - mean) * inverse * gamma[dim] + beta[dim];
            }
            let token = row % DA3_BASE_TOKENS_504X336;
            for axis in 0..2 {
                let base = axis * 32;
                let rotation = &rotations[(token * 2 * QUARTER + axis * QUARTER) * 2
                    ..(token * 2 * QUARTER + axis * QUARTER + QUARTER) * 2];
                for i in 0..QUARTER {
                    let sin = rotation[i * 2];
                    let cos = rotation[i * 2 + 1];
                    let first = data[base + i];
                    let second = data[base + i + QUARTER];
                    data[base + i] = first * cos - second * sin;
                    data[base + i + QUARTER] = second * cos + first * sin;
                }
            }
        });
}

/// Multiplies one F(2x2,3x3) Winograd tile block in the filter layout used by
/// ggml: `u[position][input_channel][output_channel]` and
/// `v[position][input_channel][tile]`.  It is deliberately a small, explicit
/// building block; layout creation and the exact Winograd transforms remain
/// owned by the runtime.
pub fn winograd_f2_blocked_f32(
    u: &[f32],
    v: &[f32],
    m: &mut [f32],
    input_channels: usize,
    output_channels: usize,
    tiles: usize,
) -> bool {
    if std::env::var_os("DA3_KERNELS_DISABLE_WINO").is_some()
        || tiles == 0
        || tiles > 8
        || output_channels % 16 != 0
        || u.len() != 16 * input_channels * output_channels
        || v.len() != 16 * input_channels * tiles
        || m.len() != 16 * tiles * output_channels
    {
        return false;
    }
    #[cfg(target_arch = "x86_64")]
    if std::is_x86_feature_detected!("avx512f") && std::is_x86_feature_detected!("fma") {
        // SAFETY: the exact contiguous layouts and AVX-512 availability were
        // checked above. Each output vector is independent.
        unsafe {
            winograd_f2_blocked_avx512(u, v, m, input_channels, output_channels, tiles);
        }
        return true;
    }
    false
}

/// Multiplies one F(4x4,3x3) Winograd tile block in the same blocked filter
/// layout as [`winograd_f2_blocked_f32`], except with the six-by-six (36
/// position) transform domain.  The F(4) runtime owns the transforms and
/// falls back to its scalar product if this AVX-512 building block is not
/// available.
pub fn winograd_f4_blocked_f32(
    u: &[f32],
    v: &[f32],
    m: &mut [f32],
    input_channels: usize,
    output_channels: usize,
    tiles: usize,
) -> bool {
    if std::env::var_os("DA3_KERNELS_DISABLE_WINO").is_some()
        || std::env::var_os("DA3_KERNELS_DISABLE_WINO_F4").is_some()
        || tiles == 0
        || tiles > 8
        || output_channels % 16 != 0
        || u.len() != 36 * input_channels * output_channels
        || v.len() != 36 * input_channels * tiles
        || m.len() != 36 * tiles * output_channels
    {
        return false;
    }
    #[cfg(target_arch = "x86_64")]
    if std::is_x86_feature_detected!("avx512f") && std::is_x86_feature_detected!("fma") {
        // SAFETY: the contiguous layouts and CPU feature requirements above
        // are exactly those required by the vector product loop.
        unsafe {
            winograd_f4_blocked_avx512(u, v, m, input_channels, output_channels, tiles);
        }
        return true;
    }
    false
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,fma")]
unsafe fn winograd_f4_blocked_avx512(
    u: &[f32],
    v: &[f32],
    m: &mut [f32],
    input_channels: usize,
    output_channels: usize,
    tiles: usize,
) {
    use core::arch::x86_64::*;
    for position in 0..36 {
        let u_position = &u[position * input_channels * output_channels..];
        let v_position = &v[position * input_channels * tiles..];
        let m_position = &mut m[position * tiles * output_channels..];
        for output0 in (0..output_channels).step_by(16) {
            let mut accumulators = [_mm512_setzero_ps(); 8];
            for input in 0..input_channels {
                let filter = unsafe {
                    _mm512_loadu_ps(u_position.as_ptr().add(input * output_channels + output0))
                };
                let values = &v_position[input * tiles..(input + 1) * tiles];
                for tile in 0..tiles {
                    accumulators[tile] =
                        _mm512_fmadd_ps(filter, _mm512_set1_ps(values[tile]), accumulators[tile]);
                }
            }
            for tile in 0..tiles {
                unsafe {
                    _mm512_storeu_ps(
                        m_position
                            .as_mut_ptr()
                            .add(tile * output_channels + output0),
                        accumulators[tile],
                    );
                }
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,fma")]
unsafe fn winograd_f2_blocked_avx512(
    u: &[f32],
    v: &[f32],
    m: &mut [f32],
    input_channels: usize,
    output_channels: usize,
    tiles: usize,
) {
    // Keep the historically rejected all-convolution variant opt-in, but
    // allow the exact 64->32 final DPT head product to be isolated: it has
    // only two output ZMMs and can behave differently from the wider layers.
    let use_tiles4_special = tiles == 4
        && (std::env::var_os("DA3_KERNELS_ENABLE_WINO_TILES4_SPECIAL").is_some()
            || (input_channels == 64
                && output_channels == 32
                && std::env::var_os("DA3_KERNELS_ENABLE_FINAL_F2_TILES4_SPECIAL").is_some()));
    if use_tiles4_special {
        unsafe { winograd_f2_blocked_avx512_tiles4(u, v, m, input_channels, output_channels) };
    } else {
        unsafe {
            winograd_f2_blocked_avx512_generic(u, v, m, input_channels, output_channels, tiles)
        };
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,fma")]
unsafe fn winograd_f2_blocked_avx512_generic(
    u: &[f32],
    v: &[f32],
    m: &mut [f32],
    input_channels: usize,
    output_channels: usize,
    tiles: usize,
) {
    use core::arch::x86_64::*;
    for position in 0..16 {
        let u_position = &u[position * input_channels * output_channels..];
        let v_position = &v[position * input_channels * tiles..];
        let m_position = &mut m[position * tiles * output_channels..];
        for output0 in (0..output_channels).step_by(16) {
            let mut accumulators = [_mm512_setzero_ps(); 8];
            for input in 0..input_channels {
                let filter = unsafe {
                    _mm512_loadu_ps(u_position.as_ptr().add(input * output_channels + output0))
                };
                let values = &v_position[input * tiles..input * tiles + tiles];
                for tile in 0..tiles {
                    accumulators[tile] =
                        _mm512_fmadd_ps(filter, _mm512_set1_ps(values[tile]), accumulators[tile]);
                }
            }
            for tile in 0..tiles {
                unsafe {
                    _mm512_storeu_ps(
                        m_position
                            .as_mut_ptr()
                            .add(tile * output_channels + output0),
                        accumulators[tile],
                    );
                }
            }
        }
    }
}

/// Specialization for the production four-tile Winograd path. The fixed
/// accumulator count lets LLVM retain the four independent tile accumulators
/// without the generic eight-lane temporary array. Per-tile input order and
/// every FMA match [`winograd_f2_blocked_avx512_generic`] exactly.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,fma")]
unsafe fn winograd_f2_blocked_avx512_tiles4(
    u: &[f32],
    v: &[f32],
    m: &mut [f32],
    input_channels: usize,
    output_channels: usize,
) {
    use core::arch::x86_64::*;
    const TILES: usize = 4;
    for position in 0..16 {
        let u_position = &u[position * input_channels * output_channels..];
        let v_position = &v[position * input_channels * TILES..];
        let m_position = &mut m[position * TILES * output_channels..];
        for output0 in (0..output_channels).step_by(16) {
            let mut accumulators = [_mm512_setzero_ps(); TILES];
            for input in 0..input_channels {
                let filter = unsafe {
                    _mm512_loadu_ps(u_position.as_ptr().add(input * output_channels + output0))
                };
                let values = &v_position[input * TILES..(input + 1) * TILES];
                accumulators[0] =
                    _mm512_fmadd_ps(filter, _mm512_set1_ps(values[0]), accumulators[0]);
                accumulators[1] =
                    _mm512_fmadd_ps(filter, _mm512_set1_ps(values[1]), accumulators[1]);
                accumulators[2] =
                    _mm512_fmadd_ps(filter, _mm512_set1_ps(values[2]), accumulators[2]);
                accumulators[3] =
                    _mm512_fmadd_ps(filter, _mm512_set1_ps(values[3]), accumulators[3]);
            }
            for tile in 0..TILES {
                unsafe {
                    _mm512_storeu_ps(
                        m_position
                            .as_mut_ptr()
                            .add(tile * output_channels + output0),
                        accumulators[tile],
                    );
                }
            }
        }
    }
}

/// Transformer projection shapes eligible for a future specialised kernel.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Da3BaseProjection {
    pub tokens: usize,
    pub input_channels: usize,
    pub output_channels: usize,
}

/// Selects the row micro-tile for the fixed projection kernels. Unsupported
/// values deliberately retain the six-row production control.
fn linear_rows_from_env(value: Option<&str>) -> usize {
    match value {
        Some("4") => 4,
        Some("8") => 8,
        _ => 6,
    }
}

fn linear_rows() -> usize {
    linear_rows_from_env(std::env::var("DA3_KERNELS_LINEAR_ROWS").ok().as_deref())
}

/// Opt-in row micro-tile selector for the FC2 bias/LayerScale epilogue.
///
/// This deliberately has a separate switch from the generic projection
/// selector: FC2's 3072x768 weight matrix has a materially different cache
/// footprint from FC1/QKV.  The production six-row kernel remains the
/// default until a controlled end-to-end study accepts another value.
fn fc2_rows_from_env(value: Option<&str>) -> usize {
    match value {
        Some("4") => 4,
        Some("8") => 8,
        _ => 6,
    }
}

fn fc2_rows() -> usize {
    fc2_rows_from_env(std::env::var("DA3_KERNELS_FC2_ROWS").ok().as_deref())
}

/// Fused F32 attention for DA3-BASE's `[heads, tokens, 64]` layout.
///
/// The native one-view raster has 865 tokens. PR #2's global cross-view
/// layers concatenate an integral number of those same rasters, so the exact
/// AVX-512 flash kernel is also valid for `views * 865` tokens. Keeping this
/// gate explicit prevents unrelated 64-wide attention workloads from entering
/// a DA3-specific unsafe implementation. The function returns `false` without
/// changing `out` when the CPU or shape is unsupported so its caller can
/// safely use its established fallback.
pub fn flash_attention_f32_da3_base(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    heads: usize,
    tokens: usize,
    out: &mut [f32],
) -> bool {
    if std::env::var_os("DA3_KERNELS_DISABLE_FLASH").is_some() {
        return false;
    }
    // Retain a narrowly scoped control for the multi-view A/B study. It must
    // never affect one-view inference, whose established flash path remains
    // independently benchmarked.
    if tokens > DA3_BASE_TOKENS_504X336
        && std::env::var_os("DA3_KERNELS_DISABLE_MULTIVIEW_FLASH").is_some()
    {
        return false;
    }
    if tokens == 0
        || !tokens.is_multiple_of(DA3_BASE_TOKENS_504X336)
        || q.len() != heads * tokens * 64
        || k.len() != q.len()
        || v.len() != q.len()
        || out.len() != q.len()
    {
        return false;
    }
    #[cfg(target_arch = "x86_64")]
    if std::is_x86_feature_detected!("avx512f") && std::is_x86_feature_detected!("fma") {
        // SAFETY: feature checks above and shape validation before entering.
        unsafe { flash_attention_avx512(q, k, v, heads, tokens, out) };
        return true;
    }
    false
}

/// Computes one of DA3-BASE's fixed transformer projections in row-major
/// form. Returns false for every other shape so callers retain their normal
/// GEMM backend without a behavioural fork.
pub fn linear_f32_da3_base(
    m: usize,
    n: usize,
    k: usize,
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
) -> bool {
    if std::env::var_os("DA3_KERNELS_DISABLE_LINEAR").is_some()
        || m != DA3_BASE_TOKENS_504X336
        || !matches!((k, n), (768, 2304) | (768, 768) | (768, 3072) | (3072, 768))
        || a.len() != m * k
        || b.len() != k * n
        || c.len() != m * n
    {
        return false;
    }
    #[cfg(da3_blis)]
    if std::env::var_os("DA3_KERNELS_BLIS_LINEAR").is_some()
        && blis_sgemm_row_major(m, n, k, a, b, c)
    {
        return true;
    }
    #[cfg(target_arch = "x86_64")]
    if std::is_x86_feature_detected!("avx512f") && std::is_x86_feature_detected!("fma") {
        // SAFETY: ISA and exact contiguous matrix dimensions were validated.
        unsafe {
            if std::env::var_os("DA3_KERNELS_DISABLE_LINEAR_COLUMN_SPLIT").is_some() {
                linear_avx512(m, n, k, a, b, c);
            } else {
                linear_avx512_column_split(m, n, k, a, b, c);
            }
        };
        return true;
    }
    false
}

/// The output-projection/FC2 form of the fixed DA3 kernel, with its two
/// rowwise epilogue passes folded into the final stores.
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
    if std::env::var_os("DA3_KERNELS_DISABLE_LINEAR_EPILOGUE").is_some()
        || m != DA3_BASE_TOKENS_504X336
        || !matches!((k, n), (768, 768) | (3072, 768))
        || a.len() != m * k
        || b.len() != k * n
        || bias.len() != n
        || scale.len() != n
        || c.len() != m * n
    {
        return false;
    }
    #[cfg(da3_blis)]
    if std::env::var_os("DA3_KERNELS_BLIS_LINEAR").is_some()
        && blis_sgemm_row_major(m, n, k, a, b, c)
    {
        for row in c.chunks_exact_mut(n) {
            for col in 0..n {
                row[col] = (row[col] + bias[col]) * scale[col];
            }
        }
        return true;
    }
    #[cfg(target_arch = "x86_64")]
    if std::is_x86_feature_detected!("avx512f") && std::is_x86_feature_detected!("fma") {
        unsafe {
            if std::env::var_os("DA3_KERNELS_BIAS_SCALE_COLUMN_SPLIT").is_some() {
                linear_bias_scale_avx512_column_split(m, n, k, a, b, bias, scale, c);
            } else {
                linear_bias_scale_avx512(m, n, k, a, b, bias, scale, c);
            }
        };
        return true;
    }
    false
}

/// DA3-BASE's FC1 projection with its bias and exact-erf GELU epilogue
/// folded into the final store. This avoids two full activation-memory passes
/// over the 865×3072 intermediate without changing the K-major FMA order.
pub fn linear_bias_gelu_f32_da3_base(
    m: usize,
    n: usize,
    k: usize,
    a: &[f32],
    b: &[f32],
    bias: &[f32],
    c: &mut [f32],
) -> bool {
    if std::env::var_os("DA3_KERNELS_DISABLE_FC1_EPILOGUE").is_some()
        || m != DA3_BASE_TOKENS_504X336
        || (k, n) != (768, 3072)
        || a.len() != m * k
        || b.len() != k * n
        || bias.len() != n
        || c.len() != m * n
    {
        return false;
    }
    #[cfg(target_arch = "x86_64")]
    if std::is_x86_feature_detected!("avx512f") && std::is_x86_feature_detected!("fma") {
        unsafe { linear_bias_gelu_avx512(m, n, k, a, b, bias, c) };
        return true;
    }
    false
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,fma")]
unsafe fn linear_bias_gelu_avx512(
    m: usize,
    n: usize,
    k: usize,
    a: &[f32],
    b: &[f32],
    bias: &[f32],
    c: &mut [f32],
) {
    use core::arch::x86_64::*;
    use rayon::prelude::*;
    const ROWS: usize = 6;
    c.par_chunks_mut(ROWS * n)
        .enumerate()
        .for_each(|(tile, c_tile)| {
            let row0 = tile * ROWS;
            let rows = (m - row0).min(ROWS);
            for col0 in (0..n).step_by(64) {
                let mut acc = [[_mm512_setzero_ps(); 4]; ROWS];
                for kk in 0..k {
                    let bp = unsafe { b.as_ptr().add(kk * n + col0) };
                    let bv = unsafe {
                        [
                            _mm512_loadu_ps(bp),
                            _mm512_loadu_ps(bp.add(16)),
                            _mm512_loadu_ps(bp.add(32)),
                            _mm512_loadu_ps(bp.add(48)),
                        ]
                    };
                    for row in 0..rows {
                        let av = _mm512_set1_ps(a[(row0 + row) * k + kk]);
                        for block in 0..4 {
                            acc[row][block] = _mm512_fmadd_ps(av, bv[block], acc[row][block]);
                        }
                    }
                }
                for row in 0..rows {
                    for block in 0..4 {
                        let offset = col0 + block * 16;
                        let bias_v = unsafe { _mm512_loadu_ps(bias.as_ptr().add(offset)) };
                        let out = unsafe { gelu_avx512(_mm512_add_ps(acc[row][block], bias_v)) };
                        unsafe { _mm512_storeu_ps(c_tile.as_mut_ptr().add(row * n + offset), out) };
                    }
                }
            }
        });
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,fma")]
unsafe fn gelu_avx512(x: core::arch::x86_64::__m512) -> core::arch::x86_64::__m512 {
    use core::arch::x86_64::*;
    let one = _mm512_set1_ps(1.0);
    let half = _mm512_set1_ps(0.5);
    let inv_sqrt2 = _mm512_set1_ps(0.707_106_78);
    let x = x;
    let abs_mask = _mm512_set1_epi32(0x7fff_ffff);
    let x_abs = _mm512_castsi512_ps(_mm512_and_epi32(
        _mm512_castps_si512(_mm512_mul_ps(x, inv_sqrt2)),
        abs_mask,
    ));
    let arg = _mm512_mul_ps(x, inv_sqrt2);
    let neg_mask = _mm512_cmp_ps_mask(arg, _mm512_setzero_ps(), _CMP_LT_OQ);
    let sign = _mm512_mask_blend_ps(neg_mask, one, _mm512_set1_ps(-1.0));
    let t = _mm512_div_ps(
        one,
        _mm512_fmadd_ps(_mm512_set1_ps(0.327_591_1), x_abs, one),
    );
    let mut poly = _mm512_set1_ps(1.061_405_4);
    poly = _mm512_fmadd_ps(poly, t, _mm512_set1_ps(-1.453_152_0));
    poly = _mm512_fmadd_ps(poly, t, _mm512_set1_ps(1.421_413_7));
    poly = _mm512_fmadd_ps(poly, t, _mm512_set1_ps(-0.284_496_74));
    poly = _mm512_fmadd_ps(poly, t, _mm512_set1_ps(0.254_829_59));
    poly = _mm512_mul_ps(poly, t);
    let erf = _mm512_mul_ps(
        sign,
        _mm512_fnmadd_ps(
            poly,
            unsafe {
                exp_avx512(_mm512_sub_ps(
                    _mm512_setzero_ps(),
                    _mm512_mul_ps(x_abs, x_abs),
                ))
            },
            one,
        ),
    );
    _mm512_mul_ps(_mm512_mul_ps(half, x), _mm512_add_ps(one, erf))
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,fma")]
unsafe fn exp_avx512(x: core::arch::x86_64::__m512) -> core::arch::x86_64::__m512 {
    use core::arch::x86_64::*;
    let x = _mm512_max_ps(
        _mm512_min_ps(x, _mm512_set1_ps(88.376_26)),
        _mm512_set1_ps(-88.376_26),
    );
    let one = _mm512_set1_ps(1.0);
    let fx0 = _mm512_fmadd_ps(x, _mm512_set1_ps(1.442_695_04), _mm512_set1_ps(0.5));
    let fx_trunc = _mm512_cvtepi32_ps(_mm512_cvttps_epi32(fx0));
    let fx = _mm512_mask_sub_ps(
        fx_trunc,
        _mm512_cmp_ps_mask(fx_trunc, fx0, _CMP_GT_OQ),
        fx_trunc,
        one,
    );
    let x = _mm512_fnmadd_ps(fx, _mm512_set1_ps(0.693_359_375), x);
    let x = _mm512_fnmadd_ps(fx, _mm512_set1_ps(-2.121_944_4e-4), x);
    let z = _mm512_mul_ps(x, x);
    let mut y = _mm512_set1_ps(1.987_569_15e-4);
    for coefficient in [
        1.398_199_95e-3,
        8.333_451_9e-3,
        4.166_579_6e-2,
        1.666_666_5e-1,
        5.000_000_1e-1,
    ] {
        y = _mm512_fmadd_ps(y, x, _mm512_set1_ps(coefficient));
    }
    y = _mm512_fmadd_ps(y, z, x);
    let y = _mm512_add_ps(y, one);
    let exponent = _mm512_slli_epi32(
        _mm512_add_epi32(_mm512_cvttps_epi32(fx), _mm512_set1_epi32(0x7f)),
        23,
    );
    _mm512_mul_ps(y, _mm512_castsi512_ps(exponent))
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,fma")]
unsafe fn linear_bias_scale_avx512(
    m: usize,
    n: usize,
    k: usize,
    a: &[f32],
    b: &[f32],
    bias: &[f32],
    scale: &[f32],
    c: &mut [f32],
) {
    match fc2_rows() {
        4 => unsafe { linear_bias_scale_avx512_rows::<4>(m, n, k, a, b, bias, scale, c) },
        8 => unsafe { linear_bias_scale_avx512_rows::<8>(m, n, k, a, b, bias, scale, c) },
        _ => unsafe { linear_bias_scale_avx512_rows::<6>(m, n, k, a, b, bias, scale, c) },
    }
}

/// Row-parallel FC2 kernel.  The accumulation order is identical for every
/// row-tile size; only independent output rows are scheduled differently.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,fma")]
unsafe fn linear_bias_scale_avx512_rows<const ROWS: usize>(
    m: usize,
    n: usize,
    k: usize,
    a: &[f32],
    b: &[f32],
    bias: &[f32],
    scale: &[f32],
    c: &mut [f32],
) {
    use core::arch::x86_64::*;
    use rayon::prelude::*;
    c.par_chunks_mut(ROWS * n)
        .enumerate()
        .for_each(|(tile, c_tile)| {
            let row0 = tile * ROWS;
            let rows = (m - row0).min(ROWS);
            for col0 in (0..n).step_by(64) {
                let mut acc = [[_mm512_setzero_ps(); 4]; ROWS];
                for kk in 0..k {
                    let bp = unsafe { b.as_ptr().add(kk * n + col0) };
                    let bv = unsafe {
                        [
                            _mm512_loadu_ps(bp),
                            _mm512_loadu_ps(bp.add(16)),
                            _mm512_loadu_ps(bp.add(32)),
                            _mm512_loadu_ps(bp.add(48)),
                        ]
                    };
                    for row in 0..rows {
                        let av = _mm512_set1_ps(a[(row0 + row) * k + kk]);
                        for block in 0..4 {
                            acc[row][block] = _mm512_fmadd_ps(av, bv[block], acc[row][block]);
                        }
                    }
                }
                for row in 0..rows {
                    for block in 0..4 {
                        let offset = col0 + block * 16;
                        let bias_v = unsafe { _mm512_loadu_ps(bias.as_ptr().add(offset)) };
                        let scale_v = unsafe { _mm512_loadu_ps(scale.as_ptr().add(offset)) };
                        let out = _mm512_mul_ps(_mm512_add_ps(acc[row][block], bias_v), scale_v);
                        unsafe {
                            _mm512_storeu_ps(c_tile.as_mut_ptr().add(row * n + offset), out);
                        }
                    }
                }
            }
        });
}

/// Column-panel work partition for DA3's output-projection/FC2 epilogue.
/// Each worker owns a disjoint 64-output panel, preserving ascending-K FMA
/// accumulation and applying the same final `(acc + bias) * scale` store.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,fma")]
unsafe fn linear_bias_scale_avx512_column_split(
    m: usize,
    n: usize,
    k: usize,
    a: &[f32],
    b: &[f32],
    bias: &[f32],
    scale: &[f32],
    c: &mut [f32],
) {
    use core::arch::x86_64::*;
    use rayon::prelude::*;
    const ROWS: usize = 6;
    let c_ptr = c.as_mut_ptr() as usize;
    (0..n)
        .step_by(64)
        .collect::<Vec<_>>()
        .into_par_iter()
        .for_each(|col0| {
            for row0 in (0..m).step_by(ROWS) {
                let rows = (m - row0).min(ROWS);
                let mut acc = [[_mm512_setzero_ps(); 4]; ROWS];
                for kk in 0..k {
                    let bp = unsafe { b.as_ptr().add(kk * n + col0) };
                    let bv = unsafe {
                        [
                            _mm512_loadu_ps(bp),
                            _mm512_loadu_ps(bp.add(16)),
                            _mm512_loadu_ps(bp.add(32)),
                            _mm512_loadu_ps(bp.add(48)),
                        ]
                    };
                    for row in 0..rows {
                        let av = _mm512_set1_ps(a[(row0 + row) * k + kk]);
                        for block in 0..4 {
                            acc[row][block] = _mm512_fmadd_ps(av, bv[block], acc[row][block]);
                        }
                    }
                }
                for row in 0..rows {
                    for block in 0..4 {
                        let offset = col0 + block * 16;
                        let bias_v = unsafe { _mm512_loadu_ps(bias.as_ptr().add(offset)) };
                        let scale_v = unsafe { _mm512_loadu_ps(scale.as_ptr().add(offset)) };
                        let out = _mm512_mul_ps(_mm512_add_ps(acc[row][block], bias_v), scale_v);
                        unsafe {
                            _mm512_storeu_ps(
                                (c_ptr as *mut f32).add((row0 + row) * n + offset),
                                out,
                            );
                        }
                    }
                }
            }
        });
}

/// Computes DA3-BASE's fused QKV projection directly into the head-major
/// buffers consumed by attention, avoiding the token-major 3×768 staging
/// tensor and its subsequent transpose.
pub fn qkv_f32_da3_base(
    a: &[f32],
    weight: &[f32],
    bias: &[f32],
    q: &mut [f32],
    k: &mut [f32],
    v: &mut [f32],
) -> bool {
    let tokens = DA3_BASE_TOKENS_504X336;
    if std::env::var_os("DA3_KERNELS_DISABLE_QKV_DIRECT").is_some()
        || a.len() != tokens * 768
        || weight.len() != 768 * 2304
        || bias.len() != 2304
        || q.len() != tokens * 768
        || k.len() != q.len()
        || v.len() != q.len()
    {
        return false;
    }
    #[cfg(target_arch = "x86_64")]
    if std::is_x86_feature_detected!("avx512f") && std::is_x86_feature_detected!("fma") {
        unsafe {
            if std::env::var_os("DA3_KERNELS_DISABLE_QKV_COLUMN_SPLIT").is_some() {
                qkv_avx512(a, weight, bias, q, k, v);
            } else {
                qkv_avx512_column_split(a, weight, bias, q, k, v);
            }
        };
        return true;
    }
    false
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,fma")]
unsafe fn qkv_avx512(
    a: &[f32],
    weight: &[f32],
    bias: &[f32],
    q: &mut [f32],
    k: &mut [f32],
    v: &mut [f32],
) {
    use core::arch::x86_64::*;
    use rayon::prelude::*;
    const ROWS: usize = 6;
    const N: usize = 2304;
    const K: usize = 768;
    let q_ptr = q.as_mut_ptr() as usize;
    let k_ptr = k.as_mut_ptr() as usize;
    let v_ptr = v.as_mut_ptr() as usize;
    // One 64-column tile is exactly one attention head within Q, K, or V.
    // Rows are independent; the accumulator order stays K-major FMA.
    (0..DA3_BASE_TOKENS_504X336)
        .step_by(ROWS)
        .collect::<Vec<_>>()
        .into_par_iter()
        .for_each(|row0| {
            let rows = (DA3_BASE_TOKENS_504X336 - row0).min(ROWS);
            for col0 in (0..N).step_by(64) {
                let mut acc = [[_mm512_setzero_ps(); 4]; ROWS];
                for input in 0..K {
                    let bp = unsafe { weight.as_ptr().add(input * N + col0) };
                    let bv = unsafe {
                        [
                            _mm512_loadu_ps(bp),
                            _mm512_loadu_ps(bp.add(16)),
                            _mm512_loadu_ps(bp.add(32)),
                            _mm512_loadu_ps(bp.add(48)),
                        ]
                    };
                    for row in 0..rows {
                        let av = _mm512_set1_ps(a[(row0 + row) * K + input]);
                        for block in 0..4 {
                            acc[row][block] = _mm512_fmadd_ps(av, bv[block], acc[row][block]);
                        }
                    }
                }
                let group = col0 / 768;
                let head = (col0 % 768) / 64;
                for row in 0..rows {
                    let destination = match group {
                        0 => q_ptr,
                        1 => k_ptr,
                        _ => v_ptr,
                    } as *mut f32;
                    let p = unsafe {
                        destination.add((head * DA3_BASE_TOKENS_504X336 + row0 + row) * 64)
                    };
                    for block in 0..4 {
                        let b = unsafe { _mm512_loadu_ps(bias.as_ptr().add(col0 + block * 16)) };
                        unsafe {
                            _mm512_storeu_ps(p.add(block * 16), _mm512_add_ps(acc[row][block], b));
                        }
                    }
                }
            }
        });
}

/// Column-panel work partition for DA3's direct QKV projection.  A 64-column
/// panel maps exactly to one head in Q, K, or V, so each Rayon task writes a
/// disjoint head-major output region while retaining K-ascending FMAs.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,fma")]
unsafe fn qkv_avx512_column_split(
    a: &[f32],
    weight: &[f32],
    bias: &[f32],
    q: &mut [f32],
    k: &mut [f32],
    v: &mut [f32],
) {
    use core::arch::x86_64::*;
    use rayon::prelude::*;
    const ROWS: usize = 6;
    const N: usize = 2304;
    const K: usize = 768;
    let q_ptr = q.as_mut_ptr() as usize;
    let k_ptr = k.as_mut_ptr() as usize;
    let v_ptr = v.as_mut_ptr() as usize;

    (0..N)
        .step_by(64)
        .collect::<Vec<_>>()
        .into_par_iter()
        .for_each(|col0| {
            let group = col0 / 768;
            let head = (col0 % 768) / 64;
            let destination = match group {
                0 => q_ptr,
                1 => k_ptr,
                _ => v_ptr,
            } as *mut f32;
            for row0 in (0..DA3_BASE_TOKENS_504X336).step_by(ROWS) {
                let rows = (DA3_BASE_TOKENS_504X336 - row0).min(ROWS);
                let mut acc = [[_mm512_setzero_ps(); 4]; ROWS];
                for input in 0..K {
                    let bp = unsafe { weight.as_ptr().add(input * N + col0) };
                    let bv = unsafe {
                        [
                            _mm512_loadu_ps(bp),
                            _mm512_loadu_ps(bp.add(16)),
                            _mm512_loadu_ps(bp.add(32)),
                            _mm512_loadu_ps(bp.add(48)),
                        ]
                    };
                    for row in 0..rows {
                        let av = _mm512_set1_ps(a[(row0 + row) * K + input]);
                        for block in 0..4 {
                            acc[row][block] = _mm512_fmadd_ps(av, bv[block], acc[row][block]);
                        }
                    }
                }
                for row in 0..rows {
                    let p = unsafe {
                        destination.add((head * DA3_BASE_TOKENS_504X336 + row0 + row) * 64)
                    };
                    for block in 0..4 {
                        let bias_v =
                            unsafe { _mm512_loadu_ps(bias.as_ptr().add(col0 + block * 16)) };
                        unsafe {
                            _mm512_storeu_ps(
                                p.add(block * 16),
                                _mm512_add_ps(acc[row][block], bias_v),
                            );
                        }
                    }
                }
            }
        });
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,fma")]
unsafe fn linear_avx512(m: usize, n: usize, k: usize, a: &[f32], b: &[f32], c: &mut [f32]) {
    match linear_rows() {
        4 => unsafe { linear_avx512_rows::<4>(m, n, k, a, b, c) },
        8 => unsafe { linear_avx512_rows::<8>(m, n, k, a, b, c) },
        _ => unsafe { linear_avx512_rows::<6>(m, n, k, a, b, c) },
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,fma")]
unsafe fn linear_avx512_rows<const ROWS: usize>(
    m: usize,
    n: usize,
    k: usize,
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
) {
    use core::arch::x86_64::*;
    use rayon::prelude::*;
    c.par_chunks_mut(ROWS * n)
        .enumerate()
        .for_each(|(tile, c_tile)| {
            let row0 = tile * ROWS;
            let rows = (m - row0).min(ROWS);
            for col0 in (0..n).step_by(64) {
                let mut acc = [[_mm512_setzero_ps(); 4]; ROWS];
                for kk in 0..k {
                    let bp = unsafe { b.as_ptr().add(kk * n + col0) };
                    let bv = unsafe {
                        [
                            _mm512_loadu_ps(bp),
                            _mm512_loadu_ps(bp.add(16)),
                            _mm512_loadu_ps(bp.add(32)),
                            _mm512_loadu_ps(bp.add(48)),
                        ]
                    };
                    for row in 0..rows {
                        let av = _mm512_set1_ps(a[(row0 + row) * k + kk]);
                        for block in 0..4 {
                            acc[row][block] = _mm512_fmadd_ps(av, bv[block], acc[row][block]);
                        }
                    }
                }
                for row in 0..rows {
                    for block in 0..4 {
                        unsafe {
                            _mm512_storeu_ps(
                                c_tile.as_mut_ptr().add(row * n + col0 + block * 16),
                                acc[row][block],
                            )
                        };
                    }
                }
            }
        });
}

/// Column-panel work partition for the same fixed-shape projection GEMMs.
/// It keeps a 64-column weight panel private to one worker, trading repeated
/// small activation scans for substantially less shared-L3 weight traffic.
/// Every output still accumulates K in ascending order with the same FMAs.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,fma")]
unsafe fn linear_avx512_column_split(
    m: usize,
    n: usize,
    k: usize,
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
) {
    match linear_rows() {
        4 => unsafe { linear_avx512_column_split_rows::<4>(m, n, k, a, b, c) },
        8 => unsafe { linear_avx512_column_split_rows::<8>(m, n, k, a, b, c) },
        _ => unsafe { linear_avx512_column_split_rows::<6>(m, n, k, a, b, c) },
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,fma")]
unsafe fn linear_avx512_column_split_rows<const ROWS: usize>(
    m: usize,
    n: usize,
    k: usize,
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
) {
    use core::arch::x86_64::*;
    use rayon::prelude::*;
    let c_ptr = c.as_mut_ptr() as usize;
    (0..n)
        .step_by(64)
        .collect::<Vec<_>>()
        .into_par_iter()
        .for_each(|col0| {
            for row0 in (0..m).step_by(ROWS) {
                let rows = (m - row0).min(ROWS);
                let mut acc = [[_mm512_setzero_ps(); 4]; ROWS];
                for kk in 0..k {
                    let bp = unsafe { b.as_ptr().add(kk * n + col0) };
                    let bv = unsafe {
                        [
                            _mm512_loadu_ps(bp),
                            _mm512_loadu_ps(bp.add(16)),
                            _mm512_loadu_ps(bp.add(32)),
                            _mm512_loadu_ps(bp.add(48)),
                        ]
                    };
                    for row in 0..rows {
                        let av = _mm512_set1_ps(a[(row0 + row) * k + kk]);
                        for block in 0..4 {
                            acc[row][block] = _mm512_fmadd_ps(av, bv[block], acc[row][block]);
                        }
                    }
                }
                for row in 0..rows {
                    for block in 0..4 {
                        unsafe {
                            _mm512_storeu_ps(
                                (c_ptr as *mut f32).add((row0 + row) * n + col0 + block * 16),
                                acc[row][block],
                            );
                        }
                    }
                }
            }
        });
}

/// The same 4-query × 64-key streaming tile used by ggml's CPU flash path.
/// K is packed once per tile into a dimension-major layout; a ZMM then scores
/// sixteen keys at once. The online softmax keeps the memory footprint O(ND).
#[inline]
fn pack_key_head_dim_major(k_head: &[f32], tokens: usize, stride: usize) -> Vec<f32> {
    const D: usize = 64;
    debug_assert_eq!(k_head.len(), tokens * D);
    debug_assert!(stride >= tokens);
    let mut packed = vec![0.0f32; D * stride];
    for key in 0..tokens {
        for dim in 0..D {
            packed[dim * stride + key] = k_head[key * D + dim];
        }
    }
    packed
}

const FLASH_QUERY_TILE: usize = 8;

struct FlashProfile {
    enabled: bool,
    k_pack_ns: AtomicU64,
    qk_gemm_ns: AtomicU64,
    softmax_ns: AtomicU64,
    v_gemm_ns: AtomicU64,
}

impl FlashProfile {
    fn from_env() -> Self {
        Self {
            enabled: std::env::var_os("DA3_FLASH_PROFILE").is_some(),
            k_pack_ns: AtomicU64::new(0),
            qk_gemm_ns: AtomicU64::new(0),
            softmax_ns: AtomicU64::new(0),
            v_gemm_ns: AtomicU64::new(0),
        }
    }

    fn add(counter: &AtomicU64, started: Instant) {
        counter.fetch_add(started.elapsed().as_nanos() as u64, Ordering::Relaxed);
    }

    fn report(&self, started: Instant) {
        eprintln!(
            "da3_flash_profile total_ms={:.3} k_pack_sum_ms={:.3} qk_gemm_sum_ms={:.3} softmax_sum_ms={:.3} v_gemm_sum_ms={:.3}",
            started.elapsed().as_secs_f64() * 1_000.0,
            self.k_pack_ns.load(Ordering::Relaxed) as f64 / 1_000_000.0,
            self.qk_gemm_ns.load(Ordering::Relaxed) as f64 / 1_000_000.0,
            self.softmax_ns.load(Ordering::Relaxed) as f64 / 1_000_000.0,
            self.v_gemm_ns.load(Ordering::Relaxed) as f64 / 1_000_000.0,
        );
    }
}

fn flash_query_tile_from_env(value: Option<&str>) -> usize {
    match value {
        Some("4") => 4,
        // QT12 is intentionally A/B-only.  It sits between the established
        // eight-query control and the already measured 16/20-query variants;
        // it changes scheduling/cache granularity only, not tile arithmetic.
        Some("12") => 12,
        Some("16") => 16,
        Some("20") => 20,
        _ => FLASH_QUERY_TILE,
    }
}

fn flash_query_tile() -> usize {
    flash_query_tile_from_env(
        std::env::var("DA3_KERNELS_FLASH_QUERY_TILE")
            .ok()
            .as_deref(),
    )
}

fn flash_head_only_from_env(value: Option<&str>) -> bool {
    value.is_some()
}

fn flash_nested_super16_from_env(value: Option<&str>) -> bool {
    value.is_some()
}

fn flash_query_tile_count(tokens: usize) -> usize {
    tokens.div_ceil(FLASH_QUERY_TILE)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,fma")]
unsafe fn flash_attention_tile_avx512<const QT: usize>(
    q_head: &[f32],
    k_head: &[f32],
    v_head: &[f32],
    packed_k_head: &[f32],
    packed_k_stride: usize,
    tokens: usize,
    q0: usize,
    out_tile: &mut [f32],
    profile: &FlashProfile,
    gemm_rows6: bool,
    gemm_8x32: bool,
) {
    const D: usize = 64;
    const KVT: usize = 64;
    let scale = 1.0f32 / 8.0;
    let rows = out_tile.len() / D;
    debug_assert!(rows <= QT);
    let use_8x32 = rows == 8 && gemm_8x32;
    let mut accum = [[0.0f32; D]; QT];
    let mut sums = [0.0f32; QT];
    let mut maxima = [f32::NEG_INFINITY; QT];

    for k0 in (0..tokens).step_by(KVT) {
        let cols = (tokens - k0).min(KVT);
        let mut scores = [[0.0f32; KVT]; QT];
        let qk_started = profile.enabled.then(Instant::now);
        if packed_k_head.is_empty() {
            let mut packed_k = [0.0f32; KVT * D];
            for key in 0..cols {
                let source = &k_head[(k0 + key) * D..(k0 + key + 1) * D];
                for dim in 0..D {
                    packed_k[dim * KVT + key] = source[dim];
                }
            }
            // SAFETY: fixed 64-wide packed matrices, and `rows` never
            // exceeds the query tile.
            unsafe {
                let scores =
                    core::slice::from_raw_parts_mut(scores.as_mut_ptr().cast::<f32>(), QT * KVT);
                if use_8x32 && cols == KVT {
                    gemm_8x32_overwrite(&q_head[q0 * D..(q0 + rows) * D], &packed_k, scores);
                } else if gemm_rows6 {
                    gemm_6x64_overwrite(rows, &q_head[q0 * D..(q0 + rows) * D], &packed_k, scores);
                } else {
                    gemm_4x64_overwrite(rows, &q_head[q0 * D..(q0 + rows) * D], &packed_k, scores);
                }
            }
        } else {
            // SAFETY: masked loads keep the dense final key tile in-bounds.
            unsafe {
                let scores =
                    core::slice::from_raw_parts_mut(scores.as_mut_ptr().cast::<f32>(), QT * KVT);
                if use_8x32 && cols == KVT {
                    gemm_8x32_overwrite_stride(
                        &q_head[q0 * D..(q0 + rows) * D],
                        packed_k_head,
                        packed_k_stride,
                        k0,
                        scores,
                    );
                } else if gemm_rows6 {
                    gemm_6x64_overwrite_stride(
                        rows,
                        &q_head[q0 * D..(q0 + rows) * D],
                        packed_k_head,
                        packed_k_stride,
                        k0,
                        cols,
                        scores,
                    );
                } else {
                    gemm_4x64_overwrite_stride(
                        rows,
                        &q_head[q0 * D..(q0 + rows) * D],
                        packed_k_head,
                        packed_k_stride,
                        k0,
                        cols,
                        scores,
                    );
                }
            }
        }
        if let Some(started) = qk_started {
            FlashProfile::add(&profile.qk_gemm_ns, started);
        }

        let softmax_started = profile.enabled.then(Instant::now);
        for row in 0..rows {
            let mut tile_max = f32::NEG_INFINITY;
            for value in &mut scores[row][..cols] {
                *value *= scale;
                tile_max = tile_max.max(*value);
            }
            let new_max = maxima[row].max(tile_max);
            let correction = if maxima[row].is_finite() {
                (maxima[row] - new_max).exp()
            } else {
                0.0
            };
            sums[row] *= correction;
            for value in &mut accum[row] {
                *value *= correction;
            }
            for value in &mut scores[row][..cols] {
                *value -= new_max;
            }
            let score_row = &mut scores[row];
            // SAFETY: every score row is the fixed 64-key tile.
            unsafe { exp_64_avx512(score_row) };
            for value in &score_row[..cols] {
                sums[row] += *value;
            }
            maxima[row] = new_max;
        }
        if let Some(started) = softmax_started {
            FlashProfile::add(&profile.softmax_ns, started);
        }

        // SAFETY: scores and V are 64-column matrices and the accumulator has
        // one 64-float row per valid query.
        let v_gemm_started = profile.enabled.then(Instant::now);
        unsafe {
            if cols == KVT {
                let scores = core::slice::from_raw_parts(scores.as_ptr().cast::<f32>(), QT * KVT);
                let accum =
                    core::slice::from_raw_parts_mut(accum.as_mut_ptr().cast::<f32>(), QT * D);
                if use_8x32 {
                    gemm_8x32_accumulate(scores, &v_head[k0 * D..(k0 + KVT) * D], accum)
                } else if gemm_rows6 {
                    gemm_6x64_accumulate(rows, scores, &v_head[k0 * D..(k0 + KVT) * D], accum)
                } else {
                    gemm_4x64_accumulate(rows, scores, &v_head[k0 * D..(k0 + KVT) * D], accum)
                }
            } else {
                let mut packed_v = [0.0f32; KVT * D];
                for key in 0..cols {
                    packed_v[key * D..(key + 1) * D]
                        .copy_from_slice(&v_head[(k0 + key) * D..(k0 + key + 1) * D]);
                }
                let scores = core::slice::from_raw_parts(scores.as_ptr().cast::<f32>(), QT * KVT);
                let accum =
                    core::slice::from_raw_parts_mut(accum.as_mut_ptr().cast::<f32>(), QT * D);
                if gemm_rows6 {
                    gemm_6x64_accumulate(rows, scores, &packed_v, accum)
                } else {
                    gemm_4x64_accumulate(rows, scores, &packed_v, accum)
                }
            }
        };
        if let Some(started) = v_gemm_started {
            FlashProfile::add(&profile.v_gemm_ns, started);
        }
    }

    for row in 0..rows {
        let inverse_sum = 1.0 / sums[row];
        for dim in 0..D {
            out_tile[row * D + dim] = accum[row][dim] * inverse_sum;
        }
    }
}

/// Hot DA3-BASE flash path for a complete eight-query tile with persistent,
/// dimension-major K packing.  Keeping this separate from the diagnostic
/// fallback above is intentional: the generic helper must reserve stack
/// storage for per-tile K/V packing even though production never takes that
/// branch.  There are 108 such full query tiles per head at 865 tokens.
///
/// The arithmetic order is deliberately identical to the QT8 generic route:
/// key panels and K dimensions remain ascending, and the short final key
/// panel still performs all 64 FMA steps with zero-filled V rows.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,fma")]
unsafe fn flash_attention_tile_packed_8x32_avx512(
    q_head: &[f32],
    v_head: &[f32],
    packed_k_head: &[f32],
    packed_k_stride: usize,
    tokens: usize,
    q0: usize,
    out_tile: &mut [f32],
    profile: &FlashProfile,
) {
    use core::mem::MaybeUninit;

    const D: usize = 64;
    const KVT: usize = 64;
    debug_assert_eq!(out_tile.len(), 8 * D);
    debug_assert!(q0 + 8 <= tokens);
    debug_assert_eq!(packed_k_head.len(), D * packed_k_stride);

    let scale = 1.0f32 / 8.0;
    let mut accum = [[0.0f32; D]; 8];
    let mut sums = [0.0f32; 8];
    let mut maxima = [f32::NEG_INFINITY; 8];

    for k0 in (0..tokens).step_by(KVT) {
        let cols = (tokens - k0).min(KVT);
        // Both overwrite kernels write every score element, including masked
        // tail lanes. Avoiding eager zero-initialisation is safe here and
        // removes another 2 KiB clear from each of the fourteen key panels.
        let mut scores = MaybeUninit::<[[f32; KVT]; 8]>::uninit();
        let score_ptr = scores.as_mut_ptr().cast::<f32>();
        let qk_started = profile.enabled.then(Instant::now);
        unsafe {
            gemm_8x32_overwrite_stride(
                &q_head[q0 * D..(q0 + 8) * D],
                packed_k_head,
                packed_k_stride,
                k0,
                core::slice::from_raw_parts_mut(score_ptr, 8 * KVT),
            );
        }
        if let Some(started) = qk_started {
            FlashProfile::add(&profile.qk_gemm_ns, started);
        }
        // The overwrite kernel above assigned all 8×64 elements (masked key
        // tail lanes are explicitly zeroed), so it is now sound to access the
        // MaybeUninit backing storage as an initialized score matrix.
        let scores = unsafe { &mut *scores.as_mut_ptr() };

        let softmax_started = profile.enabled.then(Instant::now);
        for row in 0..8 {
            let mut tile_max = f32::NEG_INFINITY;
            for value in &mut scores[row][..cols] {
                *value *= scale;
                tile_max = tile_max.max(*value);
            }
            let new_max = maxima[row].max(tile_max);
            let correction = if maxima[row].is_finite() {
                (maxima[row] - new_max).exp()
            } else {
                0.0
            };
            sums[row] *= correction;
            for value in &mut accum[row] {
                *value *= correction;
            }
            for value in &mut scores[row][..cols] {
                *value -= new_max;
            }
            unsafe { exp_64_avx512(&mut scores[row]) };
            for value in &scores[row][..cols] {
                sums[row] += *value;
            }
            maxima[row] = new_max;
        }
        if let Some(started) = softmax_started {
            FlashProfile::add(&profile.softmax_ns, started);
        }

        let v_gemm_started = profile.enabled.then(Instant::now);
        let score_flat =
            unsafe { core::slice::from_raw_parts(scores.as_ptr().cast::<f32>(), 8 * KVT) };
        let accum_flat =
            unsafe { core::slice::from_raw_parts_mut(accum.as_mut_ptr().cast::<f32>(), 8 * D) };
        unsafe {
            if cols == KVT {
                gemm_8x32_accumulate(score_flat, &v_head[k0 * D..(k0 + KVT) * D], accum_flat);
            } else {
                gemm_8x32_accumulate_zero_tail(
                    score_flat,
                    &v_head[k0 * D..(k0 + cols) * D],
                    cols,
                    accum_flat,
                );
            }
        }
        if let Some(started) = v_gemm_started {
            FlashProfile::add(&profile.v_gemm_ns, started);
        }
    }

    for row in 0..8 {
        let inverse_sum = 1.0 / sums[row];
        for dim in 0..D {
            out_tile[row * D + dim] = accum[row][dim] * inverse_sum;
        }
    }
}

/// The actual CPU tiling used by ggml's current F32 flash-attention path:
/// one task owns up to 64 query rows and streams the same 64-key panels
/// through two 64x64 GEMMs. Unlike the historical QT8 path, all query state
/// is shared by one coarse work item, so the scheduler sees exactly
/// `heads * ceil(tokens/64)` independent jobs.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,fma")]
unsafe fn flash_attention_ggml64_tile_avx512(
    q_head: &[f32],
    v_head: &[f32],
    packed_k_head: &[f32],
    packed_k_stride: usize,
    tokens: usize,
    q0: usize,
    out_tile: &mut [f32],
    profile: &FlashProfile,
) {
    const D: usize = 64;
    const QT: usize = 64;
    const KVT: usize = 64;
    let rows = out_tile.len() / D;
    debug_assert!((1..=QT).contains(&rows));
    let mut accum = [[0.0f32; D]; QT];
    let mut sums = [0.0f32; QT];
    let mut maxima = [f32::NEG_INFINITY; QT];

    for k0 in (0..tokens).step_by(KVT) {
        let cols = (tokens - k0).min(KVT);
        let mut scores = [[0.0f32; KVT]; QT];
        let qk_started = profile.enabled.then(Instant::now);
        unsafe {
            gemm_4x64_overwrite_stride(
                rows,
                &q_head[q0 * D..(q0 + rows) * D],
                packed_k_head,
                packed_k_stride,
                k0,
                cols,
                core::slice::from_raw_parts_mut(scores.as_mut_ptr().cast::<f32>(), QT * KVT),
            );
        }
        if let Some(started) = qk_started {
            FlashProfile::add(&profile.qk_gemm_ns, started);
        }

        let softmax_started = profile.enabled.then(Instant::now);
        for row in 0..rows {
            let mut tile_max = f32::NEG_INFINITY;
            for score in &mut scores[row][..cols] {
                *score *= 0.125;
                tile_max = tile_max.max(*score);
            }
            let new_max = maxima[row].max(tile_max);
            let correction = if maxima[row].is_finite() {
                (maxima[row] - new_max).exp()
            } else {
                0.0
            };
            sums[row] *= correction;
            for value in &mut accum[row] {
                *value *= correction;
            }
            for score in &mut scores[row][..cols] {
                *score -= new_max;
            }
            unsafe { exp_64_avx512(&mut scores[row]) };
            for score in &scores[row][..cols] {
                sums[row] += *score;
            }
            maxima[row] = new_max;
        }
        if let Some(started) = softmax_started {
            FlashProfile::add(&profile.softmax_ns, started);
        }

        let v_gemm_started = profile.enabled.then(Instant::now);
        let scores =
            unsafe { core::slice::from_raw_parts(scores.as_ptr().cast::<f32>(), QT * KVT) };
        let accum =
            unsafe { core::slice::from_raw_parts_mut(accum.as_mut_ptr().cast::<f32>(), QT * D) };
        if cols == KVT {
            unsafe { gemm_4x64_accumulate(rows, scores, &v_head[k0 * D..(k0 + KVT) * D], accum) };
        } else {
            let mut packed_v = [0.0f32; KVT * D];
            for key in 0..cols {
                packed_v[key * D..(key + 1) * D]
                    .copy_from_slice(&v_head[(k0 + key) * D..(k0 + key + 1) * D]);
            }
            unsafe { gemm_4x64_accumulate(rows, scores, &packed_v, accum) };
        }
        if let Some(started) = v_gemm_started {
            FlashProfile::add(&profile.v_gemm_ns, started);
        }
    }

    for row in 0..rows {
        let inverse = 1.0 / sums[row];
        for dim in 0..D {
            out_tile[row * D + dim] = accum[row][dim] * inverse;
        }
    }
}

/// Four QT8 query tiles share each packed 64-key panel. Per-query state stays
/// private, and every row still visits K panels in ascending order.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,fma")]
unsafe fn flash_attention_superblock32_avx512(
    q_head: &[f32],
    v_head: &[f32],
    packed_k: &[f32],
    stride: usize,
    tokens: usize,
    q0: usize,
    out: &mut [f32],
    gemm_rows6: bool,
) {
    const D: usize = 64;
    const QT: usize = 8;
    const KVT: usize = 64;
    const SUB: usize = 4;
    let total_rows = out.len() / D;
    let subtiles = total_rows.div_ceil(QT);
    let mut accum = [[[0.0f32; D]; QT]; SUB];
    let mut sums = [[0.0f32; QT]; SUB];
    let mut maxima = [[f32::NEG_INFINITY; QT]; SUB];
    for k0 in (0..tokens).step_by(KVT) {
        let cols = (tokens - k0).min(KVT);
        for sub in 0..subtiles {
            let row0 = sub * QT;
            let rows = (total_rows - row0).min(QT);
            let mut scores = [[0.0f32; KVT]; QT];
            let score_flat = unsafe {
                core::slice::from_raw_parts_mut(scores.as_mut_ptr().cast::<f32>(), QT * KVT)
            };
            if gemm_rows6 {
                unsafe {
                    gemm_6x64_overwrite_stride(
                        rows,
                        &q_head[(q0 + row0) * D..(q0 + row0 + rows) * D],
                        packed_k,
                        stride,
                        k0,
                        cols,
                        score_flat,
                    )
                };
            } else {
                unsafe {
                    gemm_4x64_overwrite_stride(
                        rows,
                        &q_head[(q0 + row0) * D..(q0 + row0 + rows) * D],
                        packed_k,
                        stride,
                        k0,
                        cols,
                        score_flat,
                    )
                };
            }
            for row in 0..rows {
                let mut tile_max = f32::NEG_INFINITY;
                for value in &mut scores[row][..cols] {
                    *value *= 0.125;
                    tile_max = tile_max.max(*value);
                }
                let new_max = maxima[sub][row].max(tile_max);
                let correction = if maxima[sub][row].is_finite() {
                    (maxima[sub][row] - new_max).exp()
                } else {
                    0.0
                };
                sums[sub][row] *= correction;
                for value in &mut accum[sub][row] {
                    *value *= correction;
                }
                for value in &mut scores[row][..cols] {
                    *value -= new_max;
                }
                unsafe { exp_64_avx512(&mut scores[row]) };
                for value in &scores[row][..cols] {
                    sums[sub][row] += *value;
                }
                maxima[sub][row] = new_max;
            }
            let score_flat =
                unsafe { core::slice::from_raw_parts(scores.as_ptr().cast::<f32>(), QT * KVT) };
            let accum_flat = unsafe {
                core::slice::from_raw_parts_mut(accum[sub].as_mut_ptr().cast::<f32>(), QT * D)
            };
            if cols == KVT {
                let v = &v_head[k0 * D..(k0 + KVT) * D];
                if gemm_rows6 {
                    unsafe { gemm_6x64_accumulate(rows, score_flat, v, accum_flat) };
                } else {
                    unsafe { gemm_4x64_accumulate(rows, score_flat, v, accum_flat) };
                }
            } else {
                let mut packed_v = [0.0f32; KVT * D];
                for key in 0..cols {
                    packed_v[key * D..(key + 1) * D]
                        .copy_from_slice(&v_head[(k0 + key) * D..(k0 + key + 1) * D]);
                }
                if gemm_rows6 {
                    unsafe { gemm_6x64_accumulate(rows, score_flat, &packed_v, accum_flat) };
                } else {
                    unsafe { gemm_4x64_accumulate(rows, score_flat, &packed_v, accum_flat) };
                }
            }
        }
    }
    for sub in 0..subtiles {
        for row in 0..(total_rows - sub * QT).min(QT) {
            let inverse = 1.0 / sums[sub][row];
            for dim in 0..D {
                out[(sub * QT + row) * D + dim] = accum[sub][row][dim] * inverse;
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,fma")]
unsafe fn flash_attention_avx512(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    heads: usize,
    tokens: usize,
    out: &mut [f32],
) {
    use rayon::prelude::*;

    const D: usize = 64;
    let profile = FlashProfile::from_env();
    let profile_started = profile.enabled.then(Instant::now);
    // The final logical tile is shorter than 64 keys. Its score loads use
    // AVX-512 masks, so the persistent K layout can stay dense.
    let packed_k_stride = tokens;
    let head_pack_disabled = std::env::var_os("DA3_KERNELS_DISABLE_FLASH_HEAD_PACK").is_some();
    let gemm_rows6 = std::env::var_os("DA3_KERNELS_FLASH_GEMM_ROWS6").is_some();
    // Eight query rows by two 32-column panels is the production DA3-BASE
    // flash shape: it keeps 16 accumulators live without the register spill
    // of 6x64, while sharing each K/V panel across the whole QT8 tile.  The
    // old 4x64 path remains available for diagnostic A/B through this
    // explicit disable switch.
    let gemm_8x32 = std::env::var_os("DA3_KERNELS_DISABLE_FLASH_GEMM_8X32").is_none();
    // The packed QT8 implementation is an experimental single-view fast path.
    // It is intentionally opt-in: multi-view inference alternates local and
    // global attention and must never enter an unvalidated unsafe kernel.
    // `DA3_KERNELS_FLASH_PACKED_QT8=1` is reserved for isolated benchmark
    // experiments with explicit parity and crash testing.
    let packed_qt8 = gemm_8x32 && std::env::var_os("DA3_KERNELS_FLASH_PACKED_QT8").is_some();
    let nested_super16 = flash_nested_super16_from_env(
        std::env::var("DA3_KERNELS_FLASH_NESTED_SUPER16")
            .ok()
            .as_deref(),
    );
    let head_only =
        flash_head_only_from_env(std::env::var("DA3_KERNELS_FLASH_HEAD_ONLY").ok().as_deref());

    let superblock_queries = if std::env::var_os("DA3_KERNELS_FLASH_SUPERBLOCK32").is_some() {
        Some(32)
    } else if std::env::var_os("DA3_KERNELS_FLASH_SUPERBLOCK16").is_some() {
        Some(16)
    } else {
        None
    };
    if std::env::var_os("DA3_KERNELS_FLASH_GGML64").is_some() {
        const QT: usize = 64;
        let q_blocks = tokens.div_ceil(QT);
        let packed_k_heads: Vec<Vec<f32>> = (0..heads)
            .into_par_iter()
            .map(|head| {
                let base = head * tokens * D;
                let pack_started = profile.enabled.then(Instant::now);
                let packed =
                    pack_key_head_dim_major(&k[base..base + tokens * D], tokens, packed_k_stride);
                if let Some(started) = pack_started {
                    FlashProfile::add(&profile.k_pack_ns, started);
                }
                packed
            })
            .collect();
        let out_ptr = out.as_mut_ptr() as usize;
        (0..heads * q_blocks).into_par_iter().for_each(|job| {
            let head = job / q_blocks;
            let block = job % q_blocks;
            let q0 = block * QT;
            let rows = (tokens - q0).min(QT);
            let base = head * tokens * D;
            let out_offset = base + q0 * D;
            // SAFETY: `(head, block)` identifies one unique contiguous
            // output range. All source buffers and packed panels are shared
            // immutably across the flat dynamic Rayon work queue.
            let out_tile = unsafe {
                core::slice::from_raw_parts_mut((out_ptr as *mut f32).add(out_offset), rows * D)
            };
            unsafe {
                flash_attention_ggml64_tile_avx512(
                    &q[base..base + tokens * D],
                    &v[base..base + tokens * D],
                    &packed_k_heads[head],
                    packed_k_stride,
                    tokens,
                    q0,
                    out_tile,
                    &profile,
                )
            };
        });
        if let Some(started) = profile_started {
            profile.report(started);
        }
        return;
    }
    if let Some(superblock_queries) = superblock_queries.filter(|_| !head_pack_disabled) {
        let packed_k_heads: Vec<Vec<f32>> = (0..heads)
            .into_par_iter()
            .map(|head| {
                let base = head * tokens * D;
                pack_key_head_dim_major(&k[base..base + tokens * D], tokens, packed_k_stride)
            })
            .collect();
        out.par_chunks_mut(tokens * D)
            .enumerate()
            .take(heads)
            .flat_map(|(head, out_head)| {
                let base = head * tokens * D;
                let q_head = &q[base..base + tokens * D];
                let v_head = &v[base..base + tokens * D];
                let packed_k = &packed_k_heads[head];
                out_head
                    .par_chunks_mut(superblock_queries * D)
                    .enumerate()
                    .map(move |(block, out_block)| unsafe {
                        flash_attention_superblock32_avx512(
                            q_head,
                            v_head,
                            packed_k,
                            packed_k_stride,
                            tokens,
                            block * superblock_queries,
                            out_block,
                            gemm_rows6,
                        );
                    })
            })
            .for_each(|_| {});
        if let Some(started) = profile_started {
            profile.report(started);
        }
        return;
    }

    if std::env::var_os("DA3_KERNELS_FLAT_FLASH_TILES").is_some() {
        // Phase one keeps K packing separate from execution. Phase two is one
        // flattened Rayon iterator over all head/query tiles, avoiding nested
        // scheduler entry while retaining the same per-tile arithmetic.
        let packed_k_heads: Vec<Vec<f32>> = (0..heads)
            .into_par_iter()
            .map(|head| {
                let base = head * tokens * D;
                let pack_started = profile.enabled.then(Instant::now);
                let mut packed =
                    pack_key_head_dim_major(&k[base..base + tokens * D], tokens, packed_k_stride);
                if let Some(started) = pack_started {
                    FlashProfile::add(&profile.k_pack_ns, started);
                }
                if head_pack_disabled {
                    packed.clear();
                }
                packed
            })
            .collect();
        let profile_ref = &profile;

        out.par_chunks_mut(tokens * D)
            .enumerate()
            .take(heads)
            .flat_map(|(head, out_head)| {
                let base = head * tokens * D;
                let q_head = &q[base..base + tokens * D];
                let k_head = &k[base..base + tokens * D];
                let v_head = &v[base..base + tokens * D];
                let packed_k_head = &packed_k_heads[head];
                out_head
                    .par_chunks_mut(FLASH_QUERY_TILE * D)
                    .enumerate()
                    .map(move |(tile, out_tile)| {
                        // SAFETY: every parallel item owns a disjoint query
                        // tile; all input and packed-K slices are immutable.
                        unsafe {
                            flash_attention_tile_avx512::<FLASH_QUERY_TILE>(
                                q_head,
                                k_head,
                                v_head,
                                packed_k_head,
                                packed_k_stride,
                                tokens,
                                tile * FLASH_QUERY_TILE,
                                out_tile,
                                profile_ref,
                                gemm_rows6,
                                gemm_8x32,
                            );
                        }
                    })
            })
            .for_each(|_| {});
        if let Some(started) = profile_started {
            profile.report(started);
        }
        return;
    }

    let query_tile = flash_query_tile();
    out.par_chunks_mut(tokens * D)
        .enumerate()
        .take(heads)
        .for_each(|(head, out_head)| {
            let base = head * tokens * D;
            let q_head = &q[base..base + tokens * D];
            let k_head = &k[base..base + tokens * D];
            let v_head = &v[base..base + tokens * D];

            // K is immutable for every query tile of this head. Pack it once
            // in dimension-major form instead of repeating the same transpose
            // fourteen times. The final physical padding keeps 64-wide SIMD
            // loads memory-safe; its lanes are excluded by `cols`.
            let pack_started = profile.enabled.then(Instant::now);
            let mut packed_k_head = pack_key_head_dim_major(k_head, tokens, packed_k_stride);
            if let Some(started) = pack_started {
                FlashProfile::add(&profile.k_pack_ns, started);
            }
            if head_pack_disabled {
                // Retain the old per-tile path below when explicitly A/Bing.
                packed_k_head.clear();
            }

            if nested_super16 {
                // Keep the per-head K-pack locality, but schedule 16-query
                // K-major superblocks rather than individual QT8 tiles.
                out_head
                    .par_chunks_mut(16 * D)
                    .enumerate()
                    .for_each(|(block, out_block)| unsafe {
                        flash_attention_superblock32_avx512(
                            q_head,
                            v_head,
                            &packed_k_head,
                            packed_k_stride,
                            tokens,
                            block * 16,
                            out_block,
                            gemm_rows6,
                        );
                    });
            } else if head_only {
                // A/B candidate: one Rayon job per head, with its QT8 tiles
                // run serially to avoid nested tiny task scheduling.
                for (tile, out_tile) in out_head.chunks_mut(FLASH_QUERY_TILE * D).enumerate() {
                    unsafe {
                        flash_attention_tile_avx512::<FLASH_QUERY_TILE>(
                            q_head,
                            k_head,
                            v_head,
                            &packed_k_head,
                            packed_k_stride,
                            tokens,
                            tile * FLASH_QUERY_TILE,
                            out_tile,
                            &profile,
                            gemm_rows6,
                            gemm_8x32,
                        );
                    }
                }
            } else {
                // Preserve the established nested Rayon scheduling as the
                // default control for A/B evaluation.
                out_head
                    .par_chunks_mut(query_tile * D)
                    .enumerate()
                    .for_each(|(tile, out_tile)| {
                        // SAFETY: the nested scheduler gives each closure one
                        // disjoint output tile; all sources are immutable.
                        unsafe {
                            match query_tile {
                                4 => flash_attention_tile_avx512::<4>(
                                    q_head,
                                    k_head,
                                    v_head,
                                    &packed_k_head,
                                    packed_k_stride,
                                    tokens,
                                    tile * 4,
                                    out_tile,
                                    &profile,
                                    gemm_rows6,
                                    gemm_8x32,
                                ),
                                8 if packed_qt8 && out_tile.len() == 8 * D => {
                                    flash_attention_tile_packed_8x32_avx512(
                                        q_head,
                                        v_head,
                                        &packed_k_head,
                                        packed_k_stride,
                                        tokens,
                                        tile * 8,
                                        out_tile,
                                        &profile,
                                    )
                                }
                                8 => flash_attention_tile_avx512::<8>(
                                    q_head,
                                    k_head,
                                    v_head,
                                    &packed_k_head,
                                    packed_k_stride,
                                    tokens,
                                    tile * 8,
                                    out_tile,
                                    &profile,
                                    gemm_rows6,
                                    gemm_8x32,
                                ),
                                12 => flash_attention_tile_avx512::<12>(
                                    q_head,
                                    k_head,
                                    v_head,
                                    &packed_k_head,
                                    packed_k_stride,
                                    tokens,
                                    tile * 12,
                                    out_tile,
                                    &profile,
                                    gemm_rows6,
                                    gemm_8x32,
                                ),
                                16 => flash_attention_tile_avx512::<16>(
                                    q_head,
                                    k_head,
                                    v_head,
                                    &packed_k_head,
                                    packed_k_stride,
                                    tokens,
                                    tile * 16,
                                    out_tile,
                                    &profile,
                                    gemm_rows6,
                                    gemm_8x32,
                                ),
                                _ => flash_attention_tile_avx512::<FLASH_QUERY_TILE>(
                                    q_head,
                                    k_head,
                                    v_head,
                                    &packed_k_head,
                                    packed_k_stride,
                                    tokens,
                                    tile * FLASH_QUERY_TILE,
                                    out_tile,
                                    &profile,
                                    gemm_rows6,
                                    gemm_8x32,
                                ),
                            }
                        }
                    });
            }
        });

    if let Some(started) = profile_started {
        profile.report(started);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,fma")]
unsafe fn gemm_4x64_overwrite(m: usize, a: &[f32], b: &[f32], c: &mut [f32]) {
    use core::arch::x86_64::*;
    for row0 in (0..m).step_by(4) {
        let rows = (m - row0).min(4);
        let mut acc = [[_mm512_setzero_ps(); 4]; 4];
        for kk in 0..64 {
            let bp = unsafe { b.as_ptr().add(kk * 64) };
            let bv = unsafe {
                [
                    _mm512_loadu_ps(bp),
                    _mm512_loadu_ps(bp.add(16)),
                    _mm512_loadu_ps(bp.add(32)),
                    _mm512_loadu_ps(bp.add(48)),
                ]
            };
            for row in 0..rows {
                let av = _mm512_set1_ps(a[(row0 + row) * 64 + kk]);
                for block in 0..4 {
                    acc[row][block] = _mm512_fmadd_ps(av, bv[block], acc[row][block]);
                }
            }
        }
        for row in 0..rows {
            for block in 0..4 {
                unsafe {
                    _mm512_storeu_ps(
                        c.as_mut_ptr().add((row0 + row) * 64 + block * 16),
                        acc[row][block],
                    );
                }
            }
        }
    }
}

/// Full eight-query flash microkernel with two 32-column panels.  It retains
/// the same ascending-K FMA chain as the 4x64 control, but each B panel is
/// loaded once for all eight query rows.  Sixteen accumulators fit with two B
/// vectors on AVX-512, unlike the register-spilling 8x64 shape.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,fma")]
unsafe fn gemm_8x32_overwrite(a: &[f32], b: &[f32], c: &mut [f32]) {
    use core::arch::x86_64::*;
    debug_assert_eq!(a.len(), 8 * 64);
    debug_assert_eq!(b.len(), 64 * 64);
    debug_assert_eq!(c.len(), 8 * 64);
    for col0 in (0..64).step_by(32) {
        let mut acc = [[_mm512_setzero_ps(); 2]; 8];
        for kk in 0..64 {
            let bp = unsafe { b.as_ptr().add(kk * 64 + col0) };
            let b0 = unsafe { _mm512_loadu_ps(bp) };
            let b1 = unsafe { _mm512_loadu_ps(bp.add(16)) };
            for row in 0..8 {
                let av = _mm512_set1_ps(a[row * 64 + kk]);
                acc[row][0] = _mm512_fmadd_ps(av, b0, acc[row][0]);
                acc[row][1] = _mm512_fmadd_ps(av, b1, acc[row][1]);
            }
        }
        for row in 0..8 {
            unsafe {
                _mm512_storeu_ps(c.as_mut_ptr().add(row * 64 + col0), acc[row][0]);
                _mm512_storeu_ps(c.as_mut_ptr().add(row * 64 + col0 + 16), acc[row][1]);
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,fma")]
unsafe fn gemm_8x32_overwrite_stride(
    a: &[f32],
    b: &[f32],
    stride: usize,
    column: usize,
    c: &mut [f32],
) {
    use core::arch::x86_64::*;
    debug_assert_eq!(a.len(), 8 * 64);
    debug_assert!(b.len() >= 64 * stride);
    debug_assert!(column + 64 <= stride);
    debug_assert_eq!(c.len(), 8 * 64);
    for col0 in (0..64).step_by(32) {
        let mut acc = [[_mm512_setzero_ps(); 2]; 8];
        for kk in 0..64 {
            let bp = unsafe { b.as_ptr().add(kk * stride + column + col0) };
            let b0 = unsafe { _mm512_loadu_ps(bp) };
            let b1 = unsafe { _mm512_loadu_ps(bp.add(16)) };
            for row in 0..8 {
                let av = _mm512_set1_ps(a[row * 64 + kk]);
                acc[row][0] = _mm512_fmadd_ps(av, b0, acc[row][0]);
                acc[row][1] = _mm512_fmadd_ps(av, b1, acc[row][1]);
            }
        }
        for row in 0..8 {
            unsafe {
                _mm512_storeu_ps(c.as_mut_ptr().add(row * 64 + col0), acc[row][0]);
                _mm512_storeu_ps(c.as_mut_ptr().add(row * 64 + col0 + 16), acc[row][1]);
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,fma")]
unsafe fn gemm_6x64_overwrite(m: usize, a: &[f32], b: &[f32], c: &mut [f32]) {
    use core::arch::x86_64::*;
    for row0 in (0..m).step_by(6) {
        let rows = (m - row0).min(6);
        let mut acc = [[_mm512_setzero_ps(); 4]; 6];
        for kk in 0..64 {
            let bp = unsafe { b.as_ptr().add(kk * 64) };
            let bv = unsafe {
                [
                    _mm512_loadu_ps(bp),
                    _mm512_loadu_ps(bp.add(16)),
                    _mm512_loadu_ps(bp.add(32)),
                    _mm512_loadu_ps(bp.add(48)),
                ]
            };
            for row in 0..rows {
                let av = _mm512_set1_ps(a[(row0 + row) * 64 + kk]);
                for block in 0..4 {
                    acc[row][block] = _mm512_fmadd_ps(av, bv[block], acc[row][block]);
                }
            }
        }
        for row in 0..rows {
            for block in 0..4 {
                unsafe {
                    _mm512_storeu_ps(
                        c.as_mut_ptr().add((row0 + row) * 64 + block * 16),
                        acc[row][block],
                    );
                }
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,fma")]
unsafe fn gemm_4x64_overwrite_stride(
    m: usize,
    a: &[f32],
    b: &[f32],
    stride: usize,
    column: usize,
    columns: usize,
    c: &mut [f32],
) {
    use core::arch::x86_64::*;
    for row0 in (0..m).step_by(4) {
        let rows = (m - row0).min(4);
        let mut acc = [[_mm512_setzero_ps(); 4]; 4];
        for kk in 0..64 {
            let bp = unsafe { b.as_ptr().add(kk * stride + column) };
            let bv = if columns == 64 {
                unsafe {
                    [
                        _mm512_loadu_ps(bp),
                        _mm512_loadu_ps(bp.add(16)),
                        _mm512_loadu_ps(bp.add(32)),
                        _mm512_loadu_ps(bp.add(48)),
                    ]
                }
            } else {
                let mask_for = |offset: usize| -> __mmask16 {
                    let remaining = columns.saturating_sub(offset).min(16);
                    if remaining == 16 {
                        u16::MAX
                    } else {
                        (1u16 << remaining) - 1
                    }
                };
                unsafe {
                    [
                        _mm512_maskz_loadu_ps(mask_for(0), bp),
                        _mm512_maskz_loadu_ps(mask_for(16), bp.add(16)),
                        _mm512_maskz_loadu_ps(mask_for(32), bp.add(32)),
                        _mm512_maskz_loadu_ps(mask_for(48), bp.add(48)),
                    ]
                }
            };
            for row in 0..rows {
                let av = _mm512_set1_ps(a[(row0 + row) * 64 + kk]);
                for block in 0..4 {
                    acc[row][block] = _mm512_fmadd_ps(av, bv[block], acc[row][block]);
                }
            }
        }
        for row in 0..rows {
            for block in 0..4 {
                unsafe {
                    _mm512_storeu_ps(
                        c.as_mut_ptr().add((row0 + row) * 64 + block * 16),
                        acc[row][block],
                    );
                }
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,fma")]
unsafe fn gemm_6x64_overwrite_stride(
    m: usize,
    a: &[f32],
    b: &[f32],
    stride: usize,
    column: usize,
    columns: usize,
    c: &mut [f32],
) {
    use core::arch::x86_64::*;
    let mask_for = |offset: usize| -> __mmask16 {
        let remaining = columns.saturating_sub(offset).min(16);
        if remaining == 16 {
            u16::MAX
        } else {
            (1u16 << remaining) - 1
        }
    };
    for row0 in (0..m).step_by(6) {
        let rows = (m - row0).min(6);
        let mut acc = [[_mm512_setzero_ps(); 4]; 6];
        for kk in 0..64 {
            let bp = unsafe { b.as_ptr().add(kk * stride + column) };
            let bv = if columns == 64 {
                unsafe {
                    [
                        _mm512_loadu_ps(bp),
                        _mm512_loadu_ps(bp.add(16)),
                        _mm512_loadu_ps(bp.add(32)),
                        _mm512_loadu_ps(bp.add(48)),
                    ]
                }
            } else {
                unsafe {
                    [
                        _mm512_maskz_loadu_ps(mask_for(0), bp),
                        _mm512_maskz_loadu_ps(mask_for(16), bp.add(16)),
                        _mm512_maskz_loadu_ps(mask_for(32), bp.add(32)),
                        _mm512_maskz_loadu_ps(mask_for(48), bp.add(48)),
                    ]
                }
            };
            for row in 0..rows {
                let av = _mm512_set1_ps(a[(row0 + row) * 64 + kk]);
                for block in 0..4 {
                    acc[row][block] = _mm512_fmadd_ps(av, bv[block], acc[row][block]);
                }
            }
        }
        for row in 0..rows {
            for block in 0..4 {
                unsafe {
                    _mm512_storeu_ps(
                        c.as_mut_ptr().add((row0 + row) * 64 + block * 16),
                        acc[row][block],
                    );
                }
            }
        }
    }
}

/// C[m,64] += A[m,64] × B[64,64], with the same 4×64 AVX-512 tile shape as
/// ggml's CPU flash-attention helper. `m` is at most 64.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,fma")]
unsafe fn gemm_4x64_accumulate(m: usize, a: &[f32], b: &[f32], c: &mut [f32]) {
    use core::arch::x86_64::*;
    debug_assert!(m <= 128 && a.len() >= m * 64 && b.len() >= 64 * 64 && c.len() >= m * 64);
    for row0 in (0..m).step_by(4) {
        let rows = (m - row0).min(4);
        let mut acc = [[_mm512_setzero_ps(); 4]; 4];
        for row in 0..rows {
            for block in 0..4 {
                // SAFETY: all matrices have complete 64-wide rows.
                acc[row][block] =
                    unsafe { _mm512_loadu_ps(c.as_ptr().add((row0 + row) * 64 + block * 16)) };
            }
        }
        for kk in 0..64 {
            let bp = unsafe { b.as_ptr().add(kk * 64) };
            let bv = unsafe {
                [
                    _mm512_loadu_ps(bp),
                    _mm512_loadu_ps(bp.add(16)),
                    _mm512_loadu_ps(bp.add(32)),
                    _mm512_loadu_ps(bp.add(48)),
                ]
            };
            for row in 0..rows {
                let av = _mm512_set1_ps(a[(row0 + row) * 64 + kk]);
                for block in 0..4 {
                    acc[row][block] = _mm512_fmadd_ps(av, bv[block], acc[row][block]);
                }
            }
        }
        for row in 0..rows {
            for block in 0..4 {
                unsafe {
                    _mm512_storeu_ps(
                        c.as_mut_ptr().add((row0 + row) * 64 + block * 16),
                        acc[row][block],
                    )
                };
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,fma")]
unsafe fn gemm_8x32_accumulate(a: &[f32], b: &[f32], c: &mut [f32]) {
    use core::arch::x86_64::*;
    debug_assert_eq!(a.len(), 8 * 64);
    debug_assert_eq!(b.len(), 64 * 64);
    debug_assert_eq!(c.len(), 8 * 64);
    for col0 in (0..64).step_by(32) {
        let mut acc = [[_mm512_setzero_ps(); 2]; 8];
        for row in 0..8 {
            unsafe {
                acc[row][0] = _mm512_loadu_ps(c.as_ptr().add(row * 64 + col0));
                acc[row][1] = _mm512_loadu_ps(c.as_ptr().add(row * 64 + col0 + 16));
            }
        }
        for kk in 0..64 {
            let bp = unsafe { b.as_ptr().add(kk * 64 + col0) };
            let b0 = unsafe { _mm512_loadu_ps(bp) };
            let b1 = unsafe { _mm512_loadu_ps(bp.add(16)) };
            for row in 0..8 {
                let av = _mm512_set1_ps(a[row * 64 + kk]);
                acc[row][0] = _mm512_fmadd_ps(av, b0, acc[row][0]);
                acc[row][1] = _mm512_fmadd_ps(av, b1, acc[row][1]);
            }
        }
        for row in 0..8 {
            unsafe {
                _mm512_storeu_ps(c.as_mut_ptr().add(row * 64 + col0), acc[row][0]);
                _mm512_storeu_ps(c.as_mut_ptr().add(row * 64 + col0 + 16), acc[row][1]);
            }
        }
    }
}

/// The final DA3 key panel has 33 valid keys.  This directly models the
/// generic path's zero-padded V panel, retaining the same 64 FMA operations
/// without materialising a 16 KiB temporary buffer.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,fma")]
unsafe fn gemm_8x32_accumulate_zero_tail(a: &[f32], b: &[f32], cols: usize, c: &mut [f32]) {
    use core::arch::x86_64::*;
    debug_assert_eq!(a.len(), 8 * 64);
    debug_assert!(cols < 64);
    debug_assert_eq!(b.len(), cols * 64);
    debug_assert_eq!(c.len(), 8 * 64);
    let zero = _mm512_setzero_ps();
    for col0 in (0..64).step_by(32) {
        let mut acc = [[zero; 2]; 8];
        for row in 0..8 {
            unsafe {
                acc[row][0] = _mm512_loadu_ps(c.as_ptr().add(row * 64 + col0));
                acc[row][1] = _mm512_loadu_ps(c.as_ptr().add(row * 64 + col0 + 16));
            }
        }
        for kk in 0..64 {
            let (b0, b1) = if kk < cols {
                let bp = unsafe { b.as_ptr().add(kk * 64 + col0) };
                unsafe { (_mm512_loadu_ps(bp), _mm512_loadu_ps(bp.add(16))) }
            } else {
                (zero, zero)
            };
            for row in 0..8 {
                let av = _mm512_set1_ps(a[row * 64 + kk]);
                acc[row][0] = _mm512_fmadd_ps(av, b0, acc[row][0]);
                acc[row][1] = _mm512_fmadd_ps(av, b1, acc[row][1]);
            }
        }
        for row in 0..8 {
            unsafe {
                _mm512_storeu_ps(c.as_mut_ptr().add(row * 64 + col0), acc[row][0]);
                _mm512_storeu_ps(c.as_mut_ptr().add(row * 64 + col0 + 16), acc[row][1]);
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,fma")]
unsafe fn gemm_6x64_accumulate(m: usize, a: &[f32], b: &[f32], c: &mut [f32]) {
    use core::arch::x86_64::*;
    debug_assert!(m <= 128 && a.len() >= m * 64 && b.len() >= 64 * 64 && c.len() >= m * 64);
    for row0 in (0..m).step_by(6) {
        let rows = (m - row0).min(6);
        let mut acc = [[_mm512_setzero_ps(); 4]; 6];
        for row in 0..rows {
            for block in 0..4 {
                acc[row][block] =
                    unsafe { _mm512_loadu_ps(c.as_ptr().add((row0 + row) * 64 + block * 16)) };
            }
        }
        for kk in 0..64 {
            let bp = unsafe { b.as_ptr().add(kk * 64) };
            let bv = unsafe {
                [
                    _mm512_loadu_ps(bp),
                    _mm512_loadu_ps(bp.add(16)),
                    _mm512_loadu_ps(bp.add(32)),
                    _mm512_loadu_ps(bp.add(48)),
                ]
            };
            for row in 0..rows {
                let av = _mm512_set1_ps(a[(row0 + row) * 64 + kk]);
                for block in 0..4 {
                    acc[row][block] = _mm512_fmadd_ps(av, bv[block], acc[row][block]);
                }
            }
        }
        for row in 0..rows {
            for block in 0..4 {
                unsafe {
                    _mm512_storeu_ps(
                        c.as_mut_ptr().add((row0 + row) * 64 + block * 16),
                        acc[row][block],
                    );
                }
            }
        }
    }
}

/// Cephes-style vector exponential. Keeping the score tile at 64 values lets
/// the hot softmax portion run as exactly four ZMM operations.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,fma")]
unsafe fn exp_64_avx512(values: &mut [f32; 64]) {
    use core::arch::x86_64::*;
    let hi = _mm512_set1_ps(88.376_26);
    let lo = _mm512_set1_ps(-88.376_26);
    let log2ef = _mm512_set1_ps(1.442_695_04);
    let half = _mm512_set1_ps(0.5);
    let one = _mm512_set1_ps(1.0);
    let ln2_hi = _mm512_set1_ps(0.693_359_375);
    let ln2_lo = _mm512_set1_ps(-2.121_944_4e-4);
    let p = [
        _mm512_set1_ps(1.987_569_15e-4),
        _mm512_set1_ps(1.398_199_95e-3),
        _mm512_set1_ps(8.333_451_9e-3),
        _mm512_set1_ps(4.166_579_6e-2),
        _mm512_set1_ps(1.666_666_5e-1),
        _mm512_set1_ps(5.000_000_1e-1),
    ];
    for block in 0..4 {
        // SAFETY: `values` has four contiguous sixteen-float blocks.
        let ptr = unsafe { values.as_mut_ptr().add(block * 16) };
        let x = unsafe { _mm512_max_ps(_mm512_min_ps(_mm512_loadu_ps(ptr), hi), lo) };
        let fx0 = _mm512_fmadd_ps(x, log2ef, half);
        let trunc = _mm512_cvtepi32_ps(_mm512_cvttps_epi32(fx0));
        let gt = _mm512_cmp_ps_mask(trunc, fx0, _CMP_GT_OQ);
        let fx = _mm512_mask_sub_ps(trunc, gt, trunc, one);
        let x = _mm512_fnmadd_ps(fx, ln2_lo, _mm512_fnmadd_ps(fx, ln2_hi, x));
        let z = _mm512_mul_ps(x, x);
        let mut y = p[0];
        for coefficient in &p[1..] {
            y = _mm512_fmadd_ps(y, x, *coefficient);
        }
        y = _mm512_fmadd_ps(y, z, x);
        let exponent = _mm512_slli_epi32(
            _mm512_add_epi32(_mm512_cvttps_epi32(fx), _mm512_set1_epi32(0x7f)),
            23,
        );
        // SAFETY: `ptr` addresses the same valid sixteen-float block.
        unsafe {
            _mm512_storeu_ps(
                ptr,
                _mm512_mul_ps(_mm512_add_ps(y, one), _mm512_castsi512_ps(exponent)),
            )
        };
    }
}

impl Da3BaseProjection {
    /// Returns whether this is one of DA3-BASE's four repeated F32 projection
    /// families at the locked benchmark resolution.
    pub const fn is_supported(self) -> bool {
        self.tokens == DA3_BASE_TOKENS_504X336
            && matches!(
                (self.input_channels, self.output_channels),
                (768, 2304) | (768, 768) | (768, 3072) | (3072, 768)
            )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn admits_only_the_documented_da3_base_projection_shapes() {
        assert!(
            Da3BaseProjection {
                tokens: 865,
                input_channels: 768,
                output_channels: 2304,
            }
            .is_supported()
        );
        assert!(
            !Da3BaseProjection {
                tokens: 864,
                input_channels: 768,
                output_channels: 2304,
            }
            .is_supported()
        );
    }

    #[test]
    fn linear_row_selector_accepts_only_explicit_ab_variants() {
        assert_eq!(linear_rows_from_env(None), 6);
        assert_eq!(linear_rows_from_env(Some("4")), 4);
        assert_eq!(linear_rows_from_env(Some("8")), 8);
        assert_eq!(linear_rows_from_env(Some("6")), 6);
        assert_eq!(linear_rows_from_env(Some("invalid")), 6);
    }

    #[test]
    fn fc2_row_selector_is_opt_in_and_conservative() {
        assert_eq!(fc2_rows_from_env(None), 6);
        assert_eq!(fc2_rows_from_env(Some("4")), 4);
        assert_eq!(fc2_rows_from_env(Some("8")), 8);
        assert_eq!(fc2_rows_from_env(Some("6")), 6);
        assert_eq!(fc2_rows_from_env(Some("invalid")), 6);
    }

    #[test]
    fn blocked_winograd_matches_scalar_f32_accumulation() {
        let (inputs, outputs, tiles) = (3, 16, 3);
        let u: Vec<f32> = (0..16 * inputs * outputs)
            .map(|i| (i as f32 * 0.013).sin())
            .collect();
        let v: Vec<f32> = (0..16 * inputs * tiles)
            .map(|i| (i as f32 * 0.017).cos())
            .collect();
        let mut actual = vec![0.0; 16 * tiles * outputs];
        if !winograd_f2_blocked_f32(&u, &v, &mut actual, inputs, outputs, tiles) {
            return;
        }
        let mut expected = vec![0.0; actual.len()];
        for position in 0..16 {
            for tile in 0..tiles {
                for output in 0..outputs {
                    for input in 0..inputs {
                        let slot = &mut expected[(position * tiles + tile) * outputs + output];
                        *slot = u[(position * inputs + input) * outputs + output]
                            .mul_add(v[(position * inputs + input) * tiles + tile], *slot);
                    }
                }
            }
        }
        assert_eq!(
            actual
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            expected
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
        );
    }

    #[test]
    fn f4_blocked_winograd_matches_fma_accumulation() {
        let (inputs, outputs, tiles) = (5, 32, 2);
        let u: Vec<f32> = (0..36 * inputs * outputs)
            .map(|i| (i as f32 * 0.011).sin())
            .collect();
        let v: Vec<f32> = (0..36 * inputs * tiles)
            .map(|i| (i as f32 * 0.019).cos())
            .collect();
        let mut actual = vec![0.0; 36 * tiles * outputs];
        if !winograd_f4_blocked_f32(&u, &v, &mut actual, inputs, outputs, tiles) {
            return;
        }
        let mut expected = vec![0.0; actual.len()];
        for position in 0..36 {
            for tile in 0..tiles {
                for output in 0..outputs {
                    for input in 0..inputs {
                        let slot = &mut expected[(position * tiles + tile) * outputs + output];
                        *slot = u[(position * inputs + input) * outputs + output]
                            .mul_add(v[(position * inputs + input) * tiles + tile], *slot);
                    }
                }
            }
        }
        assert_eq!(
            actual
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            expected
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
        );
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn four_tile_winograd_specialization_matches_generic_bitwise() {
        if !(std::is_x86_feature_detected!("avx512f") && std::is_x86_feature_detected!("fma")) {
            return;
        }
        let (inputs, outputs, tiles) = (3, 16, 4);
        let u: Vec<f32> = (0..16 * inputs * outputs)
            .map(|i| (i as f32 * 0.013).sin())
            .collect();
        let v: Vec<f32> = (0..16 * inputs * tiles)
            .map(|i| (i as f32 * 0.017).cos())
            .collect();
        let mut generic = vec![0.0; 16 * tiles * outputs];
        let mut specialized = generic.clone();
        unsafe {
            winograd_f2_blocked_avx512_generic(&u, &v, &mut generic, inputs, outputs, tiles);
            winograd_f2_blocked_avx512_tiles4(&u, &v, &mut specialized, inputs, outputs);
        }
        assert!(
            generic
                .iter()
                .zip(&specialized)
                .all(|(generic, specialized)| generic.to_bits() == specialized.to_bits()),
        );
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn flash_accepts_and_matches_a_two_view_da3_window() {
        if !(std::is_x86_feature_detected!("avx512f") && std::is_x86_feature_detected!("fma")) {
            return;
        }
        const HEADS: usize = 1;
        const TOKENS: usize = 2 * DA3_BASE_TOKENS_504X336;
        const DIM: usize = 64;
        let q: Vec<f32> = (0..HEADS * TOKENS * DIM)
            .map(|index| (index as f32 * 0.013).sin())
            .collect();
        let k: Vec<f32> = (0..HEADS * TOKENS * DIM)
            .map(|index| (index as f32 * 0.007).cos())
            .collect();
        let v: Vec<f32> = (0..HEADS * TOKENS * DIM)
            .map(|index| (index as f32 * 0.017).sin())
            .collect();
        let mut fast = vec![0.0; q.len()];
        let mut reference = vec![0.0; q.len()];

        assert!(flash_attention_f32_da3_base(
            &q, &k, &v, HEADS, TOKENS, &mut fast
        ));
        crate::attention::attention_serial(&q, &k, &v, HEADS, TOKENS, DIM, &mut reference);

        let (mut mae, mut max_abs) = (0.0f64, 0.0f32);
        for (candidate, control) in fast.iter().zip(&reference) {
            let error = (candidate - control).abs();
            mae += f64::from(error);
            max_abs = max_abs.max(error);
        }
        mae /= fast.len() as f64;
        assert!(
            mae <= 2.0e-5,
            "MAE {mae} exceeds the F32 attention envelope"
        );
        assert!(
            max_abs <= 2.0e-4,
            "max error {max_abs} exceeds the F32 attention envelope"
        );
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn flash_gemm_rows6_matches_rows4_bitwise() {
        if !(std::is_x86_feature_detected!("avx512f") && std::is_x86_feature_detected!("fma")) {
            return;
        }
        let m = 8;
        let a: Vec<f32> = (0..m * 64).map(|i| (i as f32 * 0.013).sin()).collect();
        let b: Vec<f32> = (0..64 * 64).map(|i| (i as f32 * 0.007).cos()).collect();
        let mut overwrite_4 = vec![0.0; m * 64];
        let mut overwrite_6 = overwrite_4.clone();
        let mut overwrite_8x32 = overwrite_4.clone();
        let stride = 68;
        let mut strided = vec![0.0; 64 * stride];
        for row in 0..64 {
            strided[row * stride..row * stride + 64].copy_from_slice(&b[row * 64..(row + 1) * 64]);
        }
        let mut stride_4 = vec![0.0; m * 64];
        let mut stride_6 = stride_4.clone();
        let mut stride_8x32 = stride_4.clone();
        let mut accumulate_4: Vec<f32> = (0..m * 64).map(|i| i as f32 * 0.001).collect();
        let mut accumulate_6 = accumulate_4.clone();
        let mut accumulate_8x32 = accumulate_4.clone();
        unsafe {
            gemm_4x64_overwrite(m, &a, &b, &mut overwrite_4);
            gemm_6x64_overwrite(m, &a, &b, &mut overwrite_6);
            gemm_8x32_overwrite(&a, &b, &mut overwrite_8x32);
            gemm_4x64_overwrite_stride(m, &a, &strided, stride, 0, 64, &mut stride_4);
            gemm_6x64_overwrite_stride(m, &a, &strided, stride, 0, 64, &mut stride_6);
            gemm_8x32_overwrite_stride(&a, &strided, stride, 0, &mut stride_8x32);
            gemm_4x64_accumulate(m, &a, &b, &mut accumulate_4);
            gemm_6x64_accumulate(m, &a, &b, &mut accumulate_6);
            gemm_8x32_accumulate(&a, &b, &mut accumulate_8x32);
        }
        for (control, candidate) in [
            (&overwrite_4, &overwrite_6),
            (&overwrite_4, &overwrite_8x32),
            (&stride_4, &stride_6),
            (&stride_4, &stride_8x32),
            (&accumulate_4, &accumulate_6),
            (&accumulate_4, &accumulate_8x32),
        ] {
            assert!(
                control
                    .iter()
                    .zip(candidate)
                    .all(|(a, b)| a.to_bits() == b.to_bits())
            );
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn flash_8x32_tile_matches_4x64_control_bitwise() {
        if !(std::is_x86_feature_detected!("avx512f") && std::is_x86_feature_detected!("fma")) {
            return;
        }
        const TOKENS: usize = DA3_BASE_TOKENS_504X336;
        let q: Vec<f32> = (0..TOKENS * 64).map(|i| (i as f32 * 0.013).sin()).collect();
        let k: Vec<f32> = (0..TOKENS * 64).map(|i| (i as f32 * 0.007).cos()).collect();
        let v: Vec<f32> = (0..TOKENS * 64).map(|i| (i as f32 * 0.017).sin()).collect();
        let packed = pack_key_head_dim_major(&k, TOKENS, TOKENS);
        let profile = FlashProfile::from_env();
        let mut control = vec![0.0; 8 * 64];
        let mut candidate = control.clone();
        unsafe {
            flash_attention_tile_avx512::<8>(
                &q,
                &k,
                &v,
                &packed,
                TOKENS,
                TOKENS,
                0,
                &mut control,
                &profile,
                false,
                false,
            );
            flash_attention_tile_avx512::<8>(
                &q,
                &k,
                &v,
                &packed,
                TOKENS,
                TOKENS,
                0,
                &mut candidate,
                &profile,
                false,
                true,
            );
        }
        assert!(
            control
                .iter()
                .zip(candidate)
                .all(|(control, candidate)| control.to_bits() == candidate.to_bits())
        );
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn packed_flash_qt8_matches_generic_qt8_bitwise_including_tail_panel() {
        if !(std::is_x86_feature_detected!("avx512f") && std::is_x86_feature_detected!("fma")) {
            return;
        }
        const TOKENS: usize = DA3_BASE_TOKENS_504X336;
        let q: Vec<f32> = (0..TOKENS * 64).map(|i| (i as f32 * 0.013).sin()).collect();
        let k: Vec<f32> = (0..TOKENS * 64).map(|i| (i as f32 * 0.007).cos()).collect();
        let v: Vec<f32> = (0..TOKENS * 64).map(|i| (i as f32 * 0.017).sin()).collect();
        let packed = pack_key_head_dim_major(&k, TOKENS, TOKENS);
        let profile = FlashProfile::from_env();
        let mut generic = vec![0.0; 8 * 64];
        let mut packed_fast = generic.clone();
        unsafe {
            flash_attention_tile_avx512::<8>(
                &q,
                &k,
                &v,
                &packed,
                TOKENS,
                TOKENS,
                0,
                &mut generic,
                &profile,
                false,
                true,
            );
            flash_attention_tile_packed_8x32_avx512(
                &q,
                &v,
                &packed,
                TOKENS,
                TOKENS,
                0,
                &mut packed_fast,
                &profile,
            );
        }
        assert!(
            generic
                .iter()
                .zip(&packed_fast)
                .all(|(generic, packed_fast)| generic.to_bits() == packed_fast.to_bits())
        );
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn flash_superblock32_matches_four_qt8_tiles_bitwise() {
        if !(std::is_x86_feature_detected!("avx512f") && std::is_x86_feature_detected!("fma")) {
            return;
        }
        const TOKENS: usize = DA3_BASE_TOKENS_504X336;
        let q: Vec<f32> = (0..TOKENS * 64).map(|i| (i as f32 * 0.013).sin()).collect();
        let k: Vec<f32> = (0..TOKENS * 64).map(|i| (i as f32 * 0.007).cos()).collect();
        let v: Vec<f32> = (0..TOKENS * 64).map(|i| (i as f32 * 0.017).sin()).collect();
        let packed = pack_key_head_dim_major(&k, TOKENS, TOKENS);
        let profile = FlashProfile::from_env();
        let mut tiled = vec![0.0; 32 * 64];
        let mut superblock = tiled.clone();
        unsafe {
            for tile in 0..4 {
                flash_attention_tile_avx512::<8>(
                    &q,
                    &k,
                    &v,
                    &packed,
                    TOKENS,
                    TOKENS,
                    tile * 8,
                    &mut tiled[tile * 8 * 64..(tile + 1) * 8 * 64],
                    &profile,
                    false,
                    false,
                );
            }
            flash_attention_superblock32_avx512(
                &q,
                &v,
                &packed,
                TOKENS,
                TOKENS,
                0,
                &mut superblock,
                false,
            );
        }
        assert!(
            tiled
                .iter()
                .zip(&superblock)
                .all(|(a, b)| a.to_bits() == b.to_bits())
        );
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn flash_ggml64_matches_eight_qt8_tiles_bitwise() {
        if !(std::is_x86_feature_detected!("avx512f") && std::is_x86_feature_detected!("fma")) {
            return;
        }
        const TOKENS: usize = DA3_BASE_TOKENS_504X336;
        let q: Vec<f32> = (0..TOKENS * 64).map(|i| (i as f32 * 0.013).sin()).collect();
        let k: Vec<f32> = (0..TOKENS * 64).map(|i| (i as f32 * 0.007).cos()).collect();
        let v: Vec<f32> = (0..TOKENS * 64).map(|i| (i as f32 * 0.017).sin()).collect();
        let packed = pack_key_head_dim_major(&k, TOKENS, TOKENS);
        let profile = FlashProfile::from_env();
        let mut tiled = vec![0.0; 64 * 64];
        let mut ggml64 = tiled.clone();
        unsafe {
            for tile in 0..8 {
                flash_attention_tile_avx512::<8>(
                    &q,
                    &k,
                    &v,
                    &packed,
                    TOKENS,
                    TOKENS,
                    tile * 8,
                    &mut tiled[tile * 8 * 64..(tile + 1) * 8 * 64],
                    &profile,
                    false,
                    false,
                );
            }
            flash_attention_ggml64_tile_avx512(
                &q,
                &v,
                &packed,
                TOKENS,
                TOKENS,
                0,
                &mut ggml64,
                &profile,
            );
        }
        assert!(
            tiled
                .iter()
                .zip(&ggml64)
                .all(|(control, candidate)| control.to_bits() == candidate.to_bits())
        );
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn flash_superblock16_matches_two_qt8_tiles_bitwise() {
        if !(std::is_x86_feature_detected!("avx512f") && std::is_x86_feature_detected!("fma")) {
            return;
        }
        const TOKENS: usize = DA3_BASE_TOKENS_504X336;
        let q: Vec<f32> = (0..TOKENS * 64).map(|i| (i as f32 * 0.013).sin()).collect();
        let k: Vec<f32> = (0..TOKENS * 64).map(|i| (i as f32 * 0.007).cos()).collect();
        let v: Vec<f32> = (0..TOKENS * 64).map(|i| (i as f32 * 0.017).sin()).collect();
        let packed = pack_key_head_dim_major(&k, TOKENS, TOKENS);
        let profile = FlashProfile::from_env();
        let mut tiled = vec![0.0; 16 * 64];
        let mut superblock = tiled.clone();
        unsafe {
            for tile in 0..2 {
                flash_attention_tile_avx512::<8>(
                    &q,
                    &k,
                    &v,
                    &packed,
                    TOKENS,
                    TOKENS,
                    tile * 8,
                    &mut tiled[tile * 8 * 64..(tile + 1) * 8 * 64],
                    &profile,
                    false,
                    false,
                );
            }
            flash_attention_superblock32_avx512(
                &q,
                &v,
                &packed,
                TOKENS,
                TOKENS,
                0,
                &mut superblock,
                false,
            );
        }
        assert!(
            tiled
                .iter()
                .zip(&superblock)
                .all(|(a, b)| a.to_bits() == b.to_bits())
        );
    }

    #[test]
    fn projection_kernel_overwrites_its_output_buffer() {
        let (m, n, k) = (DA3_BASE_TOKENS_504X336, 768, 768);
        let a = vec![0.0; m * k];
        let b = vec![0.0; k * n];
        let mut output = vec![f32::NAN; m * n];
        if linear_f32_da3_base(m, n, k, &a, &b, &mut output) {
            assert!(output.iter().all(|value| *value == 0.0));
        }
    }

    #[test]
    fn dense_key_pack_is_dimension_major() {
        const TOKENS: usize = 5;
        const D: usize = 64;
        let source: Vec<f32> = (0..TOKENS * D).map(|value| value as f32).collect();
        let packed = pack_key_head_dim_major(&source, TOKENS, TOKENS);

        assert_eq!(packed.len(), TOKENS * D);
        for key in 0..TOKENS {
            for dim in 0..D {
                assert_eq!(packed[dim * TOKENS + key], source[key * D + dim]);
            }
        }
    }

    #[test]
    fn flash_query_tiles_cover_a_head_without_overlap() {
        let tokens = DA3_BASE_TOKENS_504X336;
        let mut covered = vec![false; tokens];
        for tile in 0..flash_query_tile_count(tokens) {
            let start = tile * FLASH_QUERY_TILE;
            let end = (start + FLASH_QUERY_TILE).min(tokens);
            for slot in &mut covered[start..end] {
                assert!(!*slot);
                *slot = true;
            }
        }
        assert!(covered.into_iter().all(|slot| slot));
    }

    #[test]
    fn flash_query_tile_selector_accepts_only_the_ab_variant() {
        assert_eq!(flash_query_tile_from_env(None), FLASH_QUERY_TILE);
        assert_eq!(flash_query_tile_from_env(Some("4")), 4);
        assert_eq!(flash_query_tile_from_env(Some("8")), FLASH_QUERY_TILE);
        assert_eq!(flash_query_tile_from_env(Some("12")), 12);
        assert_eq!(flash_query_tile_from_env(Some("16")), 16);
        assert_eq!(flash_query_tile_from_env(Some("20")), 20);
        assert_eq!(flash_query_tile_from_env(Some("invalid")), FLASH_QUERY_TILE);
    }

    #[test]
    fn flash_head_only_selector_is_opt_in() {
        assert!(!flash_head_only_from_env(None));
        assert!(flash_head_only_from_env(Some("1")));
    }

    #[test]
    fn flash_nested_super16_selector_is_opt_in() {
        assert!(!flash_nested_super16_from_env(None));
        assert!(flash_nested_super16_from_env(Some("1")));
    }

    #[test]
    fn padded_key_pack_preserves_dense_prefix_and_zero_tail() {
        const TOKENS: usize = 5;
        const STRIDE: usize = 8;
        const D: usize = 64;
        let source: Vec<f32> = (0..TOKENS * D).map(|value| value as f32).collect();
        let packed = pack_key_head_dim_major(&source, TOKENS, STRIDE);

        assert_eq!(packed.len(), STRIDE * D);
        for dim in 0..D {
            assert_eq!(
                &packed[dim * STRIDE..dim * STRIDE + TOKENS],
                &source
                    .iter()
                    .skip(dim)
                    .step_by(D)
                    .copied()
                    .collect::<Vec<_>>()
            );
            assert!(
                packed[dim * STRIDE + TOKENS..(dim + 1) * STRIDE]
                    .iter()
                    .all(|value| *value == 0.0)
            );
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn column_split_projection_matches_row_split_bitwise() {
        if !(std::is_x86_feature_detected!("avx512f") && std::is_x86_feature_detected!("fma")) {
            return;
        }
        let (m, n, k) = (DA3_BASE_TOKENS_504X336, 768, 768);
        let a: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.013).sin()).collect();
        let b: Vec<f32> = (0..k * n).map(|i| (i as f32 * 0.007).cos()).collect();
        let mut rows_4 = vec![0.0f32; m * n];
        let mut rows_6 = vec![0.0f32; m * n];
        let mut rows_8 = vec![0.0f32; m * n];
        let mut columns_4 = vec![0.0f32; m * n];
        let mut columns_8 = vec![0.0f32; m * n];
        unsafe {
            linear_avx512_rows::<4>(m, n, k, &a, &b, &mut rows_4);
            linear_avx512_rows::<6>(m, n, k, &a, &b, &mut rows_6);
            linear_avx512_rows::<8>(m, n, k, &a, &b, &mut rows_8);
            linear_avx512_column_split_rows::<4>(m, n, k, &a, &b, &mut columns_4);
            linear_avx512_column_split_rows::<8>(m, n, k, &a, &b, &mut columns_8);
        }
        for candidate in [&rows_4, &rows_8, &columns_4, &columns_8] {
            assert!(
                rows_6
                    .iter()
                    .zip(candidate)
                    .all(|(control, variant)| control.to_bits() == variant.to_bits()),
            );
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn bias_scale_column_split_matches_row_split_bitwise() {
        if !(std::is_x86_feature_detected!("avx512f") && std::is_x86_feature_detected!("fma")) {
            return;
        }
        let (m, n, k) = (DA3_BASE_TOKENS_504X336, 768, 768);
        let a: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.013).sin()).collect();
        let b: Vec<f32> = (0..k * n).map(|i| (i as f32 * 0.007).cos()).collect();
        let bias: Vec<f32> = (0..n).map(|i| (i as f32 * 0.017).sin()).collect();
        let scale: Vec<f32> = (0..n).map(|i| 0.75 + (i as f32 * 0.011).cos()).collect();
        let mut rows = vec![0.0f32; m * n];
        let mut rows_4 = vec![0.0f32; m * n];
        let mut rows_8 = vec![0.0f32; m * n];
        let mut columns = vec![0.0f32; m * n];
        unsafe {
            linear_bias_scale_avx512_rows::<6>(m, n, k, &a, &b, &bias, &scale, &mut rows);
            linear_bias_scale_avx512_rows::<4>(m, n, k, &a, &b, &bias, &scale, &mut rows_4);
            linear_bias_scale_avx512_rows::<8>(m, n, k, &a, &b, &bias, &scale, &mut rows_8);
            linear_bias_scale_avx512_column_split(m, n, k, &a, &b, &bias, &scale, &mut columns);
        }
        for candidate in [&rows_4, &rows_8, &columns] {
            assert!(
                rows.iter()
                    .zip(candidate)
                    .all(|(row, candidate)| row.to_bits() == candidate.to_bits()),
            );
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn qkv_column_split_matches_row_split_bitwise() {
        if !(std::is_x86_feature_detected!("avx512f") && std::is_x86_feature_detected!("fma")) {
            return;
        }
        const TOKENS: usize = DA3_BASE_TOKENS_504X336;
        let a: Vec<f32> = (0..TOKENS * 768)
            .map(|i| (i as f32 * 0.013).sin())
            .collect();
        let weight: Vec<f32> = (0..768 * 2304).map(|i| (i as f32 * 0.007).cos()).collect();
        let bias: Vec<f32> = (0..2304).map(|i| (i as f32 * 0.017).sin()).collect();
        let mut row_q = vec![0.0f32; TOKENS * 768];
        let mut row_k = vec![0.0f32; TOKENS * 768];
        let mut row_v = vec![0.0f32; TOKENS * 768];
        let mut column_q = vec![0.0f32; TOKENS * 768];
        let mut column_k = vec![0.0f32; TOKENS * 768];
        let mut column_v = vec![0.0f32; TOKENS * 768];
        unsafe {
            qkv_avx512(&a, &weight, &bias, &mut row_q, &mut row_k, &mut row_v);
            qkv_avx512_column_split(
                &a,
                &weight,
                &bias,
                &mut column_q,
                &mut column_k,
                &mut column_v,
            );
        }
        for (control, candidate) in [
            (&row_q, &column_q),
            (&row_k, &column_k),
            (&row_v, &column_v),
        ] {
            assert!(
                control
                    .iter()
                    .zip(candidate)
                    .all(|(control, candidate)| control.to_bits() == candidate.to_bits()),
            );
        }
    }

    #[test]
    fn fused_qk_norm_rope_matches_separate_scalar_operations_bitwise() {
        const HEADS: usize = 12;
        const TOKENS: usize = DA3_BASE_TOKENS_504X336;
        const DIM: usize = 64;
        let source: Vec<f32> = (0..HEADS * TOKENS * DIM)
            .map(|i| ((i * 31 % 997) as f32 - 498.0) * 0.001)
            .collect();
        let gamma: Vec<f32> = (0..DIM).map(|i| 0.8 + i as f32 * 0.003).collect();
        let beta: Vec<f32> = (0..DIM).map(|i| -0.1 + i as f32 * 0.002).collect();
        let positions: Vec<i64> = (0..TOKENS)
            .flat_map(|i| [(i / 24) as i64, (i % 24) as i64])
            .collect();
        let mut fused_q = source.clone();
        let mut fused_k = source.clone();
        assert!(qk_norm_rope_f32_da3_base(
            &mut fused_q,
            &mut fused_k,
            &gamma,
            &beta,
            &gamma,
            &beta,
            &positions,
            100.0,
            1e-5,
        ));
        let rotations = rope_rotations(&positions, 100.0);
        let mut separate_q = source.clone();
        let mut separate_k = source;
        normalize_and_rotate(&mut separate_q, &gamma, &beta, 1e-5, &rotations);
        normalize_and_rotate(&mut separate_k, &gamma, &beta, 1e-5, &rotations);
        assert_eq!(
            fused_q.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            separate_q.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        );
        assert_eq!(
            fused_k.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            separate_k.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        );
    }

    #[test]
    fn da3_base_rope_cache_matches_generic_rotations_bitwise() {
        for layout in [Da3BaseRopeLayout::Local, Da3BaseRopeLayout::Global] {
            let positions = da3_base_rope_positions(layout);
            let cached = da3_base_cached_rope_rotations(&positions, 100.0)
                .expect("known DA3 layout must use the cache");
            let generic = rope_rotations(&positions, 100.0);
            assert_eq!(
                cached
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>(),
                generic
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>(),
            );
        }
    }

    #[test]
    fn da3_base_rope_cache_rejects_nearby_layouts_and_frequency() {
        let mut positions = da3_base_rope_positions(Da3BaseRopeLayout::Local);
        positions[2] = 9;
        assert!(da3_base_cached_rope_rotations(&positions, 100.0).is_none());
        let positions = da3_base_rope_positions(Da3BaseRopeLayout::Local);
        assert!(da3_base_cached_rope_rotations(&positions, 99.0).is_none());
    }
}

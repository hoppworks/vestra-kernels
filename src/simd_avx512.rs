//! AVX-512F kernels. This module is only ever compiled on `target_arch =
//! "x86_64"` (see the `#[cfg(...)]` gate on the `mod simd_avx512;`
//! declaration in `lib.rs`). Every path retains a scalar oracle; target-host
//! qualification and the evidence boundary are documented in
//! `BENCHMARKING.md`.

#![allow(
    clippy::approx_constant,
    clippy::excessive_precision,
    reason = "established SIMD polynomial coefficients are retained verbatim for numerical parity"
)]

use core::arch::x86_64::*;

const LANES: usize = 16; // f32 lanes per __m512

/// Vectorized `expf` (Cephes-style range reduction + minimax polynomial),
/// matching the accuracy needs of `erf_avx512` below. Operates on all 16
/// lanes of `x` at once.
#[target_feature(enable = "avx512f")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn exp_avx512(x: __m512) -> __m512 {
    let exp_hi = _mm512_set1_ps(88.376_26);
    let exp_lo = _mm512_set1_ps(-88.376_26);
    let log2ef = _mm512_set1_ps(1.442_695_04);
    let half = _mm512_set1_ps(0.5);
    let one = _mm512_set1_ps(1.0);
    let ln2_hi = _mm512_set1_ps(0.693_359_375);
    let ln2_lo = _mm512_set1_ps(-2.121_944_4e-4);

    let p0 = _mm512_set1_ps(1.987_569_15e-4);
    let p1 = _mm512_set1_ps(1.398_199_95e-3);
    let p2 = _mm512_set1_ps(8.333_451_9e-3);
    let p3 = _mm512_set1_ps(4.166_579_6e-2);
    let p4 = _mm512_set1_ps(1.666_666_5e-1);
    let p5 = _mm512_set1_ps(5.000_000_1e-1);

    let x = _mm512_min_ps(x, exp_hi);
    let x = _mm512_max_ps(x, exp_lo);

    // fx = floor(x * log2(e) + 0.5)
    let fx0 = _mm512_fmadd_ps(x, log2ef, half);
    let fx_trunc = _mm512_cvtepi32_ps(_mm512_cvttps_epi32(fx0));
    let gt_mask = _mm512_cmp_ps_mask(fx_trunc, fx0, _CMP_GT_OQ);
    let fx = _mm512_mask_sub_ps(fx_trunc, gt_mask, fx_trunc, one);

    // x -= fx * ln2 (in two steps for precision)
    let x = _mm512_fnmadd_ps(fx, ln2_hi, x);
    let x = _mm512_fnmadd_ps(fx, ln2_lo, x);

    let z = _mm512_mul_ps(x, x);

    let mut y = p0;
    y = _mm512_fmadd_ps(y, x, p1);
    y = _mm512_fmadd_ps(y, x, p2);
    y = _mm512_fmadd_ps(y, x, p3);
    y = _mm512_fmadd_ps(y, x, p4);
    y = _mm512_fmadd_ps(y, x, p5);
    y = _mm512_fmadd_ps(y, z, x);
    let y = _mm512_add_ps(y, one);

    // pow2n = 2^fx via direct exponent-bit construction
    let emm0 = _mm512_cvttps_epi32(fx);
    let emm0 = _mm512_add_epi32(emm0, _mm512_set1_epi32(0x7f));
    let emm0 = _mm512_slli_epi32(emm0, 23);
    let pow2n = _mm512_castsi512_ps(emm0);

    _mm512_mul_ps(y, pow2n)
}

/// In-place AVX-512 exponential with a scalar tail. This is the same
/// approximation used by the AVX-512 GELU path above.
#[target_feature(enable = "avx512f")]
#[allow(unsafe_op_in_unsafe_fn)]
pub(crate) unsafe fn exp_in_place_avx512(values: &mut [f32]) {
    let main = values.len() - (values.len() % LANES);
    let mut i = 0;
    while i < main {
        let ptr = values.as_mut_ptr().add(i);
        _mm512_storeu_ps(ptr, exp_avx512(_mm512_loadu_ps(ptr)));
        i += LANES;
    }
    for value in &mut values[main..] {
        *value = value.exp();
    }
}

/// AVX-512 dot product. The final scalar reduction intentionally keeps the
/// result in F32 and avoids FMA; callers still validate full-model parity.
#[inline]
#[target_feature(enable = "avx512f")]
#[allow(unsafe_op_in_unsafe_fn)]
pub(crate) unsafe fn dot_avx512(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let main = a.len() - (a.len() % LANES);
    let mut lanes = _mm512_setzero_ps();
    let mut i = 0;
    while i < main {
        lanes = _mm512_add_ps(
            lanes,
            _mm512_mul_ps(
                _mm512_loadu_ps(a.as_ptr().add(i)),
                _mm512_loadu_ps(b.as_ptr().add(i)),
            ),
        );
        i += LANES;
    }
    let mut partial = [0.0f32; LANES];
    _mm512_storeu_ps(partial.as_mut_ptr(), lanes);
    let mut sum: f32 = partial.iter().sum();
    for i in main..a.len() {
        sum += a[i] * b[i];
    }
    sum
}

#[inline]
#[target_feature(enable = "avx512f")]
#[allow(unsafe_op_in_unsafe_fn)]
pub(crate) unsafe fn scaled_add_avx512(dst: &mut [f32], scale: f32, src: &[f32]) {
    debug_assert_eq!(dst.len(), src.len());
    let scale_vec = _mm512_set1_ps(scale);
    let main = dst.len() - (dst.len() % LANES);
    let mut i = 0;
    while i < main {
        let ptr = dst.as_mut_ptr().add(i);
        let product = _mm512_mul_ps(_mm512_loadu_ps(src.as_ptr().add(i)), scale_vec);
        _mm512_storeu_ps(ptr, _mm512_add_ps(_mm512_loadu_ps(ptr), product));
        i += LANES;
    }
    for i in main..dst.len() {
        dst[i] += scale * src[i];
    }
}

/// Vectorized `erf` using the same Abramowitz-Stegun 7.1.26 approximation
/// as `scalar::erf` (|error| < 1.5e-7), so `gelu_avx512` stays within the
/// tolerance band of `scalar::gelu`.
#[target_feature(enable = "avx512f")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn erf_avx512(x: __m512) -> __m512 {
    let abs_mask = _mm512_set1_epi32(0x7fff_ffff);

    let xi = _mm512_castps_si512(x);
    let x_abs = _mm512_castsi512_ps(_mm512_and_epi32(xi, abs_mask));

    let zero = _mm512_setzero_ps();
    let one = _mm512_set1_ps(1.0);
    let neg_one = _mm512_set1_ps(-1.0);
    let neg_mask = _mm512_cmp_ps_mask(x, zero, _CMP_LT_OQ);
    let sign = _mm512_mask_blend_ps(neg_mask, one, neg_one);

    let a1 = _mm512_set1_ps(0.254_829_59);
    let a2 = _mm512_set1_ps(-0.284_496_74);
    let a3 = _mm512_set1_ps(1.421_413_7);
    let a4 = _mm512_set1_ps(-1.453_152_0);
    let a5 = _mm512_set1_ps(1.061_405_4);
    let c = _mm512_set1_ps(0.327_591_1);

    // t = 1 / (1 + c * x_abs)
    let denom = _mm512_fmadd_ps(c, x_abs, one);
    let t = _mm512_div_ps(one, denom);

    let mut poly = a5;
    poly = _mm512_fmadd_ps(poly, t, a4);
    poly = _mm512_fmadd_ps(poly, t, a3);
    poly = _mm512_fmadd_ps(poly, t, a2);
    poly = _mm512_fmadd_ps(poly, t, a1);
    poly = _mm512_mul_ps(poly, t);

    let neg_x2 = _mm512_sub_ps(zero, _mm512_mul_ps(x_abs, x_abs));
    let exp_term = exp_avx512(neg_x2);

    let y = _mm512_fnmadd_ps(poly, exp_term, one); // 1 - poly * exp_term
    _mm512_mul_ps(sign, y)
}

/// In-place GELU (exact-erf formulation): `x = 0.5*x*(1 + erf(x/sqrt(2)))`.
/// See `scalar::gelu` for the reference implementation this must match
/// within tolerance.
#[target_feature(enable = "avx512f")]
#[allow(unsafe_op_in_unsafe_fn)]
pub(crate) unsafe fn gelu_avx512(x: &mut [f32]) {
    const INV_SQRT2: f32 = 0.707_106_78;
    let inv_sqrt2 = _mm512_set1_ps(INV_SQRT2);
    let half = _mm512_set1_ps(0.5);
    let one = _mm512_set1_ps(1.0);

    let n = x.len();
    let main = n - (n % LANES);

    let mut i = 0;
    while i < main {
        let ptr = x.as_mut_ptr().add(i);
        let v = _mm512_loadu_ps(ptr);
        let arg = _mm512_mul_ps(v, inv_sqrt2);
        let e = erf_avx512(arg);
        let one_plus_e = _mm512_add_ps(one, e);
        let half_v = _mm512_mul_ps(half, v);
        let out = _mm512_mul_ps(half_v, one_plus_e);
        _mm512_storeu_ps(ptr, out);
        i += LANES;
    }

    debug_assert!(n - main < LANES);
    if main < n {
        crate::scalar::gelu(&mut x[main..n]);
    }
}

/// In-place elementwise `dst += src`, trivial AVX-512F load/add/store with
/// a scalar tail for the remainder (< 16 lanes).
#[target_feature(enable = "avx512f")]
#[allow(unsafe_op_in_unsafe_fn)]
pub(crate) unsafe fn add_avx512(dst: &mut [f32], src: &[f32]) {
    debug_assert_eq!(dst.len(), src.len());
    let n = dst.len();
    let main = n - (n % LANES);

    let mut i = 0;
    while i < main {
        let d_ptr = dst.as_mut_ptr().add(i);
        let s_ptr = src.as_ptr().add(i);
        let d = _mm512_loadu_ps(d_ptr);
        let s = _mm512_loadu_ps(s_ptr);
        let r = _mm512_add_ps(d, s);
        _mm512_storeu_ps(d_ptr, r);
        i += LANES;
    }

    debug_assert!(n - main < LANES);
    if main < n {
        crate::scalar::add(&mut dst[main..n], &src[main..n]);
    }
}

/// Dot product of two 32-lane q8_0 int8 blocks (`ax`, `by`, both signed
/// bytes in `[-127, 127]`), returned as a raw (unscaled by `d_a*d_b`) i32
/// sum, using AVX-512-VNNI's `_mm512_dpbusd_epi32` on the *pair* of 32-byte
/// blocks packed into one 64-byte/512-bit register.
///
/// `_mm512_dpbusd_epi32(src, a, b)` treats `a` as **unsigned** bytes and
/// `b` as **signed** bytes, multiplying+summing groups of 4 adjacent bytes
/// into each of 16 i32 lanes: `src[l] + sum_{t=0..3}(a[4l+t] as u8 * b[4l+t]
/// as i8)`. Both our operands are signed, so `bx` (packed from block A) is
/// shifted from signed to unsigned range via `XOR 0x80` (equivalent to
/// `+128 mod 256` for two's-complement bytes) before being fed in as the
/// "unsigned" operand; the resulting per-lane dot is therefore biased by
/// `128 * by[lane]`, which is removed afterwards by subtracting `128 *
/// sum(by)` (itself computed the same way, via `dpbusd` against an
/// all-ones "unsigned" operand — there is no cheaper horizontal byte-sum
/// instruction on AVX-512F/VNNI).
///
/// Because `dpbusd`'s 16 i32 output lanes are independent per 4-byte group
/// (lanes 0-7 come only from bytes 0-31 = block 0, lanes 8-15 only from
/// bytes 32-63 = block 1), the caller can split the result in half and
/// recover *two* independent per-block sums from one instruction — this is
/// the "2 blocks per VNNI iteration" unrolling the optimization brief
/// calls for (see `gemm_q8_0_avx512` below), not a separate optional pass.
#[target_feature(enable = "avx512f,avx512bw,avx512vnni")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn dpbusd_pair_halves(bx: __m512i, by: __m512i) -> (i32, i32) {
    let off = _mm512_set1_epi8(-128i8); // 0x80 per byte
    let ax = _mm512_xor_si512(bx, off); // ax[i] = (bx[i] as u8).wrapping_add(128)
    let ones = _mm512_set1_epi8(1i8);
    let zero = _mm512_setzero_si512();

    let dot = _mm512_dpbusd_epi32(zero, ax, by); // biased: real_dot + 128*by, per 4-byte group
    let sumb = _mm512_dpbusd_epi32(zero, ones, by); // sum(by), per 4-byte group

    let mask_lo: __mmask16 = 0x00FF; // lanes 0..8   -> block 0 (bytes 0..32)
    let mask_hi: __mmask16 = 0xFF00; // lanes 8..16  -> block 1 (bytes 32..64)

    let dot_lo = _mm512_reduce_add_epi32(_mm512_maskz_mov_epi32(mask_lo, dot));
    let dot_hi = _mm512_reduce_add_epi32(_mm512_maskz_mov_epi32(mask_hi, dot));
    let sumb_lo = _mm512_reduce_add_epi32(_mm512_maskz_mov_epi32(mask_lo, sumb));
    let sumb_hi = _mm512_reduce_add_epi32(_mm512_maskz_mov_epi32(mask_hi, sumb));

    (dot_lo - 128 * sumb_lo, dot_hi - 128 * sumb_hi)
}

/// q8_0 x q8_0 -> f32 GEMM (Task 9), AVX-512-VNNI path. See
/// `crate::scalar::gemm_q8_0` for the scalar oracle this must match and
/// `crate::Kernels::gemm_q8_0` for the dispatch entry point.
///
/// Processes blocks two at a time (64 int8 lanes = one `__m512i`) via
/// `_mm512_dpbusd_epi32`, per `dpbusd_pair_halves` above; an odd trailing
/// block (`blocks_per_row` not a multiple of 2) falls back to the scalar
/// single-block dot. Each `BlockQ8_0` is `{ d: f16, qs: [i8;32] }`
/// (`repr(C)`, 34 bytes) so consecutive blocks' `qs` are *not* contiguous
/// in memory (a `d` sits between them) — the two 32-byte `qs` arrays are
/// copied into a local 64-byte buffer before the single 512-bit load.
#[target_feature(enable = "avx512f,avx512bw,avx512vnni")]
#[allow(unsafe_op_in_unsafe_fn)]
pub(crate) unsafe fn gemm_q8_0_avx512(
    m: usize,
    n: usize,
    k: usize,
    a_q: &[crate::BlockQ8_0],
    b_q: &[crate::BlockQ8_0],
    c: &mut [f32],
) {
    debug_assert_eq!(k % crate::QK8_0, 0, "k must be a multiple of QK8_0");
    let blocks_per_row = k / crate::QK8_0;
    debug_assert_eq!(a_q.len(), m * blocks_per_row);
    debug_assert_eq!(b_q.len(), n * blocks_per_row);
    debug_assert_eq!(c.len(), m * n);

    let pairs = blocks_per_row / 2;
    let has_tail = blocks_per_row % 2 == 1;

    for i in 0..m {
        let a_row = &a_q[i * blocks_per_row..(i + 1) * blocks_per_row];
        for j in 0..n {
            let b_row = &b_q[j * blocks_per_row..(j + 1) * blocks_per_row];
            let mut acc = 0f32;

            for p in 0..pairs {
                let idx = p * 2;
                let (a0, a1) = (&a_row[idx], &a_row[idx + 1]);
                let (b0, b1) = (&b_row[idx], &b_row[idx + 1]);

                let mut a_buf = [0i8; 64];
                let mut b_buf = [0i8; 64];
                a_buf[0..32].copy_from_slice(&a0.qs);
                a_buf[32..64].copy_from_slice(&a1.qs);
                b_buf[0..32].copy_from_slice(&b0.qs);
                b_buf[32..64].copy_from_slice(&b1.qs);

                let bx = _mm512_loadu_si512(a_buf.as_ptr() as *const __m512i);
                let by = _mm512_loadu_si512(b_buf.as_ptr() as *const __m512i);
                let (isum0, isum1) = dpbusd_pair_halves(bx, by);

                acc += a0.d.to_f32() * b0.d.to_f32() * isum0 as f32;
                acc += a1.d.to_f32() * b1.d.to_f32() * isum1 as f32;
            }

            if has_tail {
                let ab = &a_row[blocks_per_row - 1];
                let bb = &b_row[blocks_per_row - 1];
                let mut isum: i32 = 0;
                for l in 0..crate::QK8_0 {
                    isum += ab.qs[l] as i32 * bb.qs[l] as i32;
                }
                acc += ab.d.to_f32() * bb.d.to_f32() * isum as f32;
            }

            c[i * n + j] = acc;
        }
    }
}

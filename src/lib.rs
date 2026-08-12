//! Independently versioned CPU kernels for the fixed DA3 inference workload.
//!
//! A kernel is admitted only after it is faster than the caller's fallback on
//! the target shape and passes the caller's end-to-end F32 parity gate.

/// The DA3-BASE token count for a 504×336 input (36×24 patches plus special
/// tokens).  This is intentionally explicit: specialised kernels must never
/// pretend to support arbitrary matrix shapes.
pub const DA3_BASE_TOKENS_504X336: usize = 865;

/// Transformer projection shapes eligible for a future specialised kernel.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Da3BaseProjection {
    pub tokens: usize,
    pub input_channels: usize,
    pub output_channels: usize,
}

/// Fused F32 attention for DA3's `[heads, tokens, 64]` layout. The function
/// returns `false` without changing `out` when the CPU or shape is unsupported
/// so its caller can safely use its established fallback.
pub fn flash_attention_f32_da3_base(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    heads: usize,
    tokens: usize,
    out: &mut [f32],
) -> bool {
    if tokens != DA3_BASE_TOKENS_504X336
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

/// The same 4-query × 64-key streaming tile used by ggml's CPU flash path.
/// K is packed once per tile into a dimension-major layout; a ZMM then scores
/// sixteen keys at once. The online softmax keeps the memory footprint O(ND).
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
    use core::arch::x86_64::*;
    use rayon::prelude::*;

    const D: usize = 64;
    const QT: usize = 64;
    const KVT: usize = 64;
    let scale = 1.0f32 / 8.0;

    out.par_chunks_mut(tokens * D)
        .enumerate()
        .take(heads)
        .for_each(|(head, out_head)| {
            let base = head * tokens * D;
            let q_head = &q[base..base + tokens * D];
            let k_head = &k[base..base + tokens * D];
            let v_head = &v[base..base + tokens * D];

            // One 64-query tile is large enough to amortise K packing. Nested
            // Rayon exposes the 14 tiles per head rather than limiting this
            // operation to twelve head-sized jobs on a 16-core benchmark.
            out_head
                .par_chunks_mut(QT * D)
                .enumerate()
                .for_each(|(tile, out_tile)| {
                    let q0 = tile * QT;
                    let rows = out_tile.len() / D;
                    let mut accum = [0.0f32; QT * D];
                    let mut sums = [0.0f32; QT];
                    let mut maxima = [f32::NEG_INFINITY; QT];

                    for k0 in (0..tokens).step_by(KVT) {
                        let cols = (tokens - k0).min(KVT);
                        let mut packed_k = [0.0f32; KVT * D];
                        let mut packed_v = [0.0f32; KVT * D];
                        for key in 0..cols {
                            let source = &k_head[(k0 + key) * D..(k0 + key + 1) * D];
                            let value = &v_head[(k0 + key) * D..(k0 + key + 1) * D];
                            for dim in 0..D {
                                packed_k[dim * KVT + key] = source[dim];
                                packed_v[key * D + dim] = value[dim];
                            }
                        }

                        let mut scores = [0.0f32; QT * KVT];
                        // SAFETY: fixed 64-wide packed matrices, and `rows`
                        // never exceeds the 64-row query tile.
                        unsafe {
                            gemm_4x64_accumulate(
                                rows,
                                &q_head[q0 * D..(q0 + rows) * D],
                                &packed_k,
                                &mut scores,
                            )
                        };

                        for row in 0..rows {
                            let mut tile_max = f32::NEG_INFINITY;
                            for value in &mut scores[row * KVT..row * KVT + cols] {
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
                            for value in &mut accum[row * D..(row + 1) * D] {
                                *value *= correction;
                            }
                            for value in &mut scores[row * KVT..row * KVT + cols] {
                                *value -= new_max;
                            }
                            let score_row: &mut [f32; KVT] = scores[row * KVT..(row + 1) * KVT]
                                .try_into()
                                .expect("fixed score tile");
                            // SAFETY: every score row is the fixed 64-key tile.
                            unsafe { exp_64_avx512(score_row) };
                            for value in &score_row[..cols] {
                                sums[row] += *value;
                            }
                            maxima[row] = new_max;
                        }

                        // SAFETY: scores and V are 64-column matrices and the
                        // accumulator has one 64-float row per valid query.
                        unsafe { gemm_4x64_accumulate(rows, &scores, &packed_v, &mut accum) };
                    }

                    for row in 0..rows {
                        let inverse_sum = 1.0 / sums[row];
                        for dim in 0..D {
                            out_tile[row * D + dim] = accum[row * D + dim] * inverse_sum;
                        }
                    }
                });
        });
}

/// C[m,64] += A[m,64] × B[64,64], with the same 4×64 AVX-512 tile shape as
/// ggml's CPU flash-attention helper. `m` is at most 64.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,fma")]
unsafe fn gemm_4x64_accumulate(m: usize, a: &[f32], b: &[f32], c: &mut [f32]) {
    use core::arch::x86_64::*;
    debug_assert!(m <= 64 && a.len() >= m * 64 && b.len() >= 64 * 64 && c.len() >= m * 64);
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
}

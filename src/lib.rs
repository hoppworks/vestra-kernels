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
    const QT: usize = 4;
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

            for q0 in (0..tokens).step_by(QT) {
                let rows = (tokens - q0).min(QT);
                let mut accum = [[0.0f32; D]; QT];
                let mut sums = [0.0f32; QT];
                let mut maxima = [f32::NEG_INFINITY; QT];

                for k0 in (0..tokens).step_by(KVT) {
                    let cols = (tokens - k0).min(KVT);
                    let mut packed_k = [[0.0f32; KVT]; D];
                    for key in 0..cols {
                        let source = &k_head[(k0 + key) * D..(k0 + key + 1) * D];
                        for dim in 0..D {
                            packed_k[dim][key] = source[dim];
                        }
                    }

                    let mut scores = [[f32::NEG_INFINITY; KVT]; QT];
                    for row in 0..rows {
                        let query = &q_head[(q0 + row) * D..(q0 + row + 1) * D];
                        let mut score_vectors = [_mm512_setzero_ps(); 4];
                        for dim in 0..D {
                            let qv = _mm512_set1_ps(query[dim]);
                            let kp = packed_k[dim].as_ptr();
                            for block in 0..4 {
                                let kv = _mm512_loadu_ps(kp.add(block * 16));
                                score_vectors[block] =
                                    _mm512_fmadd_ps(qv, kv, score_vectors[block]);
                            }
                        }
                        for block in 0..4 {
                            _mm512_storeu_ps(
                                scores[row].as_mut_ptr().add(block * 16),
                                score_vectors[block],
                            );
                        }
                    }

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
                        for dim in 0..D {
                            accum[row][dim] *= correction;
                        }
                        for value in &mut scores[row][..cols] {
                            *value = (*value - new_max).exp();
                            sums[row] += *value;
                        }
                        maxima[row] = new_max;
                    }

                    for row in 0..rows {
                        let mut result = [_mm512_setzero_ps(); 4];
                        for key in 0..cols {
                            let probability = _mm512_set1_ps(scores[row][key]);
                            let value = &v_head[(k0 + key) * D..(k0 + key + 1) * D];
                            for block in 0..4 {
                                let vv = _mm512_loadu_ps(value.as_ptr().add(block * 16));
                                result[block] = _mm512_fmadd_ps(probability, vv, result[block]);
                            }
                        }
                        for block in 0..4 {
                            let mut partial = [0.0f32; 16];
                            _mm512_storeu_ps(partial.as_mut_ptr(), result[block]);
                            for lane in 0..16 {
                                accum[row][block * 16 + lane] += partial[lane];
                            }
                        }
                    }
                }

                for row in 0..rows {
                    let inverse_sum = 1.0 / sums[row];
                    for dim in 0..D {
                        out_head[(q0 + row) * D + dim] = accum[row][dim] * inverse_sum;
                    }
                }
            }
        });
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

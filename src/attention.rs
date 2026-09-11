//! Scaled-dot-product attention: a naive GEMM+softmax+GEMM oracle and a
//! tiled, online-softmax implementation.

use rayon::prelude::*;

/// Naive reference: per head, `softmax(Q @ K^T / sqrt(head_dim)) @ V`.
///
/// `q`, `k`, `v` are `[heads, n, head_dim]` row-major; `out` is written in
/// the same layout.
pub fn attention_naive(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    heads: usize,
    n: usize,
    head_dim: usize,
    out: &mut [f32],
) {
    assert_eq!(q.len(), heads * n * head_dim);
    assert_eq!(k.len(), heads * n * head_dim);
    assert_eq!(v.len(), heads * n * head_dim);
    assert_eq!(out.len(), heads * n * head_dim);
    let scale = 1.0f32 / (head_dim as f32).sqrt();
    let mut scores = vec![0f32; n];
    for h in 0..heads {
        let qh = &q[h * n * head_dim..(h + 1) * n * head_dim];
        let kh = &k[h * n * head_dim..(h + 1) * n * head_dim];
        let vh = &v[h * n * head_dim..(h + 1) * n * head_dim];
        let oh = &mut out[h * n * head_dim..(h + 1) * n * head_dim];
        for i in 0..n {
            let qi = &qh[i * head_dim..(i + 1) * head_dim];
            let mut max_score = f32::NEG_INFINITY;
            for j in 0..n {
                let kj = &kh[j * head_dim..(j + 1) * head_dim];
                let dot: f32 = qi.iter().zip(kj.iter()).map(|(a, b)| a * b).sum();
                let s = dot * scale;
                scores[j] = s;
                if s > max_score {
                    max_score = s;
                }
            }
            let mut sum = 0f32;
            for value in &mut scores {
                *value = (*value - max_score).exp();
                sum += *value;
            }
            let inv_sum = 1.0f32 / sum;
            let oi = &mut oh[i * head_dim..(i + 1) * head_dim];
            oi.fill(0.0);
            for j in 0..n {
                let w = scores[j] * inv_sum;
                let vj = &vh[j * head_dim..(j + 1) * head_dim];
                for d in 0..head_dim {
                    oi[d] += w * vj[d];
                }
            }
        }
    }
}

const KV_TILE: usize = 64;

/// Tiled online-softmax attention.  Every `(head, query)` row is independent,
/// so Rayon distributes rows while preserving each row's exact K traversal and
/// F32 accumulation order.
pub fn attention(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    heads: usize,
    n: usize,
    head_dim: usize,
    out: &mut [f32],
) {
    assert_eq!(q.len(), heads * n * head_dim);
    assert_eq!(k.len(), heads * n * head_dim);
    assert_eq!(v.len(), heads * n * head_dim);
    assert_eq!(out.len(), heads * n * head_dim);
    if head_dim == 64 && crate::specialized::flash_attention_f32_da3_base(q, k, v, heads, n, out) {
        return;
    }

    let scale = 1.0f32 / (head_dim as f32).sqrt();
    #[cfg(target_arch = "x86_64")]
    let use_avx512 = std::is_x86_feature_detected!("avx512f");
    #[cfg(not(target_arch = "x86_64"))]
    let use_avx512 = false;
    out.par_chunks_mut(head_dim).enumerate().for_each_init(
        || vec![0.0; head_dim],
        |acc, (row, oi)| {
            let h = row / n;
            let i = row % n;
            let base = h * n * head_dim;
            attention_row(
                &q[base..base + n * head_dim],
                &k[base..base + n * head_dim],
                &v[base..base + n * head_dim],
                i,
                n,
                head_dim,
                scale,
                use_avx512,
                acc,
                oi,
            );
        },
    );
}

#[allow(clippy::too_many_arguments)]
fn attention_row(
    qh: &[f32],
    kh: &[f32],
    vh: &[f32],
    i: usize,
    n: usize,
    head_dim: usize,
    scale: f32,
    use_avx512: bool,
    acc: &mut [f32],
    oi: &mut [f32],
) {
    let qi = &qh[i * head_dim..(i + 1) * head_dim];
    let mut running_max = f32::NEG_INFINITY;
    let mut running_sum = 0f32;
    acc.fill(0.0);
    let mut j0 = 0usize;
    while j0 < n {
        let j1 = (j0 + KV_TILE).min(n);
        let mut local_scores = [0f32; KV_TILE];
        let mut tile_max = f32::NEG_INFINITY;
        for (t, j) in (j0..j1).enumerate() {
            let kj = &kh[j * head_dim..(j + 1) * head_dim];
            #[cfg(target_arch = "x86_64")]
            let dot = if use_avx512 {
                // SAFETY: runtime feature detection above guarantees AVX-512F.
                unsafe { crate::simd_avx512::dot_avx512(qi, kj) }
            } else {
                qi.iter().zip(kj.iter()).map(|(a, b)| a * b).sum()
            };
            #[cfg(not(target_arch = "x86_64"))]
            let dot: f32 = qi.iter().zip(kj.iter()).map(|(a, b)| a * b).sum();
            let score = dot * scale;
            local_scores[t] = score;
            if score > tile_max {
                tile_max = score;
            }
        }
        let new_max = running_max.max(tile_max);
        let correction = if running_max.is_finite() {
            (running_max - new_max).exp()
        } else {
            0.0
        };
        running_sum *= correction;
        for value in acc.iter_mut() {
            *value *= correction;
        }
        for score in &mut local_scores[..j1 - j0] {
            *score -= new_max;
        }
        #[cfg(target_arch = "x86_64")]
        if use_avx512 {
            // SAFETY: runtime feature detection above guarantees AVX-512F.
            unsafe { crate::simd_avx512::exp_in_place_avx512(&mut local_scores[..j1 - j0]) };
        } else {
            for score in &mut local_scores[..j1 - j0] {
                *score = score.exp();
            }
        }
        #[cfg(not(target_arch = "x86_64"))]
        for score in &mut local_scores[..j1 - j0] {
            let _ = use_avx512;
            *score = score.exp();
        }
        for (t, j) in (j0..j1).enumerate() {
            let p = local_scores[t];
            running_sum += p;
            let vj = &vh[j * head_dim..(j + 1) * head_dim];
            #[cfg(target_arch = "x86_64")]
            if use_avx512 {
                // SAFETY: runtime feature detection above guarantees AVX-512F.
                unsafe { crate::simd_avx512::scaled_add_avx512(acc, p, vj) };
            } else {
                for d in 0..head_dim {
                    acc[d] += p * vj[d];
                }
            }
            #[cfg(not(target_arch = "x86_64"))]
            for d in 0..head_dim {
                acc[d] += p * vj[d];
            }
        }
        running_max = new_max;
        j0 = j1;
    }
    let inv_sum = 1.0f32 / running_sum;
    for d in 0..head_dim {
        oi[d] = acc[d] * inv_sum;
    }
}

#[cfg(test)]
pub(crate) fn attention_serial(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    heads: usize,
    n: usize,
    head_dim: usize,
    out: &mut [f32],
) {
    let scale = 1.0f32 / (head_dim as f32).sqrt();
    let use_avx512 = false;
    let mut acc = vec![0.0; head_dim];
    for row in 0..heads * n {
        let h = row / n;
        let i = row % n;
        let base = h * n * head_dim;
        attention_row(
            &q[base..base + n * head_dim],
            &k[base..base + n * head_dim],
            &v[base..base + n * head_dim],
            i,
            n,
            head_dim,
            scale,
            use_avx512,
            &mut acc,
            &mut out[row * head_dim..(row + 1) * head_dim],
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn values(len: usize, seed: u32) -> Vec<f32> {
        let mut state = seed;
        (0..len)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 17;
                state ^= state << 5;
                (state as f32 / u32::MAX as f32) * 2.0 - 1.0
            })
            .collect()
    }

    #[test]
    fn parallel_attention_is_bitwise_serial_per_row() {
        let (heads, n, dim) = (3, 71, 16);
        let q = values(heads * n * dim, 0xA771_0001);
        let k = values(heads * n * dim, 0xA771_0002);
        let v = values(heads * n * dim, 0xA771_0003);
        let mut parallel = vec![0.0; heads * n * dim];
        let mut serial = vec![0.0; heads * n * dim];
        attention(&q, &k, &v, heads, n, dim, &mut parallel);
        attention_serial(&q, &k, &v, heads, n, dim, &mut serial);
        assert_eq!(parallel, serial);
    }
}

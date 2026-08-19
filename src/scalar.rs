use rayon::prelude::*;

pub fn gemm_f32(m: usize, n: usize, k: usize, a: &[f32], b: &[f32], c: &mut [f32]) {
    debug_assert_eq!(a.len(), m * k);
    debug_assert_eq!(b.len(), k * n);
    debug_assert_eq!(c.len(), m * n);
    for i in 0..m {
        for j in 0..n {
            c[i * n + j] = 0.0;
        }
        for p in 0..k {
            let aip = a[i * k + p];
            for j in 0..n {
                c[i * n + j] += aip * b[p * n + j];
            }
        }
    }
}

/// q8_0 x q8_0 -> f32 GEMM: A is `m x k` and B is `n x k`, both stored as
/// row-major q8_0 blocks (`k/QK8_0` blocks per row; B's rows are the *n*
/// columns of the logical `k x n` matrix, i.e. B is pre-transposed into
/// row blocks the same way A is). This is the scalar oracle: every other
/// backend (AVX-512/VNNI) must match this within the test's tolerance band.
pub fn gemm_q8_0(
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

    for i in 0..m {
        let a_row = &a_q[i * blocks_per_row..(i + 1) * blocks_per_row];
        for j in 0..n {
            let b_row = &b_q[j * blocks_per_row..(j + 1) * blocks_per_row];
            let mut acc = 0f32;
            for bi in 0..blocks_per_row {
                let ab = &a_row[bi];
                let bb = &b_row[bi];
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

pub fn add(dst: &mut [f32], src: &[f32]) {
    debug_assert_eq!(dst.len(), src.len());
    for i in 0..dst.len() {
        dst[i] += src[i];
    }
}

pub fn add_bias_rows(x: &mut [f32], rows: usize, cols: usize, bias: &[f32]) {
    debug_assert_eq!(x.len(), rows * cols);
    debug_assert_eq!(bias.len(), cols);
    if rows >= 32 {
        x.par_chunks_mut(cols).for_each(|row| {
            for c in 0..cols {
                row[c] += bias[c];
            }
        });
    } else {
        for row in x.chunks_mut(cols) {
            for c in 0..cols {
                row[c] += bias[c];
            }
        }
    }
}

pub fn layernorm(x: &mut [f32], rows: usize, cols: usize, gamma: &[f32], beta: &[f32], eps: f32) {
    debug_assert_eq!(x.len(), rows * cols);
    // Rows are independent and each keeps the exact scalar reduction order.
    // This is particularly important for Q/K normalization, which has over
    // ten thousand short rows per late DA3-BASE transformer block.
    if rows >= 32 {
        x.par_chunks_mut(cols)
            .for_each(|row| layernorm_row(row, gamma, beta, eps));
    } else {
        for row in x.chunks_mut(cols) {
            layernorm_row(row, gamma, beta, eps);
        }
    }
}

#[inline]
fn layernorm_row(row: &mut [f32], gamma: &[f32], beta: &[f32], eps: f32) {
    let cols = row.len();
    let mean = row.iter().sum::<f32>() / cols as f32;
    let var = row
        .iter()
        .map(|v| {
            let d = v - mean;
            d * d
        })
        .sum::<f32>()
        / cols as f32;
    let inv = 1.0 / (var + eps).sqrt();
    for c in 0..cols {
        row[c] = (row[c] - mean) * inv * gamma[c] + beta[c];
    }
}

pub fn gelu(x: &mut [f32]) {
    const INV_SQRT2: f32 = std::f32::consts::FRAC_1_SQRT_2;
    for v in x.iter_mut() {
        *v = 0.5 * *v * (1.0 + erf(*v * INV_SQRT2));
    }
}

// Abramowitz–Stegun 7.1.26 erf-Approximation (|error| < 1.5e-7).
fn erf(x: f32) -> f32 {
    let s = x.signum();
    let x = x.abs();
    let t = 1.0 / (1.0 + 0.327_591_1 * x);
    let y = 1.0
        - (((((1.061_405_4 * t - 1.453_152) * t) + 1.421_413_7) * t - 0.284_496_74) * t
            + 0.254_829_6)
            * t
            * (-x * x).exp();
    s * y
}

/// In-place per-column ("LayerScale") scale: `x[r,c] *= gamma[c]`. Mirrors
/// `add_bias_rows`'s shape convention exactly (row-major `[rows, cols]`,
/// one scale factor per column, broadcast over rows).
pub fn layerscale(x: &mut [f32], rows: usize, cols: usize, gamma: &[f32]) {
    debug_assert_eq!(x.len(), rows * cols);
    debug_assert_eq!(gamma.len(), cols);
    if rows >= 32 {
        x.par_chunks_mut(cols).for_each(|row| {
            for c in 0..cols {
                row[c] *= gamma[c];
            }
        });
    } else {
        for row in x.chunks_mut(cols) {
            for c in 0..cols {
                row[c] *= gamma[c];
            }
        }
    }
}

pub fn softmax_rows(x: &mut [f32], rows: usize, cols: usize) {
    debug_assert_eq!(x.len(), rows * cols);
    for r in 0..rows {
        let row = &mut x[r * cols..(r + 1) * cols];
        let m = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0;
        for v in row.iter_mut() {
            *v = (*v - m).exp();
            sum += *v;
        }
        let inv = 1.0 / sum;
        for v in row.iter_mut() {
            *v *= inv;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{layernorm, layernorm_row};

    #[test]
    fn parallel_layernorm_is_bitwise_identical_per_row() {
        let rows = 67;
        let cols = 64;
        let gamma: Vec<f32> = (0..cols).map(|i| 0.1 + i as f32 * 0.01).collect();
        let beta: Vec<f32> = (0..cols).map(|i| -0.2 + i as f32 * 0.02).collect();
        let mut parallel: Vec<f32> = (0..rows * cols)
            .map(|i| ((i * 37 % 101) as f32 - 50.0) * 0.03125)
            .collect();
        let mut sequential = parallel.clone();

        layernorm(&mut parallel, rows, cols, &gamma, &beta, 1e-5);
        for row in sequential.chunks_mut(cols) {
            layernorm_row(row, &gamma, &beta, 1e-5);
        }
        assert_eq!(
            parallel
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            sequential
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
        );
    }

    #[test]
    fn parallel_rowwise_epilogues_are_bitwise_identical() {
        let rows = 67;
        let cols = 64;
        let bias: Vec<f32> = (0..cols).map(|i| i as f32 * 0.01 - 0.2).collect();
        let gamma: Vec<f32> = (0..cols).map(|i| i as f32 * 0.02 + 0.3).collect();
        let input: Vec<f32> = (0..rows * cols)
            .map(|i| (i as f32 - 1000.0) * 0.03125)
            .collect();
        let mut parallel = input.clone();
        let mut sequential = input;

        super::add_bias_rows(&mut parallel, rows, cols, &bias);
        super::layerscale(&mut parallel, rows, cols, &gamma);
        for row in sequential.chunks_mut(cols) {
            for c in 0..cols {
                row[c] += bias[c];
            }
            for c in 0..cols {
                row[c] *= gamma[c];
            }
        }
        assert_eq!(
            parallel
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            sequential
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
        );
    }
}

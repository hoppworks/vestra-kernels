//! 2D rotary position embeddings (RoPE2D) for vision transformers.
//!
//! Convention chosen (documented here since there is currently no reference
//! dump to verify against — see `tests/rope_parity.rs`):
//!
//! `head_dim` is split into two equal halves: the first half encodes the
//! *y* coordinate, the second half encodes the *x* coordinate. Within each
//! half (size `half_dim = head_dim / 2`) we apply the standard GPT-NeoX /
//! LLaMA "rotate-half" 1D RoPE formulation: the half is itself split into
//! two quarters of size `half_dim / 2`, and dimension `i` is paired with
//! dimension `i + half_dim/2` for `i in 0..half_dim/2`. The rotation angle
//! for pair `i` is `pos * theta_i` with `theta_i = freq^(-2*i/half_dim)`.
//!
//! This is the "2D-RoPE for ViTs" convention used by EVA-02 / RoPE-ViT /
//! Qwen2-VL-style vision encoders: one rotary sub-space per axis, each
//! sub-space using the ordinary rotate-half pairing rather than adjacent-pair
//! interleaving. It is a single, consistent, well-known convention, but it
//! has **not** been verified against the C++ reference implementation here
//! (no `dumps/reference.gguf` available in this environment). Once the dump
//! exists, `tests/rope_parity.rs` will exercise this against `rope_out`.

/// Applies 2D-RoPE in-place to `x`, a `[heads, n, head_dim]` tensor stored
/// row-major (`x[((h * n + p) * head_dim) + d]`).
///
/// `pos_yx` is `[n * 2]`: `pos_yx[2*p]` = y coordinate, `pos_yx[2*p+1]` = x
/// coordinate of sequence position `p`.
///
/// `head_dim` must be a multiple of 4 (so it splits evenly into a y-half and
/// an x-half, each of which splits evenly into rotation pairs).
pub fn rope2d(x: &mut [f32], heads: usize, n: usize, head_dim: usize, pos_yx: &[i64], freq: f32) {
    assert_eq!(x.len(), heads * n * head_dim, "rope2d: x size mismatch");
    assert_eq!(pos_yx.len(), n * 2, "rope2d: pos_yx size mismatch");
    assert!(
        head_dim % 4 == 0,
        "rope2d: head_dim must be divisible by 4, got {head_dim}"
    );

    let half_dim = head_dim / 2;
    let quarter = half_dim / 2;

    // Precompute per-quarter-index inverse frequencies (shared by both axes).
    let inv_freq: Vec<f32> = (0..quarter)
        .map(|i| freq.powf(-2.0 * i as f32 / half_dim as f32))
        .collect();

    // Positions are shared by every head. Computing sin/cos inside the head
    // loop repeated the same transcendental operations `heads` times; cache
    // them per (token, axis, pair) and retain the exact per-element rotation.
    let mut rotations = vec![0.0f32; n * 2 * quarter * 2];
    for p in 0..n {
        let y = pos_yx[2 * p] as f32;
        let xc = pos_yx[2 * p + 1] as f32;
        for (axis, position) in [y, xc].into_iter().enumerate() {
            for (i, frequency) in inv_freq.iter().enumerate() {
                let (sin, cos) = (position * frequency).sin_cos();
                let offset = (p * 2 * quarter + axis * quarter + i) * 2;
                rotations[offset] = sin;
                rotations[offset + 1] = cos;
            }
        }

        for h in 0..heads {
            let base = (h * n + p) * head_dim;

            // y-half: dims [0, half_dim)
            rotate_half(
                &mut x[base..base + half_dim],
                quarter,
                &rotations[(p * 2 * quarter) * 2..(p * 2 * quarter + quarter) * 2],
            );

            // x-half: dims [half_dim, head_dim)
            rotate_half(
                &mut x[base + half_dim..base + head_dim],
                quarter,
                &rotations[(p * 2 * quarter + quarter) * 2..(p * 2 * quarter + 2 * quarter) * 2],
            );
        }
    }
}

/// Applies GPT-NeoX-style "rotate-half" RoPE to a single axis-half slice of
/// length `2 * quarter`, pairing `i` with `i + quarter`.
fn rotate_half(slice: &mut [f32], quarter: usize, rotations: &[f32]) {
    for i in 0..quarter {
        let sin = rotations[i * 2];
        let cos = rotations[i * 2 + 1];
        let a = slice[i];
        let b = slice[i + quarter];
        slice[i] = a * cos - b * sin;
        slice[i + quarter] = b * cos + a * sin;
    }
}

#[cfg(test)]
mod tests {
    use super::rope2d;

    fn rope2d_uncached(
        x: &mut [f32],
        heads: usize,
        n: usize,
        head_dim: usize,
        pos_yx: &[i64],
        freq: f32,
    ) {
        let half_dim = head_dim / 2;
        let quarter = half_dim / 2;
        let inv_freq: Vec<f32> = (0..quarter)
            .map(|i| freq.powf(-2.0 * i as f32 / half_dim as f32))
            .collect();
        for p in 0..n {
            for h in 0..heads {
                let base = (h * n + p) * head_dim;
                for (axis, position) in [pos_yx[2 * p] as f32, pos_yx[2 * p + 1] as f32]
                    .into_iter()
                    .enumerate()
                {
                    let offset = base + axis * half_dim;
                    for i in 0..quarter {
                        let (sin, cos) = (position * inv_freq[i]).sin_cos();
                        let a = x[offset + i];
                        let b = x[offset + i + quarter];
                        x[offset + i] = a * cos - b * sin;
                        x[offset + i + quarter] = b * cos + a * sin;
                    }
                }
            }
        }
    }

    #[test]
    fn cached_rotations_are_bitwise_identical_to_per_head_rotation() {
        let heads = 3;
        let n = 7;
        let head_dim = 16;
        let positions: Vec<i64> = (0..n).flat_map(|i| [i as i64, (i * 3) as i64]).collect();
        let mut cached: Vec<f32> = (0..heads * n * head_dim)
            .map(|i| (i as f32 - 43.0) * 0.0625)
            .collect();
        let mut uncached = cached.clone();
        rope2d(&mut cached, heads, n, head_dim, &positions, 100.0);
        rope2d_uncached(&mut uncached, heads, n, head_dim, &positions, 100.0);
        assert_eq!(
            cached
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            uncached
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
        );
    }
}

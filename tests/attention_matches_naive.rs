use vestra_kernels::{attention, attention_naive};

/// Deterministic xorshift32 PRNG so the test has no extra dependency and is
/// reproducible across runs.
struct Xorshift32(u32);
impl Xorshift32 {
    fn next_f32(&mut self) -> f32 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        self.0 = x;
        // map to roughly [-1, 1)
        ((x as f32) / (u32::MAX as f32)) * 2.0 - 1.0
    }
}

fn random_vec(rng: &mut Xorshift32, n: usize) -> Vec<f32> {
    (0..n).map(|_| rng.next_f32()).collect()
}

fn run_case(heads: usize, n: usize, head_dim: usize, seed: u32) {
    let mut rng = Xorshift32(seed);
    let q = random_vec(&mut rng, heads * n * head_dim);
    let k = random_vec(&mut rng, heads * n * head_dim);
    let v = random_vec(&mut rng, heads * n * head_dim);

    let mut out_naive = vec![0f32; heads * n * head_dim];
    let mut out_tiled = vec![0f32; heads * n * head_dim];

    attention_naive(&q, &k, &v, heads, n, head_dim, &mut out_naive);
    attention(&q, &k, &v, heads, n, head_dim, &mut out_tiled);

    for i in 0..out_naive.len() {
        let a = out_naive[i];
        let b = out_tiled[i];
        assert!(
            (a - b).abs() < 1e-4,
            "heads={heads} n={n} head_dim={head_dim} i={i} naive={a} tiled={b}"
        );
    }
}

#[test]
fn attention_tiled_matches_naive_small() {
    run_case(2, 8, 16, 0x1234_5678);
}

#[test]
fn attention_tiled_matches_naive_multihead() {
    run_case(4, 37, 32, 0xdead_beef);
}

/// Exercises the KV_TILE (=64) tiling boundary in the online-softmax path:
/// sequence length spans multiple tiles and crosses a tile edge exactly.
#[test]
fn attention_tiled_matches_naive_across_tile_boundary() {
    run_case(1, 130, 8, 0x0bad_f00d);
}

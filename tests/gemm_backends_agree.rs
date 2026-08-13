use vestra_kernels::gemm::{FaerGemm, Gemm, ScalarGemm};

fn rand_vec(n: usize, seed: u64) -> Vec<f32> {
    // deterministischer LCG, kein rand-crate nötig
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((s >> 33) as f32 / u32::MAX as f32) - 0.5
        })
        .collect()
}

#[test]
fn faer_matches_scalar() {
    let (m, n, k) = (64, 48, 80);
    let a = rand_vec(m * k, 1);
    let b = rand_vec(k * n, 2);
    let mut cs = vec![0.; m * n];
    let mut cf = vec![0.; m * n];
    ScalarGemm.gemm(m, n, k, &a, &b, &mut cs);
    FaerGemm.gemm(m, n, k, &a, &b, &mut cf);
    for i in 0..m * n {
        assert!(
            (cs[i] - cf[i]).abs() <= 1e-3 + 1e-3 * cs[i].abs(),
            "i={i} scalar={} faer={}",
            cs[i],
            cf[i]
        );
    }
}

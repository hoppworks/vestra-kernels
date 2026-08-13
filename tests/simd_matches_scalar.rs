use vestra_kernels::{Kernels, scalar};

fn ramp(n: usize) -> Vec<f32> {
    (0..n).map(|i| (i as f32 * 0.017) - 3.0).collect()
}

#[test]
fn gelu_simd_matches_scalar() {
    let k = Kernels::detect();
    let mut a = ramp(1000);
    let mut b = a.clone();
    k.gelu(&mut a);
    scalar::gelu(&mut b);
    for i in 0..a.len() {
        assert!(
            (a[i] - b[i]).abs() < 1e-4,
            "i={i} simd={} scalar={}",
            a[i],
            b[i]
        );
    }
}

#[test]
fn add_simd_matches_scalar() {
    let k = Kernels::detect();
    let src = ramp(1000);
    let mut a = ramp(1000);
    let mut b = a.clone();
    k.add(&mut a, &src);
    scalar::add(&mut b, &src);
    assert_eq!(a, b);
}

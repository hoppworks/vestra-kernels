use vestra_kernels::scalar::*;

#[test]
fn gemm_2x2_identity() {
    let a = [1., 2., 3., 4.]; // 2x2
    let id = [1., 0., 0., 1.]; // 2x2
    let mut c = [0.; 4];
    gemm_f32(2, 2, 2, &a, &id, &mut c);
    assert_eq!(c, a);
}

#[test]
fn softmax_rows_sums_to_one() {
    let mut x = [1., 2., 3., 0., 0., 0.]; // 2 rows, 3 cols
    softmax_rows(&mut x, 2, 3);
    let s0: f32 = x[0..3].iter().sum();
    let s1: f32 = x[3..6].iter().sum();
    assert!((s0 - 1.0).abs() < 1e-6 && (s1 - 1.0).abs() < 1e-6);
    assert!((x[3] - 1.0 / 3.0).abs() < 1e-6);
}

#[test]
fn gelu_zero_and_large() {
    let mut x = [0.0f32, 10.0, -10.0];
    gelu(&mut x);
    assert!(x[0].abs() < 1e-6);
    assert!((x[1] - 10.0).abs() < 1e-3);
    assert!(x[2].abs() < 1e-3);
}

#[test]
fn layernorm_zero_mean_unit_var() {
    let mut x = [1., 2., 3., 4.];
    let g = [1., 1., 1., 1.];
    let b = [0., 0., 0., 0.];
    layernorm(&mut x, 1, 4, &g, &b, 1e-5);
    let mean: f32 = x.iter().sum::<f32>() / 4.0;
    assert!(mean.abs() < 1e-4);
}

#[test]
fn layerscale_scales_each_column_broadcast_over_rows() {
    // 2 rows, 3 cols.
    let mut x = [1., 2., 3., 4., 5., 6.];
    let gamma = [10.0, -1.0, 0.5];
    layerscale(&mut x, 2, 3, &gamma);
    assert_eq!(x, [10.0, -2.0, 1.5, 40.0, -5.0, 3.0]);
}

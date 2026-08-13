use half::f16;
use vestra_kernels::{BlockQ8_0, QK8_0, dequantize_q8_0};
use vestra_kernels::{Kernels, quantize_row_q8_0, scalar};

fn quantize_matrix(x: &[f32], rows: usize, k: usize) -> Vec<BlockQ8_0> {
    let mut out = vec![
        BlockQ8_0 {
            d: f16::from_f32(0.0),
            qs: [0; 32]
        };
        rows * (k / QK8_0)
    ];
    for r in 0..rows {
        let blocks_per_row = k / QK8_0;
        quantize_row_q8_0(
            &x[r * k..(r + 1) * k],
            &mut out[r * blocks_per_row..(r + 1) * blocks_per_row],
        );
    }
    out
}

#[test]
fn q8_0_gemm_close_to_f32() {
    let (m, n, k) = (8, 8, 64);
    let a: Vec<f32> = (0..m * k).map(|i| ((i % 17) as f32 - 8.0) * 0.1).collect();
    let b: Vec<f32> = (0..k * n).map(|i| ((i % 13) as f32 - 6.0) * 0.1).collect();
    // B as q8_0 row blocks (B^T-blocks): B is k×n; we quantize the n columns
    // as rows -> transpose first.
    let mut bt = vec![0f32; n * k];
    for p in 0..k {
        for j in 0..n {
            bt[j * k + p] = b[p * n + j];
        }
    }
    let aq = quantize_matrix(&a, m, k);
    let bq = quantize_matrix(&bt, n, k);
    let k_dev = Kernels::detect();
    let mut c = vec![0f32; m * n];
    k_dev.gemm_q8_0(m, n, k, &aq, &bq, &mut c);
    // Reference: dequantize and f32-GEMM.
    let mut a_de = vec![0f32; m * k];
    dequantize_q8_0(&aq, &mut a_de);
    let mut bt_de = vec![0f32; n * k];
    dequantize_q8_0(&bq, &mut bt_de);
    let mut b_de = vec![0f32; k * n];
    for j in 0..n {
        for p in 0..k {
            b_de[p * n + j] = bt_de[j * k + p];
        }
    }
    let mut c_ref = vec![0f32; m * n];
    scalar::gemm_f32(m, n, k, &a_de, &b_de, &mut c_ref);
    for i in 0..m * n {
        assert!(
            (c[i] - c_ref[i]).abs() < 1e-2 + 1e-2 * c_ref[i].abs(),
            "i={i} q8={} f32={}",
            c[i],
            c_ref[i]
        );
    }
}

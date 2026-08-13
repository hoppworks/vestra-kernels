//! Baseline timing for the active DA3-specific projection kernels.
//!
//! This intentionally mirrors `blis_shape_bench.rs`: same four matrix
//! shapes, ten warm measurements and median reporting. It is a feasibility
//! tool only; the end-to-end benchmark remains the DA CLI protocol.

use std::time::Instant;
use vestra_kernels::{linear_bias_scale_f32_da3_base, linear_f32_da3_base, qkv_f32_da3_base};

const TOKENS: usize = 865;

fn values(len: usize, seed: u32) -> Vec<f32> {
    let mut state = seed;
    (0..len)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            (state as f32 / u32::MAX as f32) - 0.5
        })
        .collect()
}

fn median_ms(mut samples: Vec<f64>) -> f64 {
    samples.sort_by(f64::total_cmp);
    (samples[4] + samples[5]) * 0.5
}

fn raw_linear(name: &str, n: usize, k: usize) {
    let a = values(TOKENS * k, 0xA11C_E001);
    let b = values(k * n, 0xB11C_E002);
    let mut c = vec![0.0; TOKENS * n];
    assert!(linear_f32_da3_base(TOKENS, n, k, &a, &b, &mut c));
    let mut samples = Vec::with_capacity(10);
    for _ in 0..10 {
        let started = Instant::now();
        assert!(linear_f32_da3_base(TOKENS, n, k, &a, &b, &mut c));
        samples.push(started.elapsed().as_secs_f64() * 1_000.0);
    }
    let median = median_ms(samples);
    let gflops = 2.0 * TOKENS as f64 * n as f64 * k as f64 / 1e9 / (median / 1_000.0);
    println!("{name} m={TOKENS} n={n} k={k} median_ms={median:.3} gflops={gflops:.1}");
}

fn bias_scale_linear(name: &str, n: usize, k: usize) {
    let a = values(TOKENS * k, 0xA11C_E101);
    let b = values(k * n, 0xB11C_E102);
    let bias = values(n, 0xB11C_E103);
    let scale = values(n, 0xB11C_E104);
    let mut c = vec![0.0; TOKENS * n];
    assert!(linear_bias_scale_f32_da3_base(
        TOKENS, n, k, &a, &b, &bias, &scale, &mut c
    ));
    let mut samples = Vec::with_capacity(10);
    for _ in 0..10 {
        let started = Instant::now();
        assert!(linear_bias_scale_f32_da3_base(
            TOKENS, n, k, &a, &b, &bias, &scale, &mut c
        ));
        samples.push(started.elapsed().as_secs_f64() * 1_000.0);
    }
    let median = median_ms(samples);
    let gflops = 2.0 * TOKENS as f64 * n as f64 * k as f64 / 1e9 / (median / 1_000.0);
    println!("{name} m={TOKENS} n={n} k={k} median_ms={median:.3} gflops={gflops:.1}");
}

fn qkv() {
    let a = values(TOKENS * 768, 0xA11C_E201);
    let weight = values(2304 * 768, 0xB11C_E202);
    let bias = values(2304, 0xB11C_E203);
    let mut q = vec![0.0; 12 * TOKENS * 64];
    let mut k = vec![0.0; 12 * TOKENS * 64];
    let mut v = vec![0.0; 12 * TOKENS * 64];
    assert!(qkv_f32_da3_base(&a, &weight, &bias, &mut q, &mut k, &mut v));
    let mut samples = Vec::with_capacity(10);
    for _ in 0..10 {
        let started = Instant::now();
        assert!(qkv_f32_da3_base(&a, &weight, &bias, &mut q, &mut k, &mut v));
        samples.push(started.elapsed().as_secs_f64() * 1_000.0);
    }
    let median = median_ms(samples);
    let gflops = 2.0 * TOKENS as f64 * 2304.0 * 768.0 / 1e9 / (median / 1_000.0);
    println!("qkv m={TOKENS} n=2304 k=768 median_ms={median:.3} gflops={gflops:.1}");
}

fn main() {
    qkv();
    bias_scale_linear("projection", 768, 768);
    raw_linear("fc1", 3072, 768);
    bias_scale_linear("fc2", 768, 3072);
}

//! Standalone BLIS feasibility test for the exact DA3-BASE linear shapes.
//!
//! Build only on a host that supplies an OpenMP BLIS library, for example:
//! `RUSTFLAGS='-L native=/path/to/lib' cargo run --locked --release --example blis_shape_bench`.
//! This is deliberately not runtime wiring: it answers whether a different
//! Zen-tuned GEMM backend can clear the required microbenchmark threshold.

use std::time::Instant;

#[link(name = "bliso")]
unsafe extern "C" {
    fn bli_thread_set_num_threads(n_threads: i64);
    // BLIS is deliberately built without the optional CBLAS ABI. The
    // portable Fortran BLAS entry point below is column-major; our probe
    // allocates each operand in that layout, so this measures BLIS's actual
    // SGEMM path rather than a row-major adapter.
    fn sgemm_(
        trans_a: *const u8,
        trans_b: *const u8,
        m: *const i32,
        n: *const i32,
        k: *const i32,
        alpha: *const f32,
        a: *const f32,
        lda: *const i32,
        b: *const f32,
        ldb: *const i32,
        beta: *const f32,
        c: *mut f32,
        ldc: *const i32,
    );
}

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

fn run(name: &str, m: usize, n: usize, k: usize) {
    let a = values(m * k, 0xA11C_E001);
    let b = values(k * n, 0xB11C_E002);
    let mut c = vec![0.0; m * n];
    let (m_i32, n_i32, k_i32) = (m as i32, n as i32, k as i32);
    let (alpha, beta) = (1.0f32, 0.0f32);
    let trans = b'N';
    let mut gemm = || unsafe {
        sgemm_(
            &trans,
            &trans,
            &m_i32,
            &n_i32,
            &k_i32,
            &alpha,
            a.as_ptr(),
            &m_i32,
            b.as_ptr(),
            &k_i32,
            &beta,
            c.as_mut_ptr(),
            &m_i32,
        )
    };
    gemm();
    let mut samples = Vec::with_capacity(10);
    for _ in 0..10 {
        let started = Instant::now();
        gemm();
        samples.push(started.elapsed().as_secs_f64() * 1_000.0);
    }
    samples.sort_by(f64::total_cmp);
    let median = (samples[4] + samples[5]) * 0.5;
    let gflop = 2.0 * m as f64 * n as f64 * k as f64 / 1e9;
    println!(
        "{name} m={m} n={n} k={k} median_ms={median:.3} gflops={:.1}",
        gflop / (median / 1_000.0)
    );
}

fn main() {
    unsafe { bli_thread_set_num_threads(16) };
    run("qkv", 865, 2304, 768);
    run("projection", 865, 768, 768);
    run("fc1", 865, 3072, 768);
    run("fc2", 865, 768, 3072);
}

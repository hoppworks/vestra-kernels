#[cfg(target_arch = "x86_64")]
#[allow(
    clippy::approx_constant,
    clippy::excessive_precision,
    reason = "the benchmark must use the same established SIMD polynomial coefficients as the kernel"
)]
mod x86_64_benchmark {
    use std::{hint::black_box, time::Instant};

    use faer::Parallelism;
    use rayon::prelude::*;

    const H: usize = 12;
    const N: usize = 865;
    const D: usize = 64;

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx512f,fma")]
    unsafe fn exp16(x: core::arch::x86_64::__m512) -> core::arch::x86_64::__m512 {
        use core::arch::x86_64::*;
        let one = _mm512_set1_ps(1.0);
        let x = _mm512_max_ps(
            _mm512_min_ps(x, _mm512_set1_ps(88.376_26)),
            _mm512_set1_ps(-88.376_26),
        );
        let fx0 = _mm512_fmadd_ps(x, _mm512_set1_ps(1.442_695_04), _mm512_set1_ps(0.5));
        let trunc = _mm512_cvtepi32_ps(_mm512_cvttps_epi32(fx0));
        let gt = _mm512_cmp_ps_mask(trunc, fx0, _CMP_GT_OQ);
        let fx = _mm512_mask_sub_ps(trunc, gt, trunc, one);
        let x = _mm512_fnmadd_ps(
            fx,
            _mm512_set1_ps(-2.121_944_4e-4),
            _mm512_fnmadd_ps(fx, _mm512_set1_ps(0.693_359_375), x),
        );
        let z = _mm512_mul_ps(x, x);
        let mut y = _mm512_set1_ps(1.987_569_15e-4);
        for coefficient in [
            1.398_199_95e-3,
            8.333_451_9e-3,
            4.166_579_6e-2,
            1.666_666_5e-1,
            5.000_000_1e-1,
        ] {
            y = _mm512_fmadd_ps(y, x, _mm512_set1_ps(coefficient));
        }
        y = _mm512_fmadd_ps(y, z, x);
        let exponent = _mm512_slli_epi32(
            _mm512_add_epi32(_mm512_cvttps_epi32(fx), _mm512_set1_epi32(0x7f)),
            23,
        );
        _mm512_mul_ps(_mm512_add_ps(y, one), _mm512_castsi512_ps(exponent))
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx512f,fma")]
    unsafe fn softmax(scores: &mut [f32]) {
        use core::arch::x86_64::*;
        let scale = 1.0 / 8.0;
        for row in scores.chunks_exact_mut(N) {
            let max = row
                .iter()
                .fold(f32::NEG_INFINITY, |max, &value| max.max(value * scale));
            let mut col = 0;
            while col + 16 <= N {
                let values = unsafe { _mm512_loadu_ps(row.as_ptr().add(col)) };
                let shifted = _mm512_sub_ps(
                    _mm512_mul_ps(values, _mm512_set1_ps(scale)),
                    _mm512_set1_ps(max),
                );
                let values = unsafe { exp16(shifted) };
                unsafe { _mm512_storeu_ps(row.as_mut_ptr().add(col), values) };
                col += 16;
            }
            for value in &mut row[col..] {
                *value = (*value * scale - max).exp();
            }
            let inv_sum = 1.0 / row.iter().sum::<f32>();
            col = 0;
            let inv = _mm512_set1_ps(inv_sum);
            while col + 16 <= N {
                let values = unsafe { _mm512_loadu_ps(row.as_ptr().add(col)) };
                unsafe {
                    _mm512_storeu_ps(row.as_mut_ptr().add(col), _mm512_mul_ps(values, inv));
                }
                col += 16;
            }
            for value in &mut row[col..] {
                *value *= inv_sum;
            }
        }
    }

    fn full_score(q: &[f32], k: &[f32], v: &[f32], scores: &mut [f32], out: &mut [f32]) {
        scores
            .par_chunks_mut(N * N)
            .zip(out.par_chunks_mut(N * D))
            .enumerate()
            .take(H)
            .for_each(|(head, (scores, out))| {
                let base = head * N * D;
                let q = unsafe {
                    faer::mat::from_raw_parts::<f32>(q.as_ptr().add(base), N, D, D as isize, 1)
                };
                let kt = unsafe {
                    faer::mat::from_raw_parts::<f32>(k.as_ptr().add(base), D, N, 1, D as isize)
                };
                let score_matrix = unsafe {
                    faer::mat::from_raw_parts_mut::<f32>(scores.as_mut_ptr(), N, N, N as isize, 1)
                };
                faer::linalg::matmul::matmul(score_matrix, q, kt, None, 1.0, Parallelism::None);
                #[cfg(target_arch = "x86_64")]
                unsafe {
                    softmax(scores)
                };
                let probabilities = unsafe {
                    faer::mat::from_raw_parts::<f32>(scores.as_ptr(), N, N, N as isize, 1)
                };
                let v = unsafe {
                    faer::mat::from_raw_parts::<f32>(v.as_ptr().add(base), N, D, D as isize, 1)
                };
                let out = unsafe {
                    faer::mat::from_raw_parts_mut::<f32>(out.as_mut_ptr(), N, D, D as isize, 1)
                };
                faer::linalg::matmul::matmul(out, probabilities, v, None, 1.0, Parallelism::None);
            });
    }

    fn median(values: &mut [f64]) -> f64 {
        values.sort_by(f64::total_cmp);
        let middle = values.len() / 2;
        (values[middle - 1] + values[middle]) * 0.5
    }

    pub fn run() {
        assert!(std::is_x86_feature_detected!("avx512f"));
        assert!(std::is_x86_feature_detected!("fma"));
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(16)
            .build()
            .unwrap();
        let len = H * N * D;
        let q: Vec<f32> = (0..len)
            .map(|i| ((i % 1009) as f32 * 0.007_812_5).sin())
            .collect();
        let k: Vec<f32> = (0..len)
            .map(|i| ((i % 1013) as f32 * 0.006_835_937_5).cos())
            .collect();
        let v: Vec<f32> = (0..len)
            .map(|i| ((i % 1021) as f32 * 0.005_859_375).sin())
            .collect();
        let mut online = vec![0.0f32; len];
        let mut full = vec![0.0f32; len];
        let mut scores = vec![0.0f32; H * N * N];

        if std::env::args().nth(1).as_deref() == Some("online-long") {
            pool.install(|| {
                for _ in 0..10 {
                    assert!(vestra_kernels::flash_attention_f32_da3_base(
                        &q,
                        &k,
                        &v,
                        H,
                        N,
                        &mut online,
                    ));
                }
            });
            // Perf starts its counters after this deliberate quiet interval, so
            // allocation, data generation, and warm-up are outside the sample.
            std::thread::sleep(std::time::Duration::from_secs(1));
            let started = Instant::now();
            pool.install(|| {
                for _ in 0..1_000 {
                    assert!(vestra_kernels::flash_attention_f32_da3_base(
                        black_box(&q),
                        black_box(&k),
                        black_box(&v),
                        H,
                        N,
                        black_box(&mut online),
                    ));
                }
            });
            println!(
                "online_long iterations=1000 elapsed_ms={:.6} per_call_ms={:.6} checksum={:.9}",
                started.elapsed().as_secs_f64() * 1_000.0,
                started.elapsed().as_secs_f64(),
                black_box(online.iter().sum::<f32>()),
            );
            return;
        }

        pool.install(|| {
            for _ in 0..3 {
                assert!(vestra_kernels::flash_attention_f32_da3_base(
                    &q,
                    &k,
                    &v,
                    H,
                    N,
                    &mut online,
                ));
                full_score(&q, &k, &v, &mut scores, &mut full);
            }
        });
        let max_abs = online
            .iter()
            .zip(&full)
            .map(|(lhs, rhs)| (lhs - rhs).abs())
            .fold(0.0, f32::max);
        let mae = online
            .iter()
            .zip(&full)
            .map(|(lhs, rhs)| (lhs - rhs).abs() as f64)
            .sum::<f64>()
            / len as f64;

        let mut online_ms = Vec::with_capacity(30);
        let mut full_ms = Vec::with_capacity(30);
        pool.install(|| {
            for iteration in 0..30 {
                let full_first = iteration % 2 != 0;
                let mut run_online = || {
                    let started = Instant::now();
                    assert!(vestra_kernels::flash_attention_f32_da3_base(
                        black_box(&q),
                        black_box(&k),
                        black_box(&v),
                        H,
                        N,
                        black_box(&mut online),
                    ));
                    online_ms.push(started.elapsed().as_secs_f64() * 1_000.0);
                };
                let mut run_full = || {
                    let started = Instant::now();
                    full_score(
                        black_box(&q),
                        black_box(&k),
                        black_box(&v),
                        black_box(&mut scores),
                        black_box(&mut full),
                    );
                    full_ms.push(started.elapsed().as_secs_f64() * 1_000.0);
                };
                if full_first {
                    run_full();
                    run_online();
                } else {
                    run_online();
                    run_full();
                }
            }
        });
        let online_min = online_ms.iter().copied().fold(f64::INFINITY, f64::min);
        let full_min = full_ms.iter().copied().fold(f64::INFINITY, f64::min);
        let online_median = median(&mut online_ms);
        let full_median = median(&mut full_ms);
        println!("shape={H}x{N}x{D} threads=16 iterations=30");
        println!("online_avx512 min_ms={online_min:.6} median_ms={online_median:.6}");
        println!("full_score_faer_outer_heads min_ms={full_min:.6} median_ms={full_median:.6}");
        println!(
            "candidate_vs_online_percent={:.3}",
            (full_median / online_median - 1.0) * 100.0
        );
        println!("max_abs={max_abs:.9} mae={mae:.9}");
        println!("checksum={:.9}", black_box(full.iter().sum::<f32>()));
    }
}

#[cfg(target_arch = "x86_64")]
fn main() {
    x86_64_benchmark::run();
}

#[cfg(not(target_arch = "x86_64"))]
fn main() {
    eprintln!("attention_full_score_bench requires an x86-64 AVX-512 host");
}

/// Detected/selected instruction set architecture for the vectorized kernels.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Isa {
    Avx512,
    Avx2,
    Scalar,
}

/// Runtime kernel dispatcher. Detects the host ISA once at construction
/// (`Kernels::detect()`) and routes each kernel call to the fastest
/// implementation available on this machine, falling back to the scalar
/// reference kernels (Task 5) everywhere else — including non-x86_64 hosts.
pub struct Kernels {
    isa: Isa,
}

impl Kernels {
    /// Detect the best available ISA on this host. On non-x86_64 targets
    /// (e.g. this development machine, aarch64/Apple Silicon) this always
    /// returns `Isa::Scalar`, since `is_x86_feature_detected!` and the
    /// AVX-512/AVX2 kernels only exist on `target_arch = "x86_64"`.
    ///
    /// `Isa::Avx512` is a single unified tier gated on AVX-512F **and**
    /// AVX-512BW **and** AVX-512VNNI all being present (not just F): the
    /// q8_0 GEMM kernel (Task 9) needs VNNI's `_mm512_dpbusd_epi32`, and
    /// gating the whole tier on all three keeps `Kernels` dispatch a single
    /// two-way branch per kernel instead of tracking per-feature subsets.
    ///
    /// **Tradeoff**: gelu/add (Task 8) only need F, but will fall back to
    /// AVX2/scalar on AVX-512F/BW-capable CPUs without VNNI (e.g. original
    /// Skylake-X/Skylake-SP from 2017). VNNI shipped starting with Cascade
    /// Lake (2019). This unified gate is an accepted simplification (single
    /// dispatch tier vs. per-kernel feature detection), not an oversight.
    pub fn detect() -> Kernels {
        #[cfg(target_arch = "x86_64")]
        {
            if std::is_x86_feature_detected!("avx512f")
                && std::is_x86_feature_detected!("avx512bw")
                && std::is_x86_feature_detected!("avx512vnni")
            {
                return Kernels { isa: Isa::Avx512 };
            }
            if std::is_x86_feature_detected!("avx2") {
                return Kernels { isa: Isa::Avx2 };
            }
        }
        Kernels { isa: Isa::Scalar }
    }

    pub fn isa(&self) -> Isa {
        self.isa
    }

    /// In-place GELU. Result must be within the tolerance band of the
    /// scalar reference kernel (`scalar::gelu`).
    pub fn gelu(&self, x: &mut [f32]) {
        match self.isa {
            #[cfg(target_arch = "x86_64")]
            Isa::Avx512 => {
                use rayon::prelude::*;
                if std::env::var_os("DA3_KERNELS_DISABLE_PARALLEL_GELU").is_some() {
                    unsafe { crate::simd_avx512::gelu_avx512(x) };
                    return;
                }
                // FC1 contains 2.6M independent values at the locked DA3
                // shape. Keep the SIMD arithmetic for each value unchanged,
                // but do not leave this post-GEMM pass on a single core.
                x.par_chunks_mut(4096)
                    .for_each(|chunk| unsafe { crate::simd_avx512::gelu_avx512(chunk) });
            }
            _ => crate::scalar::gelu(x),
        }
    }

    /// In-place elementwise `dst += src`. Result must exactly match the
    /// scalar reference kernel (`scalar::add`).
    pub fn add(&self, dst: &mut [f32], src: &[f32]) {
        match self.isa {
            #[cfg(target_arch = "x86_64")]
            Isa::Avx512 => unsafe { crate::simd_avx512::add_avx512(dst, src) },
            _ => crate::scalar::add(dst, src),
        }
    }

    /// q8_0 x q8_0 -> f32 GEMM (Task 9). `a_q` is `m x k`, `b_q` is `n x k`
    /// (i.e. B is pre-transposed into row-major q8_0 blocks the same way A
    /// is), both as `k/QK8_0` blocks per row; `c` is `m x n`. Result must be
    /// within the test's tolerance band of `scalar::gemm_q8_0` run against
    /// the f32-dequantized operands. See `scalar::gemm_q8_0` for the oracle
    /// this must match.
    pub fn gemm_q8_0(
        &self,
        m: usize,
        n: usize,
        k: usize,
        a_q: &[crate::BlockQ8_0],
        b_q: &[crate::BlockQ8_0],
        c: &mut [f32],
    ) {
        match self.isa {
            #[cfg(target_arch = "x86_64")]
            Isa::Avx512 => unsafe { crate::simd_avx512::gemm_q8_0_avx512(m, n, k, a_q, b_q, c) },
            _ => crate::scalar::gemm_q8_0(m, n, k, a_q, b_q, c),
        }
    }
}

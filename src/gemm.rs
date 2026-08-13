use crate::scalar;

pub trait Gemm {
    fn gemm(&self, m: usize, n: usize, k: usize, a: &[f32], b: &[f32], c: &mut [f32]);
}

pub struct ScalarGemm;
impl Gemm for ScalarGemm {
    fn gemm(&self, m: usize, n: usize, k: usize, a: &[f32], b: &[f32], c: &mut [f32]) {
        scalar::gemm_f32(m, n, k, a, b, c);
    }
}

pub struct FaerGemm;
impl Gemm for FaerGemm {
    fn gemm(&self, m: usize, n: usize, k: usize, a: &[f32], b: &[f32], c: &mut [f32]) {
        let profile = std::env::var_os("DA_GEMM_PROFILE").is_some();
        let started = std::time::Instant::now();
        debug_assert_eq!(a.len(), m * k);
        debug_assert_eq!(b.len(), k * n);
        debug_assert_eq!(c.len(), m * n);
        // row-major Slices als faer-Views mit expliziten Strides interpretieren.
        let a = unsafe { faer::mat::from_raw_parts::<f32>(a.as_ptr(), m, k, k as isize, 1) };
        let b = unsafe { faer::mat::from_raw_parts::<f32>(b.as_ptr(), k, n, n as isize, 1) };
        let cm =
            unsafe { faer::mat::from_raw_parts_mut::<f32>(c.as_mut_ptr(), m, n, n as isize, 1) };
        // Respect faer's process-wide parallelism setting.  The previous
        // `Parallelism::None` hard-coded every model GEMM to one core even when
        // the benchmark fixed RAYON_NUM_THREADS (and the C++/PyTorch runners)
        // to a larger, identical thread count.
        faer::linalg::matmul::matmul(cm, a, b, None, 1.0, faer::get_global_parallelism());
        if profile {
            eprintln!(
                "phase: gemm m={m} n={n} k={k} elapsed={:.3}ms",
                started.elapsed().as_secs_f64() * 1e3,
            );
        }
    }
}

/// Optional bridge to the experimental BLIS build in `da3-kernels`.
///
/// The DPT head invokes its GEMMs serially, so one BLIS 16-thread team does
/// not nest inside another Rayon/Faer operation. Unsupported builds and
/// shapes retain Faer exactly, making this a reversible A/B candidate.
pub struct BlisOrFaerGemm;
impl Gemm for BlisOrFaerGemm {
    fn gemm(&self, m: usize, n: usize, k: usize, a: &[f32], b: &[f32], c: &mut [f32]) {
        if std::env::var_os("DA3_HEAD_BLIS_GEMM").is_some()
            && crate::specialized::blis_gemm_f32(m, n, k, a, b, c)
        {
            return;
        }
        FaerGemm.gemm(m, n, k, a, b, c);
    }
}

/// Narrow DA3 transformer projection dispatcher. Only the fixed BASE shapes
/// can reach the external AVX-512 candidate; every other operation uses Faer.
pub struct Da3ProjectionGemm;
impl Gemm for Da3ProjectionGemm {
    fn gemm(&self, m: usize, n: usize, k: usize, a: &[f32], b: &[f32], c: &mut [f32]) {
        if crate::specialized::linear_f32_da3_base(m, n, k, a, b, c) {
            return;
        }
        FaerGemm.gemm(m, n, k, a, b, c);
    }
}

pub struct GemmWithEpilogue<G: Gemm> {
    pub inner: G,
}
impl<G: Gemm> GemmWithEpilogue<G> {
    pub fn gemm_bias_gelu(
        &self,
        m: usize,
        n: usize,
        k: usize,
        a: &[f32],
        b: &[f32],
        bias: Option<&[f32]>,
        gelu: bool,
        c: &mut [f32],
    ) {
        self.inner.gemm(m, n, k, a, b, c);
        if let Some(bias) = bias {
            scalar::add_bias_rows(c, m, n, bias);
        }
        if gelu {
            scalar::gelu(c);
        }
    }
}

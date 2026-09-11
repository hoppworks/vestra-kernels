//! Model-owned panel packing for the fixed DA3-BASE F32 projection shapes.
//!
//! The normal row-major GGUF layout makes a 32-column panel advance by a full
//! output row for every K value. `PreparedLinearF32` stores each such panel as
//! contiguous K-major vectors. Packing is performed at model-load time, never
//! on the timed inference path.

use std::sync::Arc;

#[cfg(target_arch = "x86_64")]
use rayon::prelude::*;

use crate::specialized::DA3_BASE_TOKENS_504X336;

pub const PANEL_WIDTH: usize = 64;
#[cfg_attr(not(target_arch = "x86_64"), allow(dead_code))]
const ROW_TILE: usize = 6;

/// Immutable DA3 projection weights packed as `[n_panel][k][64]`.
///
/// The source and output matrices remain row-major. Only the persistent model
/// weight layout changes, so every output still accumulates K in ascending
/// order using the same F32 FMA sequence as the existing direct AVX route.
#[derive(Clone, Debug)]
pub struct PreparedLinearF32 {
    #[cfg_attr(not(target_arch = "x86_64"), allow(dead_code))]
    packed: Arc<[f32]>,
    input_features: usize,
    output_features: usize,
}

impl PreparedLinearF32 {
    /// Packs one `[input_features, output_features]` row-major F32 matrix.
    ///
    /// This deliberately accepts only the DA3-BASE projection dimensions. It
    /// prevents an unqualified caller from silently using this experimental
    /// layout for unrelated models.
    pub fn try_new(weight: &[f32], input_features: usize, output_features: usize) -> Option<Self> {
        if !is_da3_base_projection_shape(input_features, output_features)
            || weight.len() != input_features * output_features
        {
            return None;
        }

        let panels = output_features / PANEL_WIDTH;
        let mut packed = vec![0.0; weight.len()];
        for panel in 0..panels {
            for input in 0..input_features {
                let source = &weight[input * output_features + panel * PANEL_WIDTH
                    ..input * output_features + (panel + 1) * PANEL_WIDTH];
                let destination = &mut packed[(panel * input_features + input) * PANEL_WIDTH
                    ..(panel * input_features + input + 1) * PANEL_WIDTH];
                destination.copy_from_slice(source);
            }
        }

        Some(Self {
            packed: packed.into(),
            input_features,
            output_features,
        })
    }

    #[must_use]
    pub fn input_features(&self) -> usize {
        self.input_features
    }

    #[must_use]
    pub fn output_features(&self) -> usize {
        self.output_features
    }

    /// Reports whether this process can use the serial panel microkernels.
    ///
    /// The answer is intentionally stable for the lifetime of an inference
    /// executor: the environment switch and CPU ISA cannot change during a
    /// process. A caller that owns a validated executor may snapshot it once
    /// and use the unchecked serial entry points in its hot loop.
    #[must_use]
    pub fn serial_panel_kernel_available(&self) -> bool {
        if std::env::var_os("DA3_KERNELS_DISABLE_PACKED_LINEAR").is_some() {
            return false;
        }
        #[cfg(target_arch = "x86_64")]
        {
            std::is_x86_feature_detected!("avx512f") && std::is_x86_feature_detected!("fma")
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            false
        }
    }

    /// Computes DA3-BASE QKV directly into the head-major buffers consumed by
    /// attention using this model-owned `[panel][K][64]` weight packing.
    ///
    /// The existing direct QKV route already uses a six-row, 64-column panel
    /// schedule and writes HND output without a token-major staging tensor.
    /// This exact-order alternative changes only the immutable weight address:
    /// successive K values of one panel become contiguous rather than being
    /// separated by the full 2304-column source row stride.
    pub fn run_qkv_da3_base(
        &self,
        input: &[f32],
        bias: &[f32],
        q: &mut [f32],
        k: &mut [f32],
        v: &mut [f32],
    ) -> bool {
        let tokens = DA3_BASE_TOKENS_504X336;
        if std::env::var_os("DA3_KERNELS_DISABLE_PACKED_QKV").is_some()
            || self.input_features != 768
            || self.output_features != 2304
            || input.len() != tokens * 768
            || bias.len() != 2304
            || q.len() != tokens * 768
            || k.len() != q.len()
            || v.len() != q.len()
        {
            return false;
        }
        #[cfg(target_arch = "x86_64")]
        if std::is_x86_feature_detected!("avx512f") && std::is_x86_feature_detected!("fma") {
            // SAFETY: fixed DA3-BASE dimensions, output lengths and ISA were
            // validated above; the packed layout was created by `try_new`.
            unsafe { run_qkv_da3_base_packed_avx512(self, input, bias, q, k, v) };
            return true;
        }
        false
    }

    /// Executes an AVX-512 F32 projection at the locked DA3-BASE token count.
    ///
    /// Returns `false` without changing `output` when the host or input shape
    /// is unsupported. The current engine does not select this candidate until
    /// it wins against BLIS with runtime packing included.
    pub fn run_da3_base(&self, input: &[f32], output: &mut [f32]) -> bool {
        if std::env::var_os("DA3_KERNELS_DISABLE_PACKED_LINEAR").is_some()
            || input.len() != DA3_BASE_TOKENS_504X336 * self.input_features
            || output.len() != DA3_BASE_TOKENS_504X336 * self.output_features
        {
            return false;
        }
        #[cfg(target_arch = "x86_64")]
        if std::is_x86_feature_detected!("avx512f") && std::is_x86_feature_detected!("fma") {
            // SAFETY: fixed dimensions, buffer lengths, packed layout, and
            // required ISA were checked above.
            input
                .par_chunks(ROW_TILE * self.input_features)
                .zip(output.par_chunks_mut(ROW_TILE * self.output_features))
                .for_each(|(input_rows, output_rows)| {
                    // SAFETY: the outer dimension checks above and chunk sizes
                    // establish valid packed-projection buffers.
                    unsafe { run_rows_avx512(self, input_rows, output_rows) };
                });
            return true;
        }
        false
    }

    /// Runs a small, contiguous row range on the calling thread.
    ///
    /// This is the building block for a fused MLP schedule: one Rayon work
    /// item can retain its normalized rows and FC1 activation in private cache
    /// while it executes FC1, GELU, and FC2. The public whole-matrix entry
    /// above remains available for an isolated projection benchmark.
    pub fn run_rows_serial(&self, input: &[f32], output: &mut [f32]) -> bool {
        if std::env::var_os("DA3_KERNELS_DISABLE_PACKED_LINEAR").is_some()
            || input.is_empty()
            || input.len() % self.input_features != 0
            || input.len() / self.input_features > ROW_TILE
            || output.len() != (input.len() / self.input_features) * self.output_features
        {
            return false;
        }
        #[cfg(target_arch = "x86_64")]
        if std::is_x86_feature_detected!("avx512f") && std::is_x86_feature_detected!("fma") {
            // SAFETY: checked shapes and ISA; the routine indexes only the
            // supplied row range and immutable, model-owned packed weights.
            unsafe { run_rows_avx512(self, input, output) };
            return true;
        }
        false
    }

    /// Computes one 64-wide output panel for a small contiguous row range.
    ///
    /// This is deliberately narrower than [`run_rows_serial`].  A fused MLP
    /// can retain only one hidden-neuron strip per token slab, rather than
    /// materialising the complete 3072-wide FC1 activation.  `output_panel`
    /// is expressed in 64-wide units and the reduction still visits every
    /// input feature in ascending order.
    pub fn run_output_panel_rows_serial(
        &self,
        input: &[f32],
        output: &mut [f32],
        output_panel: usize,
    ) -> bool {
        let rows = input.len().checked_div(self.input_features).unwrap_or(0);
        if std::env::var_os("DA3_KERNELS_DISABLE_PACKED_LINEAR").is_some()
            || rows == 0
            || rows > ROW_TILE
            || input.len() != rows * self.input_features
            || output.len() != rows * PANEL_WIDTH
            || output_panel >= self.output_features / PANEL_WIDTH
        {
            return false;
        }
        #[cfg(target_arch = "x86_64")]
        if std::is_x86_feature_detected!("avx512f") && std::is_x86_feature_detected!("fma") {
            // SAFETY: shape, ISA and the selected packed panel were checked.
            unsafe { run_output_panel_rows_avx512(self, input, output, output_panel) };
            return true;
        }
        false
    }

    /// Equivalent to [`Self::run_output_panel_rows_serial`] after an owning
    /// executor has validated [`Self::serial_panel_kernel_available`] once.
    ///
    /// This avoids repeated environment and ISA dispatch in a tightly nested
    /// MLP loop. It retains all shape assertions in debug builds; production
    /// callers must use it only with the fixed DA3-BASE panel contract.
    pub fn run_output_panel_rows_serial_validated(
        &self,
        input: &[f32],
        output: &mut [f32],
        output_panel: usize,
    ) -> bool {
        debug_assert!(self.serial_panel_kernel_available());
        let rows = input.len() / self.input_features;
        debug_assert!(rows > 0 && rows <= ROW_TILE);
        debug_assert_eq!(input.len(), rows * self.input_features);
        debug_assert_eq!(output.len(), rows * PANEL_WIDTH);
        debug_assert!(output_panel < self.output_features / PANEL_WIDTH);
        #[cfg(target_arch = "x86_64")]
        {
            // SAFETY: availability and the fixed serial panel contract are
            // validated by the executor before this hot path is selected.
            unsafe { run_output_panel_rows_avx512(self, input, output, output_panel) };
            true
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            let _ = (input, output, output_panel);
            false
        }
    }

    /// Adds one 64-wide input strip to every output panel for a small row
    /// range.  It is the FC2 counterpart to
    /// [`run_output_panel_rows_serial`].  Loading the prior partial sum and
    /// then advancing the strip in ascending hidden-channel order preserves
    /// the FC2 reduction order while allowing the caller to keep the hidden
    /// activation cache-resident.
    pub fn accumulate_input_panel_rows_serial(
        &self,
        input: &[f32],
        output: &mut [f32],
        input_panel: usize,
    ) -> bool {
        let rows = input.len() / PANEL_WIDTH;
        if std::env::var_os("DA3_KERNELS_DISABLE_PACKED_LINEAR").is_some()
            || self.input_features % PANEL_WIDTH != 0
            || rows == 0
            || rows > ROW_TILE
            || input.len() != rows * PANEL_WIDTH
            || output.len() != rows * self.output_features
            || input_panel >= self.input_features / PANEL_WIDTH
        {
            return false;
        }
        #[cfg(target_arch = "x86_64")]
        if std::is_x86_feature_detected!("avx512f") && std::is_x86_feature_detected!("fma") {
            // SAFETY: shape, ISA and the selected packed K strip were checked.
            unsafe { accumulate_input_panel_rows_avx512(self, input, output, input_panel) };
            return true;
        }
        false
    }

    /// Equivalent to [`Self::accumulate_input_panel_rows_serial`] after an
    /// owning executor has validated the immutable process-level dispatch.
    pub fn accumulate_input_panel_rows_serial_validated(
        &self,
        input: &[f32],
        output: &mut [f32],
        input_panel: usize,
    ) -> bool {
        debug_assert!(self.serial_panel_kernel_available());
        let rows = input.len() / PANEL_WIDTH;
        debug_assert!(self.input_features % PANEL_WIDTH == 0);
        debug_assert!(rows > 0 && rows <= ROW_TILE);
        debug_assert_eq!(input.len(), rows * PANEL_WIDTH);
        debug_assert_eq!(output.len(), rows * self.output_features);
        debug_assert!(input_panel < self.input_features / PANEL_WIDTH);
        #[cfg(target_arch = "x86_64")]
        {
            // SAFETY: availability and the fixed serial panel contract are
            // validated by the executor before this hot path is selected.
            unsafe { accumulate_input_panel_rows_avx512(self, input, output, input_panel) };
            true
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            let _ = (input, output, input_panel);
            false
        }
    }

    /// Adds four consecutive 64-wide input strips to every output panel in
    /// one output-tile residency window.
    ///
    /// Every output lane still observes the exact same FMA sequence as four
    /// calls to [`Self::accumulate_input_panel_rows_serial`]: strip 0 through
    /// strip 3, then each strip's 64 hidden channels in ascending order. The
    /// difference is solely that the output partial sum is loaded and stored
    /// once for the four strips instead of once per strip.
    pub fn accumulate_four_input_panels_rows_serial(
        &self,
        inputs: [&[f32]; 4],
        output: &mut [f32],
        first_input_panel: usize,
    ) -> bool {
        let rows = inputs[0].len() / PANEL_WIDTH;
        if std::env::var_os("DA3_KERNELS_DISABLE_PACKED_LINEAR").is_some()
            || self.input_features % PANEL_WIDTH != 0
            || rows == 0
            || rows > ROW_TILE
            || inputs.iter().any(|input| input.len() != rows * PANEL_WIDTH)
            || output.len() != rows * self.output_features
            || first_input_panel + 4 > self.input_features / PANEL_WIDTH
        {
            return false;
        }
        #[cfg(target_arch = "x86_64")]
        if std::is_x86_feature_detected!("avx512f") && std::is_x86_feature_detected!("fma") {
            // SAFETY: shape, ISA and four consecutive packed K strips were
            // checked above.
            unsafe {
                accumulate_four_input_panels_rows_avx512(self, inputs, output, first_input_panel)
            };
            return true;
        }
        false
    }

    /// Unchecked hot-path companion to
    /// [`Self::accumulate_four_input_panels_rows_serial`].
    ///
    /// The executor must first snapshot [`Self::serial_panel_kernel_available`]
    /// and retain the fixed DA3-BASE shape contract for its lifetime.
    pub fn accumulate_four_input_panels_rows_serial_validated(
        &self,
        inputs: [&[f32]; 4],
        output: &mut [f32],
        first_input_panel: usize,
    ) -> bool {
        debug_assert!(self.serial_panel_kernel_available());
        let rows = inputs[0].len() / PANEL_WIDTH;
        debug_assert!(self.input_features % PANEL_WIDTH == 0);
        debug_assert!(rows > 0 && rows <= ROW_TILE);
        debug_assert!(inputs.iter().all(|input| input.len() == rows * PANEL_WIDTH));
        debug_assert_eq!(output.len(), rows * self.output_features);
        debug_assert!(first_input_panel + 4 <= self.input_features / PANEL_WIDTH);
        #[cfg(target_arch = "x86_64")]
        {
            // SAFETY: availability and the fixed serial panel contract are
            // validated by the executor before this hot path is selected.
            unsafe {
                accumulate_four_input_panels_rows_avx512(self, inputs, output, first_input_panel)
            };
            true
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            let _ = (inputs, output, first_input_panel);
            false
        }
    }

    #[cfg(test)]
    fn packed_panel(&self, panel: usize) -> &[f32] {
        &self.packed[panel * self.input_features * PANEL_WIDTH
            ..(panel + 1) * self.input_features * PANEL_WIDTH]
    }
}

fn is_da3_base_projection_shape(input_features: usize, output_features: usize) -> bool {
    matches!(
        (input_features, output_features),
        (128, 128) | (768, 2304) | (768, 768) | (768, 3072) | (3072, 768)
    )
}

/// Packed-weight counterpart to the production direct QKV kernel. The panel
/// schedule, K-major FMA order, bias addition, and HND stores intentionally
/// match the established column-split route exactly.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,fma")]
unsafe fn run_qkv_da3_base_packed_avx512(
    prepared: &PreparedLinearF32,
    input: &[f32],
    bias: &[f32],
    q: &mut [f32],
    k: &mut [f32],
    v: &mut [f32],
) {
    use core::arch::x86_64::*;

    const ROWS: usize = 6;
    const TOKENS: usize = DA3_BASE_TOKENS_504X336;
    const K: usize = 768;
    const N: usize = 2304;
    debug_assert_eq!(prepared.input_features, K);
    debug_assert_eq!(prepared.output_features, N);
    let q_ptr = q.as_mut_ptr() as usize;
    let k_ptr = k.as_mut_ptr() as usize;
    let v_ptr = v.as_mut_ptr() as usize;

    // One panel is one Q, K or V attention head. Every task writes a
    // disjoint HND region, while the packed K-major panel remains contiguous
    // across all 145 row tiles it serves.
    (0..N / PANEL_WIDTH).into_par_iter().for_each(|panel| {
        let col0 = panel * PANEL_WIDTH;
        let group = col0 / 768;
        let head = (col0 % 768) / PANEL_WIDTH;
        let destination = match group {
            0 => q_ptr,
            1 => k_ptr,
            _ => v_ptr,
        } as *mut f32;
        let weight_panel = &prepared.packed[panel * K * PANEL_WIDTH..(panel + 1) * K * PANEL_WIDTH];

        for row0 in (0..TOKENS).step_by(ROWS) {
            let rows = (TOKENS - row0).min(ROWS);
            let mut accumulators = [[_mm512_setzero_ps(); 4]; ROWS];
            for input_feature in 0..K {
                let weights = unsafe { weight_panel.as_ptr().add(input_feature * PANEL_WIDTH) };
                let vectors = unsafe {
                    [
                        _mm512_loadu_ps(weights),
                        _mm512_loadu_ps(weights.add(16)),
                        _mm512_loadu_ps(weights.add(32)),
                        _mm512_loadu_ps(weights.add(48)),
                    ]
                };
                for row in 0..rows {
                    let activation = _mm512_set1_ps(input[(row0 + row) * K + input_feature]);
                    for block in 0..4 {
                        accumulators[row][block] =
                            _mm512_fmadd_ps(activation, vectors[block], accumulators[row][block]);
                    }
                }
            }
            for row in 0..rows {
                let output = unsafe { destination.add((head * TOKENS + row0 + row) * PANEL_WIDTH) };
                for block in 0..4 {
                    let bias_v = unsafe { _mm512_loadu_ps(bias.as_ptr().add(col0 + block * 16)) };
                    unsafe {
                        _mm512_storeu_ps(
                            output.add(block * 16),
                            _mm512_add_ps(accumulators[row][block], bias_v),
                        );
                    }
                }
            }
        }
    });
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,fma")]
unsafe fn run_rows_avx512(prepared: &PreparedLinearF32, input: &[f32], output: &mut [f32]) {
    use core::arch::x86_64::*;

    let n = prepared.output_features;
    let k = prepared.input_features;
    let rows = input.len() / k;
    for panel in 0..n / PANEL_WIDTH {
        let weight_panel = &prepared.packed[panel * k * PANEL_WIDTH..(panel + 1) * k * PANEL_WIDTH];
        let mut accumulators = [[_mm512_setzero_ps(); 4]; ROW_TILE];
        for input_feature in 0..k {
            let weights = unsafe { weight_panel.as_ptr().add(input_feature * PANEL_WIDTH) };
            let vectors = unsafe {
                [
                    _mm512_loadu_ps(weights),
                    _mm512_loadu_ps(weights.add(16)),
                    _mm512_loadu_ps(weights.add(32)),
                    _mm512_loadu_ps(weights.add(48)),
                ]
            };
            for row in 0..rows {
                let value = _mm512_set1_ps(input[row * k + input_feature]);
                for block in 0..4 {
                    accumulators[row][block] =
                        _mm512_fmadd_ps(value, vectors[block], accumulators[row][block]);
                }
            }
        }
        for row in 0..rows {
            let destination = unsafe { output.as_mut_ptr().add(row * n + panel * PANEL_WIDTH) };
            for block in 0..4 {
                unsafe { _mm512_storeu_ps(destination.add(block * 16), accumulators[row][block]) };
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,fma")]
unsafe fn run_output_panel_rows_avx512(
    prepared: &PreparedLinearF32,
    input: &[f32],
    output: &mut [f32],
    output_panel: usize,
) {
    use core::arch::x86_64::*;

    let k = prepared.input_features;
    let rows = input.len() / k;
    let weight_panel =
        &prepared.packed[output_panel * k * PANEL_WIDTH..(output_panel + 1) * k * PANEL_WIDTH];
    let mut accumulators = [[_mm512_setzero_ps(); 4]; ROW_TILE];
    for input_feature in 0..k {
        let weights = unsafe { weight_panel.as_ptr().add(input_feature * PANEL_WIDTH) };
        let vectors = unsafe {
            [
                _mm512_loadu_ps(weights),
                _mm512_loadu_ps(weights.add(16)),
                _mm512_loadu_ps(weights.add(32)),
                _mm512_loadu_ps(weights.add(48)),
            ]
        };
        for row in 0..rows {
            let value = _mm512_set1_ps(input[row * k + input_feature]);
            for block in 0..4 {
                accumulators[row][block] =
                    _mm512_fmadd_ps(value, vectors[block], accumulators[row][block]);
            }
        }
    }
    for row in 0..rows {
        let destination = unsafe { output.as_mut_ptr().add(row * PANEL_WIDTH) };
        for block in 0..4 {
            unsafe { _mm512_storeu_ps(destination.add(block * 16), accumulators[row][block]) };
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,fma")]
unsafe fn accumulate_input_panel_rows_avx512(
    prepared: &PreparedLinearF32,
    input: &[f32],
    output: &mut [f32],
    input_panel: usize,
) {
    use core::arch::x86_64::*;

    let n = prepared.output_features;
    let rows = input.len() / PANEL_WIDTH;
    for output_panel in 0..n / PANEL_WIDTH {
        let weights = &prepared.packed[(output_panel * prepared.input_features
            + input_panel * PANEL_WIDTH)
            * PANEL_WIDTH
            ..(output_panel * prepared.input_features + (input_panel + 1) * PANEL_WIDTH)
                * PANEL_WIDTH];
        let mut accumulators = [[_mm512_setzero_ps(); 4]; ROW_TILE];
        for row in 0..rows {
            let destination = unsafe { output.as_ptr().add(row * n + output_panel * PANEL_WIDTH) };
            for block in 0..4 {
                accumulators[row][block] = unsafe { _mm512_loadu_ps(destination.add(block * 16)) };
            }
        }
        for input_feature in 0..PANEL_WIDTH {
            let weight = unsafe { weights.as_ptr().add(input_feature * PANEL_WIDTH) };
            let vectors = unsafe {
                [
                    _mm512_loadu_ps(weight),
                    _mm512_loadu_ps(weight.add(16)),
                    _mm512_loadu_ps(weight.add(32)),
                    _mm512_loadu_ps(weight.add(48)),
                ]
            };
            for row in 0..rows {
                let value = _mm512_set1_ps(input[row * PANEL_WIDTH + input_feature]);
                for block in 0..4 {
                    accumulators[row][block] =
                        _mm512_fmadd_ps(value, vectors[block], accumulators[row][block]);
                }
            }
        }
        for row in 0..rows {
            let destination = unsafe {
                output
                    .as_mut_ptr()
                    .add(row * n + output_panel * PANEL_WIDTH)
            };
            for block in 0..4 {
                unsafe { _mm512_storeu_ps(destination.add(block * 16), accumulators[row][block]) };
            }
        }
    }
}

/// Four-panel FC2 variant that retains a 6x64 output tile while consuming
/// four consecutive hidden strips. The strip loop is deliberately inside the
/// output-panel loop so each output partial sum is materialized only once.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,fma")]
unsafe fn accumulate_four_input_panels_rows_avx512(
    prepared: &PreparedLinearF32,
    inputs: [&[f32]; 4],
    output: &mut [f32],
    first_input_panel: usize,
) {
    use core::arch::x86_64::*;

    let n = prepared.output_features;
    let rows = inputs[0].len() / PANEL_WIDTH;
    for output_panel in 0..n / PANEL_WIDTH {
        let mut accumulators = [[_mm512_setzero_ps(); 4]; ROW_TILE];
        for row in 0..rows {
            let destination = unsafe { output.as_ptr().add(row * n + output_panel * PANEL_WIDTH) };
            for block in 0..4 {
                accumulators[row][block] = unsafe { _mm512_loadu_ps(destination.add(block * 16)) };
            }
        }
        for (panel_offset, input) in inputs.iter().enumerate() {
            let input_panel = first_input_panel + panel_offset;
            let weights = &prepared.packed[(output_panel * prepared.input_features
                + input_panel * PANEL_WIDTH)
                * PANEL_WIDTH
                ..(output_panel * prepared.input_features + (input_panel + 1) * PANEL_WIDTH)
                    * PANEL_WIDTH];
            for input_feature in 0..PANEL_WIDTH {
                let weight = unsafe { weights.as_ptr().add(input_feature * PANEL_WIDTH) };
                let vectors = unsafe {
                    [
                        _mm512_loadu_ps(weight),
                        _mm512_loadu_ps(weight.add(16)),
                        _mm512_loadu_ps(weight.add(32)),
                        _mm512_loadu_ps(weight.add(48)),
                    ]
                };
                for row in 0..rows {
                    let value = _mm512_set1_ps(input[row * PANEL_WIDTH + input_feature]);
                    for block in 0..4 {
                        accumulators[row][block] =
                            _mm512_fmadd_ps(value, vectors[block], accumulators[row][block]);
                    }
                }
            }
        }
        for row in 0..rows {
            let destination = unsafe {
                output
                    .as_mut_ptr()
                    .add(row * n + output_panel * PANEL_WIDTH)
            };
            for block in 0..4 {
                unsafe { _mm512_storeu_ps(destination.add(block * 16), accumulators[row][block]) };
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn packs_panels_without_losing_weight_bits() {
        let k = 768;
        let n = 768;
        let weight = (0..k * n)
            .map(|index| f32::from_bits(0x3f00_0000 | (index as u32 & 0x007f_ffff)))
            .collect::<Vec<_>>();
        let prepared = PreparedLinearF32::try_new(&weight, k, n).expect("DA3 shape");
        assert_eq!(prepared.packed.len(), weight.len());
        for panel in [0, 3, 11] {
            let packed = prepared.packed_panel(panel);
            for input in [0, 7, 511, 767] {
                for lane in [0, 15, 31, 63] {
                    assert_eq!(
                        packed[input * PANEL_WIDTH + lane].to_bits(),
                        weight[input * n + panel * PANEL_WIDTH + lane].to_bits()
                    );
                }
            }
        }
    }

    #[test]
    fn rejects_non_da3_shapes() {
        assert!(PreparedLinearF32::try_new(&[0.0; 4], 2, 2).is_none());
        assert!(PreparedLinearF32::try_new(&vec![0.0; 768 * 769], 768, 769).is_none());
    }

    #[test]
    fn serial_rows_rejects_tiles_larger_than_its_cache_contract() {
        let prepared = PreparedLinearF32::try_new(&vec![0.0; 768 * 768], 768, 768)
            .expect("DA3 projection shape");
        let input = vec![0.0; 7 * 768];
        let mut output = vec![0.0; 7 * 768];
        assert!(!prepared.run_rows_serial(&input, &mut output));
    }

    #[test]
    fn hidden_strip_primitives_preserve_full_projection_bits() {
        #[cfg(target_arch = "x86_64")]
        {
            if !std::is_x86_feature_detected!("avx512f") || !std::is_x86_feature_detected!("fma") {
                return;
            }
            let k = 768;
            let n = 768;
            let rows = 2;
            let weights = (0..k * n)
                .map(|index| ((index % 37) as f32 - 18.0) * 0.001_953_125)
                .collect::<Vec<_>>();
            let input = (0..rows * k)
                .map(|index| ((index % 29) as f32 - 14.0) * 0.007_812_5)
                .collect::<Vec<_>>();
            let prepared = PreparedLinearF32::try_new(&weights, k, n).expect("DA3 shape");
            assert!(prepared.serial_panel_kernel_available());
            let mut whole = vec![0.0; rows * n];
            assert!(prepared.run_rows_serial(&input, &mut whole));

            for panel in 0..n / PANEL_WIDTH {
                let mut one_panel = vec![0.0; rows * PANEL_WIDTH];
                let mut validated_panel = vec![0.0; rows * PANEL_WIDTH];
                assert!(prepared.run_output_panel_rows_serial(&input, &mut one_panel, panel));
                assert!(prepared.run_output_panel_rows_serial_validated(
                    &input,
                    &mut validated_panel,
                    panel,
                ));
                assert_eq!(validated_panel, one_panel);
                for row in 0..rows {
                    assert_eq!(
                        &one_panel[row * PANEL_WIDTH..(row + 1) * PANEL_WIDTH],
                        &whole[row * n + panel * PANEL_WIDTH..row * n + (panel + 1) * PANEL_WIDTH]
                    );
                }
            }

            let mut accumulated = vec![0.0; rows * n];
            for input_panel in 0..k / PANEL_WIDTH {
                let mut strip = vec![0.0; rows * PANEL_WIDTH];
                for row in 0..rows {
                    strip[row * PANEL_WIDTH..(row + 1) * PANEL_WIDTH].copy_from_slice(
                        &input[row * k + input_panel * PANEL_WIDTH
                            ..row * k + (input_panel + 1) * PANEL_WIDTH],
                    );
                }
                assert!(prepared.accumulate_input_panel_rows_serial(
                    &strip,
                    &mut accumulated,
                    input_panel,
                ));
            }
            assert_eq!(accumulated, whole);

            let mut validated_accumulated = vec![0.0; rows * n];
            for input_panel in 0..k / PANEL_WIDTH {
                let mut strip = vec![0.0; rows * PANEL_WIDTH];
                for row in 0..rows {
                    strip[row * PANEL_WIDTH..(row + 1) * PANEL_WIDTH].copy_from_slice(
                        &input[row * k + input_panel * PANEL_WIDTH
                            ..row * k + (input_panel + 1) * PANEL_WIDTH],
                    );
                }
                assert!(prepared.accumulate_input_panel_rows_serial_validated(
                    &strip,
                    &mut validated_accumulated,
                    input_panel,
                ));
            }
            assert_eq!(validated_accumulated, whole);

            let mut grouped_accumulated = vec![0.0; rows * n];
            for group in 0..k / (4 * PANEL_WIDTH) {
                let first_panel = group * 4;
                let mut strips: [Vec<f32>; 4] =
                    std::array::from_fn(|_| vec![0.0; rows * PANEL_WIDTH]);
                for (strip_offset, strip) in strips.iter_mut().enumerate() {
                    let input_panel = first_panel + strip_offset;
                    for row in 0..rows {
                        strip[row * PANEL_WIDTH..(row + 1) * PANEL_WIDTH].copy_from_slice(
                            &input[row * k + input_panel * PANEL_WIDTH
                                ..row * k + (input_panel + 1) * PANEL_WIDTH],
                        );
                    }
                }
                assert!(prepared.accumulate_four_input_panels_rows_serial(
                    [&strips[0], &strips[1], &strips[2], &strips[3]],
                    &mut grouped_accumulated,
                    first_panel,
                ));
            }
            assert_eq!(grouped_accumulated, whole);

            let mut validated_grouped_accumulated = vec![0.0; rows * n];
            for group in 0..k / (4 * PANEL_WIDTH) {
                let first_panel = group * 4;
                let mut strips: [Vec<f32>; 4] =
                    std::array::from_fn(|_| vec![0.0; rows * PANEL_WIDTH]);
                for (strip_offset, strip) in strips.iter_mut().enumerate() {
                    let input_panel = first_panel + strip_offset;
                    for row in 0..rows {
                        strip[row * PANEL_WIDTH..(row + 1) * PANEL_WIDTH].copy_from_slice(
                            &input[row * k + input_panel * PANEL_WIDTH
                                ..row * k + (input_panel + 1) * PANEL_WIDTH],
                        );
                    }
                }
                assert!(prepared.accumulate_four_input_panels_rows_serial_validated(
                    [&strips[0], &strips[1], &strips[2], &strips[3]],
                    &mut validated_grouped_accumulated,
                    first_panel,
                ));
            }
            assert_eq!(validated_grouped_accumulated, whole);
        }
    }

    #[test]
    fn packed_qkv_matches_direct_qkv_bitwise_for_every_hnd_value() {
        #[cfg(target_arch = "x86_64")]
        {
            if !std::is_x86_feature_detected!("avx512f") || !std::is_x86_feature_detected!("fma") {
                return;
            }
            let tokens = DA3_BASE_TOKENS_504X336;
            let input = (0..tokens * 768)
                .map(|index| ((index % 113) as f32 - 56.0) * 0.003_906_25)
                .collect::<Vec<_>>();
            let weight = (0..768 * 2304)
                .map(|index| ((index % 71) as f32 - 35.0) * 0.001_953_125)
                .collect::<Vec<_>>();
            let bias = (0..2304)
                .map(|index| ((index % 29) as f32 - 14.0) * 0.007_812_5)
                .collect::<Vec<_>>();
            let prepared =
                PreparedLinearF32::try_new(&weight, 768, 2304).expect("DA3 QKV shape is accepted");
            let mut direct_q = vec![f32::NAN; tokens * 768];
            let mut direct_k = vec![f32::NAN; tokens * 768];
            let mut direct_v = vec![f32::NAN; tokens * 768];
            let mut packed_q = vec![f32::NAN; tokens * 768];
            let mut packed_k = vec![f32::NAN; tokens * 768];
            let mut packed_v = vec![f32::NAN; tokens * 768];

            assert!(crate::specialized::qkv_f32_da3_base(
                &input,
                &weight,
                &bias,
                &mut direct_q,
                &mut direct_k,
                &mut direct_v,
            ));
            assert!(prepared.run_qkv_da3_base(
                &input,
                &bias,
                &mut packed_q,
                &mut packed_k,
                &mut packed_v,
            ));
            assert_eq!(packed_q, direct_q);
            assert_eq!(packed_k, direct_k);
            assert_eq!(packed_v, direct_v);

            // Run a distinct second input through the same packed model to
            // catch stale HND output or state retained across calls.
            let alternate_input = input.iter().map(|value| -*value).collect::<Vec<_>>();
            assert!(crate::specialized::qkv_f32_da3_base(
                &alternate_input,
                &weight,
                &bias,
                &mut direct_q,
                &mut direct_k,
                &mut direct_v,
            ));
            assert!(prepared.run_qkv_da3_base(
                &alternate_input,
                &bias,
                &mut packed_q,
                &mut packed_k,
                &mut packed_v,
            ));
            assert_eq!(packed_q, direct_q);
            assert_eq!(packed_k, direct_k);
            assert_eq!(packed_v, direct_v);
        }
    }
}

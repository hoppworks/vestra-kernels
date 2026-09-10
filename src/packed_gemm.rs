//! Model-owned panel packing for the fixed DA3-BASE F32 projection shapes.
//!
//! The normal row-major GGUF layout makes a 64-column panel advance by a full
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
            unsafe { run_da3_base_avx512(self, input, output) };
            return true;
        }
        false
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
        (768, 2304) | (768, 768) | (768, 3072) | (3072, 768)
    )
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,fma")]
unsafe fn run_da3_base_avx512(prepared: &PreparedLinearF32, input: &[f32], output: &mut [f32]) {
    use core::arch::x86_64::*;

    let n = prepared.output_features;
    let k = prepared.input_features;
    let panels = n / PANEL_WIDTH;
    let output_ptr = output.as_mut_ptr() as usize;
    (0..DA3_BASE_TOKENS_504X336)
        .step_by(ROW_TILE)
        .collect::<Vec<_>>()
        .into_par_iter()
        .for_each(|row0| {
            let rows = (DA3_BASE_TOKENS_504X336 - row0).min(ROW_TILE);
            for panel in 0..panels {
                let weight_panel =
                    &prepared.packed[panel * k * PANEL_WIDTH..(panel + 1) * k * PANEL_WIDTH];
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
                        let value = _mm512_set1_ps(input[(row0 + row) * k + input_feature]);
                        for block in 0..4 {
                            accumulators[row][block] =
                                _mm512_fmadd_ps(value, vectors[block], accumulators[row][block]);
                        }
                    }
                }
                for row in 0..rows {
                    let destination = unsafe {
                        (output_ptr as *mut f32).add((row0 + row) * n + panel * PANEL_WIDTH)
                    };
                    for block in 0..4 {
                        unsafe {
                            _mm512_storeu_ps(destination.add(block * 16), accumulators[row][block]);
                        }
                    }
                }
            }
        });
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
}

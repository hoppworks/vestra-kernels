//! Optional prepared oneDNN F32 convolution boundary.
//!
//! It is feature-gated and intentionally has no automatic dependency
//! discovery. See the benchmark record for the pinned native build contract.

use std::ptr::NonNull;

#[repr(C)]
struct RawPrepared;

unsafe extern "C" {
    fn vestra_onednn_conv2d_create(
        in_channels: usize,
        out_channels: usize,
        height: usize,
        width: usize,
        weight: *const f32,
        bias: *const f32,
    ) -> *mut RawPrepared;
    fn vestra_onednn_conv2d_execute(
        prepared: *mut RawPrepared,
        input: *const f32,
        output: *mut f32,
    ) -> i32;
    fn vestra_onednn_conv2d_destroy(prepared: *mut RawPrepared);
}

/// A model-owned, prepacked strict-F32 3×3/pad-1 oneDNN operation.
pub struct PreparedOneDnnConv2dF32 {
    raw: NonNull<RawPrepared>,
    input_len: usize,
    output_len: usize,
}

impl PreparedOneDnnConv2dF32 {
    /// Returns `None` for invalid shapes or when native primitive creation
    /// fails. Callers must retain their established numerical fallback.
    pub fn try_new(
        weight: &[f32],
        bias: &[f32],
        in_channels: usize,
        out_channels: usize,
        height: usize,
        width: usize,
    ) -> Option<Self> {
        if weight.len() != out_channels.checked_mul(in_channels)?.checked_mul(9)?
            || bias.len() != out_channels
        {
            return None;
        }
        let input_len = in_channels.checked_mul(height)?.checked_mul(width)?;
        let output_len = out_channels.checked_mul(height)?.checked_mul(width)?;
        let raw = unsafe {
            vestra_onednn_conv2d_create(
                in_channels,
                out_channels,
                height,
                width,
                weight.as_ptr(),
                bias.as_ptr(),
            )
        };
        Some(Self {
            raw: NonNull::new(raw)?,
            input_len,
            output_len,
        })
    }

    /// Runs exactly one prepared convolution. The native bridge waits for the
    /// stream before returning, so timing around this call includes all work.
    pub fn execute(&mut self, input: &[f32], output: &mut [f32]) -> bool {
        if input.len() != self.input_len || output.len() != self.output_len {
            return false;
        }
        unsafe {
            vestra_onednn_conv2d_execute(self.raw.as_ptr(), input.as_ptr(), output.as_mut_ptr())
                != 0
        }
    }
}

impl Drop for PreparedOneDnnConv2dF32 {
    fn drop(&mut self) {
        unsafe { vestra_onednn_conv2d_destroy(self.raw.as_ptr()) };
    }
}

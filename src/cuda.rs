//! Native CUDA driver foundation for Vestra fixed-shape kernels.
//!
//! This module intentionally exposes only device ownership and checked host ↔
//! device transfer. No Engine operator is routed here until its CUDA kernel
//! has a CPU F32 parity fixture. Dynamic driver loading keeps CPU-only builds
//! free of a CUDA toolkit dependency.

use std::sync::Arc;

use cudarc::{
    cublas::{CudaBlas, Gemm as CublasGemm, GemmConfig, sys::cublasOperation_t},
    driver::{
        CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream, LaunchConfig, PushKernelArg,
    },
    nvrtc::compile_ptx,
};

const CUDA_KERNEL_SOURCE: &str = r#"
extern "C" __global__ void vestra_add_f32(float* destination, const float* source, unsigned int len) {
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < len) {
        destination[index] += source[index];
    }
}

// Device-to-device copy used to retain the last local-token state across a
// subsequent DA3 global-attention block. Keeping this explicit avoids a host
// round trip merely to snapshot an activation.
extern "C" __global__ void vestra_copy_f32(float* destination, const float* source, unsigned int len) {
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < len) destination[index] = source[index];
}

// Device-resident tensor segment move. DA3 multi-view alternates individual
// view buffers with one flattened global-attention buffer; this primitive
// performs that layout scheduling without staging activations on the host.
extern "C" __global__ void vestra_copy_segment_f32(
    float* destination, const float* source,
    unsigned int destination_offset, unsigned int source_offset,
    unsigned int len
) {
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < len) destination[destination_offset + index] = source[source_offset + index];
}

// Replaces the leading token row from a cached camera-token parameter. DA3
// injects this at alt_start; the prefix form is intentionally general enough
// to remain useful for future register-token handling.
extern "C" __global__ void vestra_overwrite_prefix_f32(
    float* destination, const float* source, unsigned int len
) {
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < len) destination[index] = source[index];
}

extern "C" __global__ void vestra_bias_scale_f32(
    float* values,
    const float* bias,
    const float* gamma,
    unsigned int rows,
    unsigned int cols
) {
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int len = rows * cols;
    if (index < len) {
        const unsigned int column = index % cols;
        values[index] = (values[index] + bias[column]) * gamma[column];
    }
}

extern "C" __global__ void vestra_gelu_f32(float* values, unsigned int len) {
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < len) {
        const float x = values[index];
        const float ax = fabsf(x * 0.70710678f);
        const float t = 1.0f / (1.0f + 0.3275911f * ax);
        float poly = 1.0614054f * t - 1.4531520f;
        poly = poly * t + 1.4214137f;
        poly = poly * t - 0.28449674f;
        poly = poly * t + 0.25482959f;
        const float erf = copysignf(1.0f - poly * t * expf(-ax * ax), x);
        values[index] = 0.5f * x * (1.0f + erf);
    }
}

extern "C" __global__ void vestra_layernorm_f32(
    const float* input,
    const float* gamma,
    const float* beta,
    float* output,
    float epsilon,
    unsigned int cols
) {
    const unsigned int row = blockIdx.x;
    const unsigned int tid = threadIdx.x;
    const unsigned int base = row * cols;
    __shared__ float partial[256];

    float sum = 0.0f;
    for (unsigned int col = tid; col < cols; col += blockDim.x) sum += input[base + col];
    partial[tid] = sum;
    __syncthreads();
    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) partial[tid] += partial[tid + stride];
        __syncthreads();
    }
    const float mean = partial[0] / (float)cols;

    float squared = 0.0f;
    for (unsigned int col = tid; col < cols; col += blockDim.x) {
        const float delta = input[base + col] - mean;
        squared += delta * delta;
    }
    partial[tid] = squared;
    __syncthreads();
    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) partial[tid] += partial[tid + stride];
        __syncthreads();
    }
    const float inv_stddev = rsqrtf(partial[0] / (float)cols + epsilon);
    for (unsigned int col = tid; col < cols; col += blockDim.x) {
        output[base + col] = (input[base + col] - mean) * inv_stddev * gamma[col] + beta[col];
    }
}

// Strict-parity variant: one CUDA thread owns each row and deliberately
// preserves the CPU scalar reduction order. It is an oracle route, not the
// eventual throughput kernel.
extern "C" __global__ void vestra_layernorm_f32_cpu_order(
    const float* input,
    const float* gamma,
    const float* beta,
    float* output,
    float epsilon,
    unsigned int cols
) {
    const unsigned int row = blockIdx.x;
    const unsigned int base = row * cols;
    float sum = 0.0f;
    for (unsigned int col = 0; col < cols; ++col) sum += input[base + col];
    const float mean = sum / (float)cols;
    float variance_sum = 0.0f;
    for (unsigned int col = 0; col < cols; ++col) {
        const float delta = input[base + col] - mean;
        variance_sum += delta * delta;
    }
    const float inv_stddev = 1.0f / sqrtf(variance_sum / (float)cols + epsilon);
    for (unsigned int col = 0; col < cols; ++col) {
        output[base + col] = (input[base + col] - mean) * inv_stddev * gamma[col] + beta[col];
    }
}

extern "C" __global__ void vestra_attention_online_f32(
    const float* q, const float* k, const float* v, float* out,
    unsigned int tokens, unsigned int head_dim, float scale, unsigned int rows
) {
    const unsigned int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;
    const unsigned int head = row / tokens;
    const unsigned int query = row % tokens;
    const unsigned int head_base = head * tokens * head_dim;
    const float* qi = q + head_base + query * head_dim;
    float accum[64];
    for (unsigned int d = 0; d < head_dim; ++d) accum[d] = 0.0f;
    float running_max = -1.0e30f;
    float running_sum = 0.0f;
    for (unsigned int j0 = 0; j0 < tokens; j0 += 64) {
        const unsigned int j1 = min(j0 + 64, tokens);
        float scores[64];
        float tile_max = -1.0e30f;
        for (unsigned int j = j0; j < j1; ++j) {
            float dot = 0.0f;
            const float* kj = k + head_base + j * head_dim;
            for (unsigned int d = 0; d < head_dim; ++d) dot += qi[d] * kj[d];
            const float score = dot * scale;
            scores[j - j0] = score;
            tile_max = fmaxf(tile_max, score);
        }
        const float new_max = fmaxf(running_max, tile_max);
        const float correction = isfinite(running_max) ? expf(running_max - new_max) : 0.0f;
        running_sum *= correction;
        for (unsigned int d = 0; d < head_dim; ++d) accum[d] *= correction;
        for (unsigned int j = j0; j < j1; ++j) {
            const float p = expf(scores[j - j0] - new_max);
            running_sum += p;
            const float* vj = v + head_base + j * head_dim;
            for (unsigned int d = 0; d < head_dim; ++d) accum[d] += p * vj[d];
        }
        running_max = new_max;
    }
    float* oi = out + head_base + query * head_dim;
    const float inv_sum = 1.0f / running_sum;
    for (unsigned int d = 0; d < head_dim; ++d) oi[d] = accum[d] * inv_sum;
}

extern "C" __global__ void vestra_qk_norm_rope_f32(
    float* q, float* k,
    const float* q_gamma, const float* q_beta,
    const float* k_gamma, const float* k_beta,
    const float* positions_yx,
    unsigned int tokens, float frequency, float epsilon, unsigned int rows
) {
    const unsigned int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;
    const unsigned int token = row % tokens;
    const unsigned int base = row * 64;
    float qsum = 0.0f, ksum = 0.0f;
    for (unsigned int d = 0; d < 64; ++d) { qsum += q[base+d]; ksum += k[base+d]; }
    const float qmean = qsum / 64.0f, kmean = ksum / 64.0f;
    float qvar = 0.0f, kvar = 0.0f;
    for (unsigned int d = 0; d < 64; ++d) {
        const float qd = q[base+d] - qmean, kd = k[base+d] - kmean;
        qvar += qd * qd; kvar += kd * kd;
    }
    const float qinv = 1.0f / sqrtf(qvar / 64.0f + epsilon);
    const float kinv = 1.0f / sqrtf(kvar / 64.0f + epsilon);
    float qrow[64], krow[64];
    for (unsigned int d = 0; d < 64; ++d) {
        qrow[d] = (q[base+d] - qmean) * qinv * q_gamma[d] + q_beta[d];
        krow[d] = (k[base+d] - kmean) * kinv * k_gamma[d] + k_beta[d];
    }
    const float y = positions_yx[2 * token], x = positions_yx[2 * token + 1];
    for (unsigned int axis = 0; axis < 2; ++axis) {
        const float pos = axis == 0 ? y : x;
        const unsigned int offset = axis * 32;
        for (unsigned int i = 0; i < 16; ++i) {
            const float theta = powf(frequency, -2.0f * (float)i / 32.0f);
            const float sine = sinf(pos * theta), cosine = cosf(pos * theta);
            const float qa = qrow[offset+i], qb = qrow[offset+i+16];
            const float ka = krow[offset+i], kb = krow[offset+i+16];
            qrow[offset+i] = qa * cosine - qb * sine;
            qrow[offset+i+16] = qb * cosine + qa * sine;
            krow[offset+i] = ka * cosine - kb * sine;
            krow[offset+i+16] = kb * cosine + ka * sine;
        }
    }
    for (unsigned int d = 0; d < 64; ++d) { q[base+d] = qrow[d]; k[base+d] = krow[d]; }
}

// Converts a token-major fused QKV projection `[tokens, 3, heads, head_dim]`
// into the head-major layout consumed by the attention kernel. Every output
// element is independent, so this is a pure device-side layout conversion.
extern "C" __global__ void vestra_split_qkv_hnd_f32(
    const float* qkv, float* q, float* k, float* v,
    unsigned int tokens, unsigned int heads, unsigned int head_dim
) {
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int per_token = heads * head_dim;
    const unsigned int len = tokens * per_token;
    if (index >= len) return;
    const unsigned int token = index / per_token;
    const unsigned int within = index % per_token;
    const unsigned int source = token * (3 * per_token) + within;
    const unsigned int head_major = (within / head_dim) * (tokens * head_dim)
        + token * head_dim + (within % head_dim);
    q[head_major] = qkv[source];
    k[head_major] = qkv[source + per_token];
    v[head_major] = qkv[source + 2 * per_token];
}

// Inverse layout conversion for the attention output: head-major
// `[heads,tokens,head_dim]` to token-major `[tokens,heads*head_dim]`.
extern "C" __global__ void vestra_hnd_to_token_f32(
    const float* hnd, float* token_major,
    unsigned int tokens, unsigned int heads, unsigned int head_dim
) {
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int per_token = heads * head_dim;
    const unsigned int len = tokens * per_token;
    if (index >= len) return;
    const unsigned int token = index / per_token;
    const unsigned int within = index % per_token;
    const unsigned int source = (within / head_dim) * (tokens * head_dim)
        + token * head_dim + (within % head_dim);
    token_major[index] = hnd[source];
}

// Transposes token-major `[tokens, channels]` to CHW `[channels, height,
// width]`; tokens are row-major over the spatial grid. This is the device
// layout needed by the DPT resize and convolution stages after a 1x1 GEMM.
extern "C" __global__ void vestra_token_to_chw_f32(
    const float* token_major, float* chw,
    unsigned int tokens, unsigned int channels
) {
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int len = tokens * channels;
    if (index >= len) return;
    const unsigned int token = index / channels;
    const unsigned int channel = index % channels;
    chw[channel * tokens + token] = token_major[index];
}

// Lowers a contiguous NCHW image into row-major, non-overlapping patch rows.
// The row dimension follows the OIHW inner order: `[channel, patch_y, patch_x]`.
extern "C" __global__ void vestra_patchify_nchw_f32(
    const float* image, float* patches,
    unsigned int height, unsigned int width,
    unsigned int patch, unsigned int channels
) {
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int patch_area = patch * patch;
    const unsigned int row_width = channels * patch_area;
    const unsigned int grid_width = width / patch;
    const unsigned int patch_count = (height / patch) * grid_width;
    const unsigned int len = patch_count * row_width;
    if (index >= len) return;
    const unsigned int patch_index = index / row_width;
    const unsigned int within_patch = index % row_width;
    const unsigned int channel = within_patch / patch_area;
    const unsigned int spatial = within_patch % patch_area;
    const unsigned int patch_y = patch_index / grid_width;
    const unsigned int patch_x = patch_index % grid_width;
    const unsigned int y = patch_y * patch + spatial / patch;
    const unsigned int x = patch_x * patch + spatial % patch;
    patches[index] = image[(channel * height + y) * width + x];
}

// DA3-BASE token assembly for models without register tokens. It prefixes the
// CLS row and applies the already interpolated positional embedding without a
// host-side token buffer. The input patch rows and output are token-major.
extern "C" __global__ void vestra_prepend_cls_add_pos_f32(
    const float* patches, const float* cls, const float* position,
    float* tokens, unsigned int patch_count, unsigned int embed
) {
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int len = (patch_count + 1) * embed;
    if (index >= len) return;
    const unsigned int token = index / embed;
    const unsigned int channel = index % embed;
    const float value = token == 0
        ? cls[channel]
        : patches[(token - 1) * embed + channel];
    tokens[index] = value + position[index];
}
"#;

#[derive(Debug, thiserror::Error)]
pub enum CudaError {
    #[error("failed to initialize CUDA device {device}: {detail}")]
    Initialize { device: usize, detail: String },
    #[error("CUDA host-to-device transfer failed: {0}")]
    Upload(String),
    #[error("CUDA device-to-host transfer failed: {0}")]
    Download(String),
    #[error("CUDA residual add requires equally sized tensors, got {destination} and {source_len}")]
    LengthMismatch {
        destination: usize,
        source_len: usize,
    },
    #[error("CUDA kernel compilation or launch failed: {0}")]
    Kernel(String),
    #[error("CUDA CUBLAS operation failed: {0}")]
    Blas(String),
}

/// A single Engine-owned CUDA device and its default ordered stream.
#[derive(Clone)]
pub struct CudaRuntime {
    device: usize,
    context: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    _module: Arc<CudaModule>,
    residual_add: CudaFunction,
    copy: CudaFunction,
    copy_segment: CudaFunction,
    overwrite_prefix: CudaFunction,
    bias_scale: CudaFunction,
    gelu: CudaFunction,
    layernorm: CudaFunction,
    layernorm_cpu_order: CudaFunction,
    attention_online: CudaFunction,
    qk_norm_rope: CudaFunction,
    split_qkv_hnd: CudaFunction,
    hnd_to_token: CudaFunction,
    token_to_chw: CudaFunction,
    patchify_nchw: CudaFunction,
    prepend_cls_add_pos: CudaFunction,
    blas: Arc<CudaBlas>,
}

impl CudaRuntime {
    /// Retains the selected device primary context through the CUDA driver API.
    pub fn new(device: usize) -> Result<Self, CudaError> {
        let context = CudaContext::new(device).map_err(|error| CudaError::Initialize {
            device,
            detail: format!("{error:?}"),
        })?;
        let stream = context.default_stream();
        let ptx = compile_ptx(CUDA_KERNEL_SOURCE)
            .map_err(|error| CudaError::Kernel(format!("NVRTC compile: {error:?}")))?;
        let module = context
            .load_module(ptx)
            .map_err(|error| CudaError::Kernel(format!("PTX load: {error:?}")))?;
        let residual_add = module
            .load_function("vestra_add_f32")
            .map_err(|error| CudaError::Kernel(format!("function lookup: {error:?}")))?;
        let copy = module
            .load_function("vestra_copy_f32")
            .map_err(|error| CudaError::Kernel(format!("function lookup: {error:?}")))?;
        let copy_segment = module
            .load_function("vestra_copy_segment_f32")
            .map_err(|error| CudaError::Kernel(format!("function lookup: {error:?}")))?;
        let overwrite_prefix = module
            .load_function("vestra_overwrite_prefix_f32")
            .map_err(|error| CudaError::Kernel(format!("function lookup: {error:?}")))?;
        let bias_scale = module
            .load_function("vestra_bias_scale_f32")
            .map_err(|error| CudaError::Kernel(format!("function lookup: {error:?}")))?;
        let gelu = module
            .load_function("vestra_gelu_f32")
            .map_err(|error| CudaError::Kernel(format!("function lookup: {error:?}")))?;
        let layernorm = module
            .load_function("vestra_layernorm_f32")
            .map_err(|error| CudaError::Kernel(format!("function lookup: {error:?}")))?;
        let layernorm_cpu_order = module
            .load_function("vestra_layernorm_f32_cpu_order")
            .map_err(|error| CudaError::Kernel(format!("function lookup: {error:?}")))?;
        let attention_online = module
            .load_function("vestra_attention_online_f32")
            .map_err(|error| CudaError::Kernel(format!("function lookup: {error:?}")))?;
        let qk_norm_rope = module
            .load_function("vestra_qk_norm_rope_f32")
            .map_err(|error| CudaError::Kernel(format!("function lookup: {error:?}")))?;
        let split_qkv_hnd = module
            .load_function("vestra_split_qkv_hnd_f32")
            .map_err(|error| CudaError::Kernel(format!("function lookup: {error:?}")))?;
        let hnd_to_token = module
            .load_function("vestra_hnd_to_token_f32")
            .map_err(|error| CudaError::Kernel(format!("function lookup: {error:?}")))?;
        let token_to_chw = module
            .load_function("vestra_token_to_chw_f32")
            .map_err(|error| CudaError::Kernel(format!("function lookup: {error:?}")))?;
        let patchify_nchw = module
            .load_function("vestra_patchify_nchw_f32")
            .map_err(|error| CudaError::Kernel(format!("function lookup: {error:?}")))?;
        let prepend_cls_add_pos = module
            .load_function("vestra_prepend_cls_add_pos_f32")
            .map_err(|error| CudaError::Kernel(format!("function lookup: {error:?}")))?;
        let blas = Arc::new(
            CudaBlas::new(stream.clone())
                .map_err(|error| CudaError::Blas(format!("handle initialization: {error:?}")))?,
        );
        Ok(Self {
            device,
            context,
            stream,
            _module: module,
            residual_add,
            copy,
            copy_segment,
            overwrite_prefix,
            bias_scale,
            gelu,
            layernorm,
            layernorm_cpu_order,
            attention_online,
            qk_norm_rope,
            split_qkv_hnd,
            hnd_to_token,
            token_to_chw,
            patchify_nchw,
            prepend_cls_add_pos,
            blas,
        })
    }

    #[must_use]
    pub const fn device(&self) -> usize {
        self.device
    }

    /// Makes a device-resident F32 copy. Model-weight packing belongs above
    /// this primitive and is amortized at Engine load time.
    pub fn upload_f32(&self, values: &[f32]) -> Result<CudaTensorF32, CudaError> {
        let data = self
            .stream
            .clone_htod(values)
            .map_err(|error| CudaError::Upload(format!("{error:?}")))?;
        Ok(CudaTensorF32 {
            data,
            len: values.len(),
        })
    }

    /// Explicit synchronization boundary for parity fixtures and final result
    /// downloads. Production operator chains stay device-resident.
    pub fn download_f32(&self, tensor: &CudaTensorF32) -> Result<Vec<f32>, CudaError> {
        self.stream
            .clone_dtoh(&tensor.data)
            .map_err(|error| CudaError::Download(format!("{error:?}")))
    }

    /// Makes a device-to-device F32 copy without exposing an activation to
    /// host memory. This is used for DA3's `local_x` snapshot immediately
    /// before a global attention layer.
    pub fn copy_f32(&self, source: &CudaTensorF32) -> Result<CudaTensorF32, CudaError> {
        let mut destination = self
            .stream
            .clone_htod(&vec![0.0_f32; source.len])
            .map_err(|error| CudaError::Upload(format!("copy allocation: {error:?}")))?;
        let count = u32::try_from(source.len)
            .map_err(|_| CudaError::Kernel("tensor length exceeds CUDA u32 indexing".into()))?;
        unsafe {
            self.stream
                .launch_builder(&self.copy)
                .arg(&mut destination)
                .arg(&source.data)
                .arg(&count)
                .launch(LaunchConfig::for_num_elems(count))
                .map_err(|error| CudaError::Kernel(format!("copy launch: {error:?}")))?;
        }
        Ok(CudaTensorF32 {
            data: destination,
            len: source.len,
        })
    }

    /// Allocates a device tensor and copies a contiguous range from `source`.
    pub fn copy_segment_f32(
        &self,
        source: &CudaTensorF32,
        source_offset: usize,
        len: usize,
    ) -> Result<CudaTensorF32, CudaError> {
        if source_offset.saturating_add(len) > source.len {
            return Err(CudaError::LengthMismatch {
                destination: len,
                source_len: source.len.saturating_sub(source_offset),
            });
        }
        let mut destination = self
            .stream
            .clone_htod(&vec![0.0_f32; len])
            .map_err(|error| CudaError::Upload(format!("segment allocation: {error:?}")))?;
        self.copy_segment_into_raw(&mut destination, 0, &source.data, source_offset, len)?;
        Ok(CudaTensorF32 {
            data: destination,
            len,
        })
    }

    /// Copies a contiguous device range into a preallocated device tensor.
    pub fn copy_segment_into_f32(
        &self,
        destination: &mut CudaTensorF32,
        destination_offset: usize,
        source: &CudaTensorF32,
        source_offset: usize,
        len: usize,
    ) -> Result<(), CudaError> {
        if destination_offset.saturating_add(len) > destination.len
            || source_offset.saturating_add(len) > source.len
        {
            return Err(CudaError::LengthMismatch {
                destination: destination.len.saturating_sub(destination_offset),
                source_len: source.len.saturating_sub(source_offset),
            });
        }
        self.copy_segment_into_raw(
            &mut destination.data,
            destination_offset,
            &source.data,
            source_offset,
            len,
        )
    }

    fn copy_segment_into_raw(
        &self,
        destination: &mut CudaSlice<f32>,
        destination_offset: usize,
        source: &CudaSlice<f32>,
        source_offset: usize,
        len: usize,
    ) -> Result<(), CudaError> {
        let destination_offset = u32::try_from(destination_offset).map_err(|_| {
            CudaError::Kernel("destination offset exceeds CUDA u32 indexing".into())
        })?;
        let source_offset = u32::try_from(source_offset)
            .map_err(|_| CudaError::Kernel("source offset exceeds CUDA u32 indexing".into()))?;
        let count = u32::try_from(len)
            .map_err(|_| CudaError::Kernel("segment length exceeds CUDA u32 indexing".into()))?;
        unsafe {
            self.stream
                .launch_builder(&self.copy_segment)
                .arg(destination)
                .arg(source)
                .arg(&destination_offset)
                .arg(&source_offset)
                .arg(&count)
                .launch(LaunchConfig::for_num_elems(count))
                .map_err(|error| CudaError::Kernel(format!("segment copy launch: {error:?}")))?;
        }
        Ok(())
    }

    /// Replaces a device tensor's leading contiguous values from a cached
    /// device parameter. The operation is ordered on this runtime's stream.
    pub fn overwrite_prefix_f32(
        &self,
        destination: &mut CudaTensorF32,
        source: &CudaTensorF32,
        len: usize,
    ) -> Result<(), CudaError> {
        if len > destination.len || len > source.len {
            return Err(CudaError::LengthMismatch {
                destination: destination.len,
                source_len: source.len,
            });
        }
        let count = u32::try_from(len)
            .map_err(|_| CudaError::Kernel("prefix length exceeds CUDA u32 indexing".into()))?;
        unsafe {
            self.stream
                .launch_builder(&self.overwrite_prefix)
                .arg(&mut destination.data)
                .arg(&source.data)
                .arg(&count)
                .launch(LaunchConfig::for_num_elems(count))
                .map_err(|error| {
                    CudaError::Kernel(format!("overwrite prefix launch: {error:?}"))
                })?;
        }
        Ok(())
    }

    /// Adds `source` into `destination` on the device. This is the first
    /// parity-testable building block for DA3 residual and LayerScale paths;
    /// callers retain tensors on device across chained operators.
    pub fn add_f32_in_place(
        &self,
        destination: &mut CudaTensorF32,
        source: &CudaTensorF32,
    ) -> Result<(), CudaError> {
        if destination.len != source.len {
            return Err(CudaError::LengthMismatch {
                destination: destination.len,
                source_len: source.len,
            });
        }
        let count = u32::try_from(destination.len)
            .map_err(|_| CudaError::Kernel("tensor length exceeds CUDA u32 indexing".into()))?;
        // The CUDA source bounds-checks every launched index and the safe
        // driver builder tracks read/write dependencies for both allocations.
        unsafe {
            self.stream
                .launch_builder(&self.residual_add)
                .arg(&mut destination.data)
                .arg(&source.data)
                .arg(&count)
                .launch(LaunchConfig::for_num_elems(count))
                .map_err(|error| CudaError::Kernel(format!("residual add launch: {error:?}")))?;
        }
        Ok(())
    }

    /// Computes row-major `A[m,k] × B[k,n] -> C[m,n]` with F32 CUBLAS.
    /// Inputs and output stay device-resident. CUBLAS is column-major, so
    /// this uses the exact transposed identity `Cᵀ = Bᵀ × Aᵀ` without a host
    /// layout conversion.
    pub fn gemm_row_major_f32(
        &self,
        a: &CudaTensorF32,
        b: &CudaTensorF32,
        m: usize,
        k: usize,
        n: usize,
    ) -> Result<CudaTensorF32, CudaError> {
        if a.len != m.saturating_mul(k) || b.len != k.saturating_mul(n) {
            return Err(CudaError::Blas(format!(
                "row-major GEMM shape M={m}, K={k}, N={n} does not match A={} or B={}",
                a.len, b.len
            )));
        }
        let m_i32 = i32::try_from(m).map_err(|_| CudaError::Blas("M exceeds i32".into()))?;
        let k_i32 = i32::try_from(k).map_err(|_| CudaError::Blas("K exceeds i32".into()))?;
        let n_i32 = i32::try_from(n).map_err(|_| CudaError::Blas("N exceeds i32".into()))?;
        let mut output = self
            .stream
            .clone_htod(&vec![0.0_f32; m.saturating_mul(n)])
            .map_err(|error| CudaError::Upload(format!("GEMM output allocation: {error:?}")))?;
        let config = GemmConfig {
            transa: cublasOperation_t::CUBLAS_OP_N,
            transb: cublasOperation_t::CUBLAS_OP_N,
            // Column-major Cᵀ[N,M] = Bᵀ[N,K] × Aᵀ[K,M].
            m: n_i32,
            n: m_i32,
            k: k_i32,
            alpha: 1.0_f32,
            lda: n_i32,
            ldb: k_i32,
            beta: 0.0_f32,
            ldc: n_i32,
        };
        unsafe {
            self.blas
                .gemm(config, &b.data, &a.data, &mut output)
                .map_err(|error| CudaError::Blas(format!("row-major GEMM: {error:?}")))?;
        }
        Ok(CudaTensorF32 {
            data: output,
            len: m * n,
        })
    }

    /// Applies the DA3 linear epilogue `(values + bias) * gamma` per output
    /// column without moving the GEMM output off the device.
    pub fn bias_scale_f32_in_place(
        &self,
        values: &mut CudaTensorF32,
        bias: &CudaTensorF32,
        gamma: &CudaTensorF32,
        rows: usize,
        columns: usize,
    ) -> Result<(), CudaError> {
        let expected_values = rows.saturating_mul(columns);
        if values.len != expected_values || bias.len != columns || gamma.len != columns {
            return Err(CudaError::LengthMismatch {
                destination: values.len,
                source_len: expected_values,
            });
        }
        let rows = u32::try_from(rows)
            .map_err(|_| CudaError::Kernel("row count exceeds CUDA u32 indexing".into()))?;
        let columns = u32::try_from(columns)
            .map_err(|_| CudaError::Kernel("column count exceeds CUDA u32 indexing".into()))?;
        let count = rows.checked_mul(columns).ok_or_else(|| {
            CudaError::Kernel("bias-scale element count exceeds CUDA u32 indexing".into())
        })?;
        unsafe {
            self.stream
                .launch_builder(&self.bias_scale)
                .arg(&mut values.data)
                .arg(&bias.data)
                .arg(&gamma.data)
                .arg(&rows)
                .arg(&columns)
                .launch(LaunchConfig::for_num_elems(count))
                .map_err(|error| CudaError::Kernel(format!("bias-scale launch: {error:?}")))?;
        }
        Ok(())
    }

    /// Applies the DA3 exact-erf GELU approximation in place on the device.
    /// This matches Vestra Kernels' Abramowitz–Stegun F32 formulation; final
    /// Engine parity gates, rather than this primitive, own the tolerance.
    pub fn gelu_f32_in_place(&self, values: &mut CudaTensorF32) -> Result<(), CudaError> {
        let count = u32::try_from(values.len)
            .map_err(|_| CudaError::Kernel("tensor length exceeds CUDA u32 indexing".into()))?;
        unsafe {
            self.stream
                .launch_builder(&self.gelu)
                .arg(&mut values.data)
                .arg(&count)
                .launch(LaunchConfig::for_num_elems(count))
                .map_err(|error| CudaError::Kernel(format!("GELU launch: {error:?}")))?;
        }
        Ok(())
    }

    /// Computes row-major LayerNorm on the device. The one-block-per-row
    /// reduction is deliberately fixed at 256 threads so DA3's 768-wide
    /// token rows have stable launch geometry across every block.
    pub fn layernorm_f32(
        &self,
        input: &CudaTensorF32,
        gamma: &CudaTensorF32,
        beta: &CudaTensorF32,
        rows: usize,
        columns: usize,
        epsilon: f32,
    ) -> Result<CudaTensorF32, CudaError> {
        let expected = rows.saturating_mul(columns);
        if input.len != expected || gamma.len != columns || beta.len != columns {
            return Err(CudaError::LengthMismatch {
                destination: input.len,
                source_len: expected,
            });
        }
        let rows = u32::try_from(rows)
            .map_err(|_| CudaError::Kernel("row count exceeds CUDA u32 indexing".into()))?;
        let columns = u32::try_from(columns)
            .map_err(|_| CudaError::Kernel("column count exceeds CUDA u32 indexing".into()))?;
        let mut output = self
            .stream
            .clone_htod(&vec![0.0_f32; expected])
            .map_err(|error| {
                CudaError::Upload(format!("LayerNorm output allocation: {error:?}"))
            })?;
        unsafe {
            self.stream
                .launch_builder(&self.layernorm)
                .arg(&input.data)
                .arg(&gamma.data)
                .arg(&beta.data)
                .arg(&mut output)
                .arg(&epsilon)
                .arg(&columns)
                .launch(LaunchConfig {
                    grid_dim: (rows, 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                })
                .map_err(|error| CudaError::Kernel(format!("LayerNorm launch: {error:?}")))?;
        }
        Ok(CudaTensorF32 {
            data: output,
            len: expected,
        })
    }

    /// Strict CPU-order LayerNorm oracle. This intentionally launches one
    /// thread per row; use it to establish end-to-end numerical parity before
    /// replacing it with a proven parallel reduction strategy.
    pub fn layernorm_f32_cpu_order(
        &self,
        input: &CudaTensorF32,
        gamma: &CudaTensorF32,
        beta: &CudaTensorF32,
        rows: usize,
        columns: usize,
        epsilon: f32,
    ) -> Result<CudaTensorF32, CudaError> {
        let expected = rows.saturating_mul(columns);
        if input.len != expected || gamma.len != columns || beta.len != columns {
            return Err(CudaError::LengthMismatch {
                destination: input.len,
                source_len: expected,
            });
        }
        let rows = u32::try_from(rows)
            .map_err(|_| CudaError::Kernel("row count exceeds CUDA u32 indexing".into()))?;
        let columns = u32::try_from(columns)
            .map_err(|_| CudaError::Kernel("column count exceeds CUDA u32 indexing".into()))?;
        let mut output = self
            .stream
            .clone_htod(&vec![0.0_f32; expected])
            .map_err(|error| {
                CudaError::Upload(format!("CPU-order LayerNorm output allocation: {error:?}"))
            })?;
        unsafe {
            self.stream
                .launch_builder(&self.layernorm_cpu_order)
                .arg(&input.data)
                .arg(&gamma.data)
                .arg(&beta.data)
                .arg(&mut output)
                .arg(&epsilon)
                .arg(&columns)
                .launch(LaunchConfig {
                    grid_dim: (rows, 1, 1),
                    block_dim: (1, 1, 1),
                    shared_mem_bytes: 0,
                })
                .map_err(|error| {
                    CudaError::Kernel(format!("CPU-order LayerNorm launch: {error:?}"))
                })?;
        }
        Ok(CudaTensorF32 {
            data: output,
            len: expected,
        })
    }

    /// Device-resident tiled online-softmax attention oracle for head-major
    /// `[heads,tokens,head_dim]` F32 tensors. DA3-BASE uses `head_dim=64`;
    /// the kernel deliberately bounds the private accumulator at that shape.
    pub fn attention_online_f32(
        &self,
        q: &CudaTensorF32,
        k: &CudaTensorF32,
        v: &CudaTensorF32,
        heads: usize,
        tokens: usize,
        head_dim: usize,
    ) -> Result<CudaTensorF32, CudaError> {
        if head_dim == 0 || head_dim > 64 {
            return Err(CudaError::Kernel(format!(
                "CUDA online attention supports 1..=64 head dimensions, got {head_dim}"
            )));
        }
        let expected = heads.saturating_mul(tokens).saturating_mul(head_dim);
        if q.len != expected || k.len != expected || v.len != expected {
            return Err(CudaError::LengthMismatch {
                destination: q.len,
                source_len: expected,
            });
        }
        let tokens = u32::try_from(tokens)
            .map_err(|_| CudaError::Kernel("token count exceeds CUDA u32 indexing".into()))?;
        let head_dim = u32::try_from(head_dim)
            .map_err(|_| CudaError::Kernel("head dimension exceeds CUDA u32 indexing".into()))?;
        let rows = u32::try_from(heads.saturating_mul(tokens as usize))
            .map_err(|_| CudaError::Kernel("attention rows exceed CUDA u32 indexing".into()))?;
        let mut output = self
            .stream
            .clone_htod(&vec![0.0_f32; expected])
            .map_err(|error| {
                CudaError::Upload(format!("attention output allocation: {error:?}"))
            })?;
        let scale = 1.0_f32 / (head_dim as f32).sqrt();
        unsafe {
            self.stream
                .launch_builder(&self.attention_online)
                .arg(&q.data)
                .arg(&k.data)
                .arg(&v.data)
                .arg(&mut output)
                .arg(&tokens)
                .arg(&head_dim)
                .arg(&scale)
                .arg(&rows)
                .launch(LaunchConfig::for_num_elems(rows))
                .map_err(|error| {
                    CudaError::Kernel(format!("online attention launch: {error:?}"))
                })?;
        }
        Ok(CudaTensorF32 {
            data: output,
            len: expected,
        })
    }

    /// Splits a device-resident token-major fused QKV projection
    /// `[tokens, 3*heads*head_dim]` into the head-major Q/K/V tensors used by
    /// CUDA attention. The conversion never materializes on the host.
    pub fn split_qkv_hnd_f32(
        &self,
        qkv: &CudaTensorF32,
        tokens: usize,
        heads: usize,
        head_dim: usize,
    ) -> Result<(CudaTensorF32, CudaTensorF32, CudaTensorF32), CudaError> {
        let per_token = heads.saturating_mul(head_dim);
        let expected_qkv = tokens.saturating_mul(per_token).saturating_mul(3);
        if qkv.len != expected_qkv || per_token == 0 {
            return Err(CudaError::LengthMismatch {
                destination: qkv.len,
                source_len: expected_qkv,
            });
        }
        let len = tokens.saturating_mul(per_token);
        let count = u32::try_from(len)
            .map_err(|_| CudaError::Kernel("QKV layout length exceeds CUDA u32 indexing".into()))?;
        let tokens = u32::try_from(tokens)
            .map_err(|_| CudaError::Kernel("token count exceeds CUDA u32 indexing".into()))?;
        let heads = u32::try_from(heads)
            .map_err(|_| CudaError::Kernel("head count exceeds CUDA u32 indexing".into()))?;
        let head_dim = u32::try_from(head_dim)
            .map_err(|_| CudaError::Kernel("head dimension exceeds CUDA u32 indexing".into()))?;
        let make_output = || {
            self.stream
                .clone_htod(&vec![0.0_f32; len])
                .map_err(|error| {
                    CudaError::Upload(format!("QKV layout output allocation: {error:?}"))
                })
        };
        let mut q = make_output()?;
        let mut k = make_output()?;
        let mut v = make_output()?;
        unsafe {
            self.stream
                .launch_builder(&self.split_qkv_hnd)
                .arg(&qkv.data)
                .arg(&mut q)
                .arg(&mut k)
                .arg(&mut v)
                .arg(&tokens)
                .arg(&heads)
                .arg(&head_dim)
                .launch(LaunchConfig::for_num_elems(count))
                .map_err(|error| CudaError::Kernel(format!("QKV split launch: {error:?}")))?;
        }
        Ok((
            CudaTensorF32 { data: q, len },
            CudaTensorF32 { data: k, len },
            CudaTensorF32 { data: v, len },
        ))
    }

    /// Converts device-resident head-major attention output
    /// `[heads,tokens,head_dim]` into token-major `[tokens,heads*head_dim]`
    /// for the output projection without a host transfer.
    pub fn hnd_to_token_f32(
        &self,
        hnd: &CudaTensorF32,
        tokens: usize,
        heads: usize,
        head_dim: usize,
    ) -> Result<CudaTensorF32, CudaError> {
        let per_token = heads.saturating_mul(head_dim);
        let len = tokens.saturating_mul(per_token);
        if hnd.len != len || per_token == 0 {
            return Err(CudaError::LengthMismatch {
                destination: hnd.len,
                source_len: len,
            });
        }
        let count = u32::try_from(len).map_err(|_| {
            CudaError::Kernel("attention layout length exceeds CUDA u32 indexing".into())
        })?;
        let tokens = u32::try_from(tokens)
            .map_err(|_| CudaError::Kernel("token count exceeds CUDA u32 indexing".into()))?;
        let heads = u32::try_from(heads)
            .map_err(|_| CudaError::Kernel("head count exceeds CUDA u32 indexing".into()))?;
        let head_dim = u32::try_from(head_dim)
            .map_err(|_| CudaError::Kernel("head dimension exceeds CUDA u32 indexing".into()))?;
        let mut output = self
            .stream
            .clone_htod(&vec![0.0_f32; len])
            .map_err(|error| {
                CudaError::Upload(format!("attention layout output allocation: {error:?}"))
            })?;
        unsafe {
            self.stream
                .launch_builder(&self.hnd_to_token)
                .arg(&hnd.data)
                .arg(&mut output)
                .arg(&tokens)
                .arg(&heads)
                .arg(&head_dim)
                .launch(LaunchConfig::for_num_elems(count))
                .map_err(|error| {
                    CudaError::Kernel(format!("attention unpack launch: {error:?}"))
                })?;
        }
        Ok(CudaTensorF32 { data: output, len })
    }

    /// Converts token-major `[tokens, channels]` F32 to CHW, keeping the
    /// tensor device-resident for DPT convolution and resize operators.
    pub fn token_to_chw_f32(
        &self,
        token_major: &CudaTensorF32,
        tokens: usize,
        channels: usize,
    ) -> Result<CudaTensorF32, CudaError> {
        let len = tokens.saturating_mul(channels);
        if token_major.len != len || tokens == 0 || channels == 0 {
            return Err(CudaError::LengthMismatch {
                destination: token_major.len,
                source_len: len,
            });
        }
        let count = u32::try_from(len)
            .map_err(|_| CudaError::Kernel("DPT layout length exceeds CUDA u32 indexing".into()))?;
        let tokens = u32::try_from(tokens)
            .map_err(|_| CudaError::Kernel("DPT token count exceeds CUDA u32 indexing".into()))?;
        let channels = u32::try_from(channels)
            .map_err(|_| CudaError::Kernel("DPT channel count exceeds CUDA u32 indexing".into()))?;
        let mut output = self
            .stream
            .clone_htod(&vec![0.0_f32; len])
            .map_err(|error| CudaError::Upload(format!("DPT CHW allocation: {error:?}")))?;
        unsafe {
            self.stream
                .launch_builder(&self.token_to_chw)
                .arg(&token_major.data)
                .arg(&mut output)
                .arg(&tokens)
                .arg(&channels)
                .launch(LaunchConfig::for_num_elems(count))
                .map_err(|error| {
                    CudaError::Kernel(format!("DPT token-to-CHW launch: {error:?}"))
                })?;
        }
        Ok(CudaTensorF32 { data: output, len })
    }

    /// Lowers an NCHW image into row-major non-overlapping patch rows on the
    /// device. A following cached CUBLAS projection can consume the result
    /// directly; no host-side im2col buffer is created.
    pub fn patchify_nchw_f32(
        &self,
        image: &CudaTensorF32,
        height: usize,
        width: usize,
        patch: usize,
        channels: usize,
    ) -> Result<CudaTensorF32, CudaError> {
        if height == 0
            || width == 0
            || patch == 0
            || channels == 0
            || !height.is_multiple_of(patch)
            || !width.is_multiple_of(patch)
            || image.len != channels.saturating_mul(height).saturating_mul(width)
        {
            return Err(CudaError::Kernel(format!(
                "invalid NCHW patchify shape image={}, channels={channels}, height={height}, width={width}, patch={patch}",
                image.len
            )));
        }
        let rows = (height / patch).saturating_mul(width / patch);
        let columns = channels.saturating_mul(patch).saturating_mul(patch);
        let len = rows.saturating_mul(columns);
        let count = u32::try_from(len)
            .map_err(|_| CudaError::Kernel("patchified tensor exceeds CUDA u32 indexing".into()))?;
        let height = u32::try_from(height)
            .map_err(|_| CudaError::Kernel("height exceeds CUDA u32 indexing".into()))?;
        let width = u32::try_from(width)
            .map_err(|_| CudaError::Kernel("width exceeds CUDA u32 indexing".into()))?;
        let patch = u32::try_from(patch)
            .map_err(|_| CudaError::Kernel("patch exceeds CUDA u32 indexing".into()))?;
        let channels = u32::try_from(channels)
            .map_err(|_| CudaError::Kernel("channels exceeds CUDA u32 indexing".into()))?;
        let mut output = self
            .stream
            .clone_htod(&vec![0.0_f32; len])
            .map_err(|error| CudaError::Upload(format!("patchify allocation: {error:?}")))?;
        unsafe {
            self.stream
                .launch_builder(&self.patchify_nchw)
                .arg(&image.data)
                .arg(&mut output)
                .arg(&height)
                .arg(&width)
                .arg(&patch)
                .arg(&channels)
                .launch(LaunchConfig::for_num_elems(count))
                .map_err(|error| CudaError::Kernel(format!("NCHW patchify launch: {error:?}")))?;
        }
        Ok(CudaTensorF32 { data: output, len })
    }

    /// Prefixes DA3-BASE's CLS row and adds an interpolated positional grid to
    /// token-major projected patches. DA3-BASE has no register-token tensor;
    /// callers targeting a different model must use a separately qualified
    /// assembly route rather than silently dropping register rows.
    pub fn prepend_cls_add_pos_f32(
        &self,
        patches: &CudaTensorF32,
        cls: &CudaTensorF32,
        position: &CudaTensorF32,
        patch_count: usize,
        embed: usize,
    ) -> Result<CudaTensorF32, CudaError> {
        let tokens = patch_count.saturating_add(1);
        let patch_len = patch_count.saturating_mul(embed);
        let token_len = tokens.saturating_mul(embed);
        if embed == 0 || patches.len != patch_len || cls.len != embed || position.len != token_len {
            return Err(CudaError::Kernel(format!(
                "invalid CUDA token assembly patches={}, cls={}, position={}, patch_count={patch_count}, embed={embed}",
                patches.len, cls.len, position.len
            )));
        }
        let count = u32::try_from(token_len)
            .map_err(|_| CudaError::Kernel("token assembly exceeds CUDA u32 indexing".into()))?;
        let patch_count = u32::try_from(patch_count)
            .map_err(|_| CudaError::Kernel("patch count exceeds CUDA u32 indexing".into()))?;
        let embed = u32::try_from(embed)
            .map_err(|_| CudaError::Kernel("embedding width exceeds CUDA u32 indexing".into()))?;
        let mut output = self
            .stream
            .clone_htod(&vec![0.0_f32; token_len])
            .map_err(|error| CudaError::Upload(format!("token assembly allocation: {error:?}")))?;
        unsafe {
            self.stream
                .launch_builder(&self.prepend_cls_add_pos)
                .arg(&patches.data)
                .arg(&cls.data)
                .arg(&position.data)
                .arg(&mut output)
                .arg(&patch_count)
                .arg(&embed)
                .launch(LaunchConfig::for_num_elems(count))
                .map_err(|error| CudaError::Kernel(format!("token assembly launch: {error:?}")))?;
        }
        Ok(CudaTensorF32 {
            data: output,
            len: token_len,
        })
    }

    /// Fused CPU-order Q/K LayerNorm and 2D RoPE for DA3-BASE's fixed 64-wide
    /// head rows. `positions_yx` is device-resident `[tokens, 2]` F32.
    #[allow(clippy::too_many_arguments)]
    pub fn qk_norm_rope_f32_da3_base(
        &self,
        q: &mut CudaTensorF32,
        k: &mut CudaTensorF32,
        q_gamma: &CudaTensorF32,
        q_beta: &CudaTensorF32,
        k_gamma: &CudaTensorF32,
        k_beta: &CudaTensorF32,
        positions_yx: &CudaTensorF32,
        heads: usize,
        tokens: usize,
        frequency: f32,
        epsilon: f32,
    ) -> Result<(), CudaError> {
        let expected = heads.saturating_mul(tokens).saturating_mul(64);
        if q.len != expected
            || k.len != expected
            || q_gamma.len != 64
            || q_beta.len != 64
            || k_gamma.len != 64
            || k_beta.len != 64
            || positions_yx.len != tokens.saturating_mul(2)
        {
            return Err(CudaError::LengthMismatch {
                destination: q.len,
                source_len: expected,
            });
        }
        let tokens = u32::try_from(tokens)
            .map_err(|_| CudaError::Kernel("token count exceeds CUDA u32 indexing".into()))?;
        let rows = u32::try_from(heads.saturating_mul(tokens as usize))
            .map_err(|_| CudaError::Kernel("Q/K rows exceed CUDA u32 indexing".into()))?;
        unsafe {
            self.stream
                .launch_builder(&self.qk_norm_rope)
                .arg(&mut q.data)
                .arg(&mut k.data)
                .arg(&q_gamma.data)
                .arg(&q_beta.data)
                .arg(&k_gamma.data)
                .arg(&k_beta.data)
                .arg(&positions_yx.data)
                .arg(&tokens)
                .arg(&frequency)
                .arg(&epsilon)
                .arg(&rows)
                .launch(LaunchConfig::for_num_elems(rows))
                .map_err(|error| CudaError::Kernel(format!("Q/K norm+RoPE launch: {error:?}")))?;
        }
        Ok(())
    }

    /// Uploads immutable row-major linear parameters once. The returned plan
    /// accepts and returns device tensors, so callers can chain projections
    /// without re-uploading weights or materializing intermediate results.
    pub fn prepare_linear_f32(
        &self,
        input_features: usize,
        output_features: usize,
        weight: &[f32],
        bias: &[f32],
        gamma: Option<&[f32]>,
    ) -> Result<CudaLinearF32, CudaError> {
        if weight.len() != input_features.saturating_mul(output_features)
            || bias.len() != output_features
            || gamma.is_some_and(|values| values.len() != output_features)
        {
            return Err(CudaError::Blas(format!(
                "linear parameter shape K={input_features}, N={output_features} does not match weight={}, bias={}, gamma={}",
                weight.len(),
                bias.len(),
                gamma.map_or(output_features, <[f32]>::len),
            )));
        }
        let unit_gamma;
        let gamma = match gamma {
            Some(gamma) => gamma,
            None => {
                unit_gamma = vec![1.0; output_features];
                &unit_gamma
            }
        };
        Ok(CudaLinearF32 {
            runtime: self.clone(),
            input_features,
            output_features,
            weight: self.upload_f32(weight)?,
            bias: self.upload_f32(bias)?,
            gamma: self.upload_f32(gamma)?,
        })
    }

    #[must_use]
    pub fn context(&self) -> &Arc<CudaContext> {
        &self.context
    }

    #[must_use]
    pub fn stream(&self) -> &Arc<CudaStream> {
        &self.stream
    }
}

/// F32 allocation on the CUDA device. Its contents are opaque outside native
/// kernels, preventing accidental CPU fallback inside a claimed GPU path.
pub struct CudaTensorF32 {
    data: CudaSlice<f32>,
    len: usize,
}

/// Immutable device-resident parameters for one row-major F32 projection.
/// Construct it at model-load time through [`CudaRuntime::prepare_linear_f32`]
/// and feed its output directly to the next device operator.
pub struct CudaLinearF32 {
    runtime: CudaRuntime,
    input_features: usize,
    output_features: usize,
    weight: CudaTensorF32,
    bias: CudaTensorF32,
    gamma: CudaTensorF32,
}

impl CudaLinearF32 {
    #[must_use]
    pub const fn input_features(&self) -> usize {
        self.input_features
    }

    #[must_use]
    pub const fn output_features(&self) -> usize {
        self.output_features
    }

    /// Computes `(input × weight + bias) × gamma` wholly on the device.
    pub fn run(&self, input: &CudaTensorF32, rows: usize) -> Result<CudaTensorF32, CudaError> {
        if input.len != rows.saturating_mul(self.input_features) {
            return Err(CudaError::Blas(format!(
                "linear input rows={rows}, K={} does not match device input length {}",
                self.input_features, input.len
            )));
        }
        let mut output = self.runtime.gemm_row_major_f32(
            input,
            &self.weight,
            rows,
            self.input_features,
            self.output_features,
        )?;
        self.runtime.bias_scale_f32_in_place(
            &mut output,
            &self.bias,
            &self.gamma,
            rows,
            self.output_features,
        )?;
        Ok(output)
    }

    /// Computes the normal device-resident linear result and immediately
    /// emits DPT-compatible CHW order. This is intentionally a method of the
    /// cached plan so a DPT stage does not materialize the token-major result
    /// on the host merely to transpose it.
    pub fn run_chw(&self, input: &CudaTensorF32, rows: usize) -> Result<CudaTensorF32, CudaError> {
        let token_major = self.run(input, rows)?;
        self.runtime
            .token_to_chw_f32(&token_major, rows, self.output_features)
    }
}

impl CudaTensorF32 {
    #[must_use]
    pub const fn len(&self) -> usize {
        self.len
    }

    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn deterministic_values(len: usize, seed: f32) -> Vec<f32> {
        (0..len)
            .map(|index| ((index as f32 * 0.017_578_125 + seed).sin() * 0.75) + 0.1)
            .collect()
    }

    #[test]
    fn driver_round_trip_preserves_f32_values_when_explicitly_enabled() {
        if std::env::var_os("VESTRA_CUDA_TEST").is_none() {
            return;
        }
        let runtime = CudaRuntime::new(0).unwrap();
        let mut tensor = runtime.upload_f32(&[1.0, -2.0, 3.5]).unwrap();
        let addend = runtime.upload_f32(&[0.5, 2.0, -1.0]).unwrap();
        assert_eq!(tensor.len(), 3);
        runtime.add_f32_in_place(&mut tensor, &addend).unwrap();
        assert_eq!(runtime.download_f32(&tensor).unwrap(), [1.5, 0.0, 2.5]);

        let source = runtime.upload_f32(&[9.0, 8.0, 7.0, 6.0]).unwrap();
        let mut copied = runtime.copy_f32(&source).unwrap();
        let replacement = runtime.upload_f32(&[-1.0, -2.0]).unwrap();
        runtime
            .overwrite_prefix_f32(&mut copied, &replacement, 2)
            .unwrap();
        assert_eq!(
            runtime.download_f32(&copied).unwrap(),
            [-1.0, -2.0, 7.0, 6.0]
        );
        let segment = runtime.copy_segment_f32(&source, 1, 2).unwrap();
        assert_eq!(runtime.download_f32(&segment).unwrap(), [8.0, 7.0]);
        runtime
            .copy_segment_into_f32(&mut copied, 2, &segment, 0, 2)
            .unwrap();
        assert_eq!(
            runtime.download_f32(&copied).unwrap(),
            [-1.0, -2.0, 8.0, 7.0]
        );

        let left = runtime.upload_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let right = runtime
            .upload_f32(&[7.0, 8.0, 9.0, 10.0, 11.0, 12.0])
            .unwrap();
        let product = runtime.gemm_row_major_f32(&left, &right, 2, 3, 2).unwrap();
        assert_eq!(
            runtime.download_f32(&product).unwrap(),
            [58.0, 64.0, 139.0, 154.0]
        );

        let bias = runtime.upload_f32(&[1.0, -4.0]).unwrap();
        let gamma = runtime.upload_f32(&[0.5, 2.0]).unwrap();
        let mut epilogue = product;
        runtime
            .bias_scale_f32_in_place(&mut epilogue, &bias, &gamma, 2, 2)
            .unwrap();
        assert_eq!(
            runtime.download_f32(&epilogue).unwrap(),
            [29.5, 120.0, 70.0, 300.0]
        );

        let mut gelu = runtime.upload_f32(&[-3.0, -1.0, 0.0, 1.0, 3.0]).unwrap();
        runtime.gelu_f32_in_place(&mut gelu).unwrap();
        let mut expected = [-3.0, -1.0, 0.0, 1.0, 3.0];
        crate::scalar::gelu(&mut expected);
        for (actual, expected) in runtime.download_f32(&gelu).unwrap().iter().zip(expected) {
            assert!(
                (actual - expected).abs() <= 2e-6,
                "GELU diverged: actual={actual}, expected={expected}"
            );
        }

        let first = runtime
            .prepare_linear_f32(
                3,
                2,
                &[7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
                &[1.0, -4.0],
                Some(&[0.5, 2.0]),
            )
            .unwrap();
        let second = runtime
            .prepare_linear_f32(2, 1, &[2.0, -1.0], &[3.0], None)
            .unwrap();
        let chained = first.run(&left, 2).unwrap();
        let chained = second.run(&chained, 2).unwrap();
        assert_eq!(runtime.download_f32(&chained).unwrap(), [-58.0, -157.0]);
        let projected_chw = first.run_chw(&left, 2).unwrap();
        assert_eq!(
            runtime.download_f32(&projected_chw).unwrap(),
            [29.5, 70.0, 120.0, 300.0]
        );

        let norm_input = runtime
            .upload_f32(&[1.0, 2.0, 5.0, -2.0, 3.0, 4.0])
            .unwrap();
        let norm_gamma = runtime.upload_f32(&[1.5, 0.5, 2.0]).unwrap();
        let norm_beta = runtime.upload_f32(&[-1.0, 0.25, 2.0]).unwrap();
        let normalized = runtime
            .layernorm_f32(&norm_input, &norm_gamma, &norm_beta, 2, 3, 1e-5)
            .unwrap();
        let mut expected = [1.0, 2.0, 5.0, -2.0, 3.0, 4.0];
        crate::scalar::layernorm(
            &mut expected,
            2,
            3,
            &[1.5, 0.5, 2.0],
            &[-1.0, 0.25, 2.0],
            1e-5,
        );
        for (actual, expected) in runtime
            .download_f32(&normalized)
            .unwrap()
            .iter()
            .zip(expected)
        {
            assert!(
                (actual - expected).abs() <= 2e-5,
                "LayerNorm diverged: actual={actual}, expected={expected}"
            );
        }
        let normalized = runtime
            .layernorm_f32_cpu_order(&norm_input, &norm_gamma, &norm_beta, 2, 3, 1e-5)
            .unwrap();
        for (actual, expected) in runtime
            .download_f32(&normalized)
            .unwrap()
            .iter()
            .zip(expected)
        {
            assert!(
                (actual - expected).abs() <= 2e-6,
                "CPU-order LayerNorm diverged: actual={actual}, expected={expected}"
            );
        }

        let heads = 2;
        let tokens = 5;
        let head_dim = 4;
        let values = |seed: f32| {
            (0..heads * tokens * head_dim)
                .map(|index| (index as f32 * 0.03125 + seed).sin())
                .collect::<Vec<_>>()
        };
        let q = values(0.1);
        let k = values(-0.2);
        let v = values(0.7);
        let mut expected = vec![0.0; q.len()];
        crate::attention::attention_naive(&q, &k, &v, heads, tokens, head_dim, &mut expected);
        let q = runtime.upload_f32(&q).unwrap();
        let k = runtime.upload_f32(&k).unwrap();
        let v = runtime.upload_f32(&v).unwrap();
        let actual = runtime
            .attention_online_f32(&q, &k, &v, heads, tokens, head_dim)
            .unwrap();
        for (actual, expected) in runtime.download_f32(&actual).unwrap().iter().zip(expected) {
            assert!(
                (actual - expected).abs() <= 5e-5,
                "online attention diverged: actual={actual}, expected={expected}"
            );
        }

        // A deliberately non-square layout catches both token/head axes of
        // the device-only QKV split and attention-output unpack boundaries.
        let layout_tokens = 3;
        let layout_heads = 2;
        let layout_dim = 4;
        let embed = layout_heads * layout_dim;
        let qkv = deterministic_values(layout_tokens * 3 * embed, -0.4);
        let qkv_d = runtime.upload_f32(&qkv).unwrap();
        let (q_d, k_d, v_d) = runtime
            .split_qkv_hnd_f32(&qkv_d, layout_tokens, layout_heads, layout_dim)
            .unwrap();
        let mut expected_q = vec![0.0; layout_tokens * embed];
        let mut expected_k = vec![0.0; layout_tokens * embed];
        let mut expected_v = vec![0.0; layout_tokens * embed];
        for token in 0..layout_tokens {
            for head in 0..layout_heads {
                for dim in 0..layout_dim {
                    let hnd = (head * layout_tokens + token) * layout_dim + dim;
                    let source = token * 3 * embed + head * layout_dim + dim;
                    expected_q[hnd] = qkv[source];
                    expected_k[hnd] = qkv[source + embed];
                    expected_v[hnd] = qkv[source + 2 * embed];
                }
            }
        }
        assert_eq!(runtime.download_f32(&q_d).unwrap(), expected_q);
        assert_eq!(runtime.download_f32(&k_d).unwrap(), expected_k);
        assert_eq!(runtime.download_f32(&v_d).unwrap(), expected_v);
        let unpacked = runtime
            .hnd_to_token_f32(&q_d, layout_tokens, layout_heads, layout_dim)
            .unwrap();
        let mut expected_token_major = vec![0.0; layout_tokens * embed];
        for token in 0..layout_tokens {
            for head in 0..layout_heads {
                for dim in 0..layout_dim {
                    expected_token_major[token * embed + head * layout_dim + dim] =
                        expected_q[(head * layout_tokens + token) * layout_dim + dim];
                }
            }
        }
        assert_eq!(
            runtime.download_f32(&unpacked).unwrap(),
            expected_token_major
        );

        let chw = runtime
            .token_to_chw_f32(&unpacked, layout_tokens, embed)
            .unwrap();
        let mut expected_chw = vec![0.0; layout_tokens * embed];
        for token in 0..layout_tokens {
            for channel in 0..embed {
                expected_chw[channel * layout_tokens + token] =
                    expected_token_major[token * embed + channel];
            }
        }
        assert_eq!(runtime.download_f32(&chw).unwrap(), expected_chw);

        // Patch rows must retain both the NCHW source contract and the
        // OIHW-compatible inner dimension used by DA3's patch projection.
        let image = (0..3 * 4 * 4).map(|index| index as f32).collect::<Vec<_>>();
        let image = runtime.upload_f32(&image).unwrap();
        let patches = runtime.patchify_nchw_f32(&image, 4, 4, 2, 3).unwrap();
        assert_eq!(
            runtime.download_f32(&patches).unwrap(),
            vec![
                0.0, 1.0, 4.0, 5.0, 16.0, 17.0, 20.0, 21.0, 32.0, 33.0, 36.0, 37.0, 2.0, 3.0, 6.0,
                7.0, 18.0, 19.0, 22.0, 23.0, 34.0, 35.0, 38.0, 39.0, 8.0, 9.0, 12.0, 13.0, 24.0,
                25.0, 28.0, 29.0, 40.0, 41.0, 44.0, 45.0, 10.0, 11.0, 14.0, 15.0, 26.0, 27.0, 30.0,
                31.0, 42.0, 43.0, 46.0, 47.0,
            ]
        );

        let patches = runtime.upload_f32(&[10.0, 20.0, 30.0, 40.0]).unwrap();
        let cls = runtime.upload_f32(&[1.0, 2.0]).unwrap();
        let position = runtime
            .upload_f32(&[0.5, -0.5, 1.0, 2.0, -1.0, 3.0])
            .unwrap();
        let tokens = runtime
            .prepend_cls_add_pos_f32(&patches, &cls, &position, 2, 2)
            .unwrap();
        assert_eq!(
            runtime.download_f32(&tokens).unwrap(),
            [1.5, 1.5, 11.0, 22.0, 29.0, 43.0]
        );
    }

    /// Actual DA3-BASE attention geometry. This is separately gated because
    /// it intentionally executes the full production-sized CPU reference as
    /// well as the CUDA oracle; it is not a fast unit test.
    #[test]
    fn cuda_online_attention_matches_production_da3_base_shape() {
        if std::env::var_os("VESTRA_CUDA_DA3_ATTENTION_TEST").is_none() {
            return;
        }
        const HEADS: usize = 12;
        const TOKENS: usize = 865;
        const HEAD_DIM: usize = 64;
        let len = HEADS * TOKENS * HEAD_DIM;
        let q = deterministic_values(len, 0.1);
        let k = deterministic_values(len, -0.3);
        let v = deterministic_values(len, 0.7);
        let mut expected = vec![0.0; len];
        crate::attention::attention(&q, &k, &v, HEADS, TOKENS, HEAD_DIM, &mut expected);

        let runtime = CudaRuntime::new(0).unwrap();
        let q = runtime.upload_f32(&q).unwrap();
        let k = runtime.upload_f32(&k).unwrap();
        let v = runtime.upload_f32(&v).unwrap();
        let actual = runtime
            .attention_online_f32(&q, &k, &v, HEADS, TOKENS, HEAD_DIM)
            .unwrap();
        let actual = runtime.download_f32(&actual).unwrap();
        let mae = expected
            .iter()
            .zip(&actual)
            .map(|(left, right)| (left - right).abs())
            .sum::<f32>()
            / len as f32;
        let max = expected
            .iter()
            .zip(&actual)
            .map(|(left, right)| (left - right).abs())
            .fold(0.0_f32, f32::max);
        assert!(mae <= 5e-5, "DA3 attention MAE {mae} exceeds 5e-5");
        assert!(max <= 5e-4, "DA3 attention max error {max} exceeds 5e-4");
    }

    #[test]
    fn cuda_qk_norm_rope_matches_production_da3_base_shape() {
        if std::env::var_os("VESTRA_CUDA_DA3_QK_ROPE_TEST").is_none() {
            return;
        }
        const HEADS: usize = 12;
        const TOKENS: usize = 865;
        const DIM: usize = 64;
        let len = HEADS * TOKENS * DIM;
        let source = deterministic_values(len, 0.21);
        let mut expected_q = source.clone();
        let mut expected_k = deterministic_values(len, -0.37);
        let q_gamma = deterministic_values(DIM, 0.11);
        let q_beta = deterministic_values(DIM, -0.15);
        let k_gamma = deterministic_values(DIM, 0.31);
        let k_beta = deterministic_values(DIM, -0.23);
        let positions_i64 = (0..TOKENS)
            .flat_map(|token| {
                if token == 0 {
                    [0_i64, 0]
                } else {
                    [((token - 1) / 36 + 1) as i64, ((token - 1) % 36 + 1) as i64]
                }
            })
            .collect::<Vec<_>>();
        assert!(crate::qk_norm_rope_f32_da3_base(
            &mut expected_q,
            &mut expected_k,
            &q_gamma,
            &q_beta,
            &k_gamma,
            &k_beta,
            &positions_i64,
            100.0,
            1e-5,
        ));

        let positions_f32 = positions_i64
            .iter()
            .map(|value| *value as f32)
            .collect::<Vec<_>>();
        let runtime = CudaRuntime::new(0).unwrap();
        let mut q = runtime.upload_f32(&source).unwrap();
        let mut k = runtime
            .upload_f32(&deterministic_values(len, -0.37))
            .unwrap();
        let q_gamma_d = runtime.upload_f32(&q_gamma).unwrap();
        let q_beta_d = runtime.upload_f32(&q_beta).unwrap();
        let k_gamma_d = runtime.upload_f32(&k_gamma).unwrap();
        let k_beta_d = runtime.upload_f32(&k_beta).unwrap();
        let positions = runtime.upload_f32(&positions_f32).unwrap();
        runtime
            .qk_norm_rope_f32_da3_base(
                &mut q, &mut k, &q_gamma_d, &q_beta_d, &k_gamma_d, &k_beta_d, &positions, HEADS,
                TOKENS, 100.0, 1e-5,
            )
            .unwrap();
        let actual_q = runtime.download_f32(&q).unwrap();
        let actual_k = runtime.download_f32(&k).unwrap();
        let error = |expected: &[f32], actual: &[f32]| {
            let mae = expected
                .iter()
                .zip(actual)
                .map(|(a, b)| (a - b).abs())
                .sum::<f32>()
                / expected.len() as f32;
            let max = expected
                .iter()
                .zip(actual)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0_f32, f32::max);
            (mae, max)
        };
        let (q_mae, q_max) = error(&expected_q, &actual_q);
        let (k_mae, k_max) = error(&expected_k, &actual_k);
        assert!(
            q_mae <= 5e-5 && q_max <= 5e-4,
            "Q mismatch: MAE={q_mae}, max={q_max}"
        );
        assert!(
            k_mae <= 5e-5 && k_max <= 5e-4,
            "K mismatch: MAE={k_mae}, max={k_max}"
        );
    }
}

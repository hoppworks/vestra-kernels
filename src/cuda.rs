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
    bias_scale: CudaFunction,
    gelu: CudaFunction,
    layernorm: CudaFunction,
    layernorm_cpu_order: CudaFunction,
    attention_online: CudaFunction,
    qk_norm_rope: CudaFunction,
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
            bias_scale,
            gelu,
            layernorm,
            layernorm_cpu_order,
            attention_online,
            qk_norm_rope,
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
}

use crate::gemm::Gemm;
use rayon::prelude::*;
use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

pub use crate::specialized::NonoverlapTransposeF32;

#[derive(Clone, Copy, Hash, PartialEq, Eq)]
struct WinogradFilterKey {
    in_c: usize,
    out_c: usize,
    words_hash: u64,
}

static WINOGRAD_FILTERS: OnceLock<Mutex<HashMap<WinogradFilterKey, Arc<[f32]>>>> = OnceLock::new();

// Rayon workers process many independent Winograd tile blocks over the life
// of an inference. Reusing their scratch avoids allocating two sizeable
// matrices for every block while keeping each worker's storage private.
thread_local! {
    static WINOGRAD_SCRATCH: RefCell<(Vec<f32>, Vec<f32>)> = const { RefCell::new((Vec::new(), Vec::new())) };
}

/// Four tiles is the measured Zen-5 default: it keeps the transformed input
/// and product slabs within the worker cache while still exposing many
/// independent work items. Other values remain available for reproducible
/// scheduler/cache experiments without changing convolution arithmetic.
#[inline]
fn winograd_tile_block() -> usize {
    match std::env::var("DA3_WINOGRAD_TILE_BLOCK").ok().as_deref() {
        Some("4") => 4,
        Some("6") => 6,
        Some("12") => 12,
        Some("16") => 16,
        _ => 4,
    }
}

#[inline]
fn winograd_min_blocks_per_job() -> usize {
    match std::env::var("DA3_WINOGRAD_MIN_BLOCKS_PER_JOB")
        .ok()
        .as_deref()
    {
        Some("2") => 2,
        Some("4") => 4,
        Some("8") => 8,
        _ => 4,
    }
}

/// The exact-shape fused final head is the only F(2) convolution with a
/// 64->32 filter.  It can use a different block size from the rest of the
/// network without changing arithmetic; retain four until a workhorse A/B
/// demonstrates a better cache/scheduler point.
#[inline]
fn fused_final_winograd_tile_block() -> usize {
    match std::env::var("DA3_FUSED_FINAL_WINO_TILE_BLOCK")
        .ok()
        .as_deref()
    {
        Some("2") => 2,
        Some("6") => 6,
        Some("8") => 8,
        _ => 4,
    }
}

/// Prepared F(2x2,3x3) filter in the blocked layout consumed by the CPU
/// Winograd kernel. It is tied to the owning model cache, not raw pointers.
#[derive(Clone)]
pub struct WinogradF2Filter(Arc<[f32]>);

/// Prepared F(4x4, 3x3) filter. This is intentionally separate from F(2),
/// because its transform domain is 6x6 rather than 4x4 and its numerical
/// error profile must be benchmarked independently.
#[derive(Clone)]
pub struct WinogradF4Filter(Arc<[f32]>);

/// Prepares an immutable `kernel == stride` transposed-convolution filter for
/// the independently versioned AVX-512 kernel.
pub fn prepare_nonoverlap_transpose_filter(
    weight: &[f32],
    in_c: usize,
    out_c: usize,
    kh: usize,
    kw: usize,
) -> NonoverlapTransposeF32 {
    crate::specialized::prepare_nonoverlap_transpose_f32(weight, in_c, out_c, kh, kw)
}

/// Executes a prepacked non-overlapping transposed convolution. `false`
/// means that the caller should take its established generic fallback.
pub fn conv_transpose2d_prepared(
    input: &[f32],
    ih: usize,
    iw: usize,
    filter: &NonoverlapTransposeF32,
    bias: Option<&[f32]>,
    out: &mut [f32],
) -> bool {
    crate::specialized::nonoverlap_transpose_f32(input, ih, iw, filter, bias, out)
}

fn winograd_filter_key(weight: &[f32], in_c: usize, out_c: usize) -> WinogradFilterKey {
    // The model weights are immutable. Hashing their exact F32 bit patterns
    // makes cache reuse safe across separately loaded models as well.
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for value in weight {
        hash ^= u64::from(value.to_bits());
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    WinogradFilterKey {
        in_c,
        out_c,
        words_hash: hash,
    }
}

fn transformed_winograd_filter(weight: &[f32], in_c: usize, out_c: usize) -> Arc<[f32]> {
    let profile = std::env::var_os("DA_WINO_PROFILE").is_some();
    let started = std::time::Instant::now();
    let key = winograd_filter_key(weight, in_c, out_c);
    let cache = WINOGRAD_FILTERS.get_or_init(|| Mutex::new(HashMap::new()));
    let mut cache = cache.lock().expect("Winograd filter cache mutex poisoned");
    if let Some(filter) = cache.get(&key) {
        if profile {
            eprintln!(
                "phase: wino_filter {in_c}x{out_c} cache_hit={:.3}ms",
                started.elapsed().as_secs_f64() * 1e3,
            );
        }
        return Arc::clone(filter);
    }
    // ggml's cache puts output channels innermost. This is the layout the
    // AVX-512 blocked kernel consumes: U[position][input][output].
    let mut transformed = vec![0.0; 16 * in_c * out_c];
    for oc in 0..out_c {
        for ic in 0..in_c {
            let g = &weight[(oc * in_c + ic) * 9..(oc * in_c + ic + 1) * 9];
            let mut t = [[0.0; 3]; 4];
            for j in 0..3 {
                let (a, b, c) = (g[j], g[3 + j], g[6 + j]);
                t[0][j] = a;
                t[1][j] = 0.5 * (a + b + c);
                t[2][j] = 0.5 * (a - b + c);
                t[3][j] = c;
            }
            let mut u = [0.0; 16];
            for i in 0..4 {
                let (a, b, c) = (t[i][0], t[i][1], t[i][2]);
                u[i * 4] = a;
                u[i * 4 + 1] = 0.5 * (a + b + c);
                u[i * 4 + 2] = 0.5 * (a - b + c);
                u[i * 4 + 3] = c;
            }
            for position in 0..16 {
                transformed[(position * in_c + ic) * out_c + oc] = u[position];
            }
        }
    }
    let transformed: Arc<[f32]> = transformed.into();
    cache.insert(key, Arc::clone(&transformed));
    if profile {
        eprintln!(
            "phase: wino_filter {in_c}x{out_c} cache_miss={:.3}ms",
            started.elapsed().as_secs_f64() * 1e3,
        );
    }
    transformed
}

/// Converts a model-owned 3x3 OIHW filter once for repeated inference.
pub fn prepare_winograd_f2_filter(weight: &[f32], in_c: usize, out_c: usize) -> WinogradF2Filter {
    WinogradF2Filter(transformed_winograd_filter(weight, in_c, out_c))
}

/// Converts an OIHW 3x3 filter into the F(4x4,3x3) Winograd domain.
///
/// The transform is Lavin & Gray's commonly used six-point variant. It is
/// exposed only as an opt-in benchmark candidate because its larger transform
/// coefficients trade fewer multiplies for greater F32 roundoff than F(2).
pub fn prepare_winograd_f4_filter(weight: &[f32], in_c: usize, out_c: usize) -> WinogradF4Filter {
    debug_assert_eq!(weight.len(), out_c * in_c * 9);
    const G: [[f32; 3]; 6] = [
        [0.25, 0.0, 0.0],
        [-1.0 / 6.0, -1.0 / 6.0, -1.0 / 6.0],
        [-1.0 / 6.0, 1.0 / 6.0, -1.0 / 6.0],
        [1.0 / 24.0, 1.0 / 12.0, 1.0 / 6.0],
        [1.0 / 24.0, -1.0 / 12.0, 1.0 / 6.0],
        [0.0, 0.0, 1.0],
    ];
    let mut transformed = vec![0.0; 36 * in_c * out_c];
    for oc in 0..out_c {
        for ic in 0..in_c {
            let g = &weight[(oc * in_c + ic) * 9..(oc * in_c + ic + 1) * 9];
            let mut tmp = [[0.0; 3]; 6];
            for r in 0..6 {
                for c in 0..3 {
                    tmp[r][c] = G[r][0] * g[c] + G[r][1] * g[3 + c] + G[r][2] * g[6 + c];
                }
            }
            for r in 0..6 {
                for c in 0..6 {
                    transformed[(r * 6 + c) * in_c * out_c + ic * out_c + oc] =
                        tmp[r][0] * G[c][0] + tmp[r][1] * G[c][1] + tmp[r][2] * G[c][2];
                }
            }
        }
    }
    WinogradF4Filter(transformed.into())
}

/// im2col: expandiert `input` (NCHW, N=1) zu einer `(out_c_rows=kh*kw*in_c) x (oh*ow)`
/// Spaltenmatrix, sodass `conv2d` als eine einzige GEMM (`weight_mat @ col`) berechnet
/// werden kann. `col` ist row-major mit Shape `(in_c*kh*kw) x (oh*ow)`.
// Keeping all tensor dimensions explicit is intentional at the kernel
// boundary; packaging them would obscure the NCHW-to-matrix layout contract.
#[allow(clippy::too_many_arguments)]
fn im2col(
    input: &[f32],
    in_c: usize,
    ih: usize,
    iw: usize,
    kh: usize,
    kw: usize,
    stride: usize,
    pad: usize,
    oh: usize,
    ow: usize,
    col: &mut [f32],
) {
    debug_assert_eq!(input.len(), in_c * ih * iw);
    debug_assert_eq!(col.len(), in_c * kh * kw * oh * ow);
    let out_spatial = oh * ow;
    // Rows are independent.  The large high-resolution DPT convolutions
    // materialize hundreds of MiB here, so distribute the copy while keeping
    // each row's coordinate mapping and values exactly unchanged.
    col.par_chunks_mut(out_spatial)
        .enumerate()
        .for_each(|(row_idx, row)| {
            let c = row_idx / (kh * kw);
            let kernel_idx = row_idx % (kh * kw);
            let ky = kernel_idx / kw;
            let kx = kernel_idx % kw;
            let in_plane = &input[c * ih * iw..(c + 1) * ih * iw];
            for oy in 0..oh {
                let iy = oy as isize * stride as isize + ky as isize - pad as isize;
                for ox in 0..ow {
                    let ix = ox as isize * stride as isize + kx as isize - pad as isize;
                    row[oy * ow + ox] =
                        if iy >= 0 && iy < ih as isize && ix >= 0 && ix < iw as isize {
                            in_plane[iy as usize * iw + ix as usize]
                        } else {
                            0.0
                        };
                }
            }
        });
}

/// Winograd F(2x2, 3x3) convolution for the common stride-1, pad-1 DPT
/// shape.  The transforms use only additions, subtractions and halves.
#[allow(clippy::too_many_arguments)]
fn conv3x3_winograd_f2(
    input: &[f32],
    in_c: usize,
    ih: usize,
    iw: usize,
    weight: &[f32],
    out_c: usize,
    bias: Option<&[f32]>,
    out: &mut [f32],
) {
    let transformed = transformed_winograd_filter(weight, in_c, out_c);
    conv3x3_winograd_f2_impl(input, in_c, ih, iw, &transformed, out_c, bias, false, out);
}

/// Runs a 3x3 Winograd convolution using a model-owned prepared filter.
#[allow(clippy::too_many_arguments)]
pub fn conv3x3_winograd_f2_prepared(
    input: &[f32],
    in_c: usize,
    ih: usize,
    iw: usize,
    filter: &WinogradF2Filter,
    out_c: usize,
    bias: Option<&[f32]>,
    out: &mut [f32],
) {
    conv3x3_winograd_f2_impl(input, in_c, ih, iw, &filter.0, out_c, bias, false, out);
}

/// Fuses an align-corners bilinear resize directly into the input transform
/// of a prepared F(2x2, 3x3) Winograd convolution.
///
/// `add` is an optional target-sized CHW tensor.  It represents the DPT UV
/// embedding, which is added *after* resize in the materialized path.  The
/// arithmetic for every resized sample deliberately mirrors
/// `bilinear_resize_align_corners`: this function only removes the
/// target-sized activation buffer between the two operations.
#[allow(clippy::too_many_arguments)]
pub fn conv3x3_winograd_f2_prepared_resize_align_corners(
    input: &[f32],
    in_c: usize,
    ih: usize,
    iw: usize,
    oh: usize,
    ow: usize,
    add: Option<&[f32]>,
    filter: &WinogradF2Filter,
    out_c: usize,
    bias: Option<&[f32]>,
    out: &mut [f32],
) {
    debug_assert_eq!(input.len(), in_c * ih * iw);
    debug_assert!(add.is_none_or(|values| values.len() == in_c * oh * ow));
    debug_assert_eq!(out.len(), out_c * oh * ow);

    // Keep the coordinate calculation and f32 weights byte-for-byte aligned
    // with `resample::src_coord_align_corners` / its caller.
    let coords = |dst: usize, len_in: usize, len_out: usize| {
        if len_in <= 1 {
            return (0usize, 0usize, 0.0f32);
        }
        let src = if len_out <= 1 {
            0.0
        } else {
            dst as f32 * (len_in - 1) as f32 / (len_out - 1) as f32
        };
        let src = src.clamp(0.0, (len_in - 1) as f32);
        let idx0 = (src.floor() as usize).min(len_in - 1);
        let idx1 = (idx0 + 1).min(len_in - 1);
        let frac = if idx1 == idx0 { 0.0 } else { src - idx0 as f32 };
        (idx0, idx1, frac.clamp(0.0, 1.0))
    };
    let y: Vec<_> = (0..oh).map(|dst| coords(dst, ih, oh)).collect();
    let x: Vec<_> = (0..ow).map(|dst| coords(dst, iw, ow)).collect();

    let tiles_x = ow.div_ceil(2);
    let tiles = oh.div_ceil(2) * tiles_x;
    let tile_block = fused_final_winograd_tile_block();
    let min_blocks_per_job = winograd_min_blocks_per_job();
    let transformed = &filter.0;
    let out_ptr = out.as_mut_ptr() as usize;
    (0..tiles)
        .into_par_iter()
        .step_by(tile_block)
        .with_min_len(min_blocks_per_job)
        .for_each(|tile0| {
            WINOGRAD_SCRATCH.with(|scratch| {
                let mut scratch = scratch.borrow_mut();
                let (v, products) = &mut *scratch;
                v.resize(16 * in_c * tile_block, 0.0);
                products.resize(16 * tile_block * out_c, 0.0);
                let active = (tiles - tile0).min(tile_block);
                for local_tile in 0..active {
                    let tile = tile0 + local_tile;
                    let ty = tile / tiles_x;
                    let tx = tile % tiles_x;
                    let oy0 = ty * 2;
                    let ox0 = tx * 2;
                    for ic in 0..in_c {
                        let input_plane = &input[ic * ih * iw..(ic + 1) * ih * iw];
                        let add_plane = add.map(|values| &values[ic * oh * ow..(ic + 1) * oh * ow]);
                        let mut d = [0.0; 16];
                        for dy in 0..4 {
                            let oy = oy0 as isize + dy as isize - 1;
                            for dx in 0..4 {
                                let ox = ox0 as isize + dx as isize - 1;
                                let sample =
                                    if oy >= 0 && oy < oh as isize && ox >= 0 && ox < ow as isize {
                                        let (y0, y1, fy) = y[oy as usize];
                                        let (x0, x1, fx) = x[ox as usize];
                                        let row0 = &input_plane[y0 * iw..(y0 + 1) * iw];
                                        let row1 = &input_plane[y1 * iw..(y1 + 1) * iw];
                                        let top = row0[x0] * (1.0 - fx) + row0[x1] * fx;
                                        let bot = row1[x0] * (1.0 - fx) + row1[x1] * fx;
                                        top * (1.0 - fy)
                                            + bot * fy
                                            + add_plane.map_or(0.0, |plane| {
                                                plane[oy as usize * ow + ox as usize]
                                            })
                                    } else {
                                        0.0
                                    };
                                d[dy * 4 + dx] = sample;
                            }
                        }
                        let mut column = [0.0; 16];
                        for j in 0..4 {
                            let (a, b, c, d3) = (d[j], d[4 + j], d[8 + j], d[12 + j]);
                            column[j] = a - c;
                            column[4 + j] = b + c;
                            column[8 + j] = c - b;
                            column[12 + j] = b - d3;
                        }
                        for i in 0..4 {
                            let (a, b, c, d3) = (
                                column[i * 4],
                                column[i * 4 + 1],
                                column[i * 4 + 2],
                                column[i * 4 + 3],
                            );
                            v[(i * 4) * in_c * active + ic * active + local_tile] = a - c;
                            v[(i * 4 + 1) * in_c * active + ic * active + local_tile] = b + c;
                            v[(i * 4 + 2) * in_c * active + ic * active + local_tile] = c - b;
                            v[(i * 4 + 3) * in_c * active + ic * active + local_tile] = b - d3;
                        }
                    }
                }
                let used_v = &v[..16 * in_c * active];
                let used_products = &mut products[..16 * active * out_c];
                used_products.fill(0.0);
                let used_external = crate::specialized::winograd_f2_blocked_f32(
                    transformed,
                    used_v,
                    used_products,
                    in_c,
                    out_c,
                    active,
                );
                if !used_external {
                    for position in 0..16 {
                        for local_tile in 0..active {
                            for oc in 0..out_c {
                                let mut sum = 0.0;
                                for ic in 0..in_c {
                                    sum += transformed[(position * in_c + ic) * out_c + oc]
                                        * v[(position * in_c + ic) * active + local_tile];
                                }
                                products[(position * active + local_tile) * out_c + oc] = sum;
                            }
                        }
                    }
                }
                for local_tile in 0..active {
                    let tile = tile0 + local_tile;
                    let ty = tile / tiles_x;
                    let tx = tile % tiles_x;
                    for oc in 0..out_c {
                        let mut p = [0.0; 8];
                        for j in 0..4 {
                            p[j] = products[(j * active + local_tile) * out_c + oc]
                                + products[((4 + j) * active + local_tile) * out_c + oc]
                                + products[((8 + j) * active + local_tile) * out_c + oc];
                            p[4 + j] = products[((4 + j) * active + local_tile) * out_c + oc]
                                - products[((8 + j) * active + local_tile) * out_c + oc]
                                - products[((12 + j) * active + local_tile) * out_c + oc];
                        }
                        let b = bias.map_or(0.0, |values| values[oc]);
                        let plane = unsafe {
                            std::slice::from_raw_parts_mut(
                                (out_ptr as *mut f32).add(oc * oh * ow),
                                oh * ow,
                            )
                        };
                        let values = [
                            p[0] + p[1] + p[2] + b,
                            p[1] - p[2] - p[3] + b,
                            p[4] + p[5] + p[6] + b,
                            p[5] - p[6] - p[7] + b,
                        ];
                        for dy in 0..2 {
                            for dx in 0..2 {
                                let oy = ty * 2 + dy;
                                let ox = tx * 2 + dx;
                                if oy < oh && ox < ow {
                                    plane[oy * ow + ox] = values[dy * 2 + dx];
                                }
                            }
                        }
                    }
                }
            });
        });
}

/// Opt-in F(4x4, 3x3) Winograd convolution for the final DA3 head layer.
///
/// F(4) reduces transform-domain products relative to F(2), but has larger
/// coefficients and therefore a different F32 numerical error envelope. It
/// deliberately has no automatic dispatch: callers must select it only after
/// their end-to-end parity gate accepts the candidate.
#[allow(clippy::too_many_arguments)]
pub fn conv3x3_winograd_f4_prepared(
    input: &[f32],
    in_c: usize,
    ih: usize,
    iw: usize,
    filter: &WinogradF4Filter,
    out_c: usize,
    bias: Option<&[f32]>,
    out: &mut [f32],
) {
    debug_assert_eq!(input.len(), in_c * ih * iw);
    debug_assert_eq!(out.len(), out_c * ih * iw);
    debug_assert!(bias.is_none_or(|values| values.len() == out_c));
    // B^T and A^T for the six-point F(4x4,3x3) transform.
    const BT: [[f32; 6]; 6] = [
        [4.0, 0.0, -5.0, 0.0, 1.0, 0.0],
        [0.0, -4.0, -4.0, 1.0, 1.0, 0.0],
        [0.0, 4.0, -4.0, -1.0, 1.0, 0.0],
        [0.0, -2.0, -1.0, 2.0, 1.0, 0.0],
        [0.0, 2.0, -1.0, -2.0, 1.0, 0.0],
        [0.0, 4.0, 0.0, -5.0, 0.0, 1.0],
    ];
    const AT: [[f32; 6]; 4] = [
        [1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
        [0.0, 1.0, -1.0, 2.0, -2.0, 0.0],
        [0.0, 1.0, 1.0, 4.0, 4.0, 0.0],
        [0.0, 1.0, -1.0, 8.0, -8.0, 1.0],
    ];
    let tiles_x = iw.div_ceil(4);
    let tiles = ih.div_ceil(4) * tiles_x;
    let tile_block = 2usize;
    let out_ptr = out.as_mut_ptr() as usize;
    (0..tiles)
        .into_par_iter()
        .step_by(tile_block)
        .for_each(|tile0| {
            WINOGRAD_SCRATCH.with(|scratch| {
                let mut scratch = scratch.borrow_mut();
                let (v, products) = &mut *scratch;
                let active = (tiles - tile0).min(tile_block);
                v.resize(36 * in_c * active, 0.0);
                products.resize(36 * active * out_c, 0.0);
                for local_tile in 0..active {
                    let tile = tile0 + local_tile;
                    let ty = tile / tiles_x;
                    let tx = tile % tiles_x;
                    let y = ty * 4;
                    let x = tx * 4;
                    for ic in 0..in_c {
                        let mut d = [[0.0; 6]; 6];
                        for (dy, d_row) in d.iter_mut().enumerate() {
                            let sy = y as isize + dy as isize - 1;
                            for (dx, sample) in d_row.iter_mut().enumerate() {
                                let sx = x as isize + dx as isize - 1;
                                if sy >= 0 && sy < ih as isize && sx >= 0 && sx < iw as isize {
                                    *sample = input[(ic * ih + sy as usize) * iw + sx as usize];
                                }
                            }
                        }
                        let mut tmp = [[0.0; 6]; 6];
                        for r in 0..6 {
                            for c in 0..6 {
                                tmp[r][c] = BT[r][0] * d[0][c]
                                    + BT[r][1] * d[1][c]
                                    + BT[r][2] * d[2][c]
                                    + BT[r][3] * d[3][c]
                                    + BT[r][4] * d[4][c]
                                    + BT[r][5] * d[5][c];
                            }
                        }
                        for r in 0..6 {
                            for c in 0..6 {
                                v[(r * 6 + c) * in_c * active + ic * active + local_tile] =
                                    tmp[r][0] * BT[c][0]
                                        + tmp[r][1] * BT[c][1]
                                        + tmp[r][2] * BT[c][2]
                                        + tmp[r][3] * BT[c][3]
                                        + tmp[r][4] * BT[c][4]
                                        + tmp[r][5] * BT[c][5];
                            }
                        }
                    }
                }
                let used_external = crate::specialized::winograd_f4_blocked_f32(
                    &filter.0,
                    &v[..36 * in_c * active],
                    &mut products[..36 * active * out_c],
                    in_c,
                    out_c,
                    active,
                );
                if !used_external {
                    for position in 0..36 {
                        for local_tile in 0..active {
                            for oc in 0..out_c {
                                let mut sum = 0.0;
                                for ic in 0..in_c {
                                    sum += filter.0[(position * in_c + ic) * out_c + oc]
                                        * v[(position * in_c + ic) * active + local_tile];
                                }
                                products[(position * active + local_tile) * out_c + oc] = sum;
                            }
                        }
                    }
                }
                for local_tile in 0..active {
                    let tile = tile0 + local_tile;
                    let ty = tile / tiles_x;
                    let tx = tile % tiles_x;
                    for oc in 0..out_c {
                        let mut tmp = [[0.0; 6]; 4];
                        for r in 0..4 {
                            for c in 0..6 {
                                tmp[r][c] = AT[r][0]
                                    * products[(c) * active * out_c + local_tile * out_c + oc]
                                    + AT[r][1]
                                        * products
                                            [(6 + c) * active * out_c + local_tile * out_c + oc]
                                    + AT[r][2]
                                        * products
                                            [(12 + c) * active * out_c + local_tile * out_c + oc]
                                    + AT[r][3]
                                        * products
                                            [(18 + c) * active * out_c + local_tile * out_c + oc]
                                    + AT[r][4]
                                        * products
                                            [(24 + c) * active * out_c + local_tile * out_c + oc]
                                    + AT[r][5]
                                        * products
                                            [(30 + c) * active * out_c + local_tile * out_c + oc];
                            }
                        }
                        let plane = unsafe {
                            std::slice::from_raw_parts_mut(
                                (out_ptr as *mut f32).add(oc * ih * iw),
                                ih * iw,
                            )
                        };
                        let b = bias.map_or(0.0, |values| values[oc]);
                        for (dy, tmp_row) in tmp.iter().enumerate() {
                            for (dx, coefficients) in AT.iter().enumerate() {
                                let oy = ty * 4 + dy;
                                let ox = tx * 4 + dx;
                                if oy < ih && ox < iw {
                                    plane[oy * iw + ox] = tmp_row[0] * coefficients[0]
                                        + tmp_row[1] * coefficients[1]
                                        + tmp_row[2] * coefficients[2]
                                        + tmp_row[3] * coefficients[3]
                                        + tmp_row[4] * coefficients[4]
                                        + tmp_row[5] * coefficients[5]
                                        + b;
                                }
                            }
                        }
                    }
                }
            });
        });
}

/// Same F(2x2, 3x3) path as [`conv3x3_winograd_f2`], with ReLU applied
/// while each input sample is loaded for the transform. This is algebraically
/// identical to materialising `relu(input)` before the convolution.
#[allow(clippy::too_many_arguments)]
pub fn conv3x3_winograd_f2_relu_input(
    input: &[f32],
    in_c: usize,
    ih: usize,
    iw: usize,
    weight: &[f32],
    out_c: usize,
    bias: Option<&[f32]>,
    out: &mut [f32],
) {
    let transformed = transformed_winograd_filter(weight, in_c, out_c);
    conv3x3_winograd_f2_impl(input, in_c, ih, iw, &transformed, out_c, bias, true, out);
}

/// ReLU-input form of [`conv3x3_winograd_f2_prepared`].
#[allow(clippy::too_many_arguments)]
pub fn conv3x3_winograd_f2_prepared_relu_input(
    input: &[f32],
    in_c: usize,
    ih: usize,
    iw: usize,
    filter: &WinogradF2Filter,
    out_c: usize,
    bias: Option<&[f32]>,
    out: &mut [f32],
) {
    conv3x3_winograd_f2_impl(input, in_c, ih, iw, &filter.0, out_c, bias, true, out);
}

#[allow(clippy::too_many_arguments)]
fn conv3x3_winograd_f2_impl(
    input: &[f32],
    in_c: usize,
    ih: usize,
    iw: usize,
    transformed: &[f32],
    out_c: usize,
    bias: Option<&[f32]>,
    relu_input: bool,
    out: &mut [f32],
) {
    let oh = ih;
    let ow = iw;
    let tiles_y = oh.div_ceil(2);
    let tiles_x = ow.div_ceil(2);
    let tiles = tiles_y * tiles_x;
    let tile_block = winograd_tile_block();
    let min_blocks_per_job = winograd_min_blocks_per_job();
    // Every output tile owns a disjoint 2x2 region in every NCHW plane.
    // Write the inverse transform directly to it: the old staging buffer was
    // subsequently copied in a second whole-output pass.
    let out_ptr = out.as_mut_ptr() as usize;
    (0..tiles)
        .into_par_iter()
        .step_by(tile_block)
        // Keep the cache-sized arithmetic block, while the A/B selector can
        // hand several existing blocks to one Rayon task to measure scheduler
        // overhead without changing any tile arithmetic.
        .with_min_len(min_blocks_per_job)
        .for_each(|tile0| {
            WINOGRAD_SCRATCH.with(|scratch| {
                let mut scratch = scratch.borrow_mut();
                let (v, products) = &mut *scratch;
                v.resize(16 * in_c * tile_block, 0.0);
                products.resize(16 * tile_block * out_c, 0.0);
                let active = (tiles - tile0).min(tile_block);
                for local_tile in 0..active {
                    let tile = tile0 + local_tile;
                    let ty = tile / tiles_x;
                    let tx = tile % tiles_x;
                    let y = ty * 2;
                    let x = tx * 2;
                    for ic in 0..in_c {
                        let mut d = [0.0; 16];
                        for dy in 0..4 {
                            let sy = y as isize + dy as isize - 1;
                            for dx in 0..4 {
                                let sx = x as isize + dx as isize - 1;
                                let sample =
                                    if sy >= 0 && sy < ih as isize && sx >= 0 && sx < iw as isize {
                                        input[(ic * ih + sy as usize) * iw + sx as usize]
                                    } else {
                                        0.0
                                    };
                                d[dy * 4 + dx] = if relu_input { sample.max(0.0) } else { sample };
                            }
                        }
                        let mut column = [0.0; 16];
                        for j in 0..4 {
                            let (a, b, c, d3) = (d[j], d[4 + j], d[8 + j], d[12 + j]);
                            column[j] = a - c;
                            column[4 + j] = b + c;
                            column[8 + j] = c - b;
                            column[12 + j] = b - d3;
                        }
                        for i in 0..4 {
                            let (a, b, c, d3) = (
                                column[i * 4],
                                column[i * 4 + 1],
                                column[i * 4 + 2],
                                column[i * 4 + 3],
                            );
                            v[(i * 4) * in_c * active + ic * active + local_tile] = a - c;
                            v[(i * 4 + 1) * in_c * active + ic * active + local_tile] = b + c;
                            v[(i * 4 + 2) * in_c * active + ic * active + local_tile] = c - b;
                            v[(i * 4 + 3) * in_c * active + ic * active + local_tile] = b - d3;
                        }
                    }
                }
                let used_v = &v[..16 * in_c * active];
                let used_products = &mut products[..16 * active * out_c];
                // Besides providing the scalar fallback's neutral initial
                // state, this sequential touch warms the output pages before
                // the vector kernel's scattered stores and later inverse
                // transform reads.
                used_products.fill(0.0);
                let used_external = crate::specialized::winograd_f2_blocked_f32(
                    transformed,
                    used_v,
                    used_products,
                    in_c,
                    out_c,
                    active,
                );
                if !used_external {
                    for position in 0..16 {
                        for local_tile in 0..active {
                            for oc in 0..out_c {
                                let mut sum = 0.0;
                                for ic in 0..in_c {
                                    sum += transformed[(position * in_c + ic) * out_c + oc]
                                        * v[(position * in_c + ic) * active + local_tile];
                                }
                                products[(position * active + local_tile) * out_c + oc] = sum;
                            }
                        }
                    }
                }
                for local_tile in 0..active {
                    for oc in 0..out_c {
                        let mut p = [0.0; 8];
                        for j in 0..4 {
                            p[j] = products[(j * active + local_tile) * out_c + oc]
                                + products[((4 + j) * active + local_tile) * out_c + oc]
                                + products[((8 + j) * active + local_tile) * out_c + oc];
                            p[4 + j] = products[((4 + j) * active + local_tile) * out_c + oc]
                                - products[((8 + j) * active + local_tile) * out_c + oc]
                                - products[((12 + j) * active + local_tile) * out_c + oc];
                        }
                        let b = bias.map_or(0.0, |values| values[oc]);
                        let tile = tile0 + local_tile;
                        let ty = tile / tiles_x;
                        let tx = tile % tiles_x;
                        let plane = unsafe {
                            std::slice::from_raw_parts_mut(
                                (out_ptr as *mut f32).add(oc * oh * ow),
                                oh * ow,
                            )
                        };
                        let values = [
                            p[0] + p[1] + p[2] + b,
                            p[1] - p[2] - p[3] + b,
                            p[4] + p[5] + p[6] + b,
                            p[5] - p[6] - p[7] + b,
                        ];
                        for dy in 0..2 {
                            for dx in 0..2 {
                                let oy = ty * 2 + dy;
                                let ox = tx * 2 + dx;
                                if oy < oh && ox < ow {
                                    plane[oy * ow + ox] = values[dy * 2 + dx];
                                }
                            }
                        }
                    }
                }
            });
        });
}

#[cfg(test)]
#[allow(clippy::too_many_arguments)]
fn im2col_serial(
    input: &[f32],
    in_c: usize,
    ih: usize,
    iw: usize,
    kh: usize,
    kw: usize,
    stride: usize,
    pad: usize,
    oh: usize,
    ow: usize,
    col: &mut [f32],
) {
    let out_spatial = oh * ow;
    for c in 0..in_c {
        let in_plane = &input[c * ih * iw..(c + 1) * ih * iw];
        for ky in 0..kh {
            for kx in 0..kw {
                let row_idx = (c * kh + ky) * kw + kx;
                let row = &mut col[row_idx * out_spatial..(row_idx + 1) * out_spatial];
                for oy in 0..oh {
                    let iy = oy as isize * stride as isize + ky as isize - pad as isize;
                    for ox in 0..ow {
                        let ix = ox as isize * stride as isize + kx as isize - pad as isize;
                        row[oy * ow + ox] =
                            if iy >= 0 && iy < ih as isize && ix >= 0 && ix < iw as isize {
                                in_plane[iy as usize * iw + ix as usize]
                            } else {
                                0.0
                            };
                    }
                }
            }
        }
    }
}

/// Standard-Conv2D (NCHW, Batch=1) via im2col + GEMM.
///
/// - `input`: `in_c*ih*iw`
/// - `weight`: `out_c*in_c*kh*kw` (PyTorch/GGUF-Layout: OIHW)
/// - `bias`: optional `out_c`
/// - `out`: `out_c*oh*ow`, mit `oh = (ih + 2*pad - kh)/stride + 1` (analog `ow`)
#[allow(clippy::too_many_arguments)]
pub fn conv2d(
    input: &[f32],
    in_c: usize,
    ih: usize,
    iw: usize,
    weight: &[f32],
    out_c: usize,
    kh: usize,
    kw: usize,
    stride: usize,
    pad: usize,
    bias: Option<&[f32]>,
    gemm: &impl Gemm,
    out: &mut [f32],
) {
    debug_assert_eq!(input.len(), in_c * ih * iw);
    debug_assert_eq!(weight.len(), out_c * in_c * kh * kw);
    let oh = (ih + 2 * pad - kh) / stride + 1;
    let ow = (iw + 2 * pad - kw) / stride + 1;
    debug_assert_eq!(out.len(), out_c * oh * ow);

    if kh == 3 && kw == 3 && stride == 1 && pad == 1 {
        conv3x3_winograd_f2(input, in_c, ih, iw, weight, out_c, bias, out);
        return;
    }

    let k = in_c * kh * kw;
    let n = oh * ow;
    // For a 1x1 stride-1 no-padding convolution the im2col matrix is
    // exactly the existing NCHW input viewed as `[in_c, ih*iw]`.  Building
    // and filling a duplicate matrix only adds an allocation and a full
    // memory pass; use the input directly while preserving the same GEMM
    // operand order and F32 accumulation.
    if kh == 1 && kw == 1 && stride == 1 && pad == 0 {
        debug_assert_eq!(n, ih * iw);
        gemm.gemm(out_c, n, k, weight, input, out);
        if let Some(bias) = bias {
            debug_assert_eq!(bias.len(), out_c);
            for oc in 0..out_c {
                let row = &mut out[oc * n..(oc + 1) * n];
                let b = bias[oc];
                for value in row {
                    *value += b;
                }
            }
        }
        return;
    }
    let mut col = vec![0f32; k * n];
    im2col(input, in_c, ih, iw, kh, kw, stride, pad, oh, ow, &mut col);

    // weight ist bereits (out_c) x (in_c*kh*kw) row-major (OIHW geflattet) - passt
    // direkt als GEMM-Operand A.
    gemm.gemm(out_c, n, k, weight, &col, out);

    if let Some(bias) = bias {
        // Bias ist pro Ausgabekanal (Zeile in der out_c x n Matrix), nicht
        // pro Spalte - `scalar::add_bias_rows` broadcastet stattdessen
        // spaltenweise (fuer GEMM-Feature-Bias gedacht), daher hier direkt.
        debug_assert_eq!(bias.len(), out_c);
        for oc in 0..out_c {
            let row = &mut out[oc * n..(oc + 1) * n];
            let b = bias[oc];
            for v in row.iter_mut() {
                *v += b;
            }
        }
    }
}

/// Naives, direktes Conv2D ohne im2col/GEMM - Orakel zur Verifikation von
/// [`conv2d`] gegen brute-force Referenzsemantik.
#[allow(clippy::too_many_arguments)]
pub fn conv2d_naive(
    input: &[f32],
    in_c: usize,
    ih: usize,
    iw: usize,
    weight: &[f32],
    out_c: usize,
    kh: usize,
    kw: usize,
    stride: usize,
    pad: usize,
    bias: Option<&[f32]>,
    out: &mut [f32],
) {
    debug_assert_eq!(input.len(), in_c * ih * iw);
    debug_assert_eq!(weight.len(), out_c * in_c * kh * kw);
    let oh = (ih + 2 * pad - kh) / stride + 1;
    let ow = (iw + 2 * pad - kw) / stride + 1;
    debug_assert_eq!(out.len(), out_c * oh * ow);

    for oc in 0..out_c {
        for oy in 0..oh {
            for ox in 0..ow {
                let mut acc = 0f32;
                for ic in 0..in_c {
                    for ky in 0..kh {
                        let iy = oy as isize * stride as isize + ky as isize - pad as isize;
                        if iy < 0 || iy >= ih as isize {
                            continue;
                        }
                        for kx in 0..kw {
                            let ix = ox as isize * stride as isize + kx as isize - pad as isize;
                            if ix < 0 || ix >= iw as isize {
                                continue;
                            }
                            let iv = input[(ic * ih + iy as usize) * iw + ix as usize];
                            let wv = weight[((oc * in_c + ic) * kh + ky) * kw + kx];
                            acc += iv * wv;
                        }
                    }
                }
                if let Some(b) = bias {
                    acc += b[oc];
                }
                out[(oc * oh + oy) * ow + ox] = acc;
            }
        }
    }
}

/// ConvTranspose2D (NCHW, Batch=1), z.B. fuer die DPT-resize-Layer (k4s4).
/// Kein Padding-Parameter noetig fuer die hier verwendeten k=s (exaktes
/// Upsampling ohne Ueberlappung/Randbeschnitt); `output_padding` wird nicht
/// unterstuetzt.
///
/// - `weight`: `in_c*out_c*kh*kw` (PyTorch-ConvTranspose-Layout: IOHW)
/// - `bias`: optional `out_c`
/// - `oh = (ih-1)*stride + kh`, analog `ow`.
#[allow(clippy::too_many_arguments)]
pub fn conv_transpose2d(
    input: &[f32],
    in_c: usize,
    ih: usize,
    iw: usize,
    weight: &[f32],
    out_c: usize,
    kh: usize,
    kw: usize,
    stride: usize,
    bias: Option<&[f32]>,
    out: &mut [f32],
) {
    debug_assert_eq!(input.len(), in_c * ih * iw);
    debug_assert_eq!(weight.len(), in_c * out_c * kh * kw);
    let oh = (ih - 1) * stride + kh;
    let ow = (iw - 1) * stride + kw;
    debug_assert_eq!(out.len(), out_c * oh * ow);

    if kh == stride && kw == stride && in_c.is_multiple_of(16) {
        let prepared = prepare_nonoverlap_transpose_filter(weight, in_c, out_c, kh, kw);
        if conv_transpose2d_prepared(input, ih, iw, &prepared, bias, out) {
            return;
        }
    }

    // The DPT resize layers use kernel == stride with no padding.  Each input
    // spatial location therefore owns a disjoint output tile, so output
    // channels can be computed independently.  Keep the inner input-channel
    // accumulation order identical to the serial scatter path below.
    if kh == stride && kw == stride {
        out.par_chunks_mut(oh * ow)
            .enumerate()
            .for_each(|(oc, plane)| {
                for iy in 0..ih {
                    for ix in 0..iw {
                        for ky in 0..kh {
                            let oy = iy * stride + ky;
                            for kx in 0..kw {
                                let ox = ix * stride + kx;
                                let mut sum = 0.0;
                                for ic in 0..in_c {
                                    let iv = input[(ic * ih + iy) * iw + ix];
                                    let wv = weight[((ic * out_c + oc) * kh + ky) * kw + kx];
                                    sum += iv * wv;
                                }
                                plane[oy * ow + ox] = sum;
                            }
                        }
                    }
                }
                if let Some(bias) = bias {
                    for value in plane {
                        *value += bias[oc];
                    }
                }
            });
        return;
    }

    out.fill(0.0);
    for ic in 0..in_c {
        for iy in 0..ih {
            for ix in 0..iw {
                let iv = input[(ic * ih + iy) * iw + ix];
                if iv == 0.0 {
                    continue;
                }
                for oc in 0..out_c {
                    for ky in 0..kh {
                        let oy = iy * stride + ky;
                        for kx in 0..kw {
                            let ox = ix * stride + kx;
                            let wv = weight[((ic * out_c + oc) * kh + ky) * kw + kx];
                            out[(oc * oh + oy) * ow + ox] += iv * wv;
                        }
                    }
                }
            }
        }
    }
    if let Some(bias) = bias {
        for oc in 0..out_c {
            let plane = &mut out[oc * oh * ow..(oc + 1) * oh * ow];
            for v in plane.iter_mut() {
                *v += bias[oc];
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gemm::ScalarGemm;

    #[test]
    fn conv2d_matches_naive_and_direct_math_1x1() {
        // 1x1 conv is a per-pixel GEMM: in_c=3, out_c=2, spatial=2x2.
        // weight[oc][ic] chosen so output is hand-checkable.
        let in_c = 3;
        let (ih, iw) = (2, 2);
        let input: Vec<f32> = (1..=(in_c * ih * iw) as i32).map(|v| v as f32).collect();
        // input plane0: [1,2,3,4], plane1: [5,6,7,8], plane2: [9,10,11,12]
        let out_c = 2;
        // weight oc0: [1,0,0] (picks plane0), oc1: [0,0,1] (picks plane2)
        let weight = vec![
            1.0, 0.0, 0.0, // oc0
            0.0, 0.0, 1.0, // oc1
        ];
        let bias = vec![10.0, -1.0];
        let mut out = vec![0f32; out_c * ih * iw];
        conv2d(
            &input,
            in_c,
            ih,
            iw,
            &weight,
            out_c,
            1,
            1,
            1,
            0,
            Some(&bias),
            &ScalarGemm,
            &mut out,
        );
        // oc0 = plane0 + 10 = [11,12,13,14]; oc1 = plane2 - 1 = [8,9,10,11]
        assert_eq!(out, vec![11.0, 12.0, 13.0, 14.0, 8.0, 9.0, 10.0, 11.0]);

        let mut out_naive = vec![0f32; out_c * ih * iw];
        conv2d_naive(
            &input,
            in_c,
            ih,
            iw,
            &weight,
            out_c,
            1,
            1,
            1,
            0,
            Some(&bias),
            &mut out_naive,
        );
        assert_eq!(out, out_naive);
    }

    #[test]
    fn conv2d_1x1_fast_path_is_bitwise_generic_im2col() {
        let (in_c, out_c, ih, iw) = (5, 7, 4, 3);
        let mut rng = Xorshift32(0x1A11_C011);
        let input = random_vec(&mut rng, in_c * ih * iw);
        let weight = random_vec(&mut rng, out_c * in_c);
        let bias = random_vec(&mut rng, out_c);
        let mut fast = vec![0.0; out_c * ih * iw];
        conv2d(
            &input,
            in_c,
            ih,
            iw,
            &weight,
            out_c,
            1,
            1,
            1,
            0,
            Some(&bias),
            &ScalarGemm,
            &mut fast,
        );

        let mut col = vec![0.0; in_c * ih * iw];
        im2col(&input, in_c, ih, iw, 1, 1, 1, 0, ih, iw, &mut col);
        let mut generic = vec![0.0; out_c * ih * iw];
        ScalarGemm.gemm(out_c, ih * iw, in_c, &weight, &col, &mut generic);
        for oc in 0..out_c {
            for value in &mut generic[oc * ih * iw..(oc + 1) * ih * iw] {
                *value += bias[oc];
            }
        }
        assert_eq!(fast, generic);
    }

    #[test]
    fn conv2d_matches_naive_3x3_stride2_pad1() {
        let in_c = 2;
        let (ih, iw) = (7, 5);
        let out_c = 3;
        let mut rng = Xorshift32(0xC0FF_EE01);
        let input = random_vec(&mut rng, in_c * ih * iw);
        let weight = random_vec(&mut rng, out_c * in_c * 3 * 3);
        let bias = random_vec(&mut rng, out_c);

        let stride = 2;
        let pad = 1;
        let oh = (ih + 2 * pad - 3) / stride + 1;
        let ow = (iw + 2 * pad - 3) / stride + 1;

        let mut out_gemm = vec![0f32; out_c * oh * ow];
        conv2d(
            &input,
            in_c,
            ih,
            iw,
            &weight,
            out_c,
            3,
            3,
            stride,
            pad,
            Some(&bias),
            &ScalarGemm,
            &mut out_gemm,
        );
        let mut out_naive = vec![0f32; out_c * oh * ow];
        conv2d_naive(
            &input,
            in_c,
            ih,
            iw,
            &weight,
            out_c,
            3,
            3,
            stride,
            pad,
            Some(&bias),
            &mut out_naive,
        );
        for i in 0..out_gemm.len() {
            assert!(
                (out_gemm[i] - out_naive[i]).abs() < 1e-3,
                "i={i} gemm={} naive={}",
                out_gemm[i],
                out_naive[i]
            );
        }
    }

    #[test]
    fn conv_transpose2d_hand_checked_1x1_spatial_k4s4() {
        // 1x1 spatial input, 2 in-channels, 2 out-channels, k=4 s=4: this is
        // the DPT resize-layer shape family. With a single input pixel the
        // whole output IS the kernel (scaled by input and summed over ic),
        // making it hand-checkable.
        let in_c = 2;
        let (ih, iw) = (1, 1);
        let out_c = 2;
        let kh = 4;
        let kw = 4;
        let stride = 4;
        let input = vec![2.0, 3.0]; // ic0=2, ic1=3
        // weight layout IOHW: [ic][oc][kh][kw]
        let mut weight = vec![0f32; in_c * out_c * kh * kw];
        // ic0->oc0: all ones (16 elems)
        for value in weight.iter_mut().take(16) {
            *value = 1.0;
        }
        // ic0->oc1: all zeros (default)
        // ic1->oc0: all zeros
        // ic1->oc1: constant 5.0
        for i in 0..16 {
            weight[(out_c + 1) * 16 + i] = 5.0;
        }
        let bias = vec![1.0, -2.0];
        let oh = (ih - 1) * stride + kh;
        let ow = (iw - 1) * stride + kw;
        let mut out = vec![0f32; out_c * oh * ow];
        conv_transpose2d(
            &input,
            in_c,
            ih,
            iw,
            &weight,
            out_c,
            kh,
            kw,
            stride,
            Some(&bias),
            &mut out,
        );
        // oc0 = ic0*1.0 + ic1*0.0 + bias0 = 2*1 + 1 = 3 everywhere (4x4=16 elems)
        // oc1 = ic0*0.0 + ic1*5.0 + bias1 = 3*5 - 2 = 13 everywhere
        assert_eq!(&out[0..16], &[3.0; 16][..]);
        assert_eq!(&out[16..32], &[13.0; 16][..]);
    }

    #[test]
    fn conv_transpose2d_matches_naive_oracle() {
        let in_c = 3;
        let (ih, iw) = (2, 3);
        let out_c = 2;
        let kh = 4;
        let kw = 4;
        let stride = 4;
        let mut rng = Xorshift32(0xFEED_1234);
        let input = random_vec(&mut rng, in_c * ih * iw);
        let weight = random_vec(&mut rng, in_c * out_c * kh * kw);
        let bias = random_vec(&mut rng, out_c);

        let oh = (ih - 1) * stride + kh;
        let ow = (iw - 1) * stride + kw;
        let mut out = vec![0f32; out_c * oh * ow];
        conv_transpose2d(
            &input,
            in_c,
            ih,
            iw,
            &weight,
            out_c,
            kh,
            kw,
            stride,
            Some(&bias),
            &mut out,
        );

        let mut out_naive = vec![0f32; out_c * oh * ow];
        conv_transpose2d_naive(
            &input,
            in_c,
            ih,
            iw,
            &weight,
            out_c,
            kh,
            kw,
            stride,
            Some(&bias),
            &mut out_naive,
        );
        for i in 0..out.len() {
            assert!(
                (out[i] - out_naive[i]).abs() < 1e-4,
                "i={i} fast={} naive={}",
                out[i],
                out_naive[i]
            );
        }
    }

    #[test]
    fn nonoverlap_transpose_fast_path_is_bitwise_serial_scatter() {
        let (in_c, out_c, ih, iw, kernel) = (3, 4, 3, 2, 2);
        let mut rng = Xorshift32(0x7A4E_0001);
        let input = random_vec(&mut rng, in_c * ih * iw);
        let weight = random_vec(&mut rng, in_c * out_c * kernel * kernel);
        let bias = random_vec(&mut rng, out_c);
        let (oh, ow) = (ih * kernel, iw * kernel);
        let mut fast = vec![0.0; out_c * oh * ow];
        conv_transpose2d(
            &input,
            in_c,
            ih,
            iw,
            &weight,
            out_c,
            kernel,
            kernel,
            kernel,
            Some(&bias),
            &mut fast,
        );

        let mut serial = vec![0.0; out_c * oh * ow];
        for ic in 0..in_c {
            for iy in 0..ih {
                for ix in 0..iw {
                    let iv = input[(ic * ih + iy) * iw + ix];
                    for oc in 0..out_c {
                        for ky in 0..kernel {
                            for kx in 0..kernel {
                                serial[(oc * oh + iy * kernel + ky) * ow + ix * kernel + kx] +=
                                    iv * weight[((ic * out_c + oc) * kernel + ky) * kernel + kx];
                            }
                        }
                    }
                }
            }
        }
        for oc in 0..out_c {
            for value in &mut serial[oc * oh * ow..(oc + 1) * oh * ow] {
                *value += bias[oc];
            }
        }
        assert_eq!(fast, serial);
    }

    #[test]
    fn parallel_im2col_is_bitwise_serial() {
        let (in_c, ih, iw, kh, kw, stride, pad) = (3, 7, 9, 3, 3, 2, 1);
        let (oh, ow) = (
            (ih + 2 * pad - kh) / stride + 1,
            (iw + 2 * pad - kw) / stride + 1,
        );
        let mut rng = Xorshift32(0x1A2C_0001); // deterministic test seed
        let input = random_vec(&mut rng, in_c * ih * iw);
        let mut parallel = vec![0.0; in_c * kh * kw * oh * ow];
        let mut serial = vec![0.0; parallel.len()];
        im2col(
            &input,
            in_c,
            ih,
            iw,
            kh,
            kw,
            stride,
            pad,
            oh,
            ow,
            &mut parallel,
        );
        im2col_serial(
            &input,
            in_c,
            ih,
            iw,
            kh,
            kw,
            stride,
            pad,
            oh,
            ow,
            &mut serial,
        );
        assert_eq!(parallel, serial);
    }

    #[test]
    fn winograd_f2_matches_direct_3x3_oracle() {
        let (in_c, out_c, h, w) = (3, 5, 7, 9);
        let mut rng = Xorshift32(0xF2F2_0001);
        let input = random_vec(&mut rng, in_c * h * w);
        let weight = random_vec(&mut rng, out_c * in_c * 9);
        let bias = random_vec(&mut rng, out_c);
        let mut winograd = vec![0.0; out_c * h * w];
        let mut direct = vec![0.0; winograd.len()];
        conv3x3_winograd_f2(
            &input,
            in_c,
            h,
            w,
            &weight,
            out_c,
            Some(&bias),
            &mut winograd,
        );
        conv2d_naive(
            &input,
            in_c,
            h,
            w,
            &weight,
            out_c,
            3,
            3,
            1,
            1,
            Some(&bias),
            &mut direct,
        );
        for (i, (got, expected)) in winograd.iter().zip(direct.iter()).enumerate() {
            assert!(
                (got - expected).abs() < 2e-5,
                "i={i} got={got} expected={expected}"
            );
        }
    }

    #[test]
    fn winograd_relu_input_matches_materialized_relu_bitwise() {
        let (in_c, out_c, h, w) = (3, 5, 7, 9);
        let input: Vec<f32> = (0..in_c * h * w)
            .map(|i| ((i as f32 * 0.17).sin() - 0.4) * 1.3)
            .collect();
        let weight: Vec<f32> = (0..out_c * in_c * 9)
            .map(|i| (i as f32 * 0.11).cos() * 0.2)
            .collect();
        let bias: Vec<f32> = (0..out_c).map(|i| i as f32 * 0.03 - 0.05).collect();
        let mut materialized = input.clone();
        for value in &mut materialized {
            *value = value.max(0.0);
        }
        let mut expected = vec![0.0; out_c * h * w];
        let mut actual = vec![0.0; out_c * h * w];
        conv3x3_winograd_f2(
            &materialized,
            in_c,
            h,
            w,
            &weight,
            out_c,
            Some(&bias),
            &mut expected,
        );
        conv3x3_winograd_f2_relu_input(
            &input,
            in_c,
            h,
            w,
            &weight,
            out_c,
            Some(&bias),
            &mut actual,
        );
        assert_eq!(
            actual.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            expected.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        );
    }

    #[test]
    fn fused_resize_align_corners_winograd_matches_materialized_route() {
        // The head candidate must reproduce its established two-step route:
        // resize CHW, add UV, then run the prepared Winograd convolution.
        // Use 16 channels so this also exercises the AVX-512 external kernel
        // when the host provides it.
        let (in_c, out_c, ih, iw, oh, ow) = (16, 16, 7, 5, 11, 9);
        let mut rng = Xorshift32(0xF05E_0001);
        let input = random_vec(&mut rng, in_c * ih * iw);
        let add = random_vec(&mut rng, in_c * oh * ow);
        let weight = random_vec(&mut rng, out_c * in_c * 9);
        let bias = random_vec(&mut rng, out_c);
        let filter = prepare_winograd_f2_filter(&weight, in_c, out_c);

        let mut materialized = vec![0.0; in_c * oh * ow];
        crate::resample::bilinear_resize_align_corners(
            &input,
            in_c,
            ih,
            iw,
            oh,
            ow,
            &mut materialized,
        );
        for (value, offset) in materialized.iter_mut().zip(&add) {
            *value += offset;
        }
        let mut expected = vec![0.0; out_c * oh * ow];
        conv3x3_winograd_f2_prepared(
            &materialized,
            in_c,
            oh,
            ow,
            &filter,
            out_c,
            Some(&bias),
            &mut expected,
        );

        let mut actual = vec![0.0; out_c * oh * ow];
        conv3x3_winograd_f2_prepared_resize_align_corners(
            &input,
            in_c,
            ih,
            iw,
            oh,
            ow,
            Some(&add),
            &filter,
            out_c,
            Some(&bias),
            &mut actual,
        );
        assert_eq!(
            actual
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            expected
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
        );
    }

    #[test]
    fn winograd_f4_matches_direct_3x3_oracle_within_f32_envelope() {
        // F(4) has larger transform coefficients than F(2), so it is not
        // expected to be bitwise-identical to direct convolution. This bound
        // is a unit-level safety net; the DA3 four-image parity gate remains
        // the admission criterion for the opt-in production candidate.
        let (in_c, out_c, h, w) = (3, 5, 9, 11);
        let mut rng = Xorshift32(0xF4F4_0001);
        let input = random_vec(&mut rng, in_c * h * w);
        let weight = random_vec(&mut rng, out_c * in_c * 9);
        let bias = random_vec(&mut rng, out_c);
        let filter = prepare_winograd_f4_filter(&weight, in_c, out_c);
        let mut actual = vec![0.0; out_c * h * w];
        let mut expected = vec![0.0; out_c * h * w];
        conv3x3_winograd_f4_prepared(&input, in_c, h, w, &filter, out_c, Some(&bias), &mut actual);
        conv2d_naive(
            &input,
            in_c,
            h,
            w,
            &weight,
            out_c,
            3,
            3,
            1,
            1,
            Some(&bias),
            &mut expected,
        );
        let max_error = actual
            .iter()
            .zip(&expected)
            .map(|(actual, expected)| (actual - expected).abs())
            .fold(0.0f32, f32::max);
        assert!(max_error <= 0.003, "F4 max error {max_error}");
    }

    /// Deterministischer, dependency-freier PRNG (Xorshift32) fuer reproduzierbare
    /// Testdaten.
    struct Xorshift32(u32);
    impl Xorshift32 {
        fn next_f32(&mut self) -> f32 {
            let mut x = self.0;
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            self.0 = x;
            ((x as f32) / (u32::MAX as f32)) * 2.0 - 1.0
        }
    }
    fn random_vec(rng: &mut Xorshift32, n: usize) -> Vec<f32> {
        (0..n).map(|_| rng.next_f32()).collect()
    }

    /// Zweites, unabhaengig formuliertes Orakel fuer conv_transpose2d (direkte
    /// "scatter"-Definition ohne die Optimierungen der Hauptimplementierung -
    /// hier ist die Hauptimplementierung selbst schon die naive Variante,
    /// daher spiegelt dieses Orakel die mathematische Definition explizit als
    /// "gather" (aus Sicht des Outputs) statt "scatter" (aus Sicht des Inputs).
    #[allow(clippy::too_many_arguments)]
    fn conv_transpose2d_naive(
        input: &[f32],
        in_c: usize,
        ih: usize,
        iw: usize,
        weight: &[f32],
        out_c: usize,
        kh: usize,
        kw: usize,
        stride: usize,
        bias: Option<&[f32]>,
        out: &mut [f32],
    ) {
        let oh = (ih - 1) * stride + kh;
        let ow = (iw - 1) * stride + kw;
        debug_assert_eq!(out.len(), out_c * oh * ow);
        for oc in 0..out_c {
            for oy in 0..oh {
                for ox in 0..ow {
                    let mut acc = 0f32;
                    for ic in 0..in_c {
                        for ky in 0..kh {
                            if oy < ky {
                                continue;
                            }
                            let num = oy - ky;
                            if num % stride != 0 {
                                continue;
                            }
                            let iy = num / stride;
                            if iy >= ih {
                                continue;
                            }
                            for kx in 0..kw {
                                if ox < kx {
                                    continue;
                                }
                                let numx = ox - kx;
                                if numx % stride != 0 {
                                    continue;
                                }
                                let ix = numx / stride;
                                if ix >= iw {
                                    continue;
                                }
                                let iv = input[(ic * ih + iy) * iw + ix];
                                let wv = weight[((ic * out_c + oc) * kh + ky) * kw + kx];
                                acc += iv * wv;
                            }
                        }
                    }
                    if let Some(b) = bias {
                        acc += b[oc];
                    }
                    out[(oc * oh + oy) * ow + ox] = acc;
                }
            }
        }
    }
}

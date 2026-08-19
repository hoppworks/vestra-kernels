use rayon::prelude::*;

/// Bilineares Resize (NCHW, Batch=1), half-pixel-centers Konvention
/// (entspricht PyTorch `F.interpolate(..., mode="bilinear", align_corners=False)`,
/// der in DPT-artigen Netzen ueblichen Variante). Randpixel werden geklemmt
/// (clamp-to-edge), kein Padding.
///
/// - `input`: `c*ih*iw`
/// - `out`: `c*oh*ow`
pub fn bilinear_resize(
    input: &[f32],
    c: usize,
    ih: usize,
    iw: usize,
    oh: usize,
    ow: usize,
    out: &mut [f32],
) {
    debug_assert_eq!(input.len(), c * ih * iw);
    debug_assert_eq!(out.len(), c * oh * ow);

    if ih == 0 || iw == 0 || oh == 0 || ow == 0 {
        return;
    }

    let scale_y = ih as f32 / oh as f32;
    let scale_x = iw as f32 / ow as f32;

    // Pro-Zeile/Spalte vorab die Quellkoordinate + Nachbarindizes + Gewicht
    // berechnen (getrennt fuer y und x), das spart die wiederholte
    // Neuberechnung pro Kanal.
    let (y0s, y1s, wy): (Vec<usize>, Vec<usize>, Vec<f32>) =
        (0..oh).map(|oy| src_coord(oy, scale_y, ih)).fold(
            (
                Vec::with_capacity(oh),
                Vec::with_capacity(oh),
                Vec::with_capacity(oh),
            ),
            |mut acc, (a, b, w)| {
                acc.0.push(a);
                acc.1.push(b);
                acc.2.push(w);
                acc
            },
        );
    let (x0s, x1s, wx): (Vec<usize>, Vec<usize>, Vec<f32>) =
        (0..ow).map(|ox| src_coord(ox, scale_x, iw)).fold(
            (
                Vec::with_capacity(ow),
                Vec::with_capacity(ow),
                Vec::with_capacity(ow),
            ),
            |mut acc, (a, b, w)| {
                acc.0.push(a);
                acc.1.push(b);
                acc.2.push(w);
                acc
            },
        );

    out.par_chunks_mut(oh * ow)
        .enumerate()
        .for_each(|(ch, out_plane)| {
            let plane = &input[ch * ih * iw..(ch + 1) * ih * iw];
            for oy in 0..oh {
                let (y0, y1, fy) = (y0s[oy], y1s[oy], wy[oy]);
                let row0 = &plane[y0 * iw..(y0 + 1) * iw];
                let row1 = &plane[y1 * iw..(y1 + 1) * iw];
                for ox in 0..ow {
                    let (x0, x1, fx) = (x0s[ox], x1s[ox], wx[ox]);
                    let top = row0[x0] * (1.0 - fx) + row0[x1] * fx;
                    let bot = row1[x0] * (1.0 - fx) + row1[x1] * fx;
                    out_plane[oy * ow + ox] = top * (1.0 - fy) + bot * fy;
                }
            }
        });
}

/// Fuer eine Zielkoordinate `dst` entlang einer Achse mit `len_in` Quellelementen
/// und `scale = len_in/len_out`: liefert `(idx0, idx1, frac)` mit `idx0<=idx1`
/// geklemmt in `[0, len_in-1]` und `frac` das Interpolationsgewicht Richtung `idx1`.
fn src_coord(dst: usize, scale: f32, len_in: usize) -> (usize, usize, f32) {
    let src = (dst as f32 + 0.5) * scale - 0.5;
    let src_clamped = src.max(0.0);
    let idx0 = src_clamped.floor() as usize;
    let idx0 = idx0.min(len_in.saturating_sub(1));
    let idx1 = (idx0 + 1).min(len_in.saturating_sub(1));
    let frac = if idx1 == idx0 {
        0.0
    } else {
        src_clamped - idx0 as f32
    };
    let frac = frac.clamp(0.0, 1.0);
    (idx0, idx1, frac)
}

/// Unabhaengig formuliertes, ungefaltetes Referenz-Orakel: berechnet fuer
/// jedes Ausgabepixel direkt (ohne Vorab-Tabellen) die Quellkoordinate neu.
/// Dient als zweite, redundante Implementierung zur Verifikation von
/// [`bilinear_resize`].
pub fn bilinear_resize_naive(
    input: &[f32],
    c: usize,
    ih: usize,
    iw: usize,
    oh: usize,
    ow: usize,
    out: &mut [f32],
) {
    debug_assert_eq!(input.len(), c * ih * iw);
    debug_assert_eq!(out.len(), c * oh * ow);
    if ih == 0 || iw == 0 || oh == 0 || ow == 0 {
        return;
    }
    let scale_y = ih as f32 / oh as f32;
    let scale_x = iw as f32 / ow as f32;
    for ch in 0..c {
        for oy in 0..oh {
            let sy = ((oy as f32 + 0.5) * scale_y - 0.5).max(0.0);
            let y0 = (sy.floor() as usize).min(ih - 1);
            let y1 = (y0 + 1).min(ih - 1);
            let fy = if y1 == y0 {
                0.0
            } else {
                (sy - y0 as f32).clamp(0.0, 1.0)
            };
            for ox in 0..ow {
                let sx = ((ox as f32 + 0.5) * scale_x - 0.5).max(0.0);
                let x0 = (sx.floor() as usize).min(iw - 1);
                let x1 = (x0 + 1).min(iw - 1);
                let fx = if x1 == x0 {
                    0.0
                } else {
                    (sx - x0 as f32).clamp(0.0, 1.0)
                };

                let get = |y: usize, x: usize| input[(ch * ih + y) * iw + x];
                let top = get(y0, x0) * (1.0 - fx) + get(y0, x1) * fx;
                let bot = get(y1, x0) * (1.0 - fx) + get(y1, x1) * fx;
                out[(ch * oh + oy) * ow + ox] = top * (1.0 - fy) + bot * fy;
            }
        }
    }
}

/// For a target coordinate `dst` along an axis with `len_in` source elements
/// and `len_out` destination elements, under `align_corners=true` semantics:
/// `src = dst * (len_in-1)/(len_out-1)`, degenerate case `len_out==1 ->
/// src=0`. Returns `(idx0, idx1, frac)` with `idx0<=idx1` clamped into
/// `[0, len_in-1]` and `frac` the interpolation weight toward `idx1`.
fn src_coord_align_corners(dst: usize, len_in: usize, len_out: usize) -> (usize, usize, f32) {
    if len_in <= 1 {
        return (0, 0, 0.0);
    }
    let src = if len_out <= 1 {
        0.0
    } else {
        dst as f32 * (len_in - 1) as f32 / (len_out - 1) as f32
    };
    let src = src.clamp(0.0, (len_in - 1) as f32);
    let idx0 = src.floor() as usize;
    let idx0 = idx0.min(len_in - 1);
    let idx1 = (idx0 + 1).min(len_in - 1);
    let frac = if idx1 == idx0 { 0.0 } else { src - idx0 as f32 };
    (idx0, idx1, frac.clamp(0.0, 1.0))
}

/// Bilineares Resize (NCHW, Batch=1), **align_corners=true** Konvention
/// (entspricht PyTorch `F.interpolate(..., mode="bilinear",
/// align_corners=True)` / ggml's `GGML_SCALE_FLAG_ALIGN_CORNERS`) —
/// die vom DPT-Head's `feature_fusion`/finalem Upsample verwendete Variante
/// (`interp_bilinear_ac` im C++-Referenzcode, `../src/dpt_blocks.cpp`).
///
/// **This is deliberately a separate function from [`bilinear_resize`]**,
/// not a shared-with-a-flag variant: the coordinate mapping is a genuinely
/// different formula (`dst*(len_in-1)/(len_out-1)` vs. the half-pixel-center
/// `(dst+0.5)*scale-0.5`), and conflating the two silently produces a
/// slightly-different-looking but numerically WRONG upsample if the wrong
/// branch is picked — keeping them as distinct functions makes the call site
/// unambiguous about which convention it needs.
///
/// - `input`: `c*ih*iw`
/// - `out`: `c*oh*ow`
pub fn bilinear_resize_align_corners(
    input: &[f32],
    c: usize,
    ih: usize,
    iw: usize,
    oh: usize,
    ow: usize,
    out: &mut [f32],
) {
    debug_assert_eq!(input.len(), c * ih * iw);
    debug_assert_eq!(out.len(), c * oh * ow);

    if ih == 0 || iw == 0 || oh == 0 || ow == 0 {
        return;
    }

    let (y0s, y1s, wy): (Vec<usize>, Vec<usize>, Vec<f32>) =
        (0..oh).map(|oy| src_coord_align_corners(oy, ih, oh)).fold(
            (
                Vec::with_capacity(oh),
                Vec::with_capacity(oh),
                Vec::with_capacity(oh),
            ),
            |mut acc, (a, b, w)| {
                acc.0.push(a);
                acc.1.push(b);
                acc.2.push(w);
                acc
            },
        );
    let (x0s, x1s, wx): (Vec<usize>, Vec<usize>, Vec<f32>) =
        (0..ow).map(|ox| src_coord_align_corners(ox, iw, ow)).fold(
            (
                Vec::with_capacity(ow),
                Vec::with_capacity(ow),
                Vec::with_capacity(ow),
            ),
            |mut acc, (a, b, w)| {
                acc.0.push(a);
                acc.1.push(b);
                acc.2.push(w);
                acc
            },
        );

    out.par_chunks_mut(oh * ow)
        .enumerate()
        .for_each(|(ch, out_plane)| {
            let plane = &input[ch * ih * iw..(ch + 1) * ih * iw];
            for oy in 0..oh {
                let (y0, y1, fy) = (y0s[oy], y1s[oy], wy[oy]);
                let row0 = &plane[y0 * iw..(y0 + 1) * iw];
                let row1 = &plane[y1 * iw..(y1 + 1) * iw];
                for ox in 0..ow {
                    let (x0, x1, fx) = (x0s[ox], x1s[ox], wx[ox]);
                    let top = row0[x0] * (1.0 - fx) + row0[x1] * fx;
                    let bot = row1[x0] * (1.0 - fx) + row1[x1] * fx;
                    out_plane[oy * ow + ox] = top * (1.0 - fy) + bot * fy;
                }
            }
        });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bilinear_resize_identity_hand_checked() {
        // oh==ih, ow==iw => scale=1 everywhere => output must equal input
        // exactly, a fully hand-verifiable case independent of the
        // half-pixel-centers convention details.
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let mut out = vec![0f32; 9];
        bilinear_resize(&input, 1, 3, 3, 3, 3, &mut out);
        assert_eq!(out, input);
    }

    #[test]
    fn bilinear_resize_upsample_2x_hand_checked() {
        // 1x2 -> 1x4 upsample (single row, exercises only x-axis).
        // half-pixel centers: src_x(dst) = (dst+0.5)*0.5 - 0.5
        // dst=0: -0.25 -> clamp 0.0            -> idx0=0,idx1=1,frac=0.0 -> in[0]
        // dst=1: 0.25                          -> idx0=0,idx1=1,frac=0.25
        // dst=2: 0.75                          -> idx0=0,idx1=1,frac=0.75
        // dst=3: 1.25 -> clamp to idx1 (iw-1=1) -> idx0=1,idx1=1,frac=0.0 -> in[1]
        let input = vec![10.0, 20.0];
        let mut out = vec![0f32; 4];
        bilinear_resize(&input, 1, 1, 2, 1, 4, &mut out);
        let expected = [
            10.0,
            10.0 * 0.75 + 20.0 * 0.25,
            10.0 * 0.25 + 20.0 * 0.75,
            20.0,
        ];
        for i in 0..4 {
            assert!(
                (out[i] - expected[i]).abs() < 1e-6,
                "i={i} got={} want={}",
                out[i],
                expected[i]
            );
        }
    }

    #[test]
    fn bilinear_resize_matches_naive_oracle_multichannel() {
        let c = 3;
        let (ih, iw) = (5, 7);
        let (oh, ow) = (9, 4);
        let mut rng = Xorshift32(0xABCD_1234);
        let input = random_vec(&mut rng, c * ih * iw);

        let mut out = vec![0f32; c * oh * ow];
        let mut out_naive = vec![0f32; c * oh * ow];
        bilinear_resize(&input, c, ih, iw, oh, ow, &mut out);
        bilinear_resize_naive(&input, c, ih, iw, oh, ow, &mut out_naive);

        for i in 0..out.len() {
            assert!(
                (out[i] - out_naive[i]).abs() < 1e-5,
                "i={i} fast={} naive={}",
                out[i],
                out_naive[i]
            );
        }
    }

    #[test]
    fn bilinear_resize_downsample_matches_naive_oracle() {
        let c = 2;
        let (ih, iw) = (16, 16);
        let (oh, ow) = (5, 5);
        let mut rng = Xorshift32(0x1357_9BDF);
        let input = random_vec(&mut rng, c * ih * iw);

        let mut out = vec![0f32; c * oh * ow];
        let mut out_naive = vec![0f32; c * oh * ow];
        bilinear_resize(&input, c, ih, iw, oh, ow, &mut out);
        bilinear_resize_naive(&input, c, ih, iw, oh, ow, &mut out_naive);

        for i in 0..out.len() {
            assert!(
                (out[i] - out_naive[i]).abs() < 1e-5,
                "i={i} fast={} naive={}",
                out[i],
                out_naive[i]
            );
        }
    }

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

    #[test]
    fn align_corners_identity_hand_checked() {
        // oh==ih, ow==iw: align_corners src = dst*(n-1)/(n-1) = dst exactly,
        // so output must equal input exactly (same as half-pixel-centers
        // convention on the identity case, but via a different formula).
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let mut out = vec![0f32; 9];
        bilinear_resize_align_corners(&input, 1, 3, 3, 3, 3, &mut out);
        assert_eq!(out, input);
    }

    #[test]
    fn align_corners_endpoints_are_exact() {
        // Hallmark of align_corners=true: the first and last output samples
        // land exactly on the first and last input samples (unlike
        // half-pixel-centers, which does not guarantee this for up/downsampling).
        // 1x2 -> 1x5 upsample.
        let input = vec![10.0, 20.0];
        let mut out = vec![0f32; 5];
        bilinear_resize_align_corners(&input, 1, 1, 2, 1, 5, &mut out);
        assert_eq!(
            out[0], 10.0,
            "first output sample must equal first input sample exactly"
        );
        assert_eq!(
            out[4], 20.0,
            "last output sample must equal last input sample exactly"
        );
        // Hand-computed intermediate points: src(dst) = dst*(2-1)/(5-1) = dst/4.
        // dst=1: src=0.25 -> 10*0.75+20*0.25=12.5
        // dst=2: src=0.5  -> 10*0.5 +20*0.5 =15.0
        // dst=3: src=0.75 -> 10*0.25+20*0.75=17.5
        let expected = [10.0, 12.5, 15.0, 17.5, 20.0];
        for i in 0..5 {
            assert!(
                (out[i] - expected[i]).abs() < 1e-6,
                "i={i} got={} want={}",
                out[i],
                expected[i]
            );
        }
    }

    #[test]
    fn align_corners_differs_from_half_pixel_centers() {
        // Sanity check that the two conventions are NOT accidentally
        // identical (the bug this function's doc comment warns about): for
        // a non-trivial up/downsample, at least one output sample must
        // differ between the two coordinate mappings.
        let c = 2;
        let (ih, iw) = (4, 4);
        let (oh, ow) = (7, 5);
        let mut rng = Xorshift32(0x2222_AAAA);
        let input = random_vec(&mut rng, c * ih * iw);

        let mut out_ac = vec![0f32; c * oh * ow];
        let mut out_hp = vec![0f32; c * oh * ow];
        bilinear_resize_align_corners(&input, c, ih, iw, oh, ow, &mut out_ac);
        bilinear_resize(&input, c, ih, iw, oh, ow, &mut out_hp);

        let any_diff = out_ac
            .iter()
            .zip(out_hp.iter())
            .any(|(a, b)| (a - b).abs() > 1e-4);
        assert!(
            any_diff,
            "align_corners and half-pixel-centers must diverge on a non-trivial resize"
        );
    }

    #[test]
    fn align_corners_degenerate_output_size_one_uses_source_origin() {
        // len_out==1 is a degenerate case: src_coord = 0 (not a division by
        // zero), matching the formula's documented special case.
        let input = vec![5.0, 7.0, 9.0];
        let mut out = vec![0f32; 1];
        bilinear_resize_align_corners(&input, 1, 1, 3, 1, 1, &mut out);
        assert_eq!(out[0], 5.0);
    }

    #[test]
    fn align_corners_multichannel_matches_naive_reimplementation() {
        // Independent brute-force oracle recomputing align_corners
        // coordinates per-pixel (no precomputed tables), for cross-checking
        // the main (table-based) implementation above.
        let c = 3;
        let (ih, iw) = (5, 7);
        let (oh, ow) = (9, 4);
        let mut rng = Xorshift32(0x9999_5555);
        let input = random_vec(&mut rng, c * ih * iw);

        let mut out = vec![0f32; c * oh * ow];
        bilinear_resize_align_corners(&input, c, ih, iw, oh, ow, &mut out);

        let sy = |dst: usize| -> f32 {
            if oh <= 1 {
                0.0
            } else {
                dst as f32 * (ih - 1) as f32 / (oh - 1) as f32
            }
        };
        let sx = |dst: usize| -> f32 {
            if ow <= 1 {
                0.0
            } else {
                dst as f32 * (iw - 1) as f32 / (ow - 1) as f32
            }
        };
        for ch in 0..c {
            for oy in 0..oh {
                let fy_raw = sy(oy).clamp(0.0, (ih - 1) as f32);
                let y0 = (fy_raw.floor() as usize).min(ih - 1);
                let y1 = (y0 + 1).min(ih - 1);
                let fy = if y1 == y0 { 0.0 } else { fy_raw - y0 as f32 };
                for ox in 0..ow {
                    let fx_raw = sx(ox).clamp(0.0, (iw - 1) as f32);
                    let x0 = (fx_raw.floor() as usize).min(iw - 1);
                    let x1 = (x0 + 1).min(iw - 1);
                    let fx = if x1 == x0 { 0.0 } else { fx_raw - x0 as f32 };
                    let get = |y: usize, x: usize| input[(ch * ih + y) * iw + x];
                    let top = get(y0, x0) * (1.0 - fx) + get(y0, x1) * fx;
                    let bot = get(y1, x0) * (1.0 - fx) + get(y1, x1) * fx;
                    let want = top * (1.0 - fy) + bot * fy;
                    let got = out[(ch * oh + oy) * ow + ox];
                    assert!(
                        (got - want).abs() < 1e-5,
                        "ch={ch} oy={oy} ox={ox} got={got} want={want}"
                    );
                }
            }
        }
    }
}

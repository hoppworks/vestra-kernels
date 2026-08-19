# Third-party notices

Vestra Kernels is licensed under Apache-2.0. This file preserves the licenses
and provenance of upstream work used to implement or qualify individual
kernels. It does not change the license of original Vestra Kernels code.

## ggml

The Q8_0 arithmetic and AVX-512/VNNI layout used by `src/q8_0_dot.rs`,
`src/scalar.rs`, and `src/simd_avx512.rs` were adapted to Rust from ggml's
`quantize_row_q8_0` and `ggml_vec_dot_q8_0_q8_0` implementations.

- Project: <https://github.com/ggml-org/ggml>
- Source revision: `eced84c86f8b012c752c016f7fe789adea168e1e`
- License: MIT

The ggml license notice follows:

```text
MIT License

Copyright (c) 2023-2026 The ggml authors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## depth-anything.cpp

The fixed-shape tensor layouts, CPU reference behavior, and multi-view
qualification oracles were checked against `depth-anything.cpp`. Its source is
not vendored or linked into this crate, but its reference role is retained here
so derived kernel behavior remains attributable and reproducible.

- Project: <https://github.com/localai-org/depth-anything.cpp>
- Single-view reference revision: `2028b47ac75a8659c6a9aa617baf09be193eb55f`
- Multi-view PR #2 reference revision: `f56e9be43a22c12ef575584d2fa57a6a5d5be7ae`
- License: MIT

The depth-anything.cpp license notice follows:

```text
MIT License

Copyright (c) 2026 the depth-anything.cpp authors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Depth Anything 3

The DA3-BASE model architecture and its fixed 504x336 workload define the
shapes qualified by this crate.

- Project: <https://github.com/ByteDance-Seed/Depth-Anything-3>
- Reference revision: `3d835ec1a5802d64a8b8b15f817a1ab54809bfe4`
- Copyright: 2025 The Depth Anything 3 Team
- Code and DA3-BASE checkpoint license: Apache-2.0

The Apache-2.0 license text is included in this repository's `LICENSE` file.
No model weights are included in Vestra Kernels. Checkpoint licenses are
model-specific: this notice does not relicense DA3 Large, Giant, or Nested
checkpoints, some of which are restricted to non-commercial use.

## Cargo and external runtime dependencies

Rust dependencies are resolved by the checked-in `Cargo.lock`; their sources
are not vendored. Cargo package metadata remains the authority for each crate's
license. The direct dependency set is `cudarc` (MIT OR Apache-2.0), `faer`
(MIT), `half` (MIT OR Apache-2.0), `rayon` (MIT OR Apache-2.0), and `thiserror`
(MIT OR Apache-2.0); `criterion` (Apache-2.0 OR MIT) is development-only.

BLIS, the CUDA toolkit, cuBLAS, and NVIDIA drivers are optional external
runtime or development components and are not distributed by this repository.
Binary distributors remain responsible for carrying the notices required by
the exact dependency and external-library versions they ship.

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use vestra_kernels::gemm::{FaerGemm, Gemm, ScalarGemm};

fn vit_block_gemm_shapes() -> Vec<(usize, usize, usize)> {
    // Repräsentative DA3-BASE-ViT-Block-GEMMs bei 256 Tokens, embed 768, mlp 3072:
    // QKV-Projektion, Attn-Output-Projektion, MLP-fc1, MLP-fc2.
    vec![
        (256, 2304, 768),
        (256, 768, 768),
        (256, 3072, 768),
        (256, 768, 3072),
    ]
}

fn bench(c: &mut Criterion) {
    let mut g = c.benchmark_group("vit_block_gemm");
    for (m, n, k) in vit_block_gemm_shapes() {
        let a = vec![0.01f32; m * k];
        let b = vec![0.01f32; k * n];
        let mut out = vec![0f32; m * n];
        g.bench_with_input(
            BenchmarkId::new("faer", format!("{m}x{n}x{k}")),
            &(),
            |bch, _| {
                bch.iter(|| FaerGemm.gemm(m, n, k, &a, &b, &mut out));
            },
        );
        g.bench_with_input(
            BenchmarkId::new("scalar", format!("{m}x{n}x{k}")),
            &(),
            |bch, _| {
                bch.iter(|| ScalarGemm.gemm(m, n, k, &a, &b, &mut out));
            },
        );
    }
    g.finish();
}
criterion_group!(benches, bench);
criterion_main!(benches);

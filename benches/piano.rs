use criterion::{criterion_group, criterion_main, Criterion};
use fourier::engine::config::{DistributedSetupConfig, SetupConfig};
use fourier::engine::piano::PianoBackend;
use fourier::engine::piano::PianoPrecomputation;
use kzg::{Fr};
use rust_kzg_blst::types::fr::FsFr;

/// Generate random coefficients for the test polynomial in the lagrange basis
/// f(X, Y) = sum_{i=0}^{M-1} sum_{j=0}^{T-1} f_{i,j} R_i(Y) L_j(X)
fn generate_coeffs(n: usize, m: usize) -> Vec<Vec<FsFr>> {
    let t = n - m;
    let machine_count = 2usize.pow(m as u32);
    let sub_circuit_size = 2usize.pow(t as u32);
    let mut coeffs = vec![vec![FsFr::zero(); sub_circuit_size]; machine_count];
    (0..machine_count).for_each(|i| {
        (0..sub_circuit_size).for_each(|j| {
            coeffs[i][j] = FsFr::rand();
        });
    });
    coeffs
}

fn prepare() -> (PianoBackend, Vec<Vec<FsFr>>) {
    const N: usize = 8;
    const M: usize = 2;
    let cfg = DistributedSetupConfig {
        machines_scale: M,
        setup: SetupConfig {
            scale: N,
            generate_setup: true,
            generate_precompute: true,
            ..Default::default()
        },
    };
    let backend = PianoBackend::setup(&cfg).unwrap();

    let lagrange_coeffs = generate_coeffs(N, M);
    (backend, lagrange_coeffs)
}

fn bench_commit_to_poly(c: &mut Criterion) {
    let (backend, polys) = prepare();
    let mut backend_without_precompute = backend.clone();
    backend_without_precompute
        .piano_settings
        .set_precomputation(PianoPrecomputation::default());

    c.bench_function("precompute commit_to_poly", |b| {
        b.iter(|| {
            polys.iter().enumerate().for_each(|(i, poly)| {
                let _ = backend.worker_commit(i, poly);
            });
        })
    });

    c.bench_function("no precompute commit_to_poly", |b| {
        b.iter(|| {
            polys.iter().enumerate().for_each(|(i, poly)| {
                let _ = backend_without_precompute.worker_commit(i, poly);
            });
        })
    });
}

criterion_group!(benches, bench_commit_to_poly);

criterion_main!(benches);

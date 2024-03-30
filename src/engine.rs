use kzg::{FFTSettings, Fr, G1Affine, G1Fp, G1GetFp, G1Mul, KZGSettings, Poly, G1, G2};
use std::sync::Arc;

type GenerateTrustedSetup<TG1, TG2> = dyn Fn(usize, [u8; 32usize]) -> (Vec<TG1>, Vec<TG2>) + Send;

pub struct Engine<TFr, TG1, TG2, TPoly, TFFTSettings, TKZGSettings, TG1Fp, TG1Affine> {
    inner: Arc<EngineInner<TFr, TG1, TG2, TPoly, TFFTSettings, TKZGSettings, TG1Fp, TG1Affine>>,
}

pub fn arkworks_engine(
    secret: [u8; 32usize],
    scale: usize,
) -> Engine<
    rust_kzg_arkworks::kzg_types::ArkFr,
    rust_kzg_arkworks::kzg_types::ArkG1,
    rust_kzg_arkworks::kzg_types::ArkG2,
    rust_kzg_arkworks::utils::PolyData,
    rust_kzg_arkworks::kzg_proofs::FFTSettings,
    rust_kzg_arkworks::kzg_proofs::KZGSettings,
    rust_kzg_arkworks::kzg_types::ArkFp,
    rust_kzg_arkworks::kzg_types::ArkG1Affine,
> {
    Engine::new(
        scale,
        secret,
        Arc::new(rust_kzg_arkworks::kzg_proofs::generate_trusted_setup),
    )
}

#[allow(dead_code)]
impl<TFr, TG1, TG2, TPoly, TFFTSettings, TKZGSettings, TG1Fp, TG1Affine>
    Engine<TFr, TG1, TG2, TPoly, TFFTSettings, TKZGSettings, TG1Fp, TG1Affine>
where
    TFr: Fr + std::marker::Send,
    TG1: G1 + G1Mul<TFr> + G1GetFp<TG1Fp>,
    TG2: G2 + std::marker::Send + std::marker::Sync,
    TPoly: Poly<TFr> + std::marker::Send + std::marker::Sync,
    TFFTSettings: FFTSettings<TFr>,
    TKZGSettings: KZGSettings<TFr, TG1, TG2, TFFTSettings, TPoly, TG1Fp, TG1Affine>,
    TG1Fp: G1Fp,
    TG1Affine: G1Affine<TG1, TG1Fp>,
{
    pub fn new(
        scale: usize,
        secret: [u8; 32usize],
        generate_trusted_setup: Arc<GenerateTrustedSetup<TG1, TG2>>,
    ) -> Self {
        Self {
            inner: Arc::new(EngineInner::new(scale, secret, generate_trusted_setup)),
        }
    }

    pub fn commit_to_poly(&self, poly: Arc<TPoly>) -> Result<TG1, String> {
        self.inner.commit_to_poly(poly)
    }

    pub fn compute_proof_single(&self, poly: Arc<TPoly>, point: Arc<TFr>) -> Result<TG1, String> {
        self.inner.compute_proof_single(poly, point)
    }

    pub fn verify_proof_single(
        &self,
        proof: Arc<TG1>,
        x: Arc<TFr>,
        y: Arc<TFr>,
        commitment: Arc<TG1>,
    ) -> Result<bool, String> {
        self.inner.verify_proof_single(proof, x, y, commitment)
    }

    pub fn random_poly(&self) -> TPoly {
        self.inner.random_poly()
    }
}

#[allow(dead_code)]
pub struct EngineInner<TFr, TG1, TG2, TPoly, TFFTSettings, TKZGSettings, TG1Fp, TG1Affine> {
    pub generate_trusted_setup: Arc<GenerateTrustedSetup<TG1, TG2>>,
    pub fft_settings: TFFTSettings,
    pub kzg_settings: TKZGSettings,
    phantom: std::marker::PhantomData<(TFr, TG1, TG2, TPoly, TG1Fp, TG1Affine)>,
}

#[allow(dead_code)]
impl<TFr, TG1, TG2, TPoly, TFFTSettings, TKZGSettings, TG1Fp, TG1Affine>
    EngineInner<TFr, TG1, TG2, TPoly, TFFTSettings, TKZGSettings, TG1Fp, TG1Affine>
where
    TFr: Fr,
    TG1: G1 + G1Mul<TFr> + G1GetFp<TG1Fp>,
    TG2: G2,
    TPoly: Poly<TFr>,
    TFFTSettings: FFTSettings<TFr>,
    TKZGSettings: KZGSettings<TFr, TG1, TG2, TFFTSettings, TPoly, TG1Fp, TG1Affine>,
    TG1Fp: G1Fp,
    TG1Affine: G1Affine<TG1, TG1Fp>,
{
    pub fn new(
        scale: usize,
        secret: [u8; 32usize],
        generate_trusted_setup: Arc<GenerateTrustedSetup<TG1, TG2>>,
    ) -> Self {
        let fft_settings = TFFTSettings::new(scale).expect("Failed to create FFTSettings");
        let (s1, s2) = generate_trusted_setup(fft_settings.get_max_width(), secret);
        let kzg_settings = TKZGSettings::new(&s1, &s2, fft_settings.get_max_width(), &fft_settings)
            .expect("Failed to create KZGSettings");
        Self {
            generate_trusted_setup,
            fft_settings,
            kzg_settings,
            phantom: std::marker::PhantomData,
        }
    }

    pub fn commit_to_poly(&self, poly: Arc<TPoly>) -> Result<TG1, String> {
        self.kzg_settings.commit_to_poly(&poly)
    }

    pub fn compute_proof_single(&self, poly: Arc<TPoly>, point: Arc<TFr>) -> Result<TG1, String> {
        self.kzg_settings.compute_proof_single(&poly, &point)
    }

    pub fn verify_proof_single(
        &self,
        proof: Arc<TG1>,
        x: Arc<TFr>,
        y: Arc<TFr>,
        commitment: Arc<TG1>,
    ) -> Result<bool, String> {
        self.kzg_settings
            .check_proof_single(&commitment, &proof, &x, &y)
    }

    pub fn random_poly(&self) -> TPoly {
        let mut poly = TPoly::new(self.fft_settings.get_max_width());
        for i in 0..self.fft_settings.get_max_width() {
            poly.set_coeff_at(i, &TFr::rand());
        }
        poly
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rust_kzg_arkworks::kzg_types::ArkFr;
    use tracing::info;

    pub const SECRET: [u8; 32usize] = [
        0xa4, 0x73, 0x31, 0x95, 0x28, 0xc8, 0xb6, 0xea, 0x4d, 0x08, 0xcc, 0x53, 0x18, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00,
    ];

    #[tracing_test::traced_test]
    #[test]
    fn test_engine() {
        let engine = arkworks_engine(SECRET, 4);

        let poly = engine.random_poly();
        let poly = Arc::new(poly);
        let x = Arc::new(ArkFr::rand());
        let y = Arc::new(poly.eval(&x));
        let commitment = engine.commit_to_poly(poly.clone()).unwrap();
        let proof = engine
            .compute_proof_single(poly.clone(), x.clone())
            .unwrap();
        let result = engine
            .verify_proof_single(proof.into(), x.clone(), y.clone(), commitment.into())
            .unwrap();

        info!("poly: {:?}", poly);
        info!("point: {:?}", x);
        info!("commitment: {:?}", commitment);
        info!("proof: {:?}", proof);
        info!("result: {:?}", result);
    }
}

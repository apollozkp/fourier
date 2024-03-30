use kzg::{FFTSettings, Fr, G1Affine, G1Fp, G1GetFp, G1Mul, KZGSettings, Poly, G1, G2};
use std::sync::Arc;

pub trait Backend {
    type Fr: Fr;
    type G1: G1 + G1Mul<Self::Fr> + G1GetFp<Self::G1Fp>;
    type G2: G2;
    type Poly: Poly<Self::Fr>;
    type FFTSettings: FFTSettings<Self::Fr>;
    type KZGSettings: KZGSettings<
        Self::Fr,
        Self::G1,
        Self::G2,
        Self::FFTSettings,
        Self::Poly,
        Self::G1Fp,
        Self::G1Affine,
    >;
    type G1Fp: G1Fp;
    type G1Affine: G1Affine<Self::G1, Self::G1Fp>;

    fn generate_trusted_setup(
        max_width: usize,
        secret: [u8; 32usize],
    ) -> (Vec<Self::G1>, Vec<Self::G2>);

    fn new(cfg: BackendConfig) -> Self;

    fn commit_to_poly(&self, poly: Arc<Self::Poly>) -> Result<Self::G1, String>;

    fn compute_proof_single(
        &self,
        poly: Arc<Self::Poly>,
        point: Arc<Self::Fr>,
    ) -> Result<Self::G1, String>;

    fn verify_proof_single(
        &self,
        proof: Arc<Self::G1>,
        x: Arc<Self::Fr>,
        y: Arc<Self::Fr>,
        commitment: Arc<Self::G1>,
    ) -> Result<bool, String>;
}

#[derive(Debug, Clone)]
pub struct BackendConfig {
    pub scale: usize,
    pub secret: [u8; 32usize],
}

impl BackendConfig {
    pub fn new(scale: usize, secret: [u8; 32usize]) -> Self {
        Self { scale, secret }
    }
}

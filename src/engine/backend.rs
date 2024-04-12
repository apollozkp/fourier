use crate::engine::config::{BackendConfig, SetupConfig};
use kzg::{FFTSettings, Fr, G1Affine, G1Fp, G1GetFp, G1Mul, KZGSettings, Poly, G1, G2};

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

    fn new(cfg: Option<BackendConfig>) -> Self;

    fn commit_to_poly(&self, poly: Self::Poly) -> Result<Self::G1, String>;

    fn compute_proof_single(&self, poly: Self::Poly, point: Self::Fr) -> Result<Self::G1, String>;

    fn verify_proof_single(
        &self,
        proof: Self::G1,
        x: Self::Fr,
        y: Self::Fr,
        commitment: Self::G1,
    ) -> Result<bool, String>;

    fn parse_poly_from_str(&self, s: &[String]) -> Result<Self::Poly, String>;
    fn parse_point_from_str(&self, s: &str) -> Result<Self::Fr, String>;
    fn parse_g1_from_str(&self, s: &str) -> Result<Self::G1, String>;

    fn random_poly(&self, degree: usize) -> Self::Poly;
    fn random_point(&self) -> Self::Fr;
    fn evaluate(&self, poly: &Self::Poly, x: Self::Fr) -> Self::Fr;
    fn setup(cfg: SetupConfig) -> Result<Self, String>
    where
        Self: Sized;
    fn setup_and_save(cfg: SetupConfig) -> Result<(), String>;
}

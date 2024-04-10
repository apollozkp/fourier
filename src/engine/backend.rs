use kzg::{FFTSettings, Fr, G1Affine, G1Fp, G1GetFp, G1Mul, KZGSettings, Poly, G1, G2};

pub trait Backend {
    // TODO: These should probably not be constants, probably should be initialized somehow
    const SCALE: usize;

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
    fn default_config() -> BackendConfig;

    fn random_poly(&self, degree: usize) -> Self::Poly;
    fn random_point(&self) -> Self::Fr;
    fn evaluate(&self, poly: &Self::Poly, x: Self::Fr) -> Self::Fr;
}

#[derive(Debug, Clone, Default)]
pub struct BackendConfig {
    pub scale: Option<usize>,
    pub path: Option<String>,
}

impl BackendConfig {
    pub fn new(scale: Option<usize>, path: Option<String>) -> Self {
        Self { scale, path }
    }
}

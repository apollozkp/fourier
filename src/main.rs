pub mod rpc;

use kzg::{FFTSettings, Fr, G1Affine, G1Fp, G1GetFp, G1Mul, KZGSettings, Poly, G1, G2};

pub const SECRET: [u8; 32usize] = [
    0xa4, 0x73, 0x31, 0x95, 0x28, 0xc8, 0xb6, 0xea, 0x4d, 0x08, 0xcc, 0x53, 0x18, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
];

const SCALE: usize = 1;

pub fn commit_to_poly<TFr, TG1, TG2, TPoly, TFFTSettings, TKZGSettings, TG1Fp, TG1Affine>(
    generate_trusted_setup: &dyn Fn(usize, [u8; 32usize]) -> (Vec<TG1>, Vec<TG2>),
) -> Result<TG1, String>
where
    TFr: Fr,
    TG1: G1 + G1Mul<TFr> + G1GetFp<TG1Fp>,
    TG2: G2 + ?Sized,
    TPoly: Poly<TFr>,
    TFFTSettings: FFTSettings<TFr>,
    TKZGSettings: KZGSettings<TFr, TG1, TG2, TFFTSettings, TPoly, TG1Fp, TG1Affine>,
    TG1Fp: G1Fp,
    TG1Affine: G1Affine<TG1, TG1Fp>,
{
    let fs = TFFTSettings::new(SCALE).unwrap();
    let (s1, s2) = generate_trusted_setup(fs.get_max_width(), SECRET);
    let ks = TKZGSettings::new(&s1, &s2, fs.get_max_width(), &fs).unwrap();
    let mut poly = TPoly::new(fs.get_max_width());
    for i in 0..fs.get_max_width() {
        poly.set_coeff_at(i, &TFr::rand());
    }
    ks.commit_to_poly(&poly)
}

pub fn compute_proof_single<TFr, TG1, TG2, TPoly, TFFTSettings, TKZGSettings, TG1Fp, TG1Affine>(
    generate_trusted_setup: &dyn Fn(usize, [u8; 32usize]) -> (Vec<TG1>, Vec<TG2>),
) -> Result<TG1, String>
where
    TFr: Fr,
    TG1: G1 + G1Mul<TFr> + G1GetFp<TG1Fp>,
    TG2: G2 + ?Sized,
    TPoly: Poly<TFr>,
    TFFTSettings: FFTSettings<TFr>,
    TKZGSettings: KZGSettings<TFr, TG1, TG2, TFFTSettings, TPoly, TG1Fp, TG1Affine>,
    TG1Fp: G1Fp,
    TG1Affine: G1Affine<TG1, TG1Fp>,
{
    let fs = TFFTSettings::new(SCALE).unwrap();
    let (s1, s2) = generate_trusted_setup(fs.get_max_width(), SECRET);
    let ks = TKZGSettings::new(&s1, &s2, fs.get_max_width(), &fs).unwrap();
    let mut poly = TPoly::new(fs.get_max_width());
    for i in 0..fs.get_max_width() {
        poly.set_coeff_at(i, &TFr::rand());
    }
    ks.compute_proof_single(&poly, &TFr::rand())
}

#[tokio::main]
async fn main() {
    rpc::start_rpc_server().await;
}

#[cfg(test)]
mod tests {
    use super::{commit_to_poly, compute_proof_single};

    use rust_kzg_arkworks::kzg_proofs::{generate_trusted_setup, FFTSettings, KZGSettings};
    use rust_kzg_arkworks::kzg_types::{ArkFp, ArkFr, ArkG1, ArkG1Affine, ArkG2};
    use rust_kzg_arkworks::utils::PolyData;

    #[test]
    fn test_commit_to_poly() {
        let result =
            commit_to_poly::<ArkFr, ArkG1, ArkG2, PolyData, FFTSettings, KZGSettings, ArkFp, ArkG1Affine>(
                &generate_trusted_setup,
            );
        assert!(result.is_ok());
        println!("{:?}", result.unwrap());
    }

    #[test]
    fn test_proof_single() {
        let result = compute_proof_single::<
            ArkFr,
            ArkG1,
            ArkG2,
            PolyData,
            FFTSettings,
            KZGSettings,
            ArkFp,
            ArkG1Affine,
        >(&generate_trusted_setup);
        assert!(result.is_ok());
        println!("{:?}", result.unwrap());
    }
}


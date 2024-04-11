use kzg::{FFTSettings, Fr, KZGSettings, Poly, G1, G2};
use rand::Rng;
use rayon::prelude::*;
use rust_kzg_blst::types::{g1::FsG1, g2::FsG2};
use rust_kzg_blst::utils::generate_trusted_setup;
// use rust_kzg_blst::kzg_types::{FsG1, FsG2};
use crate::utils::timed;
use std::io::{Read, Write};
use tracing::{debug, info, warn};

pub struct BlstBackend {
    pub fft_settings: rust_kzg_blst::types::fft_settings::FsFFTSettings,
    pub kzg_settings: rust_kzg_blst::types::kzg_settings::FsKZGSettings,
}

impl BlstBackend {
    const DEFAULT_SCALE: usize = 20;
    fn new_from_file(path: &str) -> Result<(Vec<FsG1>, Vec<FsG2>), String> {
        let file = std::fs::File::open(path).map_err(|e| e.to_string())?;
        let mut reader = std::io::BufReader::new(file);

        let g1_raw: Result<Vec<[u8; 48]>, String> = timed("read g1", || {
            let mut g1_size_bytes = [0u8; 8];
            reader
                .read_exact(&mut g1_size_bytes)
                .map_err(|e| e.to_string())?;
            let g1_size = u64::from_le_bytes(g1_size_bytes);

            let g1_raw = (0..g1_size as usize).try_fold(
                Vec::with_capacity(g1_size as usize),
                |mut acc, _| {
                    let mut g1_bytes = [0u8; 48];
                    match reader.read_exact(&mut g1_bytes) {
                        Ok(_) => (),
                        Err(e) => return Err(e.to_string()),
                    }
                    acc.push(g1_bytes);
                    Ok(acc)
                },
            )?;

            Ok(g1_raw)
        });

        let g2_raw: Result<Vec<[u8; 96]>, String> = timed("read g2", || {
            let mut g2_size_bytes = [0u8; 8];
            reader
                .read_exact(&mut g2_size_bytes)
                .map_err(|e| e.to_string())?;
            let g2_size = u64::from_le_bytes(g2_size_bytes);

            let g2_raw = (0..g2_size as usize).try_fold(
                Vec::with_capacity(g2_size as usize),
                |mut acc, _| {
                    let mut g2_bytes = [0u8; 96];
                    match reader.read_exact(&mut g2_bytes) {
                        Ok(_) => (),
                        Err(e) => return Err(e.to_string()),
                    }
                    acc.push(g2_bytes);
                    Ok(acc)
                },
            )?;

            Ok(g2_raw)
        });

        let cores = num_cpus::get();
        debug!("splitting work over {} cores", cores);

        let g1: Result<Vec<FsG1>, String> = timed("parse g1", || {
            let g1_raw = g1_raw?;
            let chunk_size = g1_raw.len() / cores;
            let g1 = g1_raw
                .par_chunks(chunk_size)
                .map(|chunk| {
                    chunk
                        .iter()
                        .map(|x| FsG1::from_bytes(x.as_slice()).expect("Failed to parse G1"))
                        .collect::<Vec<_>>()
                })
                .flatten()
                .collect::<Vec<_>>();

            Ok(g1)
        });

        let g2: Result<Vec<FsG2>, String> = timed("parse g2", || {
            let g2_raw = g2_raw?;
            let chunk_size = g2_raw.len() / cores;
            let g2 = g2_raw
                .par_chunks(chunk_size)
                .map(|chunk| {
                    chunk
                        .iter()
                        .map(|x| FsG2::from_bytes(x.as_slice()).expect("Failed to parse G2"))
                        .collect::<Vec<_>>()
                })
                .flatten()
                .collect::<Vec<_>>();

            Ok(g2)
        });

        Ok((g1?, g2?))
    }

    fn generate_trusted_setup(max_width: usize, secret: [u8; 32usize]) -> (Vec<FsG1>, Vec<FsG2>) {
        generate_trusted_setup(max_width, secret)
    }

    fn new_fft_settings(
        scale: usize,
    ) -> Result<rust_kzg_blst::types::fft_settings::FsFFTSettings, String> {
        rust_kzg_blst::types::fft_settings::FsFFTSettings::new(scale)
    }

    fn new_kzg_settings(
        s1: &[FsG1],
        s2: &[FsG2],
        max_width: usize,
        fft_settings: &rust_kzg_blst::types::fft_settings::FsFFTSettings,
    ) -> Result<rust_kzg_blst::types::kzg_settings::FsKZGSettings, String> {
        rust_kzg_blst::types::kzg_settings::FsKZGSettings::new(s1, s2, max_width, fft_settings)
    }

    fn save_precompute(&self, path: &str) -> Result<(), String> {
        match self.kzg_settings.get_precomputation() {
            Some(precompute) => {
                let file = std::fs::File::create(path).map_err(|e| e.to_string())?;
                let mut writer = std::io::BufWriter::new(file);
                todo!()
            }
            None => return Err("Precomputation not available".to_string()),
        }

        Ok(())
    }
}

impl crate::engine::backend::Backend for BlstBackend {
    type Fr = rust_kzg_blst::types::fr::FsFr;
    type G1 = rust_kzg_blst::types::g1::FsG1;
    type G2 = rust_kzg_blst::types::g2::FsG2;
    type Poly = rust_kzg_blst::types::poly::FsPoly;
    type FFTSettings = rust_kzg_blst::types::fft_settings::FsFFTSettings;
    type KZGSettings = rust_kzg_blst::types::kzg_settings::FsKZGSettings;
    type G1Fp = rust_kzg_blst::types::fp::FsFp;
    type G1Affine = rust_kzg_blst::types::g1::FsG1Affine;

    fn new(cfg: Option<crate::engine::backend::BackendConfig>) -> Self {
        let scale = cfg
            .as_ref()
            .and_then(|cfg| cfg.scale)
            .unwrap_or(Self::DEFAULT_SCALE);
        let fft_settings = timed("Creating FFTSettings", || Self::new_fft_settings(scale))
            .expect("Failed to create FFTSettings");

        let (s1, s2) = if let Some(path) = cfg.as_ref().and_then(|cfg| cfg.path.as_ref()) {
            timed("Reading setup from file", || {
                Self::new_from_file(path).expect("Failed to read setup from file")
            })
        } else {
            warn!("No setup file provided, generating new setup");
            timed("Generating trusted setup", || {
                let secret: [u8; 32] = rand::thread_rng().gen();
                Self::generate_trusted_setup(fft_settings.get_max_width(), secret)
            })
        };

        let kzg_settings = timed("Creating KZGSettings", || {
            Self::new_kzg_settings(&s1, &s2, fft_settings.get_max_width(), &fft_settings)
                .expect("Failed to create KZGSettings")
        });

        Self {
            fft_settings,
            kzg_settings,
        }
    }

    fn commit_to_poly(&self, poly: Self::Poly) -> Result<Self::G1, String> {
        self.kzg_settings.commit_to_poly(&poly)
    }

    fn compute_proof_single(&self, poly: Self::Poly, point: Self::Fr) -> Result<Self::G1, String> {
        self.kzg_settings.compute_proof_single(&poly, &point)
    }

    fn verify_proof_single(
        &self,
        proof: Self::G1,
        x: Self::Fr,
        y: Self::Fr,
        commitment: Self::G1,
    ) -> Result<bool, String> {
        self.kzg_settings
            .check_proof_single(&commitment, &proof, &x, &y)
    }

    fn parse_point_from_str(&self, s: &str) -> Result<Self::Fr, String> {
        Fr::from_bytes(hex::decode(s).map_err(|e| e.to_string())?.as_slice())
    }

    fn parse_poly_from_str(&self, s: &[String]) -> Result<Self::Poly, String> {
        Ok(Self::Poly::from_coeffs(
            &s.iter()
                .map(|x| Fr::from_bytes(hex::decode(x).map_err(|e| e.to_string())?.as_slice()))
                .collect::<Result<Vec<Self::Fr>, String>>()?,
        ))
    }

    fn parse_g1_from_str(&self, s: &str) -> Result<Self::G1, String> {
        G1::from_bytes(hex::decode(s).map_err(|e| e.to_string())?.as_slice())
    }

    fn random_poly(&self, degree: usize) -> Self::Poly {
        let mut poly = Self::Poly::new(degree + 1);
        for i in 0..degree + 1 {
            poly.set_coeff_at(i, &Self::Fr::rand());
        }
        poly
    }

    fn random_point(&self) -> Self::Fr {
        Self::Fr::rand()
    }

    fn evaluate(&self, poly: &Self::Poly, x: Self::Fr) -> Self::Fr {
        poly.eval(&x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::backend::Backend;
    use kzg::Fr;
    use kzg::Poly;

    use rust_kzg_blst::types::fr::FsFr;
    use rust_kzg_blst::types::g1::FsG1;
    use rust_kzg_blst::types::poly::FsPoly;

    const TEST_POLY: [&str; 16] = [
        "6945DC5C4FF4DAC8A7278C9B8F0D4613320CF87FF947F21AC9BF42327EC19448",
        "68E40C088D827BCCE02CEF34BDC8C12BB025FBEA047BC6C00C0C8C5C925B7FAF",
        "67281FAC164E9348B80693BA30D5D4E311DE5878EB3D20E34A58507B484B243C",
        "5F7C377DAE6B9D9ABAD75DC15E4FFF9FE7520D1F85224C95F485F44978154C5A",
        "2D85C376A440B6E25C3F7C11559B6A27684023F36C3D7A0ACD7E7D019DE399C7",
        "4A6FB95F0241B3583771E799120C87AAE3C843ECDB50A38254A92E198968922F",
        "1005079F96EC412A719FE2E9FA67D421D98FB4DEC4181459E59430F5D502BD2A",
        "64960B8692062DCB01C0FFBAC569478A89AD880ED3C9DF710BED5CE75F484693",
        "03C2882155A447642BD21FB1CF2553F80955713F09BBBBD9724E2CBFD8B19D41",
        "0AB07FECB59EE3435F6129FCD602CB519E56D7B426941633E37A3B676A24830F",
        "12FA5861459EFFBAE654827D98BFDFEA5545DDF8BB9628579463DA21F17462B5",
        "6A6296A0376D807530DB09DC8BB069FFDEC3D7541497B82C722A199D6B7C5B06",
        "153D2C81B54D7E1C3E83EA61C7F66FD88155F1713EE581E2BE8438CA9FEE1A02",
        "216BCCC4AE97FE3E1D4B21C375C46140FA153E7868201A43480889047ACD0C2D",
        "381BD4FE924EB10E08F2A227D3DB2083AA0E5A1F661CD3C702C4B8A9385E7839",
        "723A7640FD7E65473131563AB5514916AC861C2695CE6513E5061E597E5E1A81",
    ];

    const TEST_POINT: &str = "456006fff56412d329d527901d02877a581a89cfa677ca963eb9d680766234cc";
    const TEST_EVAL: &str = "29732a1e0e074ab05ee6a9e57794c5ad1965b98b6c8c6ecde96ac776ea06ff5b";

    const EXPECTED_COMMITMENT: &str =  "8424fb9dc224ab79efccf6710edea3b936d03bbd323f052bb9c4b2efe9f98239e7c3e48148f243065cee910054a10e71";
    const EXPECTED_PROOF: &str = "895cdfe1bf26bbf10bdc0d90178ec89635269cca7c9b39836a76e91689ad3fa4d1772f8d60cdd86cd4bfd1dedbdec81d";

    impl BlstBackend {
        pub fn random_poly(&self) -> FsPoly {
            let mut poly = FsPoly::new(self.fft_settings.get_max_width());
            for i in 0..self.fft_settings.get_max_width() {
                poly.set_coeff_at(i, &Fr::rand());
            }
            poly
        }

        pub fn random_fr(&self) -> FsFr {
            FsFr::rand()
        }

        pub fn random_g1(&self) -> FsG1 {
            FsG1::rand()
        }
    }

    #[test]
    #[tracing_test::traced_test]
    fn test_arkworks_backend() {
        let cfg = crate::engine::backend::BackendConfig::new(Some(4), None);
        let backend = BlstBackend::new(Some(cfg.clone()));
        let poly = backend.random_poly();
        debug!("poly: {:?}", poly.clone());
        let x = backend.random_fr();
        let y = poly.eval(&x);
        let commitment = backend.commit_to_poly(poly.clone()).unwrap();
        let proof = backend.compute_proof_single(poly.clone(), x).unwrap();
        let result = backend
            .verify_proof_single(proof, x, y, commitment)
            .unwrap();
        assert!(result);
    }

    #[test]
    #[tracing_test::traced_test]
    fn test_arkworks_g1_serialize_deserialize() {
        let cfg = crate::engine::backend::BackendConfig::new(Some(4), None);
        let backend = BlstBackend::new(Some(cfg.clone()));
        let g1 = backend.random_g1();
        debug!("g1: {:?}", g1);
        let g1_bytes = g1.to_bytes();
        let reserialized = FsG1::from_bytes(&g1_bytes);
        assert_eq!(reserialized, Ok(g1));
    }

    // TODO: make proper test for fixed setup file and fixed secret
    // #[test]
    // #[tracing_test::traced_test]
    // fn test_pipeline() {
    //     let cfg = crate::engine::backend::BackendConfig::new(Some(4), Some("setup".to_string()));
    //     let backend = BlstBackend::new(Some(cfg.clone()));
    //
    //     // Get hardcoded poly
    //     let poly = backend
    //         .parse_poly_from_str(
    //             &TEST_POLY
    //                 .iter()
    //                 .map(|x| x.to_string())
    //                 .collect::<Vec<String>>(),
    //         )
    //         .expect("Failed to parse poly");
    //
    //     // Get hardcoded point
    //     let x = backend
    //         .parse_point_from_str(TEST_POINT)
    //         .expect("Failed to parse point");
    //     debug!("x: {:?}", hex::encode(x.to_bytes()));
    //
    //     // Evaluate poly at point and check against hardcoded value
    //     let y = poly.eval(&x);
    //     let expected = backend
    //         .parse_point_from_str(TEST_EVAL)
    //         .expect("Failed to parse point");
    //     debug!("y: {:?}", hex::encode(y.to_bytes()));
    //     assert_eq!(y, expected);
    //
    //     // Commit to poly and check against hardcoded commitment
    //     let commitment = backend
    //         .commit_to_poly(poly.clone())
    //         .expect("Failed to commit to poly");
    //     debug!("commitment hex: {:?}", hex::encode(commitment.to_bytes()));
    //     assert_eq!(hex::encode(commitment.to_bytes()), EXPECTED_COMMITMENT);
    //
    //     // Compute proof and check against hardcoded proof
    //     let proof = backend
    //         .compute_proof_single(poly.clone(), x)
    //         .expect("Failed to compute proof");
    //     debug!("proof hex: {:?}", hex::encode(proof.to_bytes()));
    //     assert_eq!(hex::encode(proof.to_bytes()), EXPECTED_PROOF);
    //
    //     // Verify proof
    //     let result = backend
    //         .verify_proof_single(proof, x, y, commitment)
    //         .expect("Failed to verify proof");
    //     assert!(result);
    // }
}

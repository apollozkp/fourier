use ark_bls12_381::{G1Affine, G2Affine};
use ark_ec::AffineRepr;
use ark_serialize::CanonicalDeserialize;
use kzg::{FFTSettings, Fr, KZGSettings, Poly, G1};
use rand::Rng;
use rust_kzg_arkworks::kzg_types::{ArkG1, ArkG2};
use std::io::Read;
use tracing::{debug, info};
use rayon::prelude::*;

pub struct ArkworksBackend {
    pub fft_settings: rust_kzg_arkworks::kzg_proofs::FFTSettings,
    pub kzg_settings: rust_kzg_arkworks::kzg_proofs::KZGSettings,
}

impl ArkworksBackend {
    fn new_from_file(path: &str) -> Result<(Vec<ArkG1>, Vec<ArkG2>), String> {
        let file = std::fs::File::open(path).map_err(|e| e.to_string())?;
        let mut reader = std::io::BufReader::new(file);

        let mut g1_size_bytes = [0u8; 8];
        reader
            .read_exact(&mut g1_size_bytes)
            .map_err(|e| e.to_string())?;
        let g1_size = u64::from_le_bytes(g1_size_bytes);
        debug!("read g1_size: {:?}", g1_size);

        let g1_raw =
            (0..g1_size as usize).try_fold(Vec::with_capacity(g1_size as usize), |mut acc, _| {
                let mut g1_bytes = [0u8; 48];
                match reader.read_exact(&mut g1_bytes) {
                    Ok(_) => (),
                    Err(e) => return Err(e.to_string()),
                }
                acc.push(g1_bytes);
                Ok(acc)
            })?;

        let mut g2_size_bytes = [0u8; 8];
        reader
            .read_exact(&mut g2_size_bytes)
            .map_err(|e| e.to_string())?;
        let g2_size = u64::from_le_bytes(g2_size_bytes);
        debug!("read g2_size: {:?}", g2_size);

        let g2_raw =
            (0..g2_size as usize).try_fold(Vec::with_capacity(g2_size as usize), |mut acc, _| {
                let mut g2_bytes = [0u8; 96];
                match reader.read_exact(&mut g2_bytes) {
                    Ok(_) => (),
                    Err(e) => return Err(e.to_string()),
                }
                acc.push(g2_bytes);
                Ok(acc)
            })?;

        let cores = num_cpus::get();
        debug!("splitting work over {} cores", cores);

        debug!("parsing g1...");
        let chunk_size = g1_size as usize / cores;
        let g1 = g1_raw.par_chunks(chunk_size).map(|chunk| {
            debug!("parsing chunk with size: {}", chunk.len());
            chunk.iter().map(|x| {
                let g1_el = G1Affine::deserialize_compressed_unchecked(x.as_slice()).unwrap();
                ArkG1(g1_el.into_group())
            }).collect::<Vec<_>>()
        }).flatten().collect::<Vec<_>>();

        debug!("parsing g2...");
        let chunk_size = g2_size as usize / cores;
        let g2 = g2_raw.par_chunks(chunk_size).map(|chunk| {
            debug!("parsing chunk with size: {}", chunk.len());
            chunk.iter().map(|x| {
                let g2_el = G2Affine::deserialize_compressed_unchecked(x.as_slice()).unwrap();
                ArkG2(g2_el.into_group())
            }).collect::<Vec<_>>()
        }).flatten().collect::<Vec<_>>();

        debug!("done parsing from file");
        Ok((g1, g2))

    }
}

impl crate::engine::backend::Backend for ArkworksBackend {
    const SCALE: usize = 20;

    type Fr = rust_kzg_arkworks::kzg_types::ArkFr;
    type G1 = rust_kzg_arkworks::kzg_types::ArkG1;
    type G2 = rust_kzg_arkworks::kzg_types::ArkG2;
    type Poly = rust_kzg_arkworks::utils::PolyData;
    type FFTSettings = rust_kzg_arkworks::kzg_proofs::FFTSettings;
    type KZGSettings = rust_kzg_arkworks::kzg_proofs::KZGSettings;
    type G1Fp = rust_kzg_arkworks::kzg_types::ArkFp;
    type G1Affine = rust_kzg_arkworks::kzg_types::ArkG1Affine;

    fn generate_trusted_setup(
        max_width: usize,
        secret: [u8; 32usize],
    ) -> (Vec<Self::G1>, Vec<Self::G2>) {
        rust_kzg_arkworks::kzg_proofs::generate_trusted_setup(max_width, secret)
    }

    fn default_config() -> crate::engine::backend::BackendConfig {
        crate::engine::backend::BackendConfig::new(Some(Self::SCALE), None)
    }

    fn new(cfg: Option<crate::engine::backend::BackendConfig>) -> Self {
        info!("Creating new ArkworksBackend...");
        let scale = cfg
            .as_ref()
            .and_then(|cfg| cfg.scale)
            .unwrap_or(Self::SCALE);
        let fft_settings = rust_kzg_arkworks::kzg_proofs::FFTSettings::new(scale)
            .expect("Failed to create FFTSettings");

        let (s1, s2) = if let Some(path) = cfg.as_ref().and_then(|cfg| cfg.path.as_ref()) {
            info!("Reading setup from file");
            Self::new_from_file(path).expect("Failed to read setup from file")
        } else {
            info!("No setup file provided, generating new setup");
            let secret: [u8; 32] = rand::thread_rng().gen();
            Self::generate_trusted_setup(fft_settings.get_max_width(), secret)
        };

        info!("Creating KZGSettings...");
        let kzg_settings = rust_kzg_arkworks::kzg_proofs::KZGSettings::new(
            &s1,
            &s2,
            fft_settings.get_max_width(),
            &fft_settings,
        )
        .expect("Failed to create KZGSettings");
        info!("Created new ArkworksBackend");
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
        Ok(rust_kzg_arkworks::utils::PolyData::from_coeffs(
            &s.iter()
                .map(|x| Fr::from_bytes(hex::decode(x).map_err(|e| e.to_string())?.as_slice()))
                .collect::<Result<Vec<Self::Fr>, String>>()?,
        ))
    }

    fn parse_g1_from_str(&self, s: &str) -> Result<Self::G1, String> {
        G1::from_bytes(hex::decode(s).map_err(|e| e.to_string())?.as_slice())
    }

    fn random_poly(&self, degree: usize) -> Self::Poly {
        let mut poly = rust_kzg_arkworks::utils::PolyData::new(degree + 1);
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

    impl ArkworksBackend {
        pub fn random_poly(&self) -> rust_kzg_arkworks::utils::PolyData {
            let mut poly =
                rust_kzg_arkworks::utils::PolyData::new(self.fft_settings.get_max_width());
            for i in 0..self.fft_settings.get_max_width() {
                poly.set_coeff_at(i, &Fr::rand());
            }
            poly
        }

        pub fn random_fr(&self) -> rust_kzg_arkworks::kzg_types::ArkFr {
            rust_kzg_arkworks::kzg_types::ArkFr::rand()
        }

        pub fn random_g1(&self) -> rust_kzg_arkworks::kzg_types::ArkG1 {
            rust_kzg_arkworks::kzg_types::ArkG1::rand()
        }
    }

    #[test]
    #[tracing_test::traced_test]
    fn test_arkworks_backend() {
        let cfg = crate::engine::backend::BackendConfig::new(Some(4), None);
        let backend = ArkworksBackend::new(Some(cfg.clone()));
        let poly = backend.random_poly();
        tracing::info!("poly: {:?}", poly.clone());
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
    fn test_arkworks_g1_serialize_deserialize() {
        let cfg = crate::engine::backend::BackendConfig::new(Some(4), None);
        let backend = ArkworksBackend::new(Some(cfg.clone()));
        let g1 = backend.random_g1();
        println!("g1: {:?}", g1);
        let g1_bytes = g1.to_bytes();
        let reserialized = rust_kzg_arkworks::kzg_types::ArkG1::from_bytes(&g1_bytes);
        assert_eq!(reserialized, Ok(g1));
    }

    #[test]
    #[tracing_test::traced_test]
    fn test_pipeline() {
        let cfg = crate::engine::backend::BackendConfig::new(Some(4), Some("setup".to_string()));
        let backend = ArkworksBackend::new(Some(cfg.clone()));

        // Get hardcoded poly
        let poly = backend
            .parse_poly_from_str(
                &TEST_POLY
                    .iter()
                    .map(|x| x.to_string())
                    .collect::<Vec<String>>(),
            )
            .expect("Failed to parse poly");

        // Get hardcoded point
        let x = backend
            .parse_point_from_str(TEST_POINT)
            .expect("Failed to parse point");
        println!("x: {:?}", hex::encode(x.to_bytes()));

        // Evaluate poly at point and check against hardcoded value
        let y = poly.eval(&x);
        let expected = backend
            .parse_point_from_str(TEST_EVAL)
            .expect("Failed to parse point");
        println!("y: {:?}", hex::encode(y.to_bytes()));
        assert_eq!(y, expected);

        // Commit to poly and check against hardcoded commitment
        let commitment = backend
            .commit_to_poly(poly.clone())
            .expect("Failed to commit to poly");
        println!("commitment hex: {:?}", hex::encode(commitment.to_bytes()));
        assert_eq!(hex::encode(commitment.to_bytes()), EXPECTED_COMMITMENT);

        // Compute proof and check against hardcoded proof
        let proof = backend
            .compute_proof_single(poly.clone(), x)
            .expect("Failed to compute proof");
        println!("proof hex: {:?}", hex::encode(proof.to_bytes()));
        assert_eq!(hex::encode(proof.to_bytes()), EXPECTED_PROOF);

        // Verify proof
        let result = backend
            .verify_proof_single(proof, x, y, commitment)
            .expect("Failed to verify proof");
        assert!(result);
    }

    #[test]
    #[tracing_test::traced_test]
    fn write_setup() {
        use kzg::G2;
        use std::io::Write;
        use tracing::info;

        info!("Generating setup");
        let cfg = crate::engine::backend::BackendConfig::new(Some(20), None);
        let backend = ArkworksBackend::new(Some(cfg.clone()));

        let mut file = std::fs::File::create("setup").unwrap();

        info!("Writing s1");
        let encoded_s1_size = backend.kzg_settings.secret_g1.len() as u64;
        Write::write(&mut file, &encoded_s1_size.to_le_bytes()).unwrap();
        for el in backend.kzg_settings.secret_g1 {
            let bytes = el.to_bytes();
            Write::write(&mut file, &bytes).unwrap();
        }

        info!("Writing s2");
        let encoded_s2_size = backend.kzg_settings.secret_g2.len() as u64;
        Write::write(&mut file, &encoded_s2_size.to_le_bytes()).unwrap();
        for el in backend.kzg_settings.secret_g2 {
            let bytes = el.to_bytes();
            Write::write(&mut file, &bytes).unwrap();
        }
    }
}

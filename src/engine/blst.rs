use crate::utils::timed;
use kzg::{FFTSettings, Fr, KZGSettings, Poly, G1};
use rand::Rng;
use rust_kzg_blst::types::fp::FsFp;
use rust_kzg_blst::types::fr::FsFr;
use rust_kzg_blst::types::g1::FsG1Affine;
use rust_kzg_blst::types::{g1::FsG1, g2::FsG2};
use rust_kzg_blst::utils::generate_trusted_setup;
use std::sync::Arc;
use tracing::{debug, warn};

use crate::engine::config::{BackendConfig, SetupConfig};

pub enum PrecomputeInstruction {
    Skip,
    Generate,
    Loaded(Option<kzg::msm::precompute::PrecomputationTable<FsFr, FsG1, FsFp, FsG1Affine>>),
}

impl From<SetupConfig> for PrecomputeInstruction {
    fn from(cfg: SetupConfig) -> Self {
        if cfg.generate_precompute() {
            PrecomputeInstruction::Generate
        } else {
            let precompute =
                BlstBackend::load_precompute_from_file(cfg.precompute_path(), cfg.compressed())
                    .expect("Failed to load precompute from file");
            PrecomputeInstruction::Loaded(precompute)
        }
    }
}

pub struct BlstBackend {
    pub kzg_settings: rust_kzg_blst::types::kzg_settings::FsKZGSettings,
}

impl BlstBackend {
    fn load_setup_from_file(
        path: &str,
        compressed: bool,
    ) -> Result<(Vec<FsG1>, Vec<FsG2>), String> {
        debug!("Reading setup from file {}", path);
        crate::utils::timed("reading secrets", || {
            rust_kzg_blst::utils::load_secrets_from_file(path, compressed)
        })
    }

    pub fn save_setup_to_file(
        file_path: &str,
        secret_g1: &[FsG1],
        secret_g2: &[FsG2],
        compressed: bool,
    ) -> Result<(), String> {
        crate::utils::timed("writing setup", || {
            rust_kzg_blst::utils::save_secrets_to_file(file_path, secret_g1, secret_g2, compressed)
        })
    }

    pub fn load_precompute_from_file(
        path: &str,
        compressed: bool,
    ) -> Result<
        Option<kzg::msm::precompute::PrecomputationTable<FsFr, FsG1, FsFp, FsG1Affine>>,
        String,
    > {
        crate::utils::timed("reading precompute", || {
            kzg::msm::precompute::precompute_from_file(path, compressed)
        })
    }

    pub fn save_precompute_to_file(
        precompute: &kzg::msm::precompute::PrecomputationTable<FsFr, FsG1, FsFp, FsG1Affine>,
        path: &str,
        compressed: bool,
    ) -> Result<(), String> {
        crate::utils::timed("writing precompute", || {
            kzg::msm::precompute::precompute_to_file(precompute, path, compressed)
        })
    }

    fn generate_trusted_setup(max_width: usize, secret: [u8; 32usize]) -> (Vec<FsG1>, Vec<FsG2>) {
        crate::utils::timed("generate trusted setup", || {
            generate_trusted_setup(max_width, secret)
        })
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
        precomputation: PrecomputeInstruction,
    ) -> Result<rust_kzg_blst::types::kzg_settings::FsKZGSettings, String> {
        match precomputation {
            PrecomputeInstruction::Skip => Ok(rust_kzg_blst::types::kzg_settings::FsKZGSettings {
                fs: fft_settings.clone(),
                secret_g1: s1.to_vec(),
                secret_g2: s2.to_vec(),
                precomputation: None,
            }),
            PrecomputeInstruction::Generate => {
                rust_kzg_blst::types::kzg_settings::FsKZGSettings::new(
                    s1,
                    s2,
                    max_width,
                    fft_settings,
                )
            }
            PrecomputeInstruction::Loaded(precomputation) => {
                Ok(rust_kzg_blst::types::kzg_settings::FsKZGSettings {
                    fs: fft_settings.clone(),
                    secret_g1: s1.to_vec(),
                    secret_g2: s2.to_vec(),
                    precomputation: precomputation.map(Arc::new),
                })
            }
        }
    }

    pub fn save_to_file(
        &self,
        setup_path: Option<&str>,
        precompute_path: Option<&str>,
        compressed: bool,
    ) -> Result<(), String> {
        if let Some(path) = setup_path {
            Self::save_setup_to_file(
                path,
                &self.kzg_settings.secret_g1,
                &self.kzg_settings.secret_g2,
                compressed,
            )
            .map_err(|e| e.to_string())?;
        } else {
            warn!("No setup path provided, skipping setup save");
        }

        if let Some(path) = precompute_path {
            if let Some(precompute) = self.kzg_settings.get_precomputation() {
                Self::save_precompute_to_file(precompute, path, compressed)
                    .map_err(|e| e.to_string())?;
            } else {
                warn!("No precompute to save, skipping");
            }
        } else {
            warn!("No precompute path provided, skipping precompute save");
        };
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

    fn new(cfg: Option<BackendConfig>) -> Self {
        let cfg = cfg.unwrap_or_default().into();
        Self::setup(cfg).expect("Failed to setup KZGSettings")
    }

    fn setup(cfg: SetupConfig) -> Result<Self, String> {
        let fft_settings = timed("Creating FFTSettings", || {
            Self::new_fft_settings(cfg.scale())
        })?;

        let (s1, s2) = if cfg.generate_setup() {
            timed("Generating trusted setup", || {
                let secret: [u8; 32] = rand::thread_rng().gen();
                Self::generate_trusted_setup(fft_settings.get_max_width(), secret)
            })
        } else {
            timed("Reading setup from file", || {
                debug!("Reading secrets from file {}", cfg.setup_path());
                Self::load_secrets_from_file(cfg.setup_path(), cfg.compressed())
                    .expect("Failed to read setup from file")
            })
        };

        let kzg_settings = timed("Creating KZGSettings", || {
            Self::new_kzg_settings(
                &s1,
                &s2,
                fft_settings.get_max_width(),
                &fft_settings,
                cfg.into(),
            )
        })?;

        Ok(Self { kzg_settings })
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

    fn setup_and_save(cfg: SetupConfig) -> Result<(), String> {
        let backend = Self::setup(cfg.clone())?;
        let (compressed, setup_path, precompute_path) = if cfg.compress_existing() {
            debug!("Compressing setup and precompute");
            (
                true,
                format!("{}.compressed", cfg.setup_path()),
                format!("{}.compressed", cfg.precompute_path()),
            )
        } else if cfg.decompress_existing() {
            debug!("Decompressing setup and precompute");
            (
                false,
                format!("{}.decompressed", cfg.setup_path()),
                format!("{}.decompressed", cfg.precompute_path()),
            )
        } else {
            debug!("Saving setup and precompute");
            (
                cfg.compressed(),
                cfg.setup_path().to_string(),
                cfg.precompute_path().to_string(),
            )
        };

        debug!("Saving to setup: {}, precompute: {}", setup_path, precompute_path);
        backend.save_to_file(
            Some(setup_path.as_str()),
            Some(precompute_path.as_str()),
            compressed,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::backend::Backend;
    use crate::RunArgs;
    use crate::SetupArgs;
    use kzg::Fr;
    use kzg::Poly;

    use rust_kzg_blst::types::fr::FsFr;
    use rust_kzg_blst::types::g1::FsG1;
    use rust_kzg_blst::types::poly::FsPoly;
    use tracing::debug;

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

    impl BlstBackend {
        pub fn random_poly(&self) -> FsPoly {
            let mut poly = FsPoly::new(self.kzg_settings.fs.get_max_width());
            for i in 0..self.kzg_settings.fs.get_max_width() {
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
    fn test_blst_backend() {
        const SCALE: usize = 4;
        let cfg = BackendConfig::new(None, None, SCALE, None, None);
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
    fn test_blst_g1_serialize_deserialize() {
        const SCALE: usize = 4;
        let cfg = BackendConfig::new(None, None, SCALE, None, None);
        let backend = BlstBackend::new(Some(cfg.clone()));
        let g1 = backend.random_g1();
        debug!("g1: {:?}", g1);
        let g1_bytes = g1.to_bytes();
        let reserialized = FsG1::from_bytes(&g1_bytes);
        assert_eq!(reserialized, Ok(g1));
    }

    #[test]
    #[tracing_test::traced_test]
    fn test_write_and_load_precompute() {

        const SETUP_PATH: &str = "test_setup_wal";
        const PRECOMPUTE_PATH: &str = "test_precompute_wal";
        const SCALE: usize = 5;
        for uncompressed in &[false, true] {
            let args = SetupArgs {
                setup_path: SETUP_PATH.to_owned(),
                precompute_path: PRECOMPUTE_PATH.to_owned(),
                scale: SCALE,
                overwrite: true,
                generate_secrets: true,
                generate_precompute: true,
                uncompressed: *uncompressed,
                decompress_existing: false,
                compress_existing: false,
            };
            BlstBackend::setup_and_save(args.into()).expect("Failed to setup KZGSettings");
            debug!("wrote setup and precompute for uncompressed: {}", *uncompressed);
            let args = RunArgs {
                host: "localhost".to_owned(),
                port: 9999,
                scale: SCALE,
                setup_path: Some(SETUP_PATH.to_owned()),
                precompute_path: Some(PRECOMPUTE_PATH.to_owned()),
                uncompressed: *uncompressed,
            };
            let backend = BlstBackend::new(Some(args.into()));
            debug!("loaded setup and precompute for uncompressed: {}", *uncompressed);
            assert!(backend.kzg_settings.get_precomputation().is_some());
        }
        let _ = std::fs::remove_file(SETUP_PATH);
        let _ = std::fs::remove_file(PRECOMPUTE_PATH);
    }

    fn test_pipeline(backend: BlstBackend) {
        let poly = backend
            .parse_poly_from_str(
                &TEST_POLY
                    .iter()
                    .map(|x| x.to_string())
                    .collect::<Vec<String>>(),
            )
            .expect("Failed to parse poly");

        let x = backend
            .parse_point_from_str(TEST_POINT)
            .expect("Failed to parse point");
        debug!("x: {:?}", hex::encode(x.to_bytes()));

        let y = poly.eval(&x);
        let expected = backend
            .parse_point_from_str(TEST_EVAL)
            .expect("Failed to parse point");
        debug!("y: {:?}", hex::encode(y.to_bytes()));
        assert_eq!(y, expected);

        let commitment = backend
            .commit_to_poly(poly.clone())
            .expect("Failed to commit to poly");
        debug!("commitment hex: {:?}", hex::encode(commitment.to_bytes()));

        let proof = backend
            .compute_proof_single(poly.clone(), x)
            .expect("Failed to compute proof");
        debug!("proof hex: {:?}", hex::encode(proof.to_bytes()));

        let result = backend
            .verify_proof_single(proof, x, y, commitment)
            .expect("Failed to verify proof");
        assert!(result);
    }

    #[test]
    #[tracing_test::traced_test]
    fn test_pipeline_from_setup() {
        // Setup defaults
        const SETUP_PATH: &str = "test_setup_pipeline";
        const PRECOMPUTE_PATH: &str = "test_precompute_pipeline";
        const SCALE: usize = 5;
        const UNCOMPRESSED: bool = false;

        // Run defaults
        const HOST: &str = "localhost";
        const PORT: usize = 9999;

        // Do setup to populate files
        let setup_args = SetupArgs {
            setup_path: SETUP_PATH.to_string(),
            precompute_path: PRECOMPUTE_PATH.to_string(),
            scale: SCALE,
            overwrite: false,
            generate_setup: true,
            generate_precompute: true,
            uncompressed: UNCOMPRESSED,
            decompress_existing: false,
            compress_existing: false,
        };

        BlstBackend::setup_and_save(setup_args.into()).expect("Failed to setup KZGSettings");

        // Files are now populated, restart with files
        let run_args = RunArgs {
            host: HOST.to_owned(),
            port: PORT,
            scale: SCALE,
            setup_path: Some(SETUP_PATH.to_owned()),
            precompute_path: Some(PRECOMPUTE_PATH.to_owned()),
            uncompressed: UNCOMPRESSED,
        };

        let backend = BlstBackend::new(Some(run_args.into()));

        test_pipeline(backend);
        let _ = std::fs::remove_file(SETUP_PATH);
        let _ = std::fs::remove_file(PRECOMPUTE_PATH);
    }

    #[test]
    #[tracing_test::traced_test]
    fn test_decompress_existing() {
        const SETUP_PATH: &str = "test_setup_de";
        const PRECOMPUTE_PATH: &str = "test_precompute_de";
        const SCALE: usize = 5;
        const UNCOMPRESSED: bool = false;

        // if files exist, remove them
        let _ = std::fs::remove_file(SETUP_PATH);
        let _ = std::fs::remove_file(PRECOMPUTE_PATH);
        let _ = std::fs::remove_file(format!("{}.decompressed", SETUP_PATH));
        let _ = std::fs::remove_file(format!("{}.decompressed", PRECOMPUTE_PATH));

        // First generate a compressed setup
        let setup_args = SetupArgs {
            setup_path: SETUP_PATH.to_string(),
            precompute_path: PRECOMPUTE_PATH.to_string(),
            scale: SCALE,
            overwrite: false,
            generate_secrets: true,
            generate_precompute: true,
            uncompressed: UNCOMPRESSED,
            decompress_existing: false,
            compress_existing: false,
        };
        BlstBackend::setup_and_save(setup_args.into()).expect("Failed to setup KZGSettings");

        // Check if files exist
        assert!(std::path::Path::new(SETUP_PATH).exists());
        assert!(std::path::Path::new(PRECOMPUTE_PATH).exists());

        // Now decompress the setup
        let setup_args = SetupArgs {
            setup_path: SETUP_PATH.to_string(),
            precompute_path: PRECOMPUTE_PATH.to_string(),
            scale: SCALE,
            overwrite: false,
            generate_secrets: false,
            generate_precompute: false,
            uncompressed: UNCOMPRESSED,
            decompress_existing: true,
            compress_existing: false,
        };
        BlstBackend::setup_and_save(setup_args.into()).expect("Failed to setup KZGSettings");
        
        // Check if files exist
        assert!(std::path::Path::new(format!("{}.decompressed", SETUP_PATH).as_str()).exists());
        assert!(std::path::Path::new(format!("{}.decompressed", PRECOMPUTE_PATH).as_str()).exists());

        // Now load the decompressed setup
        let run_args = RunArgs {
            host: "localhost".to_owned(),
            port: 9999,
            scale: SCALE,
            setup_path: Some(format!("{}.decompressed", SETUP_PATH)),
            precompute_path: Some(format!("{}.decompressed", PRECOMPUTE_PATH)),
            uncompressed: !UNCOMPRESSED,
        };

        let backend = BlstBackend::new(Some(run_args.into()));
        test_pipeline(backend);
        // cleanup
        let _ = std::fs::remove_file(SETUP_PATH);
        let _ = std::fs::remove_file(PRECOMPUTE_PATH);
        let _ = std::fs::remove_file(format!("{}.decompressed", SETUP_PATH));
        let _ = std::fs::remove_file(format!("{}.decompressed", PRECOMPUTE_PATH));
    }

    #[test]
    #[tracing_test::traced_test]
    fn test_compress_existing() {
        const SETUP_PATH: &str = "test_setup_ce";
        const PRECOMPUTE_PATH: &str = "test_precompute_ce";
        const SCALE: usize = 5;
        const UNCOMPRESSED: bool = true;

        // if files exist, remove them
        let _ = std::fs::remove_file(SETUP_PATH);
        let _ = std::fs::remove_file(PRECOMPUTE_PATH);
        let _ = std::fs::remove_file(format!("{}.compressed", SETUP_PATH));
        let _ = std::fs::remove_file(format!("{}.compressed", PRECOMPUTE_PATH));

        // First generate a decompressed setup
        let setup_args = SetupArgs {
            setup_path: SETUP_PATH.to_string(),
            precompute_path: PRECOMPUTE_PATH.to_string(),
            scale: SCALE,
            overwrite: false,
            generate_secrets: true,
            generate_precompute: true,
            uncompressed: UNCOMPRESSED,
            decompress_existing: false,
            compress_existing: false,
        };
        BlstBackend::setup_and_save(setup_args.into()).expect("Failed to setup KZGSettings");
        
        // Check if files exist
        assert!(std::path::Path::new(SETUP_PATH).exists());
        assert!(std::path::Path::new(PRECOMPUTE_PATH).exists());

        // Now compress the setup
        let setup_args = SetupArgs {
            setup_path: SETUP_PATH.to_string(),
            precompute_path: PRECOMPUTE_PATH.to_string(),
            scale: SCALE,
            overwrite: false,
            generate_secrets: false,
            generate_precompute: false,
            uncompressed: UNCOMPRESSED,
            decompress_existing: false,
            compress_existing: true,
        };
        BlstBackend::setup_and_save(setup_args.into()).expect("Failed to setup KZGSettings");
        
        // Check if files exist
        assert!(std::path::Path::new(format!("{}.compressed", SETUP_PATH).as_str()).exists());
        assert!(std::path::Path::new(format!("{}.compressed", PRECOMPUTE_PATH).as_str()).exists());

        // Now load the decompressed setup
        let run_args = RunArgs {
            host: "localhost".to_owned(),
            port: 9999,
            scale: SCALE,
            setup_path: Some(format!("{}.compressed", SETUP_PATH)),
            precompute_path: Some(format!("{}.compressed", PRECOMPUTE_PATH)),
            uncompressed: !UNCOMPRESSED,
        };

        let backend = BlstBackend::new(Some(run_args.into()));
        test_pipeline(backend);
        // cleanup
        let _ = std::fs::remove_file(SETUP_PATH);
        let _ = std::fs::remove_file(PRECOMPUTE_PATH);
        let _ = std::fs::remove_file(format!("{}.compressed", SETUP_PATH));
        let _ = std::fs::remove_file(format!("{}.compressed", PRECOMPUTE_PATH));
    }






}

use kzg::{FFTSettings, Fr, KZGSettings, Poly, G1};

pub struct ArkworksBackend {
    pub fft_settings: rust_kzg_arkworks::kzg_proofs::FFTSettings,
    pub kzg_settings: rust_kzg_arkworks::kzg_proofs::KZGSettings,
}

impl crate::engine::backend::Backend for ArkworksBackend {
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

    fn new(cfg: crate::engine::backend::BackendConfig) -> Self {
        let fft_settings = rust_kzg_arkworks::kzg_proofs::FFTSettings::new(cfg.scale)
            .expect("Failed to create FFTSettings");
        let (s1, s2) = Self::generate_trusted_setup(fft_settings.get_max_width(), cfg.secret);
        let kzg_settings = rust_kzg_arkworks::kzg_proofs::KZGSettings::new(
            &s1,
            &s2,
            fft_settings.get_max_width(),
            &fft_settings,
        )
        .expect("Failed to create KZGSettings");
        Self {
            fft_settings,
            kzg_settings,
        }
    }

    fn commit_to_poly(&self, poly: Self::Poly) -> Result<Self::G1, String> {
        self.kzg_settings.commit_to_poly(&poly)
    }

    fn compute_proof_single(
        &self,
        poly: Self::Poly,
        point: Self::Fr,
    ) -> Result<Self::G1, String> {
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

    fn parse_poly_from_str(
        &self,
        s: &[String],
    ) -> Result<Self::Poly, String> {
        Ok(rust_kzg_arkworks::utils::PolyData::from_coeffs(&s
            .iter()
            .map(|x| Fr::from_bytes(hex::decode(x).map_err(|e| e.to_string())?.as_slice()))
            .collect::<Result<Vec<Self::Fr>, String>>()?))
    }

    fn parse_g1_from_str(
        &self,
        s: &str,
    ) -> Result<Self::G1, String> {
        G1::from_bytes(hex::decode(s).map_err(|e| e.to_string())?.as_slice())
    }

}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::backend::Backend;
    use kzg::Fr;
    use kzg::Poly;

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
    }

    #[test]
    #[tracing_test::traced_test]
    fn test_arkworks_backend() {
        const SECRET: [u8; 32usize] = [
            0xa4, 0x73, 0x31, 0x95, 0x28, 0xc8, 0xb6, 0xea, 0x4d, 0x08, 0xcc, 0x53, 0x18, 0x00,
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00,
        ];
        const SCALE: usize = 4;

        let backend =
            ArkworksBackend::new(crate::engine::backend::BackendConfig::new(SCALE, SECRET));
        let poly = backend.random_poly();
        tracing::info!("poly: {:?}", poly.clone());
        let x = backend.random_fr();
        let y = poly.eval(&x);
        let commitment = backend.commit_to_poly(poly.clone()).unwrap();
        let proof = backend
            .compute_proof_single(poly.clone(), x)
            .unwrap();
        let result = backend
            .verify_proof_single(proof, x, y, commitment)
            .unwrap();
        assert!(result);
    }
}

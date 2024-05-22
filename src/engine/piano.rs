use kzg::eip_4844::hash_to_bls_field;
use kzg::msm::precompute::precompute;
use kzg::msm::precompute::PrecomputationTable;
use kzg::FFTFr;
use kzg::FFTSettings;
use kzg::Fr;
use kzg::G1Mul;
use kzg::G2Mul;
use kzg::Poly;
use kzg::G1;
use kzg::G2;
use rust_kzg_blst::types::fft_settings::FsFFTSettings;
use rust_kzg_blst::types::fp::FsFp;
use rust_kzg_blst::types::fr::FsFr;
use rust_kzg_blst::types::g1::FsG1;
use rust_kzg_blst::types::g1::FsG1Affine;
use rust_kzg_blst::types::g2::FsG2;
use rust_kzg_blst::types::poly::FsPoly;
use std::io::Write;
use std::sync::Arc;

use super::config::DistributedBackendConfig;
use super::config::DistributedSetupConfig;

#[derive(Debug, Clone)]
pub struct PianoBackend {
    pub fft_settings: PianoFFTSettings,
    pub piano_settings: PianoSettings,
}

impl PianoBackend {
    pub fn new(cfg: Option<DistributedBackendConfig>) -> Self {
        let cfg = cfg.unwrap_or_default().into();
        Self::setup(cfg).expect("Failed to setup KZGSettings")
    }

    pub fn setup(cfg: DistributedSetupConfig) -> Result<Self, String> {
        use crate::utils::timed;
        use rand::Rng;
        let fft_settings = timed("Creating FFTSettings", || PianoFFTSettings::from(&cfg));

        let mut piano_settings = if cfg.setup.generate_setup() {
            timed("Generating Trusted Setup", || {
                let secrets: [[u8; 32]; 2] = [rand::thread_rng().gen(), rand::thread_rng().gen()];
                generate_trusted_setup(&fft_settings, secrets)
            })
        } else {
            timed("Reading trusted setup from file", || {
                tracing::debug!("Reading secrets from file {}", cfg.setup.setup_path());
                PianoSettings::load_setup_from_file(cfg.setup.setup_path(), cfg.setup.compressed())
            })?
        };

        if cfg.setup.generate_precompute() {
            timed("Generating Precomputations", || {
                piano_settings.generate_precomputation()
            })?;
        } else {
            timed("Loading Precomputations from file", || {
                piano_settings.load_precomputation_from_file(
                    cfg.setup.precompute_path(),
                    cfg.setup.compressed(),
                )
            })?;
        }

        Ok(PianoBackend {
            fft_settings,
            piano_settings,
        })
    }

    /// Commit to a polynomial f_i(X) as if we are machine i
    /// 1. change the polynomial to the Lagrange basis f(X) = sum_{j=0}^{t-1} f_j L_j(X)
    /// 2. Then compute the commitment as
    ///     com_f = prod_{j=0}^{t-1} U_{i, j}^{f_j}
    ///           = prod g^{R_i(tau_Y) * L_j(tau_X) * f_j}
    ///           = g^{R_i(tau_Y) * f_i(tau_X)}
    /// NOTE: We are commiting to R_i(tau_Y) * f_i(X), not f_i(X)
    pub fn commit(&self, i: usize, poly: &FsPoly) -> Result<FsG1, String> {
        let mut out = FsG1::default();
        rust_kzg_blst::kzg_proofs::g1_linear_combination(
            &mut out,
            &(0..poly.len())
                .map(|j| self.piano_settings.u(i, j))
                .collect::<Vec<_>>(),
            &self.fft_settings.fft_left(&poly.coeffs, false)?,
            poly.len(),
            self.piano_settings.get_precomputation().get_u(i),
        );

        Ok(out)
    }

    /// Commit to a polynomial f(X) by combining the commitments of all machines
    pub fn master_commit(&self, commitments: &[FsG1]) -> FsG1 {
        let mut out = FsG1::default();
        rust_kzg_blst::kzg_proofs::g1_linear_combination(
            &mut out,
            commitments,
            &vec![FsFr::one(); commitments.len()],
            commitments.len(),
            None,
        );
        out
    }

    /// Open a polynomial f_i(X) as if we are machine i
    /// 1. Evaluate f_i(alpha)
    /// 2. Compute q_0^{(i})}(X) = (f_i(X) - f_i(alpha)) / (X - alpha)
    /// 3. Compute pi_0^{(i)} = g^{R(tau_Y) * q_0^{(i)}(tau_X)}
    /// NOTE: pi = g^{R(tau_Y) * q(tau_X)}
    ///          = g^{q'(tau_X)} where q'(X)
    ///          = R_i(tau_Y) * q(X)
    ///          = (R_i(tau_Y) * f_i(X) - R_i(tau_Y) * f_i(alpha)) / (X - alpha)
    /// So we are actually just opening f'(X) = R_i(tau_Y) * f_i(X)
    /// But we do return f_i(alpha)
    pub fn open(&self, i: usize, poly: &FsPoly, alpha: &FsFr) -> Result<(FsFr, FsG1), String> {
        // Evaluate the polynomial at alpha
        let y = poly.eval(alpha);

        // Compute the quotient polynomial

        // Numerator is (f_i(X) - y) = (f_i_0 - y) + f_i_1 X + f_i_2 X^2 + ...
        let mut numerator_coeffs = poly.coeffs.clone();
        numerator_coeffs[0] = numerator_coeffs[0].add(&y.negate());
        let mut numerator = FsPoly::from_coeffs(&numerator_coeffs);

        // Denominator is (X - alpha) = (- alpha + X)
        let denominator_coeffs = vec![alpha.negate(), FsFr::one()];
        let denominator = FsPoly::from_coeffs(&denominator_coeffs);

        let q = numerator.div(&denominator)?;

        // compute proof by using
        // g^{R(tau_Y)q(tau_X)} = g^{R_i(tau_Y)(q_0 L_0(X) + q_1 L_1(X) + ...)}
        //                      = prod g^{q_j R_i(tau_Y) L_j(tau_X)}
        //                      = prod g^{R_i(tau_Y) L_j(tau_X)}^q_j
        //                      = prod U_{i,j}^q_j

        // q needs to be of length poly.len(), so we zero-pad it
        let mut q_cob = vec![FsFr::zero(); poly.len()];
        q_cob[..q.coeffs.len()].copy_from_slice(&q.coeffs);

        let mut out = FsG1::default();
        rust_kzg_blst::kzg_proofs::g1_linear_combination(
            &mut out,
            &(0..poly.len())
                .map(|j| self.piano_settings.u(i, j))
                .collect::<Vec<_>>(),
            &self.fft_settings.fft_left(&q_cob, false).unwrap(),
            poly.len(),
            self.piano_settings.get_precomputation().get_u(i),
        );
        Ok((y, out))
    }

    /// Open a polynomial f(X) by combining the openings of all machines
    /// 1. Compute the proof pi_0 = prod pi_0^{(i)}
    ///                           = prod g^{R_i(tau_Y) q_0^{(i)}(tau_X)}
    ///                           = prod g^{R_i(tau_Y) (f_i(tau_X) - f_i(alpha)) / (tau_X - alpha)}
    ///                           = g^{sum R_i(tau_Y) f_i(tau_X) / (tau_X - alpha)}
    /// 2. Recover f(Y, alpha) = sum R_i(Y) f_i(alpha)
    /// 3. Evaluate z = f(beta, alpha)
    /// 4. Compute the quotient polynomial q_1(X) = (f(Y, alpha) - z) / (Y - beta)
    /// 5. Compute the proof pi_1 = g^{q_1(tau_Y)}
    /// 6. Return (z, pi_f) = (z, (pi_0, pi_1))
    pub fn master_open(
        &self,
        evals: &[FsFr],
        proofs: &[FsG1],
        beta: &FsFr,
    ) -> Result<(FsFr, (FsG1, FsG1)), String> {
        // Compute master proof by adding all the proofs together
        let mut pi0 = FsG1::default();
        rust_kzg_blst::kzg_proofs::g1_linear_combination(
            &mut pi0,
            proofs,
            &vec![FsFr::one(); proofs.len()],
            proofs.len(),
            None,
        );

        // Recover f(Y, alpha) = sum R_i(Y) f_i(alpha)
        let poly = FsPoly::from_coeffs(&self.fft_settings.fft_right(evals, true).unwrap());

        // Evaluate f(Y, alpha) at beta
        let z = poly.eval(beta);

        // Compute quotient poly
        // Numerator is (f(Y, alpha) - z)
        let mut numerator_coeffs = poly.coeffs.clone();
        numerator_coeffs[0] = numerator_coeffs[0].sub(&z);
        let mut numerator = FsPoly::from_coeffs(&numerator_coeffs);

        // Denominator is (Y - beta)
        let denominator = FsPoly::from_coeffs(&[beta.negate(), FsFr::one()]);

        let q = numerator.div(&denominator)?;

        // compute g^q(tau_Y)
        // using g^{q(tau_Y)} = prod g^{q_j tau_Y^j} = prod (g^{tau_Y})^q_j
        let mut pi1 = FsG1::default();
        rust_kzg_blst::kzg_proofs::g1_linear_combination(
            &mut pi1,
            &(0..q.len())
                .map(|j| self.piano_settings.g_tau_y(j))
                .collect::<Vec<_>>(),
            &q.coeffs,
            q.len(),
            self.piano_settings.get_precomputation().get_g1_tau_y(),
        );

        Ok((z, (pi0, pi1)))
    }

    /// Verify a single opening of a polynomial f_i(X) from machine i
    /// Our commitment and proof are actually of f_i'(X) = g^{R_i(tau_Y) * f_i(X)}
    /// I.e. we have commitment = g^{R_i(tau_Y) * f_i(alpha)}
    /// And we have pi = g^{R_i(tau_Y) * q_i(tau_X)} where q_i(X) = (f_i(X) - f(alpha)) / (X - alpha)
    /// But y = f_i(alpha)
    /// We need y' = f'(alpha) = g^{R_i(tau_Y) * f_i(alpha)}
    /// We have g^{tau_Y}, so we can compute g^{R_i(tau_Y)} and then g^{R_i(tau_Y) * y}
    /// as g^{R_i(tau_Y)}^{y}
    /// Then we validate that e(commitment/ g^{y'}, g) == e(pi, g^{tau_X - alpha})
    pub fn verify_single(
        &self,
        i: usize,
        commitment: &FsG1,
        alpha: &FsFr,
        y: &FsFr,
        pi: &FsG1,
    ) -> bool {
        // Compute y' = g^{R_i(tau_Y) * y}

        // Get R_i(X) = (omega_Y^i / M) * (X^M - 1) / (X - omega_Y^i) in the standard basis
        let right_lagrange_poly = self.fft_settings.right_lagrange_poly(i).unwrap();

        // Compute g^{R_i(tau_Y)} using the coefficients of R_i(tau_Y) and g^{tau_Y}
        // I.e. let R_i(X) = sum r_j X^j, then g^{R_i(tau_Y)} = prod g^{r_j tau_Y^j} = g^{ sum r_j tau_Y^j }
        // NOTE: No base change here!
        let mut r_i_tau_y = FsG1::default();
        rust_kzg_blst::kzg_proofs::g1_linear_combination(
            &mut r_i_tau_y,
            &(0..right_lagrange_poly.len())
                .map(|j| self.piano_settings.g_tau_y(j))
                .collect::<Vec<_>>(),
            &right_lagrange_poly.coeffs,
            right_lagrange_poly.len(),
            self.piano_settings.get_precomputation().get_g1_tau_y(),
        );

        // Compute g^{y'} = g^{R_i(tau_Y) * y} = (g^{R_i(tau_Y)})^y
        let g_y_prime = r_i_tau_y.mul(y);

        // Compute g^{R_i(tau_Y) * f_i(tau_X) - y'} = commitment / g^{y'}
        let commitment_minus_y_prime = commitment.sub(&g_y_prime);

        // Compute g^{tau_X - alpha}
        let g_tau_x_minus_alpha = self
            .piano_settings
            .g2_tau_x()
            .sub(&self.piano_settings.g2().mul(alpha));

        // Compute e(commitment / g^{y'}, g) == e(pi, g^{tau_X - alpha})
        Self::pairings_verify_single(
            &commitment_minus_y_prime,
            &self.piano_settings.g2(),
            pi,
            &g_tau_x_minus_alpha,
        )
    }

    fn pairings_verify_single(a1: &FsG1, a2: &FsG2, b1: &FsG1, b2: &FsG2) -> bool {
        use blst::{
            blst_fp12_is_one, blst_p1_affine, blst_p1_cneg, blst_p1_to_affine, blst_p2_affine,
            blst_p2_to_affine, Pairing,
        };

        let mut aa1 = blst_p1_affine::default();
        let mut aa2 = blst_p2_affine::default();
        let mut bb1 = blst_p1_affine::default();
        let mut bb2 = blst_p2_affine::default();

        // As an optimisation, we want to invert one of the pairings
        // so we negate one of the points
        let mut a1neg: FsG1 = *a1;
        unsafe {
            blst_p1_cneg(&mut a1neg.0, true);
            blst_p1_to_affine(&mut aa1, &a1neg.0);
            blst_p2_to_affine(&mut aa2, &a2.0);

            blst_p1_to_affine(&mut bb1, &b1.0);
            blst_p2_to_affine(&mut bb2, &b2.0);

            let dst = [0u8; 3];
            let mut pairing_blst = Pairing::new(false, &dst);
            pairing_blst.raw_aggregate(&aa2, &aa1);
            pairing_blst.raw_aggregate(&bb2, &bb1);
            let gt_point = pairing_blst.as_fp12().final_exp();

            blst_fp12_is_one(&gt_point)
        }
    }

    /// Verify an opening of a polynomial f(X) from any machine
    /// Verify that e(commitment / g^z, g) == e(pi_0, g^{tau_X - alpha}) * e(pi_1, g^{tau_Y - beta})
    /// commitment / g^z = g^{f(tau_X) - z}
    pub fn verify(
        &self,
        commitment: &FsG1,
        beta: &FsFr,
        alpha: &FsFr,
        z: &FsFr,
        pi_f: &(FsG1, FsG1),
    ) -> bool {
        let commitment_minux_z = commitment.sub(&self.piano_settings.g().mul(z));
        let g_tau_x_minus_alpha = self
            .piano_settings
            .g2_tau_x()
            .sub(&self.piano_settings.g2().mul(alpha));
        let g_tau_y_minus_beta = self
            .piano_settings
            .g2_tau_y()
            .sub(&self.piano_settings.g2().mul(beta));

        Self::pairings_verify(
            &commitment_minux_z,
            &self.piano_settings.g2(),
            &pi_f.0,
            &g_tau_x_minus_alpha,
            &pi_f.1,
            &g_tau_y_minus_beta,
        )
    }

    fn pairings_verify(
        a1: &FsG1,
        a2: &FsG2,
        b11: &FsG1,
        b12: &FsG2,
        b21: &FsG1,
        b22: &FsG2,
    ) -> bool {
        use blst::{
            blst_fp12_is_one, blst_p1_affine, blst_p1_cneg, blst_p1_to_affine, blst_p2_affine,
            blst_p2_to_affine, Pairing,
        };

        let mut aa1 = blst_p1_affine::default();
        let mut aa2 = blst_p2_affine::default();
        let mut bb11 = blst_p1_affine::default();
        let mut bb12 = blst_p2_affine::default();
        let mut bb21 = blst_p1_affine::default();
        let mut bb22 = blst_p2_affine::default();

        // As an optimisation, we want to invert one of the pairings
        // so we negate one of the points
        let mut a1neg: FsG1 = *a1;
        unsafe {
            blst_p1_cneg(&mut a1neg.0, true);
            blst_p1_to_affine(&mut aa1, &a1neg.0);
            blst_p2_to_affine(&mut aa2, &a2.0);

            blst_p1_to_affine(&mut bb11, &b11.0);
            blst_p2_to_affine(&mut bb12, &b12.0);
            blst_p1_to_affine(&mut bb21, &b21.0);
            blst_p2_to_affine(&mut bb22, &b22.0);

            let dst = [0u8; 3];
            let mut pairing_blst = Pairing::new(false, &dst);
            pairing_blst.raw_aggregate(&aa2, &aa1);
            pairing_blst.raw_aggregate(&bb12, &bb11);
            pairing_blst.raw_aggregate(&bb22, &bb21);
            let gt_point = pairing_blst.as_fp12().final_exp();

            blst_fp12_is_one(&gt_point)
        }
    }
}

type OptionalPrecomputationTable = Option<Arc<PrecomputationTable<FsFr, FsG1, FsFp, FsG1Affine>>>;

#[derive(Debug, Default, PartialEq, Clone)]
pub struct PianoPrecomputation {
    pub g1_tau_y: OptionalPrecomputationTable,

    pub u: Vec<OptionalPrecomputationTable>,
}

impl PianoPrecomputation {
    pub fn generate(settings: &PianoSettings) -> Result<Self, String> {
        let g1_tau_y = precompute(&settings.g_tau_y).ok().flatten().map(Arc::new);
        let mut u = vec![None; settings.u.len()];
        for (i, row) in settings.u.iter().enumerate() {
            u[i] = precompute(row).ok().flatten().map(Arc::new);
        }
        Ok(PianoPrecomputation { g1_tau_y, u })
    }

    pub fn get_g1_tau_y(&self) -> Option<&PrecomputationTable<FsFr, FsG1, FsFp, FsG1Affine>> {
        self.g1_tau_y.as_ref().map(|v| v.as_ref())
    }

    pub fn get_u(&self, i: usize) -> Option<&PrecomputationTable<FsFr, FsG1, FsFp, FsG1Affine>> {
        if i >= self.u.len() {
            return None;
        }
        self.u[i].as_ref().map(|v| v.as_ref())
    }

    /// Load precomputation tables from a file
    /// The file format is:
    /// - g1_tau_y
    /// - size of u
    /// - u_0
    /// - u_1
    /// - ...
    /// - u_n
    /// where each u_i is a precomputation table
    pub fn load_from_file(file_path: &str, compressed: bool) -> Result<Self, String> {
        use std::io::Read;

        let file = std::fs::File::open(file_path).map_err(|e| e.to_string())?;
        let mut reader = std::io::BufReader::new(file);

        let g1_tau_y = PrecomputationTable::read_from_reader(&mut reader, compressed)?;

        let mut size_bytes = [0u8; 8];
        reader
            .read_exact(&mut size_bytes)
            .map_err(|e| e.to_string())?;
        let size = u64::from_le_bytes(size_bytes) as usize;
        let u = (0..size)
            .map(|_| {
                PrecomputationTable::read_from_reader(&mut reader, compressed)
                    .ok()
                    .map(Arc::new)
            })
            .collect();
        Ok(PianoPrecomputation {
            g1_tau_y: Some(Arc::new(g1_tau_y)),
            u,
        })
    }

    /// Save precomputation tables to a file
    /// The file format is:
    /// - g1_tau_y
    /// - size of u
    /// - u_0
    /// - u_1
    /// - ...
    /// - u_n
    /// where each u_i is a precomputation table
    pub fn save_to_file(&self, file_path: &str, compressed: bool) -> Result<(), String> {
        let file = std::fs::File::create(file_path).map_err(|e| e.to_string())?;
        let mut writer = std::io::BufWriter::new(file);

        if let Some(g1_tau_y) = self.g1_tau_y.as_ref() {
            g1_tau_y.write_to_writer(&mut writer, compressed)?;
        } else {
            return Err("g1_tau_y is missing".to_string());
        }

        let size = self.u.len() as u64;
        writer
            .write_all(&size.to_le_bytes())
            .map_err(|e| e.to_string())?;
        for table in self.u.iter() {
            if let Some(table) = table {
                table.write_to_writer(&mut writer, compressed)?;
            } else {
                return Err("u table is missing".to_string());
            }
        }

        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct PianoSettings {
    g: FsG1,
    g_tau_x: Vec<FsG1>, //NOTE: this one is never used?
    g_tau_y: Vec<FsG1>,
    u: Vec<Vec<FsG1>>,

    g2: FsG2,
    g2_tau_x: FsG2,
    g2_tau_y: FsG2,

    precomputation: PianoPrecomputation,
}

impl PianoSettings {
    pub fn g(&self) -> FsG1 {
        self.g
    }

    /// Get g^{tau_X^i}
    pub fn g_tau_x(&self, i: usize) -> FsG1 {
        self.g_tau_x[i]
    }

    /// Get g^{tau_Y^i}
    pub fn g_tau_y(&self, i: usize) -> FsG1 {
        self.g_tau_y[i]
    }

    pub fn u(&self, i: usize, j: usize) -> FsG1 {
        self.u[i][j]
    }

    pub fn g2(&self) -> FsG2 {
        self.g2
    }

    pub fn g2_tau_x(&self) -> FsG2 {
        self.g2_tau_x
    }

    pub fn g2_tau_y(&self) -> FsG2 {
        self.g2_tau_y
    }

    pub fn set_precomputation(&mut self, precomputation: PianoPrecomputation) {
        self.precomputation = precomputation;
    }

    pub fn get_precomputation(&self) -> &PianoPrecomputation {
        &self.precomputation
    }

    pub fn generate_precomputation(&mut self) -> Result<(), String> {
        self.precomputation = PianoPrecomputation::generate(self)?;
        Ok(())
    }

    pub fn save_setup_to_file(&self, file_path: &str, compressed: bool) -> Result<(), String> {
        fn write_g1(file: &mut std::fs::File, el: &FsG1, compressed: bool) -> Result<(), String> {
            if compressed {
                let bytes = el.to_bytes();
                std::io::Write::write(file, &bytes).map_err(|e| e.to_string())?;
            } else {
                let bytes = el.serialize();
                std::io::Write::write(file, &bytes).map_err(|e| e.to_string())?;
            }
            Ok(())
        }
        fn write_g2(file: &mut std::fs::File, el: &FsG2, compressed: bool) -> Result<(), String> {
            if compressed {
                let bytes = el.to_bytes();
                std::io::Write::write(file, &bytes).map_err(|e| e.to_string())?;
            } else {
                let bytes = el.serialize();
                std::io::Write::write(file, &bytes).map_err(|e| e.to_string())?;
            }
            Ok(())
        }
        let mut file = std::fs::File::create(file_path).unwrap();

        // Write the G1 generator
        write_g1(&mut file, &self.g(), compressed)?;

        // Write the g^{tau_X}^i
        let encoded_g_tau_x_size = self.g_tau_x.len() as u64;
        std::io::Write::write(&mut file, &encoded_g_tau_x_size.to_le_bytes()).unwrap();
        for el in self.g_tau_x.iter() {
            write_g1(&mut file, el, compressed)?;
        }

        // Write the g^{tau_Y}^i
        let encoded_g_tau_y_size = self.g_tau_y.len() as u64;
        std::io::Write::write(&mut file, &encoded_g_tau_y_size.to_le_bytes()).unwrap();
        for el in self.g_tau_y.iter() {
            write_g1(&mut file, el, compressed)?;
        }

        // Write the U_{i, j}
        let encoded_u_rows_size = self.u.len() as u64;
        let encoded_u_cols_size = self.u[0].len() as u64;
        std::io::Write::write(&mut file, &encoded_u_rows_size.to_le_bytes()).unwrap();
        std::io::Write::write(&mut file, &encoded_u_cols_size.to_le_bytes()).unwrap();
        for row in self.u.iter() {
            for el in row.iter() {
                write_g1(&mut file, el, compressed)?;
            }
        }

        // Write the G2 generator, g2^{tau_X}, g2^{tau_Y}
        write_g2(&mut file, &self.g2, compressed)?;
        write_g2(&mut file, &self.g2_tau_x, compressed)?;
        write_g2(&mut file, &self.g2_tau_y, compressed)?;

        Ok(())
    }

    pub fn load_setup_from_file(
        file_path: &str,
        compressed: bool,
    ) -> Result<PianoSettings, String> {
        use std::io::Read;
        use std::sync::Arc;
        fn load_g1(
            reader: &mut std::io::BufReader<std::fs::File>,
            compressed: bool,
        ) -> Result<FsG1, String> {
            if compressed {
                let mut bytes = [0u8; 48];
                reader.read_exact(&mut bytes).unwrap();
                FsG1::from_bytes(&bytes).map_err(|e| e.to_string())
            } else {
                let mut bytes = [0u8; 96];
                reader.read_exact(&mut bytes).unwrap();
                FsG1::deserialize(&bytes).map_err(|e| e.to_string())
            }
        }

        fn load_g2(
            reader: &mut std::io::BufReader<std::fs::File>,
            compressed: bool,
        ) -> Result<FsG2, String> {
            if compressed {
                let mut bytes = [0u8; 96];
                reader.read_exact(&mut bytes).unwrap();
                FsG2::from_bytes(&bytes).map_err(|e| e.to_string())
            } else {
                let mut bytes = [0u8; 192];
                reader.read_exact(&mut bytes).unwrap();
                FsG2::deserialize(&bytes).map_err(|e| e.to_string())
            }
        }

        fn load_g1_array(
            reader: &mut std::io::BufReader<std::fs::File>,
            compressed: bool,
        ) -> Result<Vec<FsG1>, String> {
            const COMPRESSED_SIZE: usize = 48;
            const UNCOMPRESSED_SIZE: usize = 96;
            let mut array_size_bytes = [0u8; 8];
            reader.read_exact(&mut array_size_bytes).unwrap();
            let array_size = u64::from_le_bytes(array_size_bytes) as usize;

            if compressed {
                fn g1_handler(bytes: &[u8; COMPRESSED_SIZE]) -> FsG1 {
                    FsG1::from_bytes(bytes).expect("failed to deserialize G1 element")
                }

                kzg::io_utils::batch_reader::<COMPRESSED_SIZE, FsG1>(
                    reader,
                    array_size,
                    Arc::new(g1_handler),
                    None,
                )
            } else {
                fn g1_handler(bytes: &[u8; UNCOMPRESSED_SIZE]) -> FsG1 {
                    FsG1::deserialize(bytes).expect("failed to deserialize G1 element")
                }

                kzg::io_utils::batch_reader::<UNCOMPRESSED_SIZE, FsG1>(
                    reader,
                    array_size,
                    Arc::new(g1_handler),
                    None,
                )
            }
        }

        fn load_g1_matrix(
            reader: &mut std::io::BufReader<std::fs::File>,
            compressed: bool,
        ) -> Result<Vec<Vec<FsG1>>, String> {
            const COMPRESSED_SIZE: usize = 48;
            const UNCOMPRESSED_SIZE: usize = 96;
            let mut rows_bytes = [0u8; 8];
            reader.read_exact(&mut rows_bytes).unwrap();
            let rows = u64::from_le_bytes(rows_bytes) as usize;

            let mut cols_bytes = [0u8; 8];
            reader.read_exact(&mut cols_bytes).unwrap();
            let cols = u64::from_le_bytes(cols_bytes) as usize;

            if compressed {
                fn g1_handler(bytes: &[u8; COMPRESSED_SIZE]) -> FsG1 {
                    FsG1::from_bytes(bytes).expect("failed to deserialize G1 element")
                }

                kzg::io_utils::batch_reader::<COMPRESSED_SIZE, FsG1>(
                    reader,
                    rows * cols,
                    Arc::new(g1_handler),
                    None,
                )
                .map(|v| v.chunks(cols).map(|c| c.to_vec()).collect())
            } else {
                fn g1_handler(bytes: &[u8; UNCOMPRESSED_SIZE]) -> FsG1 {
                    FsG1::deserialize(bytes).expect("failed to deserialize G1 element")
                }

                kzg::io_utils::batch_reader::<UNCOMPRESSED_SIZE, FsG1>(
                    reader,
                    rows * cols,
                    Arc::new(g1_handler),
                    None,
                )
                .map(|v| v.chunks(cols).map(|c| c.to_vec()).collect())
            }
        }

        let file = std::fs::File::open(file_path).unwrap();
        let mut reader = std::io::BufReader::new(file);
        let (g, g_tau_x, g_tau_y, u, g2, g2_tau_x, g2_tau_y) = (
            load_g1(&mut reader, compressed)?,
            load_g1_array(&mut reader, compressed)?,
            load_g1_array(&mut reader, compressed)?,
            load_g1_matrix(&mut reader, compressed)?,
            load_g2(&mut reader, compressed)?,
            load_g2(&mut reader, compressed)?,
            load_g2(&mut reader, compressed)?,
        );

        Ok(PianoSettings {
            g,
            g_tau_x,
            g_tau_y,
            u,
            g2,
            g2_tau_x,
            g2_tau_y,

            precomputation: PianoPrecomputation::default(),
        })
    }

    pub fn save_precomputation_to_file(
        &self,
        file_path: &str,
        compressed: bool,
    ) -> Result<(), String> {
        crate::utils::timed("Saving precomputation", || {
            self.precomputation.save_to_file(file_path, compressed)
        })
    }

    pub fn load_precomputation_from_file(
        &mut self,
        file_path: &str,
        compressed: bool,
    ) -> Result<(), String> {
        self.precomputation = crate::utils::timed("Loading precomputation", || {
            PianoPrecomputation::load_from_file(file_path, compressed)
        })?;
        Ok(())
    }
}

/// We need to generate the following:
/// 2. two secret field elements tau_X, tau_Y
/// 3. g^{tau_X}
/// 4. g^{tau_Y}
/// 5. U_{i, j} = g^{R_i(tau_Y) * L_j(tau_X)}
/// Then we forget tau_X, tau_Y
pub fn generate_trusted_setup(
    fft_settings: &PianoFFTSettings,
    secrets: [[u8; 32usize]; 2],
) -> PianoSettings {
    // Generate tau_X, tau_Y
    let tau_x = hash_to_bls_field(&secrets[0]);
    let tau_y = hash_to_bls_field(&secrets[1]);

    // G1
    let g = rust_kzg_blst::consts::G1_GENERATOR;

    let sub_circuit_size = 2usize.pow(fft_settings.t() as u32);
    let machine_count = 2usize.pow(fft_settings.m() as u32);

    let g_tau_x = (0..sub_circuit_size)
        .fold((vec![], FsFr::one()), |(mut acc, mut pow), _| {
            acc.push(g.mul(&pow));
            pow = pow.mul(&tau_x);
            (acc, pow)
        })
        .0;
    let g_tau_y = (0..machine_count)
        .fold((vec![], FsFr::one()), |(mut acc, mut pow), _| {
            acc.push(g.mul(&pow));
            pow = pow.mul(&tau_y);
            (acc, pow)
        })
        .0;

    let u = (0..machine_count)
        .map(|i| {
            (0..sub_circuit_size)
                .map(|j| {
                    let r = fft_settings.right_lagrange_poly(i).unwrap();
                    let l = fft_settings.left_lagrange_poly(j).unwrap();
                    let r_tau_y = r.eval(&tau_y);
                    let l_tau_x = l.eval(&tau_x);
                    g.mul(&r_tau_y.mul(&l_tau_x))
                })
                .collect()
        })
        .collect();

    // G2
    let g2 = rust_kzg_blst::consts::G2_GENERATOR;

    let g2_tau_x = g2.mul(&tau_x);
    let g2_tau_y = g2.mul(&tau_y);

    PianoSettings {
        g,
        g_tau_x,
        g_tau_y,
        u,

        g2,
        g2_tau_x,
        g2_tau_y,

        precomputation: PianoPrecomputation::default(),
    }
}

#[derive(Debug, Clone)]
pub struct PianoFFTSettings {
    left: FsFFTSettings,
    right: FsFFTSettings,

    n: usize,
    m: usize,
    t: usize,
}

impl From<&DistributedSetupConfig> for PianoFFTSettings {
    fn from(config: &DistributedSetupConfig) -> Self {
        let m = config.machine_scale;
        let n = config.setup.scale;
        Self::new(n, m).unwrap()
    }
}

impl PianoFFTSettings {
    /// Create a new FFT settings for the Piano protocol
    /// 2**n is the circuit size
    /// 2**m is the number of machines
    pub fn new(n: usize, m: usize) -> Result<PianoFFTSettings, String> {
        if m > n {
            return Err("m must be less than or equal to n".to_string());
        }

        let t = n - m;
        let left = FsFFTSettings::new(t)?;
        let right = FsFFTSettings::new(m)?;

        Ok(PianoFFTSettings {
            left,
            right,
            n,
            m,
            t,
        })
    }

    /// Circuit size
    pub fn n(&self) -> usize {
        self.n
    }

    /// Number of machines
    pub fn m(&self) -> usize {
        self.m
    }

    /// Sub-circuit size
    pub fn t(&self) -> usize {
        self.t
    }

    /// Perform FFT on the left side
    /// If data is not of length 2**t, it will be zero-padded
    pub fn fft_left(&self, data: &[FsFr], inverse: bool) -> Result<Vec<FsFr>, String> {
        self.left.fft_fr(data, inverse)
    }

    /// Perform FFT on the right side
    pub fn fft_right(&self, data: &[FsFr], inverse: bool) -> Result<Vec<FsFr>, String> {
        self.right.fft_fr(data, inverse)
    }

    /// Get the j-th root of unity for the left FFT
    /// omega_X^j = omega_t^j
    pub fn left_expanded_root_of_unity(&self, j: usize) -> FsFr {
        self.left.get_expanded_roots_of_unity_at(j)
    }

    /// Get the i-th root of unity for the right FFT
    /// omega_Y^i = omega_m^i
    pub fn right_expanded_root_of_unity(&self, i: usize) -> FsFr {
        self.right.get_expanded_roots_of_unity_at(i)
    }

    /// Get the j-th lagrange polynomial for the left FFT
    /// Let omega_X be a primitive T-th root of unity
    /// L_j(X) = (omega_X^j / t) * (X^t - 1) / (X - omega_X^j)
    pub fn left_lagrange_poly(&self, i: usize) -> Result<FsPoly, String> {
        let mut coeffs = vec![FsFr::zero(); self.left.get_max_width()];
        coeffs[i] = FsFr::one();
        coeffs = self.fft_left(&coeffs, true)?;
        Ok(FsPoly::from_coeffs(&coeffs))
    }

    /// Get the i-th lagrange polynomial for the right FFT
    /// Let omega_Y be a primitive M-th root of unity
    /// R_i(Y) = (omega_Y^i / M) * (Y^M - 1) / (Y - omega_Y^i)
    pub fn right_lagrange_poly(&self, i: usize) -> Result<FsPoly, String> {
        let mut coeffs = vec![FsFr::zero(); self.right.get_max_width()];
        coeffs[i] = FsFr::one();
        coeffs = self.fft_right(&coeffs, true)?;
        Ok(FsPoly::from_coeffs(&coeffs))
    }
}

#[cfg(test)]
mod tests {
    use crate::bipoly::BivariateFsPolynomial;
    // Polynomial for testing:
    // 1 + xy + x**2 + y**2
    // = (1 + y**2) + x(y) + x**2(1)
    fn small_poly() -> BivariateFsPolynomial {
        let polys = vec![
            FsPoly::from_coeffs(&[FsFr::one(), FsFr::zero(), FsFr::one()]), // 1 + y**2
            FsPoly::from_coeffs(&[FsFr::zero(), FsFr::one()]),              // 1 + y
            FsPoly::from_coeffs(&[FsFr::one()]),                            // 1
        ];
        BivariateFsPolynomial::from_polys(polys)
    }
    use kzg::FFTSettings;

    use super::*;

    #[test]
    #[tracing_test::traced_test]
    fn stupid_test() {
        let tau = FsFr::rand();
        let g = rust_kzg_blst::consts::G1_GENERATOR;
        let g_tau = g.mul(&tau);
        let g_tau_i = |i: usize| {
            if i == 0 {
                return g;
            }
            (0..i).fold(FsG1::default(), |acc, _| acc.add(&g_tau))
        };

        assert_eq!(g_tau_i(0), g);
        assert_eq!(g_tau_i(1), g_tau);
    }

    #[test]
    #[tracing_test::traced_test]
    fn evaluate_constant_poly_test() {
        let poly = FsPoly::from_coeffs(&[FsFr::one(), FsFr::zero()]);
        // constant polynomial f(x, y) = 1
        let bipoly = BivariateFsPolynomial::from_poly_as_x(&poly);
        assert_eq!(
            poly.eval(&FsFr::zero()),
            bipoly.eval(&FsFr::zero(), &FsFr::zero())
        );
        assert_eq!(
            poly.eval(&FsFr::one()),
            bipoly.eval(&FsFr::one(), &FsFr::one())
        );
    }

    #[test]
    #[tracing_test::traced_test]
    fn evaluate_linear_poly_test() {
        // 1 + x
        let poly = FsPoly::from_coeffs(&[FsFr::one(), FsFr::one()]);
        // 1 + x
        let bipoly = BivariateFsPolynomial::from_poly_as_x(&poly);
        assert_eq!(
            poly.eval(&FsFr::zero()),
            bipoly.eval(&FsFr::zero(), &FsFr::zero())
        );
        assert_eq!(
            poly.eval(&FsFr::one()),
            bipoly.eval(&FsFr::one(), &FsFr::zero())
        );
        assert_eq!(
            poly.eval(&FsFr::from_u64(2)),
            bipoly.eval(&FsFr::from_u64(2), &FsFr::zero())
        );
    }

    #[test]
    #[tracing_test::traced_test]
    fn evaluate_poly_test() {
        let poly = small_poly();
        let x = FsFr::from_u64(2);
        let y = FsFr::from_u64(3);
        let result = poly.eval(&x, &y);
        // expected = 2**2 + 2*3 + 3**2 + 1 = 4 + 6 + 9 + 1 = 20
        let expected = x.mul(&x).add(&x.mul(&y)).add(&y.mul(&y)).add(&FsFr::one());
        assert_eq!(result, expected);
    }

    #[test]
    #[tracing_test::traced_test]
    fn test_poly_mul() {
        let coeffs_1 = vec![FsFr::one(), FsFr::one()];
        let coeffs_2 = vec![FsFr::one(), FsFr::one()];
        let poly_1 = FsPoly::from_coeffs(&coeffs_1);
        let poly_2 = FsPoly::from_coeffs(&coeffs_2);

        // (1 + x)(1 + x) = 1 + 2x + x**2
        let result = BivariateFsPolynomial::from_poly_as_x(&poly_1)
            .mul(&BivariateFsPolynomial::from_poly_as_x(&poly_2));
        let expected = BivariateFsPolynomial::from_coeffs(vec![
            vec![FsFr::one()],
            vec![FsFr::from_u64(2)],
            vec![FsFr::one()],
        ]);
        assert_eq!(result.coeffs(), expected.coeffs());

        // (1 + x)(1 + y) = 1 + x + y + xy
        // = (1 + y) + x(1 + y)
        let result = BivariateFsPolynomial::from_poly_as_x(&poly_1)
            .mul(&BivariateFsPolynomial::from_poly_as_y(&poly_2));
        let expected = BivariateFsPolynomial::from_coeffs(vec![
            vec![FsFr::one(), FsFr::one()],
            vec![FsFr::one(), FsFr::one()],
        ]);
        assert_eq!(result.coeffs(), expected.coeffs());

        // (1 + y)(1 + y) = 1 + 2y + y**2
        let result = BivariateFsPolynomial::from_poly_as_y(&poly_1)
            .mul(&BivariateFsPolynomial::from_poly_as_y(&poly_2));
        let expected = BivariateFsPolynomial::from_coeffs(vec![vec![
            FsFr::one(),
            FsFr::from_u64(2),
            FsFr::one(),
        ]]);
        assert_eq!(result.coeffs(), expected.coeffs());
    }

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

    #[test]
    #[tracing_test::traced_test]
    fn roots_of_unity_test() {
        fn gcd(mut n: usize, mut m: usize) -> usize {
            if n == 0 {
                return m;
            }
            if m == 0 {
                return n;
            }
            assert!(n != 0 && m != 0);
            while m != 0 {
                if m < n {
                    std::mem::swap(&mut m, &mut n);
                }
                m %= n;
            }
            n
        }
        const N: usize = 8;
        const M: usize = 4;
        const T: usize = N - M;
        let fft_settings = PianoFFTSettings::new(N, M).unwrap();

        // Left side
        tracing::debug!("testing left side");
        const MAX_ORDER_LEFT: usize = 2usize.pow(T as u32);
        for i in 0..MAX_ORDER_LEFT {
            let root = fft_settings.left.get_expanded_roots_of_unity_at(i);

            // For n-th root of unity w, w^i has order i / gcd(i, n)
            let order = MAX_ORDER_LEFT / gcd(i, MAX_ORDER_LEFT);

            // The n-th root of unity raised to the power of n should be equal to 1
            assert_eq!(root.pow(order), FsFr::one());

            // All powers less than order should not be equal to 1
            for k in 1..order {
                assert_ne!(root.pow(k), FsFr::one());
            }
        }

        // Right side
        const MAX_ORDER_RIGHT: usize = 2usize.pow(M as u32);
        for i in 0..MAX_ORDER_RIGHT {
            let root = fft_settings.right.get_expanded_roots_of_unity_at(i);

            // For n-th root of unity w, w^i has order i / gcd(i, n)
            let order = MAX_ORDER_RIGHT / gcd(i, MAX_ORDER_RIGHT);

            // The n-th root of unity raised to the power of n should be equal to 1
            assert_eq!(root.pow(order), FsFr::one());

            // All powers less than order should not be equal to 1
            for k in 1..order {
                assert_ne!(root.pow(k), FsFr::one());
            }
        }
    }

    #[test]
    #[tracing_test::traced_test]
    fn lagrange_test() {
        const N: usize = 8;
        const M: usize = 4;
        const T: usize = N - M;

        let fft_settings = PianoFFTSettings::new(N, M).unwrap();

        // Left side
        tracing::debug!("testing left side");
        for i in 0..2usize.pow(T as u32) {
            const MAX_ORDER: usize = 2usize.pow(T as u32);
            tracing::debug!("testing lagrange poly at {}", i);

            // Construct i-th lagrange poly
            let mut coeffs = vec![FsFr::zero(); MAX_ORDER];
            coeffs[i] = FsFr::one();
            let lagrange = FsPoly::from_coeffs(&fft_settings.left.fft_fr(&coeffs, true).unwrap());

            // Check that the i-th lagrange poly is correct
            // L_i(omega^j) = 1 if i == j, 0 otherwise
            for k in 0..MAX_ORDER {
                let root = fft_settings.left.get_expanded_roots_of_unity_at(k);
                if k == i {
                    assert_eq!(lagrange.eval(&root), FsFr::one());
                } else {
                    assert_eq!(lagrange.eval(&root), FsFr::zero());
                }
            }
        }

        // Right side
        tracing::debug!("testing right side");
        for i in 0..2usize.pow(M as u32) {
            const MAX_ORDER: usize = 2usize.pow(M as u32);
            tracing::debug!("testing lagrange poly at {}", i);

            // Construct i-th lagrange poly
            let mut coeffs = vec![FsFr::zero(); MAX_ORDER];
            coeffs[i] = FsFr::one();
            let lagrange = FsPoly::from_coeffs(&fft_settings.right.fft_fr(&coeffs, true).unwrap());

            // Check that the i-th lagrange poly is correct
            // L_i(omega^j) = 1 if i == j, 0 otherwise
            for k in 0..MAX_ORDER {
                let root = fft_settings.right.get_expanded_roots_of_unity_at(k);
                if k == i {
                    assert_eq!(lagrange.eval(&root), FsFr::one());
                } else {
                    assert_eq!(lagrange.eval(&root), FsFr::zero());
                }
            }
        }
    }

    #[test]
    #[tracing_test::traced_test]
    fn test_verify_default() {
        const N: usize = 8;
        const M: usize = 1;
        let backend = PianoBackend::new(Some(DistributedBackendConfig {
            backend: crate::engine::config::BackendConfig {
                scale: N,
                ..Default::default()
            },
            machine_scale: M,
        }));

        let i = 0;
        let commitment = FsG1::default();
        let alpha = FsFr::rand();
        let eval = FsFr::zero();
        let proof = FsG1::default();

        assert!(backend.verify_single(i, &commitment, &alpha, &eval, &proof));
    }


    #[test]
    #[tracing_test::traced_test]
    fn manual_commit_test() {
        const N: usize = 8;
        const MACHINES: usize = 4;
        const M: usize = MACHINES;

        let coeffs = generate_coeffs(N, M);
        let polys = coeffs
            .iter()
            .map(|coeffs| FsPoly::from_coeffs(coeffs))
            .collect::<Vec<_>>();

        let fft_settings = PianoFFTSettings::new(N, M).unwrap();
        let cfg = DistributedBackendConfig {
            backend: crate::engine::config::BackendConfig {
                scale: N,
                ..Default::default()
            },
            machine_scale: MACHINES,
        };
        let backend = PianoBackend::new(cfg.into());

        // Commit to the polynomial using the backend
        let commitments = (0..M)
            .map(|i| backend.commit(i, &polys[i]).unwrap())
            .collect::<Vec<_>>();

        // Compute commitment manually
        // Commitment = prod U_{0, j}^{f_j} where f_j is the j-th coefficient of the polynomial in the lagrange basis

        for i in 0..M {
            let cob_coeffs = fft_settings.fft_left(&coeffs[i], false).unwrap();
            let result = cob_coeffs
                .iter()
                .enumerate()
                .fold(FsG1::default(), |acc, (j, c)| {
                    acc.add(&backend.piano_settings.u(i, j).mul(c))
                });
            assert_eq!(commitments[i], result);
        }

        // Compute master commitment
        let master_commitment = backend.master_commit(&commitments);

        // Compute master commitment manually
        let master_result = commitments
            .iter()
            .fold(FsG1::default(), |acc, c| acc.add(c));

        assert_eq!(master_commitment, master_result);
    }

    #[test]
    #[tracing_test::traced_test]
    fn test_piano_setup() {
        let m = 3;
        let n = 5;
        let secrets = [[0u8; 32usize], [1u8; 32usize]];
        let fft_settings = PianoFFTSettings::new(n, m).unwrap();
        let setup = generate_trusted_setup(&fft_settings, secrets);

        let t = n - m;
        for i in 0..m {
            for j in 0..t {
                println!("U[{}, {}] = {:?}", i, j, setup.u(i, j));
            }
        }
    }

    #[test]
    #[tracing_test::traced_test]
    fn small_poly_reconstruction_test() {
        let lagrange_coeffs = [
            vec![FsFr::one(), FsFr::zero()],
            vec![FsFr::zero(), FsFr::one()],
        ];

        let fft_settings = PianoFFTSettings::new(2, 1).unwrap();
        let mut f = BivariateFsPolynomial::zero();
        for (i, row) in lagrange_coeffs.iter().enumerate() {
            for (j, c) in row.iter().enumerate() {
                let left = fft_settings.left_lagrange_poly(j).unwrap();
                let right = fft_settings.right_lagrange_poly(i).unwrap();
                let mut term = BivariateFsPolynomial::from_poly_as_x(&left)
                    .mul(&BivariateFsPolynomial::from_poly_as_y(&right));
                term = term.scale(c);
                f = f.add(&term);
            }
        }

        let alpha = FsFr::rand();
        let beta = FsFr::rand();

        tracing::debug!("f = {:?}", f);
        tracing::debug!("f(alpha, Y) = {:?}", f.eval_x(&alpha));

        let f_1 = FsPoly::from_coeffs(&fft_settings.fft_left(&lagrange_coeffs[0], true).unwrap());
        let f_2 = FsPoly::from_coeffs(&fft_settings.fft_left(&lagrange_coeffs[1], true).unwrap());
        tracing::debug!("f_1 = {:?}", f_1);
        tracing::debug!("f_2 = {:?}", f_2);

        let eval_1 = f_1.eval(&alpha);
        let eval_2 = f_2.eval(&alpha);

        let reconstructed =
            FsPoly::from_coeffs(&fft_settings.fft_right(&[eval_1, eval_2], true).unwrap());
        tracing::debug!("reconstructed = {:?}", reconstructed);
        let z = reconstructed.eval(&beta);
        let z_manual = f.eval(&alpha, &beta);
        tracing::debug!("z = {:?}, z_manual = {:?}", z, z_manual);
        assert_eq!(z, z_manual);
    }

    #[test]
    #[tracing_test::traced_test]
    fn poly_reconstruction_test() {
        let n = 4;
        let m = 2;
        let t = n - m;

        let fft_settings = PianoFFTSettings::new(n, m).unwrap();

        let lagrange_coeffs = generate_coeffs(n, m);
        let mut f = BivariateFsPolynomial::zero();
        (0..2usize.pow(m as u32)).for_each(|i| {
            (0..2usize.pow(t as u32)).for_each(|j| {
                let left = fft_settings.left_lagrange_poly(j).unwrap();
                let right = fft_settings.right_lagrange_poly(i).unwrap();
                let mut term = BivariateFsPolynomial::from_poly_as_x(&left)
                    .mul(&BivariateFsPolynomial::from_poly_as_y(&right));
                term = term.scale(&lagrange_coeffs[i][j]);
                f = f.add(&term);
            });
        });

        let sub_polynomials = (0..2usize.pow(m as u32))
            .map(|i| {
                let coeffs = lagrange_coeffs[i].clone();
                FsPoly::from_coeffs(&fft_settings.fft_left(&coeffs, true).unwrap())
            })
            .collect::<Vec<_>>();

        let alpha = FsFr::rand();
        let beta = FsFr::rand();

        let evals = sub_polynomials
            .iter()
            .map(|p| p.eval(&alpha))
            .collect::<Vec<_>>();

        let reconstructed = FsPoly::from_coeffs(&fft_settings.fft_right(&evals, true).unwrap());
        let z = reconstructed.eval(&beta);
        let z_manual = f.eval(&alpha, &beta);
        assert_eq!(z, z_manual);
    }

    #[test]
    #[tracing_test::traced_test]
    fn test_kzg_proof_linearity() {
        use crate::engine::backend::Backend;
        use crate::engine::blst::BlstBackend;
        const SCALE: usize = 5;
        let cfg = crate::engine::config::BackendConfig::new(None, None, SCALE, None, None);
        let backend = BlstBackend::new(Some(cfg));

        let coeffs = generate_coeffs(SCALE, 0);
        let poly = FsPoly::from_coeffs(&coeffs[0]);

        tracing::debug!("normal verification");
        let x = FsFr::rand();
        let y = poly.eval(&x);
        let commitment = backend.commit_to_poly(poly.clone()).unwrap();
        let proof = backend.compute_proof_single(poly.clone(), x).unwrap();
        let verify = backend
            .verify_proof_single(proof, x, y, commitment)
            .unwrap();
        assert!(verify);
        tracing::debug!("normal verification OK");

        tracing::debug!("scaled verification");
        let scalar = FsFr::rand();
        let new_y = y.mul(&scalar);
        let mut new_poly = poly.clone();
        let new_poly = new_poly
            .mul(&FsPoly::from_coeffs(&[scalar, FsFr::zero()]), poly.len())
            .unwrap();
        let new_commitment = backend.commit_to_poly(new_poly.clone()).unwrap();
        let new_proof = backend.compute_proof_single(new_poly, x).unwrap();
        let new_verify = backend
            .verify_proof_single(new_proof, x, new_y, new_commitment)
            .unwrap();
        assert!(new_verify);
        tracing::debug!("scaled verification OK");
    }

    #[test]
    #[tracing_test::traced_test]
    // TODO: this only works for M = 1
    fn pianist_test() {
        for n in 2..8 {
            for m in 1..n - 1 {
                tracing::debug!("Running pianist test for n = {}, m = {}", n, m);
                let result = pianist(n, m);
                assert!(result.is_ok());
            }
        }

        /// Demonstrates the Pianist protocol
        /// N = 2^n circuit size
        /// M = 2^m number of machines
        /// T = M / N
        /// t = n - m
        fn pianist(n: usize, m: usize) -> Result<(), String> {
            // Basic variables
            let t = n - m;
            let machines_count = 2usize.pow(m as u32);
            let sub_circuit_size = 2usize.pow(t as u32);

            tracing::debug!(
                "Piano setup: circuit size = {}, machines = {}",
                2usize.pow(n as u32),
                2usize.pow(m as u32)
            );

            // Start one backend
            let cfg = DistributedBackendConfig {
                backend: crate::engine::config::BackendConfig {
                    scale: n,
                    ..Default::default()
                },
                machine_scale: m,
            };
            let backend = PianoBackend::new(cfg.into());

            // Generate the polynomial we'll be working with in the standard basis
            // f(x, y) = sum_{i=0}^{n} sum_{j=0}^{m} f_{i,j} L(j) R(i)
            let lagrange_coeffs = generate_coeffs(n, m);
            tracing::debug!("Lagrange coeffs size = {}", lagrange_coeffs.len());
            let mut f = BivariateFsPolynomial::zero();
            (0..machines_count).for_each(|i| {
                (0..sub_circuit_size).for_each(|j| {
                    let left = backend.fft_settings.left_lagrange_poly(j).unwrap();
                    let right = backend.fft_settings.right_lagrange_poly(i).unwrap();
                    let mut term = BivariateFsPolynomial::from_poly_as_x(&left)
                        .mul(&BivariateFsPolynomial::from_poly_as_y(&right));
                    term = term.scale(&lagrange_coeffs[i][j]);
                    f = f.add(&term);
                });
            });

            // Compute sub-polynomials in standard basis
            let polynomials = (0..machines_count)
                .map(|i| {
                    let coeffs = lagrange_coeffs[i].clone();
                    FsPoly::from_coeffs(&backend.fft_settings.fft_left(&coeffs, true).unwrap())
                })
                .collect::<Vec<_>>();

            // WORKER NODES
            // Commitments
            tracing::debug!("Committing...");
            let commitments = (0..machines_count)
                .map(|i| backend.commit(i, &polynomials[i]).unwrap())
                .collect::<Vec<_>>();

            // Openings
            tracing::debug!("Opening...");
            let alpha = FsFr::rand();
            let beta = FsFr::rand();
            let proofs = (0..machines_count)
                .map(|i| backend.open(i, &polynomials[i], &alpha).unwrap())
                .collect::<Vec<_>>();

            // MASTER NODE
            // Verify each individual commitment and opening before proceeding
            (0..machines_count).for_each(|i| {
                let commitment = commitments[i];
                let (y, pi_0) = proofs[i];
                tracing::debug!("Checking proof for machine {}", i);
                let verify = backend.verify_single(i, &commitment, &alpha, &y, &pi_0);
                if !verify {
                    tracing::error!("Verification failed for machine {}", i);
                } else {
                    tracing::debug!("Verification OK for machine {}", i);
                }
            });

            // Compute Master Commitment
            let master_commitment = backend.master_commit(&commitments);

            // Compute Master Opening
            let (evals, proofs) = proofs.iter().fold(
                (Vec::new(), Vec::new()),
                |(mut evals, mut proofs), (y, pi)| {
                    evals.push(*y);
                    proofs.push(*pi);
                    (evals, proofs)
                },
            );

            let (z, pi_f) = backend.master_open(&evals, &proofs, &beta)?;

            // NOTE: We check the reconstruction here as a sanity check
            // Check z manually, z = f(alpha, beta)
            let z_manual = f.eval(&alpha, &beta);
            assert_eq!(z, z_manual);
            tracing::debug!("Reconstruction OK!");

            // Master Verification
            tracing::debug!("Verifying...");
            let result = backend.verify(&master_commitment, &beta, &alpha, &z, &pi_f);
            if result {
                Ok(())
            } else {
                Err("Verification failed".to_string())
            }
        }
    }

    #[test]
    #[tracing_test::traced_test]
    fn partial_commit_test() {
        const N: usize = 8;
        const M: usize = 2;

        // This test demonstrates that a trusted setup with parameters N and M can also function
        // when M' < M machines are active. 
        // We can still generate commitments provided the circuit size is smaller than N' = M' + (N - M).
        // All that needs to happen is that the data sent to the inactive machines is zeroes out,
        // so that the individual commitments and proofs become the identity element and therefore
        // trivially valid.

        const ACTIVE_MACHINES: [usize; 2] = [0, 1];

        let backend = PianoBackend::new(Some(DistributedBackendConfig {
            backend: crate::engine::config::BackendConfig {
                scale: N,
                ..Default::default()
            },
            machine_scale: M,
        }));

        let coeffs = generate_coeffs(N, M);

        let commitments = coeffs
            .iter()
            .enumerate()
            .map(|(i, coeffs)| {
                if ACTIVE_MACHINES.contains(&i) {
                    FsG1::default()
                } else {
                    let poly = FsPoly::from_coeffs(coeffs);
                    backend.commit(i, &poly).unwrap()
                }
            })
            .collect::<Vec<_>>();


        let master_commitment = backend.master_commit(&commitments);

        let alpha = FsFr::rand();
        let beta = FsFr::rand();


        let proofs = (0..2usize.pow(M as u32))
            .map(|i| {
                if ACTIVE_MACHINES.contains(&i) {
                    (FsFr::zero(), FsG1::default())
                } else {
                    tracing::debug!("opening machine {}", i);
                    let poly = FsPoly::from_coeffs(&coeffs[i]);
                    backend.open(i, &poly, &alpha).unwrap()
                }
            })
            .collect::<Vec<_>>();


        // verify that the commitments are correct
        for (i, (commitment, (eval, proof))) in commitments.iter().zip(proofs.iter()).enumerate() {
            assert!(backend.verify_single(i, commitment, &alpha, eval, proof))
        }

        let (evals, proofs): (Vec<FsFr>, Vec<FsG1>) = proofs.iter().cloned().unzip();

        let (z, pi_f) = backend.master_open(&evals, &proofs, &beta).unwrap();

        assert!(backend.verify(&master_commitment, &beta, &alpha, &z, &pi_f));

    }

    #[test]
    #[tracing_test::traced_test]
    fn test_save_and_load_setup() {
        const N: usize = 8;
        const M: usize = 2;
        const FILENAME: &str = "test_piano_setup.json";
        std::fs::remove_file(FILENAME).unwrap_or_default();
        fn assert_eq(a: &PianoSettings, b: &PianoSettings) {
            assert_eq!(a.g, b.g);
            assert_eq!(a.g_tau_x, b.g_tau_x);
            assert_eq!(a.g_tau_y, b.g_tau_y);
            assert_eq!(a.u, b.u);
            assert_eq!(a.g2, b.g2);
            assert_eq!(a.g2_tau_x, b.g2_tau_x);
            assert_eq!(a.g2_tau_y, b.g2_tau_y);
        }
        fn test_save_load(backend: &PianoBackend, compressed: bool) {
            crate::utils::timed("Saving Setup", || {
                assert!(backend
                    .piano_settings
                    .save_setup_to_file(FILENAME, compressed)
                    .is_ok());
            });
            let loaded = crate::utils::timed("Loading Setup", || {
                PianoSettings::load_setup_from_file(FILENAME, compressed).unwrap()
            });
            assert_eq(&backend.piano_settings, &loaded);
        }

        // Generate a new backend
        let cfg = DistributedSetupConfig {
            setup: crate::engine::config::SetupConfig {
                scale: N,
                generate_setup: true,
                ..Default::default()
            },
            machine_scale: M,
        };
        let backend = PianoBackend::setup(cfg).unwrap();

        test_save_load(&backend, true);
        test_save_load(&backend, false);

        std::fs::remove_file(FILENAME).unwrap();
    }

    #[test]
    #[tracing_test::traced_test]
    fn test_save_and_load_precompute() {
        const N: usize = 8;
        const M: usize = 2;
        const FILENAME: &str = "test_piano_precompute.json";
        std::fs::remove_file(FILENAME).unwrap_or_default();
        fn test_save_load(
            backend: &PianoBackend,
            compressed: bool,
            polys: &[FsPoly],
            commitments: &[FsG1],
        ) {
            crate::utils::timed("Saving Precomputations", || {
                assert!(backend
                    .piano_settings
                    .save_precomputation_to_file(FILENAME, compressed)
                    .is_ok());
            });
            let loaded = crate::utils::timed("Loading Precomputations", || {
                PianoPrecomputation::load_from_file(FILENAME, compressed).unwrap()
            });
            let mut new_backend = backend.clone();
            new_backend.piano_settings.set_precomputation(loaded);

            let new_commitments =
                crate::utils::timed(format!("Committing {} polynomials", M).as_str(), || {
                    (0..M)
                        .map(|i| new_backend.commit(i, &polys[i]).unwrap())
                        .collect::<Vec<_>>()
                });

            crate::utils::timed(format!("Checking {} commitments", M).as_str(), || {
                new_commitments
                    .iter()
                    .zip(commitments.iter())
                    .for_each(|(a, b)| {
                        assert_eq!(a, b);
                    });
            });
        }

        let cfg = DistributedSetupConfig {
            setup: crate::engine::config::SetupConfig {
                generate_precompute: true,
                generate_setup: true,
                scale: N,
                ..Default::default()
            },
            machine_scale: M,
        };
        let backend = PianoBackend::setup(cfg).unwrap();

        let coeffs = generate_coeffs(N, M);
        let polynomials = coeffs
            .iter()
            .map(|coeffs| FsPoly::from_coeffs(coeffs))
            .collect::<Vec<_>>();
        let commitments = (0..M)
            .map(|i| backend.commit(i, &polynomials[i]).unwrap())
            .collect::<Vec<_>>();

        test_save_load(&backend, false, &polynomials, &commitments);

        std::fs::remove_file(FILENAME).unwrap();
    }
}

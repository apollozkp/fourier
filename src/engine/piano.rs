use kzg::eip_4844::hash_to_bls_field;
use kzg::FFTFr;
use kzg::FFTSettings;
use kzg::Fr;
use kzg::G1Mul;
use kzg::G2Mul;
use kzg::Poly;
use kzg::G1;
use kzg::G2;
use rust_kzg_blst::types::fft_settings::FsFFTSettings;
use rust_kzg_blst::types::fr::FsFr;
use rust_kzg_blst::types::g1::FsG1;
use rust_kzg_blst::types::g2::FsG2;
use rust_kzg_blst::types::poly::FsPoly;

fn polysum(poly: &mut FsPoly, other: &FsPoly) {
    if poly.len() < other.len() {
        poly.coeffs.resize(other.len(), FsFr::zero());
    }
    other.coeffs.iter().enumerate().for_each(|(i, c)| {
        poly.coeffs[i] = poly.coeffs[i].add(c);
    });
}

fn polymul_and_sum(poly: &mut FsPoly, other: &FsPoly, x: &FsFr) {
    let mut result = other.clone();
    result.coeffs.iter_mut().for_each(|c| *c = c.mul(x));
    polysum(poly, &result);
}

fn polyeval(poly: &FsPoly, x: &FsFr) -> FsFr {
    if poly.len() == 0 {
        return FsFr::zero();
    }
    if poly.len() == 1 {
        return poly.coeffs[0];
    }
    poly.eval(x)
}

/// A bivariate polynomial in the form:
/// f(x, y) = sum_{i=0}^{n} x^i sum_{j=0}^{m} a_{i,j} y^j
/// where a_{i,j} are the coefficients of the polynomial.
/// The coefficients are stored in a 2D vector, where the first index is the
/// power of x and the second index is the power of y.
#[derive(Debug)]
pub struct BivariateFsPolynomial {
    parts: Vec<FsPoly>,
}

impl BivariateFsPolynomial {
    pub fn eval(&self, x: &FsFr, y: &FsFr) -> FsFr {
        polyeval(&self.eval_x(x), y)
    }

    pub fn eval_x(&self, x: &FsFr) -> FsPoly {
        let max_degree = self.parts.iter().map(|p| p.len()).max().unwrap();
        let mut result = FsPoly::from_coeffs(&vec![FsFr::zero(); max_degree]);
        let mut pow = FsFr::one();
        for f in self.parts.iter() {
            polymul_and_sum(&mut result, f, &pow);
            pow = pow.mul(x);
        }
        result
    }

    pub fn eval_y(&self, y: &FsFr) -> FsPoly {
        FsPoly::from_coeffs(&self.parts.iter().map(|f| f.eval(y)).collect::<Vec<FsFr>>())
    }

    pub fn from_poly_as_x(poly: &FsPoly) -> BivariateFsPolynomial {
        BivariateFsPolynomial::from_coeffs(poly.coeffs.iter().map(|c| vec![*c]).collect())
    }

    pub fn from_poly_as_y(poly: &FsPoly) -> BivariateFsPolynomial {
        BivariateFsPolynomial::from_polys(vec![poly.clone()])
    }

    pub fn from_coeffs(coeffs: Vec<Vec<FsFr>>) -> BivariateFsPolynomial {
        BivariateFsPolynomial {
            parts: coeffs.iter().map(|c| FsPoly::from_coeffs(c)).collect(),
        }
    }

    pub fn from_polys(polys: Vec<FsPoly>) -> BivariateFsPolynomial {
        BivariateFsPolynomial { parts: polys }
    }

    pub fn mul(&self, other: &BivariateFsPolynomial) -> BivariateFsPolynomial {
        let mut result =
            vec![FsPoly::from_coeffs(&[FsFr::zero()]); self.parts.len() + other.parts.len() - 1];
        for (i, f) in self.parts.iter().enumerate() {
            for (j, g) in other.parts.iter().enumerate() {
                let mut h = f.clone();
                let degree = h.len() + g.len() - 1;
                h = h.mul(g, degree).unwrap();
                polysum(&mut result[i + j], &h);
            }
        }
        BivariateFsPolynomial::from_coeffs(result.iter().map(|p| p.coeffs.clone()).collect())
    }

    pub fn add(&self, other: &BivariateFsPolynomial) -> BivariateFsPolynomial {
        let size = self.parts.len().max(other.parts.len());
        let mut result = vec![FsPoly::from_coeffs(&[FsFr::zero()]); size];
        for (i, f) in self.parts.iter().enumerate() {
            polysum(&mut result[i], f);
        }
        for (i, f) in other.parts.iter().enumerate() {
            polysum(&mut result[i], f);
        }
        BivariateFsPolynomial::from_coeffs(result.iter().map(|p| p.coeffs.clone()).collect())
    }

    pub fn coeffs(&self) -> Vec<Vec<FsFr>> {
        self.parts.iter().map(|p| p.coeffs.clone()).collect()
    }

    pub fn scale(&self, scalar: &FsFr) -> BivariateFsPolynomial {
        BivariateFsPolynomial::from_coeffs(
            self.parts
                .iter()
                .map(|p| {
                    let coeffs = p.coeffs.iter().map(|c| c.mul(scalar)).collect();
                    coeffs
                })
                .collect(),
        )
    }

    pub fn zero() -> BivariateFsPolynomial {
        BivariateFsPolynomial::from_coeffs(vec![vec![FsFr::zero()]])
    }
}

// Suppose we have M = 2**m machines and circuit size N = 2**n
// Let M * T = N, and t = n - m
// Let omega_X be an T-th root of unity
// Let omega_Y be an M-th root of unity
// Let L_j(x) = (omega_X^j / T) * (X^T - 1) / (X - omega_X^j) (Lagrange poly)
// Let R_i(y) = (omega_Y^i / M) * (Y^M - 1) / (Y - omega_Y^i) (Lagrange poly)
// Let g be a generator of G1
//
// We need to generate the following:
// 2. two secret field elements tau_X, tau_Y
// 3. g^{tau_X}
// 4. g^{tau_Y}
// 5. U_{i, j} = g^{R_i(tau_Y) * L_j(tau_X)}
// Then we forget tau_X, tau_Y

#[derive(Debug)]
pub struct PianoSettings {
    g: FsG1,
    g_tau_x: FsG1,
    g_tau_y: FsG1,
    u: Vec<Vec<FsG1>>,

    g2: FsG2,
    g2_tau_x: FsG2,
    g2_tau_y: FsG2,
}

impl PianoSettings {
    pub fn g(&self) -> FsG1 {
        self.g
    }

    /// Get g^{tau_X^i}
    pub fn g_tau_x(&self, i: usize) -> FsG1 {
        if i == 0 {
            return self.g();
        }
        (0..i).fold(self.g_tau_x, |acc, _| acc.add(&self.g_tau_x))
    }

    /// Get g^{tau_Y^i}
    pub fn g_tau_y(&self, i: usize) -> FsG1 {
        if i == 0 {
            return self.g();
        }
        (0..i).fold(self.g_tau_y, |acc, _| acc.add(&self.g_tau_y))
    }

    // TODO: out of bounds check
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
}

pub fn generate_trusted_setup(
    fft_settings: &PianoFFTSettings,
    secrets: [[u8; 32usize]; 2],
) -> PianoSettings {
    let tau_x = hash_to_bls_field(&secrets[0]);
    let tau_y = hash_to_bls_field(&secrets[1]);
    let g = rust_kzg_blst::consts::G1_GENERATOR;
    let g2 = rust_kzg_blst::consts::G2_GENERATOR;

    let sub_circuit_size = 2usize.pow(fft_settings.t() as u32);
    let machine_count = 2usize.pow(fft_settings.m() as u32);

    let g_tau_x = g.mul(&tau_x);
    let g_tau_y = g.mul(&tau_y);

    let g2_tau_x = g2.mul(&tau_x);
    let g2_tau_y = g2.mul(&tau_y);

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

    PianoSettings {
        g,
        g_tau_x,
        g_tau_y,
        u,

        g2,
        g2_tau_x,
        g2_tau_y,
    }
}

pub struct PianoFFTSettings {
    left: FsFFTSettings,
    right: FsFFTSettings,

    n: usize,
    m: usize,
    t: usize,
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
        tracing::debug!("initializing FFT settings, left = {}, right = {}", t, m);
        let left = FsFFTSettings::new(t)?;
        tracing::debug!("left max width {:?}", left.get_max_width());
        let right = FsFFTSettings::new(m)?;
        tracing::debug!("right max width {:?}", right.get_max_width());

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

pub struct PianoBackend {
    fft_settings: PianoFFTSettings,
    piano_settings: PianoSettings,
}

impl PianoBackend {
    pub fn new(n: usize, m: usize, secrets: &[[u8; 32usize]; 2]) -> PianoBackend {
        let fft_settings = PianoFFTSettings::new(n, m).unwrap();
        let piano_settings = generate_trusted_setup(&fft_settings, *secrets);
        PianoBackend {
            fft_settings,
            piano_settings,
        }
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
            None,
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
    /// NOTE: We are opeining R_i(tau_Y) * f_i(X), not f_i(X), but we do supply f_i(alpha)
    pub fn open(&self, i: usize, poly: &FsPoly, alpha: &FsFr) -> Result<(FsFr, FsG1), String> {
        tracing::debug!("poly coeffs len = {}", poly.coeffs.len());
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
            None,
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
        let pi0 = proofs
            .iter()
            .fold(None, |acc, pi| {
                acc.map_or(Some(*pi), |acc: FsG1| Some(acc.add(pi)))
            })
            .unwrap();

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
        let pi1 = q
            .coeffs
            .iter()
            .enumerate()
            .fold(None, |acc, (j, c)| {
                acc.map_or(Some(self.piano_settings.g_tau_y(j).mul(c)), |acc: FsG1| {
                    Some(acc.add(&self.piano_settings.g_tau_y(j).mul(c)))
                })
            })
            .unwrap();

        Ok((z, (pi0, pi1)))
    }

    /// Verify a single opening of a polynomial f_i(X) from machine i
    /// Our commitment and proof are actually of f'(X) = g^{R_i(tau_Y) * f_i(X)}
    /// But y = f_i(alpha)
    /// We need y' = f'(alpha) = g^{R_i(tau_Y) * f_i(alpha)}
    /// We have g^{tau_Y}, so we can compute g^{R_i(tau_Y)} and then g^{R_i(tau_Y) * y}
    /// as g^{R_i(tau_Y)}^{y}
    /// Then we validate that e(commitment - y', g) == e(pi, g^{tau_X - alpha})
    pub fn verify_single(
        &self,
        i: usize,
        commitment: &FsG1,
        alpha: &FsFr,
        y: &FsFr,
        pi: &FsG1,
    ) -> bool {
        // Compute y' = g^{R_i(tau_Y) * y}
        let right_lagrange_poly = self.fft_settings.right_lagrange_poly(i).unwrap();
        let eval = right_lagrange_poly
            .coeffs
            .iter()
            .enumerate()
            .fold(None, |acc, (j, c)| {
                let term = self.piano_settings.g_tau_y(j).mul(c);
                acc.map_or(Some(term), |a: FsG1| Some(a.add(&term)))
            })
            .unwrap();
        let g_y_prime = eval.mul(y);

        // Compute g^{R_i(tau_Y) * f_i(tau_X) - y'}
        let commitment_minus_y_prime = commitment.sub(&g_y_prime);

        // Compute g^{tau_X - alpha}
        let tau_minus_alpha = self
            .piano_settings
            .g2_tau_x()
            .sub(&self.piano_settings.g2().mul(alpha));

        Self::pairings_verify_single(
            &commitment_minus_y_prime,
            &self.piano_settings.g2(),
            pi,
            &tau_minus_alpha,
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
        let commitment_minux_z = commitment.sub(&self.piano_settings.g.mul(z));
        let tau_x_minus_alpha = self
            .piano_settings
            .g2_tau_x()
            .sub(&self.piano_settings.g2().mul(alpha));
        let tau_y_minus_beta = self
            .piano_settings
            .g2_tau_y()
            .sub(&self.piano_settings.g2().mul(beta));

        Self::pairings_verify(
            &commitment_minux_z,
            &self.piano_settings.g2(),
            &pi_f.0,
            &tau_x_minus_alpha,
            &pi_f.1,
            &tau_y_minus_beta,
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

#[cfg(test)]
mod tests {
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

    use crate::engine::blst::BlstBackend;

    use super::*;

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
        let M = 2usize.pow(m as u32);
        let T = 2usize.pow(t as u32);
        let mut coeffs = vec![vec![FsFr::zero(); T]; M];
        (0..M).for_each(|i| {
            (0..T).for_each(|j| {
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
        const SCALE: usize = 2;
        const MAX_ORDER: usize = 2usize.pow(SCALE as u32);
        let fft_settings = PianoFFTSettings::new(SCALE, 0).unwrap();
        for i in 0..MAX_ORDER {
            let root = fft_settings.left.get_expanded_roots_of_unity_at(i);

            // For n-th root of unity w, w^i has order i / gcd(i, n)
            let order = MAX_ORDER / gcd(i, MAX_ORDER);

            // The n-th root of unity raised to the power of n should be equal to 1
            assert_eq!(root.pow(order), FsFr::one());

            // All powers less than order should not be equal to 1
            for j in 1..order {
                assert_ne!(root.pow(j), FsFr::one());
            }
        }
    }

    #[test]
    #[tracing_test::traced_test]
    fn lagrange_test() {
        const SCALE: usize = 2;

        let fft_settings = PianoFFTSettings::new(SCALE, 0).unwrap();

        for i in 0..2usize.pow(SCALE as u32) {
            tracing::debug!("testing lagrange poly at {}", i);

            // Construct i-th lagrange poly
            let mut coeffs = vec![FsFr::zero(); 2usize.pow(SCALE as u32)];
            coeffs[i] = FsFr::one();
            let lagrange = FsPoly::from_coeffs(&fft_settings.left.fft_fr(&coeffs, true).unwrap());

            // Check that the i-th lagrange poly is correct
            // L_i(omega^j) = 1 if i == j, 0 otherwise
            for j in 0..2usize.pow(SCALE as u32) {
                let root = fft_settings.left.get_expanded_roots_of_unity_at(j);
                if i == j {
                    assert_eq!(lagrange.eval(&root), FsFr::one());
                } else {
                    assert_eq!(lagrange.eval(&root), FsFr::zero());
                }
            }
        }
    }

    #[test]
    #[tracing_test::traced_test]
    fn manual_commit_test() {
        const N: usize = 6;
        const MACHINES: usize = 2;
        const T: usize = N - MACHINES;
        const M: usize = MACHINES;

        let coeffs = generate_coeffs(N, M);
        let polys = coeffs
            .iter()
            .map(|coeffs| FsPoly::from_coeffs(coeffs))
            .collect::<Vec<_>>();

        let fft_settings = PianoFFTSettings::new(N, M).unwrap();
        let backend = PianoBackend::new(N, M, &[[0u8; 32usize], [1u8; 32usize]]);

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

    /// Demonstrate the Pianist protocol
    /// N = 2^n circuit size
    /// M = 2^m number of machines
    /// T = M / N
    /// t = n - m
    fn pianist(n: usize, m: usize) -> Result<(), String> {
        use std::sync::Arc;

        // Basic variables
        let t = n - m;
        let M = 2usize.pow(m as u32);
        let T = 2usize.pow(t as u32);

        struct Machine {
            id: usize,
            backend: Arc<PianoBackend>,
        }

        impl Machine {
            fn commit(&self, poly: &FsPoly) -> Result<FsG1, String> {
                tracing::debug!("Machine {} committing", self.id);
                self.backend.commit(self.id, poly)
            }

            fn prove(&self, poly: &FsPoly, x: &FsFr) -> Result<(FsFr, FsG1), String> {
                tracing::debug!("Machine {} proving", self.id);
                self.backend.open(self.id, poly, x)
            }
        }

        tracing::debug!(
            "Piano setup: circuit size = {}, machines = {}",
            2usize.pow(n as u32),
            2usize.pow(m as u32)
        );

        let fft_settings = PianoFFTSettings::new(n, m).unwrap();

        let lagrange_coeffs = generate_coeffs(n, m);
        tracing::debug!("Lagrange coeffs size = {}", lagrange_coeffs.len());
        let mut f = BivariateFsPolynomial::zero();
        (0..M).for_each(|i| {
            (0..T).for_each(|j| {
                let left = fft_settings.left_lagrange_poly(j).unwrap();
                let right = fft_settings.right_lagrange_poly(i).unwrap();
                let mut term = BivariateFsPolynomial::from_poly_as_x(&left)
                    .mul(&BivariateFsPolynomial::from_poly_as_y(&right));
                term = term.scale(&lagrange_coeffs[i][j]);
                f = f.add(&term);
            });
        });

        // Start one backend
        let backend = Arc::new(PianoBackend::new(n, m, &[[0u8; 32usize], [1u8; 32usize]]));

        // Generate the polynomial we'll be working with
        // f(x, y) = sum_{i=0}^{n} sum_{j=0}^{m} f_{i,j} L(j) R(i)

        // NOTE: We are working with the coefficients in the Lagrange basis
        // but we FFT those into the standard basis before supplying them to the backend
        tracing::debug!("Starting machines");
        let polynomials = (0..M)
            .map(|i| {
                let coeffs = lagrange_coeffs[i].clone();
                FsPoly::from_coeffs(&fft_settings.fft_left(&coeffs, true).unwrap())
            })
            .collect::<Vec<_>>();

        let machines = (0..M)
            .map(|i| {
                let backend = backend.clone();
                tracing::debug!("coeffs size = {}", lagrange_coeffs[i].len());
                tracing::debug!("coeffs {:?}", lagrange_coeffs[i]);

                Machine { id: i, backend }
            })
            .collect::<Vec<_>>();

        // Commitments
        tracing::debug!("Committing...");
        let commitments = machines
            .iter()
            .map(|m| m.commit(polynomials.get(m.id).unwrap()))
            .collect::<Result<Vec<_>, _>>()?;
        tracing::debug!("commitments size = {}", commitments.len());
        tracing::debug!("commitments = {:?}", commitments);

        let commitment = backend.master_commit(&commitments);
        tracing::debug!("master commitment = {:?}", commitment);

        // Openings
        tracing::debug!("Opening...");
        let alpha = FsFr::rand();
        let beta = FsFr::rand();

        let proofs = machines
            .iter()
            .map(|m| m.prove(polynomials.get(m.id).unwrap(), &alpha))
            .collect::<Result<Vec<_>, _>>()
            .unwrap();

        // // Check the intermediate proofs
        // for (i, (y, pi_0)) in proofs.iter().enumerate() {
        //     let commitment = commitments[i];
        //     tracing::debug!("Checking proof for machine {}", i);
        //     assert!(backend.verify_single(i, &commitment, &alpha, y, pi_0));
        // }

        let evals = proofs
            .iter()
            .map(|(f_i_alpha, _)| *f_i_alpha)
            .collect::<Vec<_>>();

        tracing::debug!("evals size = {}", evals.len());
        tracing::debug!("evals = {:?}", evals);
        tracing::debug!("proofs size = {}", proofs.len());
        tracing::debug!("proofs = {:?}", proofs);

        let proofs = proofs.iter().map(|(_, pi_0_i)| *pi_0_i).collect::<Vec<_>>();
        let (z, pi_f) = backend.master_open(&evals, &proofs, &beta)?;

        // Check z manually, z = f(alpha, beta)
        let z_manual = f.eval(&alpha, &beta);
        tracing::debug!("z_manual = {:?}", z_manual);
        tracing::debug!("z = {:?}", z);
        assert_eq!(z, z_manual);
        tracing::debug!("Reconstruction OK!");

        tracing::debug!("pi_f = {:?}", pi_f);

        // Verification
        tracing::debug!("Verifying...");
        let result = backend.verify(&commitment, &beta, &alpha, &z, &pi_f);
        if result {
            Ok(())
        } else {
            Err("Verification failed".to_string())
        }
    }

    #[test]
    #[tracing_test::traced_test]
    fn pianist_test() {
        let result = pianist(4, 1);
        assert!(result.is_ok());
    }
}

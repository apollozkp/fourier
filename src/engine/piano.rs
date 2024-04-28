use kzg::Fr;
use kzg::G1Affine;
use kzg::G1Mul;
use kzg::Poly;
use kzg::G1;
use rust_kzg_blst::types::fr::FsFr;
use rust_kzg_blst::types::g1::FsG1;
use rust_kzg_blst::types::g1::FsG1Affine;
use rust_kzg_blst::types::poly::FsPoly;

use super::backend::Backend;
use super::blst::BlstBackend;
use super::config::BackendConfig;

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
    pub fn eval(&self, x: FsFr, y: FsFr) -> FsFr {
        polyeval(&self.eval_x(&x), &y)
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
        let mut result = vec![FsPoly::from_coeffs(&[FsFr::zero()]); self.parts.len()];
        for (i, f) in self.parts.iter().enumerate() {
            for (j, g) in other.parts.iter().enumerate() {
                let mut h = f.clone();
                h.mul(g, h.len()).unwrap();
                polysum(&mut result[i + j], &h);
            }
        }
        BivariateFsPolynomial::from_coeffs(result.iter().map(|p| p.coeffs.clone()).collect())
    }

    pub fn add(&self, other: &BivariateFsPolynomial) -> BivariateFsPolynomial {
        let mut result = vec![FsPoly::from_coeffs(&[FsFr::zero()]); self.parts.len()];
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

fn root_of_unity(n: usize) -> FsFr {
    FsFr::from_u64_arr(&rust_kzg_blst::consts::SCALE2_ROOT_OF_UNITY[n])
}

// i-th Lagrange poly using 2**n-th root of unity
fn lagrange_poly(i: usize, n: usize) -> Result<FsPoly, String> {
    let omega = root_of_unity(n);
    let pow = omega.pow(i);

    let mut num_coeffs = vec![FsFr::zero(); n + 1];
    num_coeffs[0] = FsFr::zero().sub(&FsFr::one());
    num_coeffs[n] = FsFr::one();
    let mut numerator = FsPoly::from_coeffs(&num_coeffs);

    let den_coeffs = vec![FsFr::zero().sub(&pow), FsFr::one()];
    let denominator = FsPoly::from_coeffs(&den_coeffs);

    let scalar = FsPoly::from_coeffs(&[pow.div(&FsFr::from_u64(n as u64))?]);

    let mut poly = numerator.div(&denominator)?;
    poly.mul(&scalar, poly.len())
}

fn l_poly(t: usize) -> impl Fn(usize) -> BivariateFsPolynomial {
    move |j: usize| BivariateFsPolynomial::from_poly_as_x(&lagrange_poly(j, t).unwrap())
}

fn r_poly(m: usize) -> impl Fn(usize) -> BivariateFsPolynomial {
    move |i: usize| BivariateFsPolynomial::from_poly_as_y(&lagrange_poly(i, m).unwrap())
}

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

fn generate_coeffs(n: usize, m: usize) -> Vec<Vec<FsFr>> {
    let t = m / n;
    let mut coeffs = vec![vec![FsFr::zero(); m]; t];
    for i in 0..t {
        for j in 0..m {
            coeffs[i][j] = FsFr::rand();
        }
    }
    coeffs
}

// N: circuit size
// M: number of machines
// T: sub-circle size
// T = M / N
fn pianist(circuit_size: usize, num_machines: usize) {
    struct Machine {
        id: usize,
        poly: FsPoly,
        backend: BlstBackend,
    }

    impl Machine {
        fn commit(&self) -> Result<FsG1, String> {
            self.backend.commit_to_poly(self.poly.clone())
        }

        fn prove(&self, x: FsFr) -> Result<FsG1, String> {
            self.backend.compute_proof_single(self.poly.clone(), x)
        }
    }

    let n = circuit_size;
    let m = num_machines;
    let t = m / n;

    let L = l_poly(t);
    let R = r_poly(m);

    let coeffs = generate_coeffs(n, m);
    let f = |i: usize, j: usize| coeffs[j][i];

    // Generate the polynomial we'll be working with
    // f(x, y) = sum_{i=0}^{n} sum_{j=0}^{m} f_{i,j} L(j) R(i)
    let mut poly = BivariateFsPolynomial::zero();
    for i in 0..m {
        for j in 0..t {
            let result = L(j).mul(&R(i)).scale(&f(i, j));
            poly = poly.add(&result);
        }
    }

    // Generate the machines
    let cfg = BackendConfig {
        scale: 5,
        ..Default::default()
    };
    let machines = (0..m)
        .map(|i| {
            let backend = BlstBackend::new(Some(cfg.clone()));
            let poly = (0..t).fold(FsPoly::from_coeffs(&[FsFr::zero()]), |acc, j| {
                polysum(
                    &mut acc.clone(),
                    &L(j).scale(&f(i, j)).eval_y(&FsFr::zero()),
                );
                acc
            });
            Machine {
                id: i,
                poly,
                backend,
            }
        })
        .collect::<Vec<_>>();

    // Commit to the polynomial
    let commitments = machines
        .iter()
        .map(|m| m.commit())
        .collect::<Result<Vec<_>, _>>()
        .unwrap();

    let commitment = commitments.iter().fold(None, |acc, c| {
        acc.map_or(Some(*c), |a: FsG1| Some(a.add(c)))
    });

    // Evaluate the polynomial at a point
    let alpha = FsFr::rand();
    let beta = FsFr::rand();

    let evaluations = machines
        .iter()
        .map(|m| m.poly.eval(&alpha))
        .collect::<Vec<_>>();

    let poly_y = evaluations
        .iter()
        .enumerate()
        .fold(BivariateFsPolynomial::zero(), |acc, (i, e)| {
            acc.add(&R(i).scale(e))
        }).eval_x(&FsFr::zero());

    let result = poly_y.eval(&beta);

    // Open the polynomial at a point
    let proofs = machines
        .iter()
        .map(|m| m.prove(alpha))
        .collect::<Result<Vec<_>, _>>()
        .unwrap();

    let proof = proofs.iter().fold(None, |acc, p| {
        acc.map_or(Some(*p), |a: FsG1| Some(a.add(p)))
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[tracing_test::traced_test]
    fn evaluate_constant_poly_test() {
        let poly = FsPoly::from_coeffs(&[FsFr::one(), FsFr::zero()]);
        // constant polynomial f(x, y) = 1
        let bipoly = BivariateFsPolynomial::from_poly_as_x(&poly);
        assert_eq!(
            poly.eval(&FsFr::zero()),
            bipoly.eval(FsFr::zero(), FsFr::zero())
        );
        assert_eq!(
            poly.eval(&FsFr::one()),
            bipoly.eval(FsFr::one(), FsFr::one())
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
            bipoly.eval(FsFr::zero(), FsFr::zero())
        );
        assert_eq!(
            poly.eval(&FsFr::one()),
            bipoly.eval(FsFr::one(), FsFr::zero())
        );
        assert_eq!(
            poly.eval(&FsFr::from_u64(2)),
            bipoly.eval(FsFr::from_u64(2), FsFr::zero())
        );
    }

    #[test]
    #[tracing_test::traced_test]
    fn evaluate_poly_test() {
        let poly = small_poly();
        let x = FsFr::from_u64(2);
        let y = FsFr::from_u64(3);
        let result = poly.eval(x, y);
        // expected = 2**2 + 2*3 + 3**2 + 1 = 4 + 6 + 9 + 1 = 20
        let expected = x.mul(&x).add(&x.mul(&y)).add(&y.mul(&y)).add(&FsFr::one());
        assert_eq!(result, expected);
    }

    #[test]
    fn piano_test() {}
}

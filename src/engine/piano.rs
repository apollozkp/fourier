use kzg::Fr;
use kzg::Poly;
use rust_kzg_blst::types::fr::FsFr;
use rust_kzg_blst::types::poly::FsPoly;

use super::backend::Backend;
use super::blst::BlstBackend;
use super::config::BackendConfig;

fn transpose<T>(matrix: Vec<Vec<T>>) -> Vec<Vec<T>>
where
    T: Clone,
{
    let mut transposed = Vec::with_capacity(matrix[0].len());
    for i in 0..matrix[0].len() {
        let mut row = Vec::with_capacity(matrix.len());
        (0..matrix.len()).for_each(|j| {
            row.push(matrix[j][i].clone());
        });
        transposed.push(row);
    }
    transposed
}

pub struct BivariateFsPolynomial {
    coefficients: Vec<Vec<FsFr>>,
}

/// A bivariate polynomial in the form:
/// f(x, y) = sum_{i=0}^{n} sum_{j=0}^{m} a_{i,j} x^i y^j
/// where a_{i,j} are the coefficients of the polynomial.
/// The coefficients are stored in a 2D vector, where the first index is the
/// power of x and the second index is the power of y.
impl BivariateFsPolynomial {
    pub fn evaluate(&self, x: FsFr, y: FsFr) -> FsFr {
        self.eval_y(&x).eval(&y)
    }

    pub fn iter_by_y(&self) -> impl Iterator<Item = &Vec<FsFr>> {
        self.coefficients.iter()
    }

    pub fn transposed(&self) -> BivariateFsPolynomial {
        BivariateFsPolynomial {
            coefficients: transpose(self.coefficients.clone()),
        }
    }

    /// Evaluate the polynomial at y, treating x as variable.
    /// This will return a polynomial in x.
    /// f(x, y) = sum_{i=0}^{n} sum_{j=0}^{m} a_{i,j} x^i y^j
    pub fn eval_y(&self, y: &FsFr) -> FsPoly {
        let (_, intermediate) = self.coefficients.iter().fold(
            (FsFr::one(), Vec::new()),
            |(y_power, mut intermediate), coeffs| {
                let coeffs = coeffs
                    .iter()
                    .map(|c| c.mul(&y_power))
                    .collect::<Vec<FsFr>>();
                intermediate.push(coeffs);
                (y_power.mul(y), intermediate)
            },
        );

        let x_coeffs = transpose(intermediate)
            .into_iter()
            .map(|coeffs| coeffs.into_iter().fold(FsFr::zero(), |acc, c| acc.add(&c)))
            .collect::<Vec<FsFr>>();

        FsPoly::from_coeffs(x_coeffs.as_slice())
    }

    pub fn eval_x(&self, x: &FsFr) -> FsPoly {
        self.transposed().eval_y(x)
    }
}

fn root_of_unity(n: usize) -> FsFr {
    FsFr::from_u64_arr(&rust_kzg_blst::consts::SCALE2_ROOT_OF_UNITY[n])
}

fn lagrange_poly(j: usize, t: usize) -> Result<FsPoly, String> {
    let omega = root_of_unity(t);

    let mut num_coeffs = vec![FsFr::zero(); t + 1];
    num_coeffs[0] = FsFr::zero().sub(&FsFr::one());
    num_coeffs[t] = FsFr::one();
    let mut numerator = FsPoly::from_coeffs(&num_coeffs);

    let den_coeffs = vec![FsFr::zero().sub(&omega), FsFr::one()];
    let denominator = FsPoly::from_coeffs(&den_coeffs);

    let scalar = FsPoly::from_coeffs(&[omega.div(&FsFr::from_u64(t as u64))?]);

    let mut poly = numerator.div(&denominator)?;
    poly.mul(&scalar, poly.len())
}

// Polynomial for testing:
// 1 + xy + x**2 + y**2
fn poly() -> BivariateFsPolynomial {
    let coefficients = vec![
        vec![FsFr::one(), FsFr::zero(), FsFr::one()],
        vec![FsFr::zero(), FsFr::one(), FsFr::zero()],
        vec![FsFr::one(), FsFr::zero(), FsFr::zero()],
    ];

    BivariateFsPolynomial { coefficients }
}

fn pianist() {
    let cfg = BackendConfig {
        scale: 5,
        ..Default::default()
    };
    let backend = BlstBackend::new(Some(cfg.clone()));
    let poly = poly();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[tracing_test::traced_test]
    fn evaluate_constant_poly_test() {
        let poly = FsPoly::from_coeffs(&[FsFr::one(), FsFr::zero()]);
        let bipoly = BivariateFsPolynomial {
            coefficients: vec![
                vec![FsFr::one(), FsFr::zero()],
                vec![FsFr::zero(), FsFr::zero()],
            ],
        };
        assert_eq!(
            poly.eval(&FsFr::zero()),
            bipoly.evaluate(FsFr::zero(), FsFr::zero())
        );
        assert_eq!(
            poly.eval(&FsFr::one()),
            bipoly.evaluate(FsFr::one(), FsFr::one())
        );
    }

    #[test]
    #[tracing_test::traced_test]
    fn evaluate_linear_poly_test() {
        // 1 + x
        let poly = FsPoly::from_coeffs(&[FsFr::one(), FsFr::one()]);
        // 1 + x
        let bipoly = BivariateFsPolynomial {
            coefficients: vec![
                vec![FsFr::one(), FsFr::zero()],
                vec![FsFr::one(), FsFr::zero()],
            ],
        };
        assert_eq!(
            poly.eval(&FsFr::zero()),
            bipoly.evaluate(FsFr::zero(), FsFr::zero())
        );
        assert_eq!(
            poly.eval(&FsFr::one()),
            bipoly.evaluate(FsFr::one(), FsFr::zero())
        );
        assert_eq!(
            poly.eval(&FsFr::from_u64(2)),
            bipoly.evaluate(FsFr::from_u64(2), FsFr::zero())
        );
    }

    #[test]
    #[tracing_test::traced_test]
    fn evaluate_poly_test() {
        let poly = poly();
        let x = FsFr::from_u64(2);
        let y = FsFr::from_u64(3);
        let result = poly.evaluate(y, x);
        // expected = 2**2 + 2*3 + 3**2 + 1 = 4 + 6 + 9 + 1 = 20
        let expected = x.mul(&x).add(&x.mul(&y)).add(&y.mul(&y)).add(&FsFr::one());
        assert_eq!(result, expected);
    }

    #[test]
    fn piano_test() {}
}

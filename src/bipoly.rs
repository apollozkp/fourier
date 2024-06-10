use kzg::Fr;
use kzg::Poly;
use rust_kzg_blst::types::fr::FsFr;
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

use crate::engine::config::{BackendConfig, SetupConfig};
use serde::{Deserialize, Serialize};
use base64::Engine;
use kzg::{FFTSettings, Fr, G1Affine, G1Fp, G1GetFp, G1Mul, KZGSettings, Poly, G1, G2};

pub trait Backend {
    type Fr: Fr;
    type G1: G1 + G1Mul<Self::Fr> + G1GetFp<Self::G1Fp>;
    type G2: G2;
    type Poly: Poly<Self::Fr>;
    type FFTSettings: FFTSettings<Self::Fr>;
    type KZGSettings: KZGSettings<
        Self::Fr,
        Self::G1,
        Self::G2,
        Self::FFTSettings,
        Self::Poly,
        Self::G1Fp,
        Self::G1Affine,
    >;
    type G1Fp: G1Fp;
    type G1Affine: G1Affine<Self::G1, Self::G1Fp>;

    fn new(cfg: Option<BackendConfig>) -> Self;

    fn commit_to_poly(&self, poly: Self::Poly) -> Result<Self::G1, String>;

    fn compute_proof_single(&self, poly: Self::Poly, point: Self::Fr) -> Result<Self::G1, String>;

    fn verify_proof_single(
        &self,
        proof: Self::G1,
        x: Self::Fr,
        y: Self::Fr,
        commitment: Self::G1,
    ) -> Result<bool, String>;

    fn parse_poly_from_str(&self, s: &[String], encoding: Option<Encoding>) -> Result<Self::Poly, String>;
    fn parse_point_from_str(&self, s: &str, encoding: Option<Encoding>) -> Result<Self::Fr, String>;
    fn parse_g1_from_str(&self, s: &str, encoding: Option<Encoding>) -> Result<Self::G1, String>;

    fn random_poly(&self, degree: usize) -> Self::Poly;
    fn random_point(&self) -> Self::Fr;
    fn evaluate(&self, poly: &Self::Poly, x: Self::Fr) -> Self::Fr;
    fn setup(cfg: SetupConfig) -> Result<Self, String>
    where
        Self: Sized;
    fn setup_and_save(cfg: SetupConfig) -> Result<(), String>;
}

pub fn hex_score(value: &str) -> usize {
    let is_hex = |c: char| c.is_ascii_hexdigit();
    let sum_score = |value: &str| -> usize {
        value
            .chars()
            .map(|c| if is_hex(c) { 1 } else { 0 })
            .sum::<usize>()
    };
    let score = if let Some(value) = value.strip_prefix("0x") {
        sum_score(value) + 2
    } else {
        sum_score(value)
    };

    score
}

pub fn b64_score(value: &str) -> usize {
    let is_b64 = |c: char| c.is_ascii_alphanumeric() || c == '+' || c == '/' || c == '=';
    let sum_score = |value: &str| -> usize {
        value
            .chars()
            .map(|c| if is_b64(c) { 1 } else { 0 })
            .sum::<usize>()
    };

    sum_score(value)
}

// Defaults to hex for backwards compatibility
#[derive(Debug, PartialEq, Default, Deserialize, Serialize, Copy, Clone)]
#[serde(rename_all = "camelCase")]
pub enum Encoding {
    #[default]
    Hex,
    Base64,
}

impl Encoding {
    const BASE_64_ENGINE: base64::engine::GeneralPurpose = base64::engine::GeneralPurpose::new(
        &base64::alphabet::STANDARD,
        base64::engine::general_purpose::NO_PAD,
    );

    pub fn decode_array_unknown(values: &[String]) -> Result<Vec<Vec<u8>>, String> {
        values.iter().map(|v| Self::decode_unknown(v)).collect()
    }

    pub fn decode_unknown(value: &str) -> Result<Vec<u8>, String> {
        Self::from(value).decode(value)
    }

    pub fn decode(&self, value: &str) -> Result<Vec<u8>, String> {
        match self {
            Encoding::Hex => if let Some(value) = value.strip_prefix("0x") {
                hex::decode(value).map_err(|e| e.to_string())
            } else {
                hex::decode(value).map_err(|e| e.to_string())
            },
            Encoding::Base64 => Self::BASE_64_ENGINE
                .decode(value)
                .map_err(|e| e.to_string()),
        }
    }

    pub fn encode(&self, value: &[u8]) -> String {
        match self {
            Encoding::Hex => hex::encode(value),
            Encoding::Base64 => Self::BASE_64_ENGINE.encode(value),
        }
    }

    pub fn decode_array(&self, values: &[String]) -> Result<Vec<Vec<u8>>, String> {
        values.iter().map(|v| self.decode(v)).collect()
    }

    pub fn encode_array(&self, values: &[Vec<u8>]) -> Vec<String> {
        values.iter().map(|v| self.encode(v)).collect()
    }
}

impl From<&str> for Encoding {
    fn from(s: &str) -> Self {
        match (hex_score(s), b64_score(s)) {
            (hex, b64) if hex >= b64 => Encoding::Hex,
            _ => Encoding::Base64,
        }
    }
}

impl From<&Vec<String>> for Encoding {
    fn from(s: &Vec<String>) -> Self {
        if let Some(s) = s.iter().next() {
            Encoding::from(s.as_str())
        } else {
            Encoding::default()
        }
    }
}

impl From<&[String]> for Encoding {
    fn from(s: &[String]) -> Self {
        if let Some(s) = s.iter().next() {
            Encoding::from(s.as_str())
        } else {
            Encoding::default()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_hex() {
        assert_eq!(Encoding::from("0x1234567890abcdef"), Encoding::Hex);
        assert_ne!(Encoding::from("1234567890abcdefg"), Encoding::Hex);
    }

    #[test]
    fn test_is_b64() {
        assert_eq!(Encoding::from("aGVsbG8gd29ybGQ="), Encoding::Base64);
    }

    #[test]
    fn test_decode_hex() {
        let value = "0x1234567890abcdef";
        let decoded = Encoding::Hex.decode(value).unwrap();
        assert_eq!(decoded, vec![0x12, 0x34, 0x56, 0x78, 0x90, 0xab, 0xcd, 0xef]);
    }

    #[test]
    fn test_decode_b64() {
        let value = "aGVsbG8gd29ybGQ";
        let decoded = Encoding::Base64.decode(value).unwrap();
        assert_eq!(decoded, b"hello world".to_vec());
    }
}

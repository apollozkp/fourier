use http_body_util::{BodyExt, Full};
use hyper::body::{Buf, Bytes};
use hyper::Response;
use kzg::{Fr, Poly, G1};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use tracing::{error, info};

use crate::engine::backend::Encoding;
use crate::RunArgs;

#[derive(Debug, Serialize, Deserialize)]
struct JsonRpcRequest {
    jsonrpc: String,
    method: Method,
    #[serde(skip_serializing_if = "Option::is_none")]
    params: Option<JsonRpcParams>,
    #[serde(skip_serializing_if = "Option::is_none")]
    id: Option<Value>,
}

impl JsonRpcRequest {
    fn response(
        &self,
        result: Option<JsonRpcResult>,
        error: Option<JsonRpcError>,
    ) -> JsonRpcResponse {
        JsonRpcResponse {
            jsonrpc: "2.0".to_owned(),
            result,
            error,
            id: self.id.clone(),
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(untagged)]
pub enum JsonRpcParams {
    Ping,
    Verify {
        proof: String,
        x: String,
        y: String,
        commitment: String,
        encoding: Option<Encoding>,
    },
    Open {
        poly: Vec<String>,
        x: String,
        encoding: Option<Encoding>,
    },
    Commit {
        poly: Vec<String>,
        encoding: Option<Encoding>,
    },
    RandomPoly {
        degree: usize,
        #[serde(skip_serializing_if = "Option::is_none")]
        encoding: Option<Encoding>,
    },
    RandomPoint {
        encoding: Option<Encoding>,
    },
}

#[derive(Debug, Serialize, Deserialize)]
struct JsonRpcResponse {
    jsonrpc: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    result: Option<JsonRpcResult>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<JsonRpcError>,
    id: Option<Value>,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(untagged)]
pub enum JsonRpcResult {
    Commit {
        commitment: String,
    },
    Open {
        proof: String,
    },
    Verify {
        valid: bool,
    },
    RandomPoly {
        poly: Vec<String>,
    },
    RandomPoint {
        point: String,
    },
    Evaluate {
        y: String,
    },
    Prove {
        commitment: String,
        y: String,
        x: String,
        proof: String,
    },
    Pong,
}

#[derive(Debug, Serialize, Deserialize)]
struct JsonRpcError {
    code: i32,
    message: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub enum Method {
    #[serde(rename = "ping")]
    Ping,

    #[serde(rename = "commit")]
    Commit,

    #[serde(rename = "open")]
    Open,

    #[serde(rename = "verify")]
    Verify,

    #[serde(rename = "randomPoly")]
    RandomPoly,

    #[serde(rename = "randomPoint")]
    RandomPoint,

    #[serde(rename = "evaluate")]
    Evaluate,

    #[serde(rename = "prove")]
    Prove,

    #[serde(other)]
    NotSupported,
}

pub struct RpcHandler<B>
where
    B: crate::engine::backend::Backend,
{
    backend: Arc<B>,
}

impl<B> Clone for RpcHandler<B>
where
    B: crate::engine::backend::Backend,
{
    fn clone(&self) -> Self {
        RpcHandler {
            backend: self.backend.clone(),
        }
    }
}

impl<B> RpcHandler<B>
where
    B: crate::engine::backend::Backend,
{
    pub fn new(backend: Arc<B>) -> Self {
        RpcHandler { backend }
    }

    async fn handle(&self, req: JsonRpcRequest) -> JsonRpcResponse {
        match req.method {
            Method::Ping => Self::handle_ping(req),
            Method::Commit => self.handle_commit(req),
            Method::Open => self.handle_open(req),
            Method::Verify => self.handle_verify(req),
            Method::RandomPoly => self.handle_random_poly(req),
            Method::RandomPoint => self.handle_random_point(req),
            Method::Evaluate => self.handle_evaluate(req),
            Method::Prove => self.handle_prove(req),
            Method::NotSupported => Self::handle_not_supported(Some(req)),
        }
    }

    fn handle_ping(req: JsonRpcRequest) -> JsonRpcResponse {
        req.response(Some(JsonRpcResult::Pong), None)
    }

    fn handle_commit(&self, req: JsonRpcRequest) -> JsonRpcResponse {
        let (result, error) = if let Some(JsonRpcParams::Commit { ref poly, encoding }) = req.params
        {
            let encoding = encoding.unwrap_or(Encoding::from(poly));
            match self.backend.parse_poly_from_str(poly, Some(encoding)) {
                Ok(poly) => (
                    Some(JsonRpcResult::Commit {
                        commitment: encoding
                            .encode(&self.backend.commit_to_poly(poly).unwrap().to_bytes()),
                    }),
                    None,
                ),
                Err(err) => (
                    None,
                    Some(JsonRpcError {
                        code: -32000,
                        message: err,
                    }),
                ),
            }
        } else {
            (
                None,
                Some(JsonRpcError {
                    code: -32602,
                    message: "Invalid params".to_owned(),
                }),
            )
        };
        req.response(result, error)
    }

    fn handle_open(&self, req: JsonRpcRequest) -> JsonRpcResponse {
        let (result, err) = if let Some(JsonRpcParams::Open {
            ref poly,
            ref x,
            encoding,
        }) = req.params
        {
            let encoding = encoding.unwrap_or(Encoding::from(poly));
            match (|| {
                Ok((
                    self.backend.parse_poly_from_str(poly, Some(encoding))?,
                    self.backend.parse_point_from_str(x, Some(encoding))?,
                ))
            })() {
                Ok((poly, x)) => (
                    Some(JsonRpcResult::Open {
                        proof: encoding.encode(
                            &self
                                .backend
                                .compute_proof_single(poly, x)
                                .unwrap()
                                .to_bytes(),
                        ),
                    }),
                    None,
                ),
                Err(err) => (
                    None,
                    Some(JsonRpcError {
                        code: -32000,
                        message: err,
                    }),
                ),
            }
        } else {
            (
                None,
                Some(JsonRpcError {
                    code: -32602,
                    message: "Invalid params".to_owned(),
                }),
            )
        };
        req.response(result, err)
    }

    fn handle_verify(&self, req: JsonRpcRequest) -> JsonRpcResponse {
        let (result, err) = if let Some(JsonRpcParams::Verify {
            ref proof,
            ref x,
            ref y,
            ref commitment,
            encoding,
        }) = req.params
        {
            match (|| {
                Ok((
                    self.backend.parse_g1_from_str(proof, encoding)?,
                    self.backend.parse_point_from_str(x, encoding)?,
                    self.backend.parse_point_from_str(y, encoding)?,
                    self.backend.parse_g1_from_str(commitment, encoding)?,
                ))
            })() {
                Ok((proof, x, y, commitment)) => (
                    Some(JsonRpcResult::Verify {
                        valid: self
                            .backend
                            .verify_proof_single(proof, x, y, commitment)
                            .unwrap(),
                    }),
                    None,
                ),
                Err(err) => (
                    None,
                    Some(JsonRpcError {
                        code: -32000,
                        message: err,
                    }),
                ),
            }
        } else {
            (
                None,
                Some(JsonRpcError {
                    code: -32602,
                    message: "Invalid params".to_owned(),
                }),
            )
        };
        req.response(result, err)
    }

    fn handle_random_poly(&self, req: JsonRpcRequest) -> JsonRpcResponse {
        let (result, err) = if let Some(JsonRpcParams::RandomPoly { degree, encoding }) = req.params
        {
            let encoding = encoding.unwrap_or(Encoding::default());
            (
                Some(JsonRpcResult::RandomPoly {
                    poly: self
                        .backend
                        .random_poly(degree)
                        .get_coeffs()
                        .iter()
                        .map(|x| encoding.encode(&x.to_bytes()))
                        .collect(),
                }),
                None,
            )
        } else {
            (
                None,
                Some(JsonRpcError {
                    code: -32602,
                    message: "Invalid params".to_owned(),
                }),
            )
        };
        req.response(result, err)
    }

    fn handle_random_point(&self, req: JsonRpcRequest) -> JsonRpcResponse {
        let encoding = Encoding::default();
        req.response(
            Some(JsonRpcResult::RandomPoint {
                point: encoding.encode(&self.backend.random_point().to_bytes()),
            }),
            None,
        )
    }

    fn handle_evaluate(&self, req: JsonRpcRequest) -> JsonRpcResponse {
        info!("Handling evaluate request{:?}", req);
        // TODO: This is gross, params are matched to Open since the enum is untagged
        // and the Open variant is the first one. This should be fixed.
        let (result, err) = if let Some(JsonRpcParams::Open {
            ref poly,
            ref x,
            encoding,
        }) = req.params
        {
            let encoding = encoding.unwrap_or(Encoding::from(poly));
            match (|| {
                Ok((
                    self.backend.parse_poly_from_str(poly, Some(encoding))?,
                    self.backend.parse_point_from_str(x, Some(encoding))?,
                ))
            })() {
                Ok((poly, x)) => (
                    Some(JsonRpcResult::Evaluate {
                        y: encoding.encode(&self.backend.evaluate(&poly, x).to_bytes()),
                    }),
                    None,
                ),
                Err(err) => (
                    None,
                    Some(JsonRpcError {
                        code: -32000,
                        message: err,
                    }),
                ),
            }
        } else {
            (
                None,
                Some(JsonRpcError {
                    code: -32602,
                    message: "Invalid params".to_owned(),
                }),
            )
        };
        req.response(result, err)
    }

    fn handle_prove(&self, req: JsonRpcRequest) -> JsonRpcResponse {
        let (result, err) = if let Some(JsonRpcParams::Commit { ref poly, encoding }) = req.params {
            let encoding = encoding.unwrap_or(Encoding::from(poly));
            match self.backend.parse_poly_from_str(poly, Some(encoding)) {
                Ok(poly) => {
                    let x = Fr::rand();
                    let y = poly.eval(&x);
                    (
                        Some(JsonRpcResult::Prove {
                            commitment: encoding.encode(
                                &self
                                    .backend
                                    .commit_to_poly(poly.clone())
                                    .unwrap()
                                    .to_bytes(),
                            ),
                            y: encoding.encode(&y.to_bytes()),
                            x: encoding.encode(&x.to_bytes()),
                            proof: encoding.encode(
                                &self
                                    .backend
                                    .compute_proof_single(poly, x)
                                    .unwrap()
                                    .to_bytes(),
                            ),
                        }),
                        None,
                    )
                }
                Err(err) => (
                    None,
                    Some(JsonRpcError {
                        code: -32000,
                        message: err,
                    }),
                ),
            }
        } else {
            (
                None,
                Some(JsonRpcError {
                    code: -32602,
                    message: "Invalid params".to_owned(),
                }),
            )
        };
        req.response(result, err)
    }

    fn handle_not_supported(req: Option<JsonRpcRequest>) -> JsonRpcResponse {
        let (result, err) = (
            None,
            Some(JsonRpcError {
                code: -32000,
                message: "Method not implemented".to_owned(),
            }),
        );
        JsonRpcResponse {
            jsonrpc: "2.0".to_owned(),
            result,
            error: err,
            id: req.and_then(|r| r.id),
        }
    }
}

impl<B> hyper::service::Service<hyper::Request<hyper::body::Incoming>> for RpcHandler<B>
where
    B: crate::engine::backend::Backend + Send + Sync + 'static,
{
    type Response = Response<Full<Bytes>>;
    type Error = hyper::Error;
    type Future = Pin<Box<dyn Future<Output = Result<Self::Response, Self::Error>> + Send>>;

    fn call(&self, req: hyper::Request<hyper::body::Incoming>) -> Self::Future {
        let make_response = |res: JsonRpcResponse| {
            let serialized = serde_json::to_vec(&res).unwrap();
            Response::new(Full::from(serialized))
        };

        let handler = self.clone();

        let future = async move {
            match req.collect().await {
                Ok(body) => {
                    info!("Received request: {:?}", body);
                    let whole_body = body.aggregate();
                    match serde_json::from_reader(whole_body.reader()) {
                        Ok(req) => Ok(make_response(handler.handle(req).await)),
                        Err(err) => {
                            error!("Error: {}", err);
                            Ok(make_response(Self::handle_not_supported(None)))
                        }
                    }
                }
                Err(err) => {
                    error!("Error: {}", err);
                    Ok(make_response(Self::handle_not_supported(None)))
                }
            }
        };

        Box::pin(future)
    }
}

#[derive(Debug, Clone, Default)]
pub struct Config {
    pub host: String,
    pub port: usize,
    pub backend: crate::engine::config::BackendConfig,
}

impl From<RunArgs> for Config {
    fn from(args: RunArgs) -> Self {
        Config {
            host: args.host.clone(),
            port: args.port,
            backend: args.into(),
        }
    }
}

#[derive(Debug, Default)]
pub struct Server {
    cfg: Config,
}

impl Server {
    pub fn new(cfg: Config) -> Self {
        Server { cfg }
    }

    fn addr(&self) -> String {
        format!("{}:{}", self.cfg.host, self.cfg.port)
    }

    fn new_handler<B>(&self) -> RpcHandler<B>
    where
        B: crate::engine::backend::Backend,
    {
        let backend = B::new(Some(self.cfg.backend.clone()));
        RpcHandler::new(Arc::new(backend))
    }

    pub async fn run<B>(&self) -> std::io::Result<()>
    where
        B: crate::engine::backend::Backend + Send + Sync + 'static,
    {
        info!("Starting RPC server...");
        let listener = tokio::net::TcpListener::bind(&self.addr()).await?;
        info!("Listening on: {}", self.addr());
        let handler = crate::utils::timed("start handler", || self.new_handler::<B>());

        loop {
            let (stream, _) = listener.accept().await?;
            info!("Accepted connection from: {}", stream.peer_addr().unwrap());
            let io = hyper_util::rt::TokioIo::new(stream);

            let handler = handler.clone();

            tokio::task::spawn(async move {
                if let Err(err) = hyper::server::conn::http1::Builder::new()
                    .serve_connection(io, handler)
                    .await
                {
                    error!("Error: {}", err);
                }
            });
        }
    }
}

pub async fn start_rpc_server<B>(cfg: Config)
where
    B: crate::engine::backend::Backend + Send + Sync + 'static,
{
    let server = Server::new(cfg);
    if let Err(e) = server.run::<B>().await {
        error!("Error: {}", e);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::{backend::Backend, config::BackendConfig};

    const POLY: [&str; 6] = [
        "0cd70a3a63e25a07f3068874e8fdf9a5238b5b391b3b88d69aeb63815d83d6bf",
        "347600725caf9aa43827b6a61f44a28b94441178ee5b4bd76bc9875ec3e9b4f7",
        "689323bfea2caecbc9deb55057ad82939d62ce7087c5df26845bfb352b76bb43",
        "1475ce3ccbbe2534b6bda56880d5cabca486521a06be0b119a398f047f201135",
        "40480c2ba4f4d4d6951bf385eacdee2e62deaea98de2fbb6fa4b579203b66a70",
        "2fd31a457aa075a3be8c9ebd982f4544f477f0d7730a02a63807b19c9ba62e53",
    ];
    const X: &str = "3970b257e06b2db2e040dee0d8be7242af39e009121b7527e2359c1adc7c35db";
    const Y: &str = "0eacb75ad74a19f048530e1587df7c5e746f834f3c2e17dbbbba17df0fe91f3c";

    const POLY_BASE64: [&str; 6] = [
        "DNcKOmPiWgfzBoh06P35pSOLWzkbO4jWmutjgV2D1r8",
        "NHYAclyvmqQ4J7amH0Sii5REEXjuW0vXa8mHXsPptPc",
        "aJMjv+osrsvJ3rVQV62Ck51iznCHxd8mhFv7NSt2u0M",
        "FHXOPMu+JTS2vaVogNXKvKSGUhoGvgsRmjmPBH8gETU",
        "QEgMK6T01NaVG/OF6s3uLmLerqmN4vu2+ktXkgO2anA",
        "L9MaRXqgdaO+jJ69mC9FRPR38NdzCgKmOAexnJumLlM",
    ];

    const X_BASE64: &str = "OXCyV+BrLbLgQN7g2L5yQq854AkSG3Un4jWcGtx8Nds";
    const Y_BASE64: &str = "Dqy3WtdKGfBIUw4Vh998XnRvg088Lhfbu7oX3w/pHzw";

    fn test_config() -> BackendConfig {
        BackendConfig {
            setup_path: None,
            precompute_path: None,
            scale: 5,
            skip_precompute: false,
            compressed: true,
        }
    }

    #[test]
    #[tracing_test::traced_test]
    fn test_serialize_deserialize_request() {
        let raw_request = r#"{"jsonrpc":"2.0","method":"ping","id":1}"#;
        let serialized = serde_json::from_str::<JsonRpcRequest>(raw_request);
        assert!(serialized.is_ok());
        let deserialized = serde_json::to_string(&serialized.unwrap());
        assert!(deserialized.is_ok());
        assert_eq!(raw_request, deserialized.unwrap());
    }

    #[tokio::test]
    #[tracing_test::traced_test]
    async fn test_random_polynomial_hex() {
        let config = test_config();
        let backend = crate::engine::blst::BlstBackend::new(Some(config));
        let handler = RpcHandler::new(Arc::new(backend));
        let raw_request = r#"{"jsonrpc":"2.0","method":"randomPoly","params":{"degree":3},"id":1}"#;
        let req = serde_json::from_str::<JsonRpcRequest>(raw_request).unwrap();
        tracing::debug!("Request: {:?}", req);
        let res = handler.handle(req).await;
        tracing::debug!("Response: {:?}", res);
        assert!(res.result.is_some());
        assert!(res.error.is_none());
    }

    #[tokio::test]
    #[tracing_test::traced_test]
    async fn test_random_polynomial_base64() {
        let config = test_config();
        let backend = crate::engine::blst::BlstBackend::new(Some(config));
        let handler = RpcHandler::new(Arc::new(backend));
        let raw_request = r#"{"jsonrpc":"2.0","method":"randomPoly","params":{"degree":3, "encoding": "base64"},"id":1}"#;
        let req = serde_json::from_str::<JsonRpcRequest>(raw_request).unwrap();
        tracing::debug!("Request: {:?}", req);
        let res = handler.handle(req).await;
        tracing::debug!("Response: {:?}", res);
        assert!(res.result.is_some());
        assert!(res.error.is_none());
    }

    #[tokio::test]
    #[tracing_test::traced_test]
    async fn test_commit_default() {
        let config = test_config();
        let backend = crate::engine::blst::BlstBackend::new(Some(config));
        let handler = RpcHandler::new(Arc::new(backend));
        let raw_request = format!(
            r#"{{"jsonrpc":"2.0","method":"commit","params":{{"poly":{:?}}},"id":1}}"#,
            POLY
        );
        let req = serde_json::from_str::<JsonRpcRequest>(&raw_request).unwrap();
        tracing::debug!("Request: {:?}", req);
        let res = handler.handle(req).await;
        tracing::debug!("Response: {:?}", res);
        assert!(res.result.is_some());
        assert!(res.error.is_none());
    }

    #[tokio::test]
    #[tracing_test::traced_test]
    async fn test_commit_base64_inferred() {
        let config = test_config();
        let backend = crate::engine::blst::BlstBackend::new(Some(config));
        let handler = RpcHandler::new(Arc::new(backend));
        let raw_request = format!(
            r#"{{"jsonrpc":"2.0","method":"commit","params":{{"poly":{:?}}},"id":1}}"#,
            POLY_BASE64
        );
        let req = serde_json::from_str::<JsonRpcRequest>(&raw_request).unwrap();
        tracing::debug!("Request: {:?}", req);
        let res = handler.handle(req).await;
        tracing::debug!("Response: {:?}", res);
        assert!(res.result.is_some());
        assert!(res.error.is_none());
    }

    #[tokio::test]
    #[tracing_test::traced_test]
    async fn test_commit_base64_explicit() {
        let config = test_config();
        let backend = crate::engine::blst::BlstBackend::new(Some(config));
        let handler = RpcHandler::new(Arc::new(backend));
        let raw_request = format!(
            r#"{{"jsonrpc":"2.0","method":"commit","params":{{"poly":{:?}, "encoding": "base64"}},"id":1}}"#,
            POLY_BASE64
        );
        let req = serde_json::from_str::<JsonRpcRequest>(&raw_request).unwrap();
        tracing::debug!("Request: {:?}", req);
        let res = handler.handle(req).await;
        tracing::debug!("Response: {:?}", res);
        assert!(res.result.is_some());
        assert!(res.error.is_none());
    }

    #[tokio::test]
    #[tracing_test::traced_test]
    async fn test_commit_base64_explicit_wrong() {
        let config = test_config();
        let backend = crate::engine::blst::BlstBackend::new(Some(config));
        let handler = RpcHandler::new(Arc::new(backend));
        let raw_request = format!(
            r#"{{"jsonrpc":"2.0","method":"commit","params":{{"poly":{:?}, "encoding": "hex"}},"id":1}}"#,
            POLY_BASE64
        );
        let req = serde_json::from_str::<JsonRpcRequest>(&raw_request).unwrap();
        tracing::debug!("Request: {:?}", req);
        let res = handler.handle(req).await;
        tracing::debug!("Response: {:?}", res);
        assert!(res.result.is_none());
        assert!(res.error.is_some());
    }

}

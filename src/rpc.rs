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
    },
    Open {
        poly: Vec<String>,
        x: String,
    },
    Commit {
        poly: Vec<String>,
    },
    RandomPoly {
        degree: usize,
    },
    RandomPoint,
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
        let (result, error) = if let Some(JsonRpcParams::Commit { ref poly }) = req.params {
            match self.backend.parse_poly_from_str(poly) {
                Ok(poly) => (
                    Some(JsonRpcResult::Commit {
                        commitment: hex::encode(
                            self.backend.commit_to_poly(poly).unwrap().to_bytes(),
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
        req.response(result, error)
    }

    fn handle_open(&self, req: JsonRpcRequest) -> JsonRpcResponse {
        let (result, err) = if let Some(JsonRpcParams::Open { ref poly, ref x }) = req.params {
            match (|| {
                Ok((
                    self.backend.parse_poly_from_str(poly)?,
                    self.backend.parse_point_from_str(x)?,
                ))
            })() {
                Ok((poly, x)) => (
                    Some(JsonRpcResult::Open {
                        proof: hex::encode(
                            self.backend
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
        }) = req.params
        {
            match (|| {
                Ok((
                    self.backend.parse_g1_from_str(proof)?,
                    self.backend.parse_point_from_str(x)?,
                    self.backend.parse_point_from_str(y)?,
                    self.backend.parse_g1_from_str(commitment)?,
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
        let (result, err) = if let Some(JsonRpcParams::RandomPoly { degree }) = req.params {
            (
                Some(JsonRpcResult::RandomPoly {
                    poly: self
                        .backend
                        .random_poly(degree)
                        .get_coeffs()
                        .iter()
                        .map(|x| hex::encode(x.to_bytes()))
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
        req.response(
            Some(JsonRpcResult::RandomPoint {
                point: hex::encode(self.backend.random_point().to_bytes()),
            }),
            None,
        )
    }

    fn handle_evaluate(&self, req: JsonRpcRequest) -> JsonRpcResponse {
        info!("Handling evaluate request{:?}", req);
        // TODO: This is gross, params are matched to Open since the enum is untagged
        // and the Open variant is the first one. This should be fixed.
        let (result, err) = if let Some(JsonRpcParams::Open { ref poly, ref x }) = req.params {
            match (|| {
                Ok((
                    self.backend.parse_poly_from_str(poly)?,
                    self.backend.parse_point_from_str(x)?,
                ))
            })() {
                Ok((poly, x)) => (
                    Some(JsonRpcResult::Evaluate {
                        y: hex::encode(self.backend.evaluate(&poly, x).to_bytes()),
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
        let (result, err) = if let Some(JsonRpcParams::Commit { ref poly }) = req.params {
            match self.backend.parse_poly_from_str(poly) {
                Ok(poly) => {
                    let x = Fr::rand();
                    let y = poly.eval(&x);
                    (
                        Some(JsonRpcResult::Prove {
                            commitment: hex::encode(
                                self.backend
                                    .commit_to_poly(poly.clone())
                                    .unwrap()
                                    .to_bytes(),
                            ),
                            y: hex::encode(y.to_bytes()),
                            x: hex::encode(x.to_bytes()),
                            proof: hex::encode(
                                self.backend
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

#[derive(Debug, Clone)]
pub struct ServerConfig {
    pub host: String,
    pub port: u16,
    pub backend: Option<crate::engine::backend::BackendConfig>,
}

impl ServerConfig {
    pub fn new(port: u16) -> Self {
        ServerConfig {
            host: "127.0.0.1".to_owned(),
            port,
            backend: None,
        }
    }
}

impl Default for ServerConfig {
    fn default() -> Self {
        ServerConfig {
            host: "127.0.0.1".to_owned(),
            port: 1337,
            backend: None,
        }
    }
}

#[derive(Debug, Default)]
pub struct Server {
    cfg: ServerConfig,
}

impl Server {
    pub fn new(cfg: ServerConfig) -> Self {
        Server { cfg }
    }

    fn addr(&self) -> String {
        format!("{}:{}", self.cfg.host, self.cfg.port)
    }

    fn new_handler<B>(&self, path: Option<String>) -> RpcHandler<B>
    where
        B: crate::engine::backend::Backend,
    {
        let backend = B::new(self.cfg.backend.clone(), path);
        RpcHandler::new(Arc::new(backend))
    }

    pub async fn run<B>(&self, path: Option<String>) -> std::io::Result<()>
    where
        B: crate::engine::backend::Backend + Send + Sync + 'static,
    {
        info!("Starting server...");
        let listener = tokio::net::TcpListener::bind(&self.addr()).await?;
        info!("Listening on: {}", self.addr());
        let handler = self.new_handler::<B>(path);

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

pub async fn start_rpc_server<B>(port: u16, path: Option<String>)
where
    B: crate::engine::backend::Backend + Send + Sync + 'static,
{
    // init tracing subscriber and write to stdout
    tracing_subscriber::fmt()
        .with_writer(std::io::stdout)
        .init();
    info!("Starting RPC server...");
    let config = ServerConfig::new(port);
    let server = Server::new(config);
    server.run::<B>(path).await.unwrap();
}

#[cfg(test)]
mod tests {
    use super::*;

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
}

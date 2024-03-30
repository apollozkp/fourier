use http_body_util::{BodyExt, Full};
use hyper::body::{Buf, Bytes};
use hyper::Response;
use kzg::G1;
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

#[derive(Debug, Serialize, Deserialize)]
#[serde(untagged)]
pub enum JsonRpcParams {
    Ping,
    Commit {
        poly: Vec<String>,
    },
    Prove {
        poly: Vec<String>,
        x: String,
        y: String,
    },
    Verify {
        proof: String,
        x: String,
        y: String,
        commitment: String,
    },
}

#[derive(Debug, Serialize, Deserialize)]
struct JsonRpcResponse {
    jsonrpc: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    result: Option<JsonRpcResult>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<JsonRpcError>,
    id: Value,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(untagged)]
pub enum JsonRpcResult {
    Commit { commitment: String },
    Prove { proof: String },
    Verify { valid: bool },
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

    #[serde(rename = "prove")]
    Prove,

    #[serde(rename = "verify")]
    Verify,

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
            Method::Ping => Self::handle_ping(),
            Method::Commit => self.handle_commit(req),
            Method::Prove => Self::handle_prove(req),
            Method::Verify => Self::handle_verify(req),
            Method::NotSupported => Self::handle_not_supported(),
        }
    }

    fn handle_ping() -> JsonRpcResponse {
        JsonRpcResponse {
            jsonrpc: "2.0".to_owned(),
            result: Some(JsonRpcResult::Pong),
            error: None,
            id: Value::Null,
        }
    }

    fn handle_commit(&self, _req: JsonRpcRequest) -> JsonRpcResponse {
        if let Some(JsonRpcParams::Commit { poly }) = _req.params {
            let poly = self.backend.parse_poly_from_str(&poly);
            let (result, error) = match poly {
                Ok(poly) => {
                    let result = self.backend.commit_to_poly(poly);
                    match result {
                        Ok(commitment) => (
                            Some(JsonRpcResult::Commit {
                                commitment: hex::encode(commitment.to_bytes()),
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
                }
                Err(err) => (
                    None,
                    Some(JsonRpcError {
                        code: -32000,
                        message: err,
                    }),
                ),
            };

            JsonRpcResponse {
                jsonrpc: "2.0".to_owned(),
                result,
                error,
                id: Value::Null,
            }
        } else {
            JsonRpcResponse {
                jsonrpc: "2.0".to_owned(),
                result: None,
                error: Some(JsonRpcError {
                    code: -32602,
                    message: "Invalid params".to_owned(),
                }),
                id: Value::Null,
            }
        }
    }

    fn handle_prove(_req: JsonRpcRequest) -> JsonRpcResponse {
        JsonRpcResponse {
            jsonrpc: "2.0".to_owned(),
            result: None,
            error: Some(JsonRpcError {
                code: -32601,
                message: "Method not implemented".to_owned(),
            }),
            id: Value::Null,
        }
    }

    fn handle_verify(_req: JsonRpcRequest) -> JsonRpcResponse {
        JsonRpcResponse {
            jsonrpc: "2.0".to_owned(),
            result: None,
            error: Some(JsonRpcError {
                code: -32601,
                message: "Method not implemented".to_owned(),
            }),
            id: Value::Null,
        }
    }

    fn handle_not_supported() -> JsonRpcResponse {
        JsonRpcResponse {
            jsonrpc: "2.0".to_owned(),
            result: None,
            error: Some(JsonRpcError {
                code: -32601,
                message: "Method not found".to_owned(),
            }),
            id: Value::Null,
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
                            Ok(make_response(Self::handle_not_supported()))
                        }
                    }
                }
                Err(err) => {
                    error!("Error: {}", err);
                    Ok(make_response(Self::handle_not_supported()))
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
    pub backend: crate::engine::backend::BackendConfig,
}

impl Default for ServerConfig {
    fn default() -> Self {
        ServerConfig {
            host: "127.0.0.1".to_owned(),
            port: 1337,
            backend: crate::engine::backend::BackendConfig::new(4, [0u8; 32]),
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

    fn new_handler<B>(&self) -> RpcHandler<B>
    where
        B: crate::engine::backend::Backend,
    {
        let backend = B::new(self.cfg.backend.clone());
        RpcHandler::new(Arc::new(backend))
    }

    pub async fn run<B>(&self) -> std::io::Result<()>
    where
        B: crate::engine::backend::Backend + Send + Sync + 'static,
    {
        info!("Starting server...");
        let listener = tokio::net::TcpListener::bind(&self.addr()).await?;
        info!("Listening on: {}", self.addr());
        let handler = self.new_handler::<B>();

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

pub async fn start_rpc_server<B>()
where
    B: crate::engine::backend::Backend + Send + Sync + 'static,
{
    // init tracing subscriber and write to stdout
    tracing_subscriber::fmt()
        .with_writer(std::io::stdout)
        .init();
    info!("Starting RPC server...");
    let config = ServerConfig::default();
    let server = Server::new(config);
    server.run::<B>().await.unwrap();
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::backend::Backend;

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
    async fn test_rpc_handler_all_methods() {
        async fn test_method(
            method: &str,
            _expected_result: Option<&str>,
            expected_error: Option<&str>,
        ) {
            let raw_request = format!(r#"{{"jsonrpc":"2.0","method":"{}","id":1}}"#, method);
            let req = serde_json::from_str::<JsonRpcRequest>(&raw_request).unwrap();
            let cfg = crate::engine::backend::BackendConfig::new(4, [0u8; 32]);
            let handler =
                RpcHandler::new(Arc::new(crate::engine::arkworks::ArkworksBackend::new(cfg)));
            let res = handler.handle(req).await;
            if res.result.is_some() {
                // assert_eq!(
                //     res.result.unwrap().as_str().unwrap(),
                //     expected_result.unwrap()
                // );
            }
            if res.error.is_some() {
                assert_eq!(res.error.unwrap().message, expected_error.unwrap());
            }
        }

        test_method("ping", Some("pong"), None).await;
        test_method("commit", None, Some("Method not implemented")).await;
        test_method("prove", None, Some("Method not implemented")).await;
        test_method("verify", None, Some("Method not implemented")).await;
        test_method("any", None, Some("Method not found")).await;
    }

    #[tokio::test]
    #[tracing_test::traced_test]
    async fn test_server() {
        type Backend = crate::engine::arkworks::ArkworksBackend;

        async fn assert_response(address: &str, port: u16, method: &str, should_succeed: bool) {
            let client = reqwest::Client::new();
            let res = client
                .post(&format!("http://{}:{}/", address, port))
                .body(format!(
                    r#"{{"jsonrpc":"2.0","method":"{}","id":1}}"#,
                    method
                ))
                .send()
                .await
                .unwrap();
            assert_eq!(res.status(), 200);
            let resp = serde_json::from_str::<JsonRpcResponse>(&res.text().await.unwrap()).unwrap();
            if should_succeed {
                assert!(resp.result.is_some());
            } else {
                assert!(resp.error.is_some());
            }
        }

        const ADDRESS: &str = "127.0.0.1";
        const PORT: u16 = 1337;
        let server = Server::new(ServerConfig {
            host: ADDRESS.to_owned(),
            port: PORT,
            backend: crate::engine::backend::BackendConfig::new(4, [0u8; 32]),
        });
        tokio::spawn(async move {
            server.run::<Backend>().await.unwrap();
        });

        tokio::time::sleep(std::time::Duration::from_secs(1)).await;

        assert_response(ADDRESS, PORT, "ping", true).await;
        assert_response(ADDRESS, PORT, "commit", false).await;
        assert_response(ADDRESS, PORT, "prove", false).await;
        assert_response(ADDRESS, PORT, "verify", false).await;
        assert_response(ADDRESS, PORT, "any", false).await;
    }
}

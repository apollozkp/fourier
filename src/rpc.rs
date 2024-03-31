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
    Commit { commitment: String },
    Open { proof: String },
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

    #[serde(rename = "open")]
    Open,

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
            Method::Ping => Self::handle_ping(req),
            Method::Commit => self.handle_commit(req),
            Method::Open => self.handle_open(req),
            Method::Verify => self.handle_verify(req),
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
                        commitment: hex::encode(self.backend.commit_to_poly(poly).unwrap().to_bytes()),
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

    // #[test]
    // #[tracing_test::traced_test]
    // fn test_commit() {
    //     let raw_request = r#"{\"id\": 0, \"method\": \"commit\", \"params\": {\"poly\": [\"6945DC5C4FF4DA C8A7278C9B8F0D4613320CF87FF947F21AC9BF42327EC19448\", \"68E40C088D827BCCE02CEF34BDC8C12BB025FBEA047BC6C00C0C8C5C925B7FAF\", \"67281FAC164E9348B80693BA30D5D4E311DE5878EB3D20E34A585 07B484B243C\", \"5F7C377DAE6B9D9ABAD75DC15E4FFF9FE7520D1F85224C95F485F44978154C5A\", \"2D85C376A440B6E25C3F7C11559B6A27684023F36C3D7A0ACD7E7D019DE399C7\", \"4A6FB95F0241B3583771E7 99120C87AAE3C843ECDB50A38254A92E198968922F\", \"1005079F96EC412A719FE2E9FA67D421D98FB4DEC4181459E59430F5D502BD2A\", \"64960B8692062DCB01C0FFBAC569478A89AD880ED3C9DF710BED5CE75F484 693\", \"03C2882155A447642BD21FB1CF2553F80955713F09BBBBD9724E2CBFD8B19D41\", \"0AB07FECB59EE3435F6129FCD602CB519E56D7B426941633E37A3B676A24830F\", \"12FA5861459EFFBAE654827D98BFDF EA5545DDF8BB9628579463DA21F17462B5\", \"6A6296A0376D807530DB09DC8BB069FFDEC3D7541497B82C722A199D6B7C5B06\", \"153D2C81B54D7E1C3E83EA61C7F66FD88155F1713EE581E2BE8438CA9FEE1A02\", \ "216BCCC4AE97FE3E1D4B21C375C46140FA153E7868201A43480889047ACD0C2D\", \"381BD4FE924EB10E08F2A227D3DB2083AA0E5A1F661CD3C702C4B8A9385E7839\", \"723A7640FD7E65473131563AB5514916AC861C 2695CE6513E5061E597E5E1A81\"#;
    //     let serialized = serde_json::from_str::<JsonRpcRequest>(raw_request);
    //     assert!(serialized.is_ok());
    //     info!("Request: {:?}", serialized.unwrap());
    //     let cfg = crate::engine::backend::BackendConfig::new(4, [0u8; 32]);
    //     let handler =
    //         RpcHandler::new(Arc::new(crate::engine::arkworks::ArkworksBackend::new(cfg)));


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
                RpcHandler::new(Arc::new(crate::engine::arkworks::ArkworksBackend::new(Some(cfg))));
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
        test_method("open", None, Some("Method not implemented")).await;
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
            backend: None,
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

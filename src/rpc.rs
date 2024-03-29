use http_body_util::{BodyExt, Full};
use hyper::body::{Buf, Bytes};
use hyper::Response;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::convert::Infallible;
use tracing::{error, info};

#[derive(Debug, Serialize, Deserialize)]
struct JsonRpcRequest {
    jsonrpc: String,
    method: Method,
    #[serde(skip_serializing_if = "Option::is_none")]
    params: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    id: Option<Value>,
}

#[derive(Debug, Serialize, Deserialize)]
struct JsonRpcResponse {
    jsonrpc: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    result: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<JsonRpcError>,
    id: Value,
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

pub struct RpcHandler {}

impl RpcHandler {
    async fn handle(req: JsonRpcRequest) -> JsonRpcResponse {
        match req.method {
            Method::Ping => Self::handle_ping(),
            Method::Commit => Self::handle_commit(),
            Method::Prove => Self::handle_prove(),
            Method::Verify => Self::handle_verify(),
            Method::NotSupported => Self::handle_not_supported(),
        }
    }

    fn handle_ping() -> JsonRpcResponse {
        JsonRpcResponse {
            jsonrpc: "2.0".to_owned(),
            result: Some(Value::String("pong".to_owned())),
            error: None,
            id: Value::Null,
        }
    }

    fn handle_commit() -> JsonRpcResponse {
        JsonRpcResponse {
            jsonrpc: "2.0".to_owned(),
            result: None,
            error: Some(JsonRpcError {
                code: -32601,
                message: "commit request received".to_owned(),
            }),
            id: Value::Null,
        }
    }

    fn handle_prove() -> JsonRpcResponse {
        JsonRpcResponse {
            jsonrpc: "2.0".to_owned(),
            result: None,
            error: Some(JsonRpcError {
                code: -32601,
                message: "prove request received".to_owned(),
            }),
            id: Value::Null,
        }
    }

    fn handle_verify() -> JsonRpcResponse {
        JsonRpcResponse {
            jsonrpc: "2.0".to_owned(),
            result: None,
            error: Some(JsonRpcError {
                code: -32601,
                message: "verify method received".to_owned(),
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

#[derive(Debug)]
pub struct Config {
    pub host: String,
    pub port: u16,
}

impl Default for Config {
    fn default() -> Self {
        Config {
            host: "127.0.0.1".to_owned(),
            port: 1337,
        }
    }
}

#[derive(Debug, Default)]
pub struct Server {
    cfg: Config,
}

impl Server {
    pub fn new(cfg: Config) -> Self {
        Server {
            cfg,
        }
    }

    fn addr(&self) -> String {
        format!("{}:{}", self.cfg.host, self.cfg.port)
    }

    pub async fn run(&self) -> std::io::Result<()> {
        info!("Starting server...");
        let listener = tokio::net::TcpListener::bind(&self.addr()).await?;
        info!("Listening on: {}", self.addr());

        loop {
            let (stream, _) = listener.accept().await?;
            info!("Accepted connection from: {}", stream.peer_addr().unwrap());
            let io = hyper_util::rt::TokioIo::new(stream);
            tokio::task::spawn(async move {
                if let Err(err) = hyper::server::conn::http1::Builder::new()
                    .serve_connection(io, hyper::service::service_fn(Self::handle_request))
                    .await
                {
                    error!("Error: {}", err);
                }
            });
        }
    }

    async fn handle_request(
        req: hyper::Request<hyper::body::Incoming>,
    ) -> Result<Response<Full<Bytes>>, Infallible> {
        info!("Handling request: {:?}", req);
        match req.collect().await {
            Ok(body) => {
                info!("Received body: {:?}", body);
                let whole_body = body.aggregate();
                match serde_json::from_reader(whole_body.reader()) {
                    Ok(req) => {
                        info!("Deserialized request: {:?}", req);
                        let res = RpcHandler::handle(req).await;
                        match serde_json::to_vec(&res) {
                            Ok(serialized) => Ok(Response::new(Full::from(serialized))),
                            Err(err) => {
                                error!("Error: {}", err);
                                Ok(Response::new(Full::from("Error")))
                            }
                        }
                    }
                    Err(err) => {
                        error!("Error: {}", err);
                        Ok(Response::new(Full::from("Error")))
                    }
                }
            }
            Err(err) => {
                error!("Error: {}", err);
                Ok(Response::new(Full::from("Error")))
            }
        }
    }
}


pub async fn start_rpc_server() {
    // init tracing subscriber and write to stdout
    tracing_subscriber::fmt().with_writer(std::io::stdout).init();
    info!("Starting RPC server...");
    let server = Server::default();
    server.run().await.unwrap();
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

    #[tokio::test]
    #[tracing_test::traced_test]
    async fn test_rpc_handler() {
        let req = JsonRpcRequest {
            jsonrpc: "2.0".to_owned(),
            method: Method::Ping,
            params: None,
            id: None,
        };

        let res = RpcHandler::handle(req).await;
        assert_eq!(res.result, Some(Value::String("pong".to_owned())));
    }

    #[tokio::test]
    #[tracing_test::traced_test]
    async fn test_server() {
        const ADDRESS: &str = "127.0.0.1";
        const PORT: u16 = 1337;
        let server = Server::new(Config {
            host: ADDRESS.to_owned(),
            port: PORT,
        });
        tokio::spawn(async move {
            server.run().await.unwrap();
        });

        tokio::time::sleep(std::time::Duration::from_secs(1)).await;

        let client = reqwest::Client::new();
        let res = client
            .post(&format!("http://{}:{}/", ADDRESS, PORT))
            .body(r#"{"jsonrpc":"2.0","method":"say_hello","id":1}"#)
            .send()
            .await
            .unwrap();
        assert_eq!(res.status(), 200);
        println!("{:?}", res.text().await.unwrap());
    }

    #[tokio::test]
    #[tracing_test::traced_test]
    async fn test_start_rpc_server() {
        Server::default().run().await.unwrap();
    }

}

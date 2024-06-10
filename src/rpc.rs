use http_body_util::{BodyExt, Full};
use hyper::body::{Buf, Bytes};
use hyper::Response;
use kzg::{Fr, G1};
use rust_kzg_blst::types::fr::FsFr;
use rust_kzg_blst::types::g1::FsG1;
use serde::{Deserialize, Serialize};
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use tracing::{error, info};

use crate::cli::RunArgs;
use crate::engine::piano::PianoBackend;

#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "method", content = "params")]
#[serde(rename_all = "camelCase")]
pub enum RpcRequest {
    /// Worker Node Methods
    WorkerCommit {
        i: usize,
        poly: Vec<String>,
    },
    WorkerOpen {
        i: usize,
        poly: Vec<String>,
        x: String,
    },
    WorkerVerify {
        i: usize,
        alpha: String,
        proof: String,
        eval: String,
        commitment: String,
    },

    /// Master Node Methods
    MasterCommit {
        commitments: Vec<String>,
    },
    MasterOpen {
        evals: Vec<String>,
        proofs: Vec<String>,
        beta: String,
    },
    MasterVerify {
        commitment: String,
        beta: String,
        alpha: String,
        z: String,
        pi_0: String,
        pi_1: String,
    },

    /// Utils
    RandomPoly,
    RandomPoint,
    Evaluate {
        poly: Vec<String>,
        x: String,
    },
    Fft {
        poly: Vec<String>,
        left: bool,
        inverse: bool,
    },
    Ping,
}

#[derive(Debug, Serialize, Deserialize, PartialEq)]
#[serde(untagged)]
pub enum RpcResponse {
    Error { jsonrpc: String, error: String },
    Result { jsonrpc: String, result: RpcResult },
}

impl From<RpcResult> for RpcResponse {
    fn from(result: RpcResult) -> Self {
        RpcResponse::Result {
            jsonrpc: "2.0".to_owned(),
            result,
        }
    }
}

impl From<String> for RpcResponse {
    fn from(error: String) -> Self {
        RpcResponse::Error {
            jsonrpc: "2.0".to_owned(),
            error,
        }
    }
}

#[derive(Debug, Serialize, Deserialize, PartialEq)]
#[serde(untagged)]
pub enum RpcResult {
    /// Worker Node responses
    WorkerCommit {
        commitment: String,
    },
    WorkerOpen {
        proof: String,
        eval: String,
    },
    WorkerVerify {
        valid: bool,
    },

    /// Master Node responses
    MasterCommit {
        commitment: String,
    },
    MasterOpen {
        z: String,
        pi_0: String,
        pi_1: String,
    },
    MasterVerify {
        valid: bool,
    },

    /// Utils
    Pong,
    RandomPoly {
        poly: Vec<Vec<String>>,
    },
    RandomPoint {
        point: String,
    },
    Evaluate {
        y: String,
    },
    Fft {
        poly: Vec<String>,
    },
    Error {
        message: String,
    },
}

impl RpcResult {
    pub fn method(&self) -> &str {
        match self {
            RpcResult::WorkerCommit { .. } => "WorkerCommit",
            RpcResult::WorkerOpen { .. } => "WorkerOpen",
            RpcResult::WorkerVerify { .. } => "WorkerVerify",
            RpcResult::MasterCommit { .. } => "MasterCommit",
            RpcResult::MasterOpen { .. } => "MasterOpen",
            RpcResult::MasterVerify { .. } => "MasterVerify",
            RpcResult::RandomPoly { .. } => "RandomPoly",
            RpcResult::RandomPoint { .. } => "RandomPoint",
            RpcResult::Evaluate { .. } => "Evaluate",
            RpcResult::Fft { .. } => "Fft",
            RpcResult::Pong => "Pong",
            RpcResult::Error { .. } => "Error",
        }
    }
}

pub struct RpcHandler {
    backend: Arc<PianoBackend>,
}

impl Clone for RpcHandler {
    fn clone(&self) -> Self {
        RpcHandler {
            backend: self.backend.clone(),
        }
    }
}

impl RpcHandler {
    pub fn new(backend: Arc<PianoBackend>) -> Self {
        RpcHandler { backend }
    }

    async fn handle(&self, req: RpcRequest) -> Result<RpcResult, String> {
        match req {
            // Worker Node
            RpcRequest::WorkerCommit { .. } => self.handle_worker_commit(req),
            RpcRequest::WorkerOpen { .. } => self.handle_worker_open(req),
            RpcRequest::WorkerVerify { .. } => self.handle_worker_verify(req),

            // Master Node
            RpcRequest::MasterCommit { .. } => self.handle_master_commit(req),
            RpcRequest::MasterOpen { .. } => self.handle_master_open(req),
            RpcRequest::MasterVerify { .. } => self.handle_master_verify(req),

            // Utils
            RpcRequest::RandomPoly { .. } => self.handle_random_poly(req),
            RpcRequest::RandomPoint { .. } => self.handle_random_point(req),
            RpcRequest::Evaluate { .. } => self.handle_evaluate(req),
            RpcRequest::Fft { .. } => self.handle_fft(req),
            RpcRequest::Ping { .. } => Self::handle_ping(),
        }
    }

    fn handle_ping() -> Result<RpcResult, String> {
        Ok(RpcResult::Pong)
    }

    fn handle_fft(&self, req: RpcRequest) -> Result<RpcResult, String> {
        if let RpcRequest::Fft {
            ref poly,
            left,
            inverse,
        } = req
        {
            let poly = poly
                .iter()
                .map(|p| self.backend.parse_point_from_str(p))
                .collect::<Result<Vec<FsFr>, _>>()?;
            let result = if left {
                self.backend.fft_settings.fft_left(&poly, inverse)
            } else {
                self.backend.fft_settings.fft_right(&poly, inverse)
            }?;
            Ok(RpcResult::Fft {
                poly: result.iter().map(|x| hex::encode(x.to_bytes())).collect(),
            })
        } else {
            Err("Invalid params".to_owned())
        }
    }

    fn handle_master_commit(&self, req: RpcRequest) -> Result<RpcResult, String> {
        if let RpcRequest::MasterCommit { ref commitments } = req {
            let commitments = commitments
                .iter()
                .map(|c| self.backend.parse_g1_from_str(c))
                .collect::<Result<Vec<FsG1>, _>>()?;
            let commitment = self.backend.master_commit(&commitments);
            Ok(RpcResult::MasterCommit {
                commitment: hex::encode(commitment.to_bytes()),
            })
        } else {
            Err("Invalid params".to_owned())
        }
    }

    fn handle_master_open(&self, req: RpcRequest) -> Result<RpcResult, String> {
        if let RpcRequest::MasterOpen {
            ref evals,
            ref proofs,
            ref beta,
            ..
        } = req
        {
            let evals = evals
                .iter()
                .map(|e| self.backend.parse_point_from_str(e))
                .collect::<Result<Vec<FsFr>, _>>()?;
            let proofs = proofs
                .iter()
                .map(|p| self.backend.parse_g1_from_str(p))
                .collect::<Result<Vec<FsG1>, _>>()?;
            let beta = self.backend.parse_point_from_str(beta)?;
            let (eval, (proof_0, proof_1)) = self.backend.master_open(&evals, &proofs, &beta)?;
            Ok(RpcResult::MasterOpen {
                z: hex::encode(eval.to_bytes()),
                pi_0: hex::encode(proof_0.to_bytes()),
                pi_1: hex::encode(proof_1.to_bytes()),
            })
        } else {
            Err("Invalid params".to_owned())
        }
    }

    fn handle_master_verify(&self, req: RpcRequest) -> Result<RpcResult, String> {
        if let RpcRequest::MasterVerify {
            ref commitment,
            ref beta,
            ref alpha,
            ref z,
            ref pi_0,
            ref pi_1,
        } = req
        {
            let commitment = self.backend.parse_g1_from_str(commitment)?;
            let beta = self.backend.parse_point_from_str(beta)?;
            let alpha = self.backend.parse_point_from_str(alpha)?;
            let z = self.backend.parse_point_from_str(z)?;
            let pi_0 = self.backend.parse_g1_from_str(pi_0)?;
            let pi_1 = self.backend.parse_g1_from_str(pi_1)?;

            let valid = self
                .backend
                .master_verify(&commitment, &beta, &alpha, &z, &(pi_0, pi_1));
            Ok(RpcResult::MasterVerify { valid })
        } else {
            Err("Invalid params".to_owned())
        }
    }

    fn handle_worker_commit(&self, req: RpcRequest) -> Result<RpcResult, String> {
        if let RpcRequest::WorkerCommit { i, ref poly, .. } = req {
            let poly = self.backend.parse_poly_from_str(poly)?;
            let commitment = self.backend.commit(i, &poly)?;
            Ok(RpcResult::WorkerCommit {
                commitment: hex::encode(commitment.to_bytes()),
            })
        } else {
            Err("Invalid params".to_owned())
        }
    }

    fn handle_worker_open(&self, req: RpcRequest) -> Result<RpcResult, String> {
        if let RpcRequest::WorkerOpen {
            i, ref poly, ref x, ..
        } = req
        {
            let poly = self.backend.parse_poly_from_str(poly)?;
            let x = self.backend.parse_point_from_str(x)?;
            let (eval, proof) = self.backend.open(i, &poly, &x)?;
            Ok(RpcResult::WorkerOpen {
                proof: hex::encode(proof.to_bytes()),
                eval: hex::encode(eval.to_bytes()),
            })
        } else {
            Err("Invalid params".to_owned())
        }
    }

    fn handle_worker_verify(&self, req: RpcRequest) -> Result<RpcResult, String> {
        if let RpcRequest::WorkerVerify {
            i,
            ref proof,
            ref eval,
            ref alpha,
            ref commitment,
            ..
        } = req
        {
            let proof = self.backend.parse_g1_from_str(proof)?;
            let alpha = self.backend.parse_point_from_str(alpha)?;
            let eval = self.backend.parse_point_from_str(eval)?;
            let commitment = self.backend.parse_g1_from_str(commitment)?;
            let valid = self.backend.verify(i, &commitment, &alpha, &eval, &proof);
            Ok(RpcResult::WorkerVerify { valid })
        } else {
            Err("Invalid params".to_owned())
        }
    }

    fn handle_random_poly(&self, req: RpcRequest) -> Result<RpcResult, String> {
        if let RpcRequest::RandomPoly = req {
            let poly = self.backend.random_bivariate_polynomial();
            Ok(RpcResult::RandomPoly {
                poly: poly
                    .iter()
                    .map(|p| p.iter().map(|x| hex::encode(x.to_bytes())).collect())
                    .collect(),
            })
        } else {
            Err("Invalid params".to_owned())
        }
    }

    fn handle_random_point(&self, req: RpcRequest) -> Result<RpcResult, String> {
        if let RpcRequest::RandomPoint = req {
            let point = self.backend.random_point();
            Ok(RpcResult::RandomPoint {
                point: hex::encode(point.to_bytes()),
            })
        } else {
            Err("Invalid params".to_owned())
        }
    }

    fn handle_evaluate(&self, req: RpcRequest) -> Result<RpcResult, String> {
        if let RpcRequest::Evaluate {
            ref poly, ref x, ..
        } = req
        {
            let poly = self.backend.parse_poly_from_str(poly)?;
            let x = self.backend.parse_point_from_str(x)?;
            let y = self.backend.evaluate(&poly, &x);
            Ok(RpcResult::Evaluate {
                y: hex::encode(y.to_bytes()),
            })
        } else {
            Err("Invalid params".to_owned())
        }
    }
}

impl hyper::service::Service<hyper::Request<hyper::body::Incoming>> for RpcHandler {
    type Response = Response<Full<Bytes>>;
    type Error = hyper::Error;
    type Future = Pin<Box<dyn Future<Output = Result<Self::Response, Self::Error>> + Send>>;

    fn call(&self, req: hyper::Request<hyper::body::Incoming>) -> Self::Future {
        let make_response = |res: RpcResult| {
            let serialized = serde_json::to_vec(&res).unwrap();
            Response::new(Full::from(serialized))
        };

        let handler = self.clone();

        let future = async move {
            match req.collect().await {
                Ok(body) => {
                    info!("Received request");
                    let whole_body = body.aggregate();
                    match serde_json::from_reader(whole_body.reader()) {
                        Ok(req) => match handler.handle(req).await {
                            Ok(res) => {
                                tracing::debug!("Sending back response {}", res.method());
                                Ok(make_response(res))
                            }
                            Err(err) => {
                                tracing::error!("Error: {}", err);
                                Ok(make_response(RpcResult::Error {
                                    message: err.to_string(),
                                }))
                            }
                        },
                        Err(err) => {
                            tracing::error!("Error: {}", err);
                            Ok(make_response(RpcResult::Error {
                                message: err.to_string(),
                            }))
                        }
                    }
                }
                Err(e) => {
                    tracing::error!("Error: {}", e);
                    Ok(Response::new(Full::from("Error")))
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
    pub backend: crate::engine::config::DistributedBackendConfig,
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

    fn new_handler(&self) -> RpcHandler {
        let backend = PianoBackend::new(Some(self.cfg.backend.clone()));
        RpcHandler::new(Arc::new(backend))
    }

    pub async fn run(&self) -> std::io::Result<()> {
        info!("Starting RPC server...");
        let listener = tokio::net::TcpListener::bind(&self.addr()).await?;
        info!("Listening on: {}", self.addr());
        let handler = crate::utils::timed("start handler", || self.new_handler());

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

pub async fn start_rpc_server(cfg: Config) {
    let server = Server::new(cfg);
    if let Err(e) = server.run().await {
        error!("Error: {}", e);
    }
}

/// If this is your first time running these tests
/// OR
/// You have made changes to the constant values below
/// THEN
/// You should FIRST run the `setup_and_save` function to generate the setup and precompute
/// files.
/// If these files are not present, the tests will complain.
/// If the wrong files are presnet, the tests will also complain.
#[cfg(test)]
mod tests {
    use kzg::Poly;
    use rust_kzg_blst::types::poly::FsPoly;

    use crate::{bipoly::BivariateFsPolynomial, engine::config::DistributedBackendConfig};

    use super::*;

    const HOST: &str = "localhost";
    const PORT: usize = 1337;

    const SCALE: usize = 6;
    const MACHINES_SCALE: usize = 2;
    const COMPRESSED: bool = false;

    const SETUP_PATH: &str = "data/test_setup";
    const PRECOMPUTE_PATH: &str = "data/test_precompute";

    #[test]
    #[tracing_test::traced_test]
    fn test_serialize_deserialize() {
        const RAW_REQUESTS: [&str; 11] = [
            r#"{"method":"ping"}"#,
            r#"{"method":"randomPoly"}"#,
            r#"{"method":"randomPoint"}"#,
            r#"{"method":"evaluate","params":{"poly":["123","456"],"x":"789"}}"#,
            r#"{"method":"workerCommit","params":{"i":0,"poly":["123","456"]}}"#,
            r#"{"method":"workerOpen","params":{"i":0,"poly":["123","456"],"x":"789"}}"#,
            r#"{"method":"workerVerify","params":{"i":0,"alpha":"123","proof":"456","eval":"789","commitment":"abc"}}"#,
            r#"{"method":"masterCommit","params":{"commitments":["123","456"]}}"#,
            r#"{"method":"masterOpen","params":{"evals":["123","456"],"proofs":["789","abc"],"beta":"def"}}"#,
            r#"{"method":"masterVerify","params":{"commitment":"123","beta":"456","alpha":"789","z":"abc","pi_0":"def","pi_1":"ghi"}}"#,
            r#"{"method":"fft","params":{"poly":["123","456"],"left":true,"inverse":false}}"#,
        ];

        RAW_REQUESTS.iter().for_each(|raw| {
            let deserialized: RpcRequest = serde_json::from_str(raw).unwrap();
            let reserialized = serde_json::to_string(&deserialized).unwrap();
            assert_eq!(raw, &reserialized);
        });
    }

    fn test_config() -> DistributedBackendConfig {
        crate::engine::config::DistributedBackendConfig {
            machines_scale: MACHINES_SCALE,
            backend: crate::engine::config::BackendConfig {
                scale: SCALE,
                skip_precompute: false,
                compressed: COMPRESSED,
                precompute_path: None,
                setup_path: None,
            },
        }
    }

    #[tokio::test]
    #[tracing_test::traced_test]
    async fn test_setup_and_save() -> Result<(), String> {
        let backend_config = test_config();

        let mut setup_config = crate::engine::config::DistributedSetupConfig::from(backend_config);
        SETUP_PATH.clone_into(&mut setup_config.setup.setup_path);
        PRECOMPUTE_PATH.clone_into(&mut setup_config.setup.precompute_path);
        setup_config.setup.generate_precompute = true;
        setup_config.setup.generate_setup = true;
        setup_config.setup.overwrite = true;

        PianoBackend::setup_and_save(&setup_config).unwrap();
        
        // Check that the files were created and delete them
        let setup_path = setup_config.setup.setup_path.clone();
        let precompute_path = setup_config.setup.precompute_path.clone();
        assert!(std::path::Path::new(&setup_path).exists());
        assert!(std::path::Path::new(&precompute_path).exists());
        std::fs::remove_file(&setup_path).unwrap();
        std::fs::remove_file(&precompute_path).unwrap();
        Ok(())
    }

    async fn start_test_server(cfg: Config) {
        tokio::spawn(async move {
            start_rpc_server(cfg).await;
        });
        tokio::time::sleep(tokio::time::Duration::from_secs(1)).await;
    }

    #[tokio::test]
    #[tracing_test::traced_test]
    async fn test_handle_ping() -> Result<(), String> {
        let cfg = Config {
            host: HOST.to_owned(),
            port: PORT,
            backend: test_config(),
        };
        start_test_server(cfg).await;

        let req = RpcRequest::Ping;

        let client = reqwest::Client::new();
        let resp = client
            .get(&format!("http://{}:{}/", HOST, PORT))
            .body(serde_json::to_string(&req).unwrap())
            .send()
            .await
            .map_err(|e| e.to_string())?;

        assert_eq!(resp.status(), reqwest::StatusCode::OK);
        Ok(())
    }

    #[tokio::test]
    #[tracing_test::traced_test]
    async fn test_generate_poly() -> Result<(), String> {
        #[derive(Debug, Serialize, Deserialize)]
        struct Response {
            poly: Vec<Vec<String>>,
        }

        let cfg = Config {
            host: HOST.to_owned(),
            port: PORT,
            backend: test_config(),
        };
        start_test_server(cfg).await;

        let req = RpcRequest::RandomPoly;
        let client = reqwest::Client::new();
        let resp = client
            .get(&format!("http://{}:{}/", HOST, PORT))
            .body(serde_json::to_string(&req).unwrap())
            .send()
            .await
            .map_err(|e| e.to_string())?;
        let body = resp.text().await.map_err(|e| e.to_string())?;
        tracing::debug!("Response: {}", body);
        let response = serde_json::from_str::<Response>(&body).map_err(|e| e.to_string())?;

        assert_eq!(response.poly.len(), 2usize.pow(MACHINES_SCALE as u32));
        assert_eq!(
            response.poly[0].len(),
            2usize.pow((SCALE - MACHINES_SCALE) as u32)
        );

        Ok(())
    }

    #[tokio::test]
    #[tracing_test::traced_test]
    async fn test_generate_point() -> Result<(), String> {
        #[derive(Debug, Serialize, Deserialize)]
        struct Response {
            point: String,
        }

        let cfg = Config {
            host: HOST.to_owned(),
            port: PORT,
            backend: test_config(),
        };
        start_test_server(cfg).await;

        let req = RpcRequest::RandomPoint;
        let client = reqwest::Client::new();
        let resp = client
            .get(&format!("http://{}:{}/", HOST, PORT))
            .body(serde_json::to_string(&req).unwrap())
            .send()
            .await
            .map_err(|e| e.to_string())?;
        let body = resp.text().await.map_err(|e| e.to_string())?;
        tracing::debug!("Response: {}", body);
        let point = serde_json::from_str::<Response>(&body).map_err(|e| e.to_string())?;
        assert_eq!(point.point.len(), 64);
        Ok(())
    }

    #[tokio::test]
    #[tracing_test::traced_test]
    async fn test_evaluate() -> Result<(), String> {
        #[derive(Debug, Serialize, Deserialize)]
        struct Response {
            y: String,
        }

        let size = 2usize.pow((SCALE - MACHINES_SCALE) as u32);

        let poly = (0..size)
            .map(|i| FsFr::from_u64(i as u64))
            .collect::<Vec<_>>();
        let x = FsFr::one();
        let y = poly
            .iter()
            .rev()
            .fold(FsFr::zero(), |acc, p| acc.mul(&x).add(p));

        let cfg = Config {
            host: HOST.to_owned(),
            port: PORT,
            backend: test_config(),
        };
        start_test_server(cfg).await;

        let req = RpcRequest::Evaluate {
            poly: poly.iter().map(|x| hex::encode(x.to_bytes())).collect(),
            x: hex::encode(x.to_bytes()),
        };
        let client = reqwest::Client::new();
        let resp = client
            .get(&format!("http://{}:{}/", HOST, PORT))
            .body(serde_json::to_string(&req).unwrap())
            .send()
            .await
            .map_err(|e| e.to_string())?;
        let body = resp.text().await.map_err(|e| e.to_string())?;
        tracing::debug!("Response: {}", body);
        let response = serde_json::from_str::<Response>(&body).map_err(|e| e.to_string())?;
        assert_eq!(response.y, hex::encode(y.to_bytes()));
        Ok(())
    }

    #[tokio::test]
    #[tracing_test::traced_test]
    async fn test_worker_commit_open_verify() -> Result<(), String> {

        // Create setup files
        const SETUP_PATH: &str = "test_pipeline_setup";
        const PRECOMPUTE_PATH: &str = "test_pipeline_precompute";
        let mut test_config = test_config();
        test_config.backend.precompute_path = Some(PRECOMPUTE_PATH.to_owned());
        test_config.backend.setup_path = Some(SETUP_PATH.to_owned());

        let mut test_setup_config = crate::engine::config::DistributedSetupConfig::from(test_config.clone());
        test_setup_config.setup.generate_setup = true;
        test_setup_config.setup.generate_precompute = true;
        SETUP_PATH.clone_into(&mut test_setup_config.setup.setup_path);
        PRECOMPUTE_PATH.clone_into(&mut test_setup_config.setup.precompute_path);
        test_setup_config.setup.overwrite = true;
        PianoBackend::setup_and_save(&test_setup_config).unwrap();


        #[derive(Debug, Serialize, Deserialize)]
        struct WorkerCommitResponse {
            commitment: String,
        }

        #[derive(Debug, Serialize, Deserialize)]
        struct WorkerOpenResponse {
            proof: String,
            eval: String,
        }

        #[derive(Debug, Serialize, Deserialize)]
        struct WorkerVerifyResponse {
            valid: bool,
        }

        #[derive(Debug, Serialize, Deserialize)]
        struct MasterCommitResponse {
            commitment: String,
        }

        #[derive(Debug, Serialize, Deserialize)]
        struct MasterOpenResponse {
            z: String,
            pi_0: String,
            pi_1: String,
        }

        #[derive(Debug, Serialize, Deserialize)]
        struct MasterVerifyResponse {
            valid: bool,
        }

        async fn query_client<T>(
            client: reqwest::Client,
            port: usize,
            req: RpcRequest,
        ) -> Result<T, String>
        where
            T: for<'de> Deserialize<'de>,
        {
            let resp = client
                .get(&format!("http://{}:{}/", HOST, port))
                .body(serde_json::to_string(&req).unwrap())
                .send()
                .await
                .map_err(|e| e.to_string())?;
            let body = resp.text().await.map_err(|e| e.to_string())?;
            tracing::debug!("Response: {}", body);
            serde_json::from_str::<T>(&body).map_err(|e| e.to_string())
        }

        // Setup environment
        let machines_count = 2usize.pow(MACHINES_SCALE as u32);
        let sub_circuit_size = 2usize.pow((SCALE - MACHINES_SCALE) as u32);
        let backend = PianoBackend::new(Some(test_config.clone()));

        // Generate the polynomial we'll be working with in the standard basis
        // f(x, y) = sum_{i=0}^{n} sum_{j=0}^{m} f_{i,j} L(j) R(i)
        let lagrange_coeffs = backend.random_bivariate_polynomial();
        let mut full_poly = BivariateFsPolynomial::zero();
        (0..machines_count).for_each(|i| {
            (0..sub_circuit_size).for_each(|j| {
                let left = backend.fft_settings.left_lagrange_poly(j).unwrap();
                let right = backend.fft_settings.right_lagrange_poly(i).unwrap();
                let mut term = BivariateFsPolynomial::from_poly_as_x(&left)
                    .mul(&BivariateFsPolynomial::from_poly_as_y(&right));
                term = term.scale(&lagrange_coeffs[i][j]);
                full_poly = full_poly.add(&term);
            });
        });

        // Compute sub-polynomials in standard basis
        // NOTE: we can get rid of the standard basis entirely and work in the FFT basis
        let worker_polynomials = (0..machines_count)
            .map(|i| {
                let coeffs = lagrange_coeffs[i].clone();
                FsPoly::from_coeffs(&backend.fft_settings.fft_left(&coeffs, true).unwrap())
            })
            .collect::<Vec<_>>();

        // Client for making requests
        let client = reqwest::Client::new();

        // We set up worker servers and a validator server
        // This is not strictly necessary since all servers have the same setup, but good practice
        // for testing
        for i in 0..machines_count + 1 {
            let cfg = Config {
                host: HOST.to_owned(),
                port: PORT + i,
                backend: test_config.clone(),
            };

            start_test_server(cfg).await;
        }
        let validator_port = PORT + machines_count;

        // QUERY WORKER NODES
        // WORKER COMMIT
        tracing::debug!("Committing...");
        let mut worker_commitments = vec![];
        for (i, poly) in worker_polynomials.iter().enumerate() {
            let req = RpcRequest::WorkerCommit {
                i,
                poly: poly
                    .coeffs
                    .iter()
                    .map(|x| hex::encode(x.to_bytes()))
                    .collect(),
            };
            let response =
                query_client::<WorkerCommitResponse>(client.clone(), PORT + i, req).await?;
            let commitment = backend.parse_g1_from_str(&response.commitment)?;
            worker_commitments.push(commitment);
        }

        // WORKER OPEN
        tracing::debug!("Opening...");
        let mut worker_proofs = vec![];
        let alpha = FsFr::rand();
        for (i, poly) in worker_polynomials.iter().enumerate() {
            let req = RpcRequest::WorkerOpen {
                i,
                poly: poly
                    .coeffs
                    .iter()
                    .map(|x| hex::encode(x.to_bytes()))
                    .collect(),
                x: hex::encode(alpha.to_bytes()),
            };
            let response =
                query_client::<WorkerOpenResponse>(client.clone(), PORT + i, req).await?;
            let eval = backend.parse_point_from_str(&response.eval)?;
            let proof = backend.parse_g1_from_str(&response.proof)?;
            worker_proofs.push((eval, proof));
        }

        // QUERY MASTER NODE
        // VERIFY COMMITMENTS
        for i in 0..machines_count {
            let commitment = worker_commitments[i];
            let (y, pi_0) = worker_proofs[i];
            let req = RpcRequest::WorkerVerify {
                i,
                alpha: hex::encode(alpha.to_bytes()),
                proof: hex::encode(pi_0.to_bytes()),
                eval: hex::encode(y.to_bytes()),
                commitment: hex::encode(commitment.to_bytes()),
            };
            let response =
                query_client::<WorkerVerifyResponse>(client.clone(), validator_port, req).await?;
            assert!(response.valid);
            tracing::debug!("Worker {} verified OK!", i);
        }

        // COMPUTE MASTER COMMITMENT
        let req = RpcRequest::MasterCommit {
            commitments: worker_commitments
                .iter()
                .map(|c| hex::encode(c.to_bytes()))
                .collect(),
        };
        let response =
            query_client::<MasterCommitResponse>(client.clone(), validator_port, req).await?;
        let master_commitment = backend.parse_g1_from_str(&response.commitment)?;

        // COMPUTE MASTER OPENING
        let (evals, proofs) = worker_proofs.iter().fold(
            (Vec::new(), Vec::new()),
            |(mut evals, mut proofs), (y, pi)| {
                evals.push(*y);
                proofs.push(*pi);
                (evals, proofs)
            },
        );
        let beta = FsFr::rand();
        let req = RpcRequest::MasterOpen {
            evals: evals.iter().map(|e| hex::encode(e.to_bytes())).collect(),
            proofs: proofs.iter().map(|p| hex::encode(p.to_bytes())).collect(),
            beta: hex::encode(beta.to_bytes()),
        };
        let response =
            query_client::<MasterOpenResponse>(client.clone(), validator_port, req).await?;

        let z = backend.parse_point_from_str(&response.z)?;
        let pi_0 = backend.parse_g1_from_str(&response.pi_0)?;
        let pi_1 = backend.parse_g1_from_str(&response.pi_1)?;

        // VERIFY MASTER OPENING
        let req = RpcRequest::MasterVerify {
            commitment: hex::encode(master_commitment.to_bytes()),
            beta: hex::encode(beta.to_bytes()),
            alpha: hex::encode(alpha.to_bytes()),
            z: hex::encode(z.to_bytes()),
            pi_0: hex::encode(pi_0.to_bytes()),
            pi_1: hex::encode(pi_1.to_bytes()),
        };
        let response =
            query_client::<MasterVerifyResponse>(client.clone(), validator_port, req).await?;
        assert!(response.valid);
        Ok(())
    }
}

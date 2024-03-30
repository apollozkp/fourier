mod engine;
mod rpc;

#[tokio::main]
async fn main() {
    rpc::start_rpc_server().await;
}



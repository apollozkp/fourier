pub mod engine;
pub mod rpc;

#[tokio::main]
async fn main() {
    type Backend = engine::arkworks::ArkworksBackend;
    rpc::start_rpc_server::<Backend>().await;
}

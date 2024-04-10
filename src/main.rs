pub mod engine;
pub mod rpc;

use clap::Parser;

#[derive(Parser, Debug)]
#[command(name = "fourier-rpc", version = "0.1.0", about = "Fourier RPC server", long_about = None)]
struct Args {
    #[arg(short = 'H', long, default_value = "127.0.0.1")]
    host: String,
    #[arg(short = 'P', long, default_value = "1337")]
    port: u16,
    #[arg(short = 'p', long, default_value = None)]
    path: Option<String>,
}

#[tokio::main]
async fn main() {
    let args = Args::parse();
    type Backend = engine::arkworks::ArkworksBackend;
    rpc::start_rpc_server::<Backend>(args.port, args.path).await;
}

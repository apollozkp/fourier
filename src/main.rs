use fourier::cli::{RunArgs, SetupArgs, SubCommand};
use tracing::error;

use fourier::engine::backend::Backend as _;
use fourier::engine::blst::BlstBackend as Backend;

use clap::Parser;

pub(crate) fn setup(args: SetupArgs) {
    assert!(args.can_proceed());
    Backend::setup_and_save(args.into())
        .map_err(|e| error!("{}", e))
        .unwrap();
}

async fn run_server(args: RunArgs) {
    fourier::rpc::start_rpc_server::<Backend>(args.into()).await;
}

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .with_writer(std::io::stdout)
        .init();
    let cli = fourier::cli::CLi::parse();
    match cli.subcmd {
        SubCommand::Setup(args) => setup(args),
        SubCommand::Run(args) => run_server(args).await,
    }
}

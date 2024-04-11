pub mod engine;
pub mod rpc;
pub mod utils;

use tracing::error;

use crate::engine::backend::Backend as _;
use crate::engine::{backend::BackendConfig, blst::BlstBackend as Backend};

use clap::Parser;

#[derive(Parser, Debug)]
#[clap(name = "fourier-rpc", version = "0.1.0", about = "Fourier RPC server", long_about = None)]
struct CLi {
    #[clap(flatten)]
    global_opts: GlobalOpts,

    #[clap(subcommand)]
    subcmd: SubCommand,
}

#[derive(Parser, Debug)]
pub struct GlobalOpts {}

#[derive(Parser, Debug)]
enum SubCommand {
    Setup(SetupArgs),
    Run(RunArgs),
}

#[derive(Parser, Debug)]
struct SetupArgs {
    #[clap(long)]
    secrets_path: Option<String>,
    #[clap(long)]
    precompute_path: Option<String>,
    #[clap(long)]
    scale: Option<usize>,
    #[clap(long)]
    overwrite: bool,
    #[clap(long)]
    skip_secrets: bool,
}

impl SetupArgs {
    fn to_config(&self) -> BackendConfig {
        let cache_config = crate::engine::backend::BackendCacheConfig::new(
            if self.skip_secrets {self.secrets_path.clone()} else {None},
            None,
        );
        BackendConfig::new(
            self.scale,
            Some(cache_config),
            None,
            Some(self.skip_secrets),
        )
    }
}

#[derive(Parser, Debug)]
struct RunArgs {
    #[clap(long)]
    host: Option<String>,
    #[clap(long)]
    port: Option<u16>,
    #[clap(long)]
    scale: Option<usize>,
    #[clap(long)]
    secrets_path: Option<String>,
    #[clap(long)]
    precompute_path: Option<String>,
    #[clap(long)]
    skip_precompute: bool,
}

impl RunArgs {
    fn to_config(&self) -> rpc::Config {
        rpc::Config::new(
            self.port,
            self.host.clone(),
            self.secrets_path.clone(),
            self.precompute_path.clone(),
            self.scale,
            Some(self.skip_precompute),
        )
    }
}

pub(crate) fn setup(args: SetupArgs) {
    fn path_exists(path: Option<String>) -> bool {
        path.map(|p| std::path::Path::new(&p).exists())
            .unwrap_or(false)
    }
    if path_exists(args.secrets_path.clone()) && !args.overwrite {
        error!("File {} already exists, use --overwrite to overwrite", args.secrets_path.unwrap());
        return;
    }
    if path_exists(args.precompute_path.clone()) && !args.overwrite {
        error!("File {} already exists, use --overwrite to overwrite", args.precompute_path.unwrap());
        return;
    }

    let backend = utils::timed("setup", || {
        Backend::new(Some(args.to_config().clone()))
    });

    if let Err(e) = backend.save_to_file(args.secrets_path, args.precompute_path) {
        error!("Failed to save to file: {}", e);
    }
}

async fn run_server(args: RunArgs) {
    rpc::start_rpc_server::<Backend>(args.to_config()).await;
}

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .with_writer(std::io::stdout)
        .init();
    let cli = CLi::parse();
    match cli.subcmd {
        SubCommand::Setup(args) => setup(args),
        SubCommand::Run(args) => run_server(args).await,
    }
}

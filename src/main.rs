pub mod engine;
pub mod rpc;
pub mod utils;

use tracing::error;

use crate::engine::backend::Backend as _;
use crate::engine::blst::BlstBackend as Backend;

use clap::Parser;

#[derive(Parser, Debug)]
#[clap(name = "fourier-rpc", version = "0.1.0", about = "Fourier RPC server", long_about = None)]
struct CLi {
    #[clap(subcommand)]
    subcmd: SubCommand,
}

#[derive(Parser, Debug)]
enum SubCommand {
    Setup(SetupArgs),
    Run(RunArgs),
}

#[derive(Parser, Debug)]
struct RunArgs {
    // Path to the file where the setup is saved, omit to generate a new setup
    #[clap(long)]
    secrets_path: Option<String>,

    // Path to the file where the precomputed values are saved, omit to generate a new setup
    #[clap(long)]
    precompute_path: Option<String>,

    // The scale of the polynomial
    #[clap(long, default_value_t = 20)]
    scale: usize,

    // The host to bind to
    #[clap(long, default_value = "localhost")]
    host: String,

    // The port to bind to
    #[clap(long, default_value = "1337")]
    port: usize,
}

#[derive(Parser, Debug)]
struct SetupArgs {
    // Path to the file where the setup is saved
    #[clap(long, default_value = "setup")]
    secrets_path: String,

    // Path to the file where the precomputed values are saved
    #[clap(long, default_value = "precompute")]
    precompute_path: String,

    // The scale of the polynomial
    #[clap(long, default_value_t = 20)]
    scale: usize,

    // Overwrite the files if they already exist
    #[clap(long, default_value_t = false)]
    overwrite: bool,

    // Generate the secrets on setup, false will attempt to load them from the file
    #[clap(long, default_value_t = false)]
    generate_secrets: bool,

    // Generate the precomputed values on setup, false will attempt to load them from the file
    #[clap(long, default_value_t = false)]
    generate_precompute: bool,
}

impl SetupArgs {
    fn can_proceed(&self) -> bool {
        fn path_exists(path: &str) -> bool {
            std::path::Path::new(path).exists()
        }
        if path_exists(&self.secrets_path) && !self.overwrite {
            error!(
                "File {} already exists, use --overwrite to overwrite",
                self.secrets_path
            );
            return false;
        }
        if path_exists(&self.precompute_path) && !self.overwrite {
            error!(
                "File {} already exists, use --overwrite to overwrite",
                self.precompute_path
            );
            return false;
        }
        true
    }
}

pub(crate) fn setup(args: SetupArgs) {
    assert!(args.can_proceed());
    Backend::setup_and_save(args.into())
        .map_err(|e| error!("{}", e))
        .unwrap();
}

async fn run_server(args: RunArgs) {
    rpc::start_rpc_server::<Backend>(args.into()).await;
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

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
    setup_path: Option<String>,

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

    // Compression on off
    #[clap(long, default_value_t = false)]
    uncompressed: bool,
}

#[derive(Parser, Debug)]
struct SetupArgs {
    // Path to the file where the setup is saved
    #[clap(long, default_value = "setup")]
    setup_path: String,

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

    #[clap(long, default_value_t = false)]
    uncompressed: bool,

    // Compressed to uncompressed
    #[clap(long, default_value_t = false)]
    decompress_existing: bool,

    // uncompressed to compressed
    #[clap(long, default_value_t = false)]
    compress_existing: bool,
}

impl SetupArgs {
    fn can_proceed(&self) -> bool {
        fn path_exists(path: &str) -> bool {
            std::path::Path::new(path).exists()
        }
        if path_exists(&self.setup_path) && self.generate_secrets && !self.overwrite {
            error!(
                "File {} already exists, use --overwrite to overwrite",
                self.setup_path
            );
            return false;
        }
        if path_exists(&self.precompute_path) && self.generate_precompute && !self.overwrite {
            error!(
                "File {} already exists, use --overwrite to overwrite",
                self.precompute_path
            );
            return false;
        }
        if self.compress_existing && self.decompress_existing {
            error!("Cannot compress and decompress at the same time, choose one");
            return false;
        }
        if self.compress_existing && !self.uncompressed {
            error!("Cannot compress an already compressed file");
            return false;
        }
        if self.decompress_existing && self.uncompressed {
            error!("Cannot decompress an already decompressed file");
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

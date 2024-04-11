pub mod engine;
pub mod rpc;
pub mod utils;

use kzg::G1;
use kzg::G2;
use std::io::Write;
use tracing::error;
use tracing::warn;

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
    #[clap(short, long)]
    path: Option<String>,
    #[clap(short, long)]
    scale: Option<usize>,
    #[clap(short, long)]
    overwrite: bool,
}

#[derive(Parser, Debug)]
struct RunArgs {
    #[clap(short, long)]
    host: Option<String>,
    #[clap(short = 'P', long)]
    port: Option<u16>,
    #[clap(short = 'S', long)]
    scale: Option<usize>,
    #[clap(short, long)]
    setup_path: Option<String>,
    #[clap(short, long)]
    precompute: bool,
}

impl RunArgs {
    fn to_config(&self) -> rpc::Config {
        rpc::Config::new(self.port, self.host.clone(), self.setup_path.clone(), self.scale, Some(self.precompute))
    }
}

pub(crate) fn setup(args: SetupArgs) {
    if let Some(path) = &args.path {
        if std::path::Path::new(path).exists() {
            if args.overwrite {
                warn!("File already exists, will be overwritten");
            } else {
                error!("File already exists, use --overwrite to overwrite");
                return;
            }
        }
    }

    let backend = utils::timed("setup", || {
        let cfg = BackendConfig::new(args.scale, None, None);
        Backend::new(Some(cfg.clone()))
    });

    let file_path = args.path.unwrap_or("setup".to_string());
    let mut file = std::fs::File::create(file_path).unwrap();

    utils::timed("writing s1", || {
        let encoded_s1_size = backend.kzg_settings.secret_g1.len() as u64;
        Write::write(&mut file, &encoded_s1_size.to_le_bytes()).unwrap();
        for el in backend.kzg_settings.secret_g1.iter() {
            let bytes = el.to_bytes();
            Write::write(&mut file, &bytes).unwrap();
        }
    });

    utils::timed("writing s2", || {
        let encoded_s2_size = backend.kzg_settings.secret_g2.len() as u64;
        Write::write(&mut file, &encoded_s2_size.to_le_bytes()).unwrap();
        for el in backend.kzg_settings.secret_g2.iter() {
            let bytes = el.to_bytes();
            Write::write(&mut file, &bytes).unwrap();
        }
    });
}

async fn run_server(args: RunArgs) {
    let cfg = args.to_config();
    rpc::start_rpc_server::<Backend>(cfg).await;
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

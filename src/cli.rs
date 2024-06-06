use clap::Parser;
use tracing::error;

#[derive(Parser, Debug)]
#[clap(name = "fourier-rpc", version = "0.1.0", about = "Fourier RPC server", long_about = None)]
pub struct CLi {
    #[clap(subcommand)]
    pub subcmd: SubCommand,
}

#[derive(Parser, Debug)]
pub enum SubCommand {
    Setup(SetupArgs),
    Run(RunArgs),
}

#[derive(Parser, Debug, Default)]
pub struct RunArgs {
    // Path to the file where the setup is saved, omit to generate a new setup
    #[clap(long)]
    pub setup_path: Option<String>,

    // Path to the file where the precomputed values are saved, omit to generate a new setup
    #[clap(long)]
    pub precompute_path: Option<String>,

    // The scale of the polynomial
    #[clap(long, default_value_t = 20)]
    pub scale: usize,

    // The scale of the number of machines
    #[clap(long, default_value_t = 1)]
    pub machines_scale: usize,

    // The host to bind to
    #[clap(long, default_value = "localhost")]
    pub host: String,

    // The port to bind to
    #[clap(long, default_value = "1337")]
    pub port: usize,

    // Compression on off
    #[clap(long, default_value_t = false)]
    pub uncompressed: bool,
}

#[derive(Parser, Debug, Default)]
pub struct SetupArgs {
    // Path to the file where the setup is saved
    #[clap(long, default_value = "data/setup")]
    pub setup_path: String,

    // Path to the file where the precomputed values are saved
    #[clap(long, default_value = "data/precompute")]
    pub precompute_path: String,

    // The scale of the polynomial
    #[clap(long, default_value_t = 20)]
    pub scale: usize,

    // The scale of the number of machines
    #[clap(long, default_value_t = 1)]
    pub machines_scale: usize,

    // Overwrite the files if they already exist
    #[clap(long, default_value_t = false)]
    pub overwrite: bool,

    // Generate the setup on setup, false will attempt to load them from the file
    #[clap(long, default_value_t = false)]
    pub generate_setup: bool,

    // Generate the precomputed values on setup, false will attempt to load them from the file
    #[clap(long, default_value_t = false)]
    pub generate_precompute: bool,

    #[clap(long, default_value_t = false)]
    pub uncompressed: bool,

    // Compressed to uncompressed
    #[clap(long, default_value_t = false)]
    pub decompress_existing: bool,

    // uncompressed to compressed
    #[clap(long, default_value_t = false)]
    pub compress_existing: bool,
}

impl SetupArgs {
    pub fn can_proceed(&self) -> bool {
        fn path_exists(path: &str) -> bool {
            std::path::Path::new(path).exists()
        }
        if path_exists(&self.setup_path) && self.generate_setup&& !self.overwrite {
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

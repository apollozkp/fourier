use crate::cli::{RunArgs, SetupArgs};

#[derive(Debug, Clone)]
pub struct DistributedBackendConfig {
    pub machine_scale: usize,
    pub backend: BackendConfig,
}

#[derive(Debug, Clone)]
pub struct DistributedSetupConfig {
    pub machine_scale: usize,
    pub setup: SetupConfig,
}

impl From<DistributedBackendConfig> for DistributedSetupConfig {
    fn from(config: DistributedBackendConfig) -> Self {
        Self {
            machine_scale: config.machine_scale,
            setup: config.backend.into(),
        }
    }
}

impl From<SetupArgs> for DistributedSetupConfig {
    fn from(args: SetupArgs) -> Self {
        Self {
            machine_scale: 1,
            setup: args.into(),
        }
    }
}

impl From<RunArgs> for DistributedBackendConfig {
    fn from(args: RunArgs) -> Self {
        Self {
            machine_scale: args.machines_scale,
            backend: args.into(),
        }
    }
}

impl From<BackendConfig> for DistributedBackendConfig {
    fn from(config: BackendConfig) -> Self {
        Self {
            machine_scale: 1,
            backend: config,
        }
    }
}

impl DistributedBackendConfig {
    pub fn distribution_scale(&self) -> usize {
        self.machine_scale
    }

    pub fn config(&self) -> &BackendConfig {
        &self.backend
    }
}

impl Default for DistributedBackendConfig {
    fn default() -> Self {
        BackendConfig::default().into()
    }
}

#[derive(Debug, Clone)]
pub struct BackendConfig {
    pub setup_path: Option<String>,
    pub precompute_path: Option<String>,

    pub scale: usize,
    pub skip_precompute: bool,

    pub compressed: bool,
}

impl Default for BackendConfig {
    fn default() -> Self {
        Self {
            setup_path: None,
            precompute_path: None,
            scale: 20,
            skip_precompute: false,
            compressed: true,
        }
    }
}

impl From<RunArgs> for BackendConfig {
    fn from(args: RunArgs) -> Self {
        Self {
            setup_path: args.setup_path,
            precompute_path: args.precompute_path,
            scale: args.scale,
            skip_precompute: false,
            compressed: !args.uncompressed,
        }
    }
}

impl BackendConfig {
    pub fn new(
        setup_path: Option<String>,
        precompute_path: Option<String>,
        scale: usize,
        skip_precompute: Option<bool>,
        compressed: Option<bool>,
    ) -> Self {
        Self {
            setup_path,
            precompute_path,
            scale,
            skip_precompute: skip_precompute.unwrap_or(false),
            compressed: compressed.unwrap_or(true),
        }
    }

    pub fn setup_path(&self) -> Option<&str> {
        self.setup_path.as_deref()
    }

    pub fn precompute_path(&self) -> Option<&str> {
        self.precompute_path.as_deref()
    }

    pub fn scale(&self) -> usize {
        self.scale
    }

    pub fn skip_precompute(&self) -> bool {
        self.skip_precompute
    }

    pub fn compressed(&self) -> bool {
        self.compressed
    }
}

#[derive(Debug, Clone)]
pub struct SetupConfig {
    pub setup_path: String,
    pub precompute_path: String,

    pub scale: usize,
    pub overwrite: bool,
    pub generate_setup: bool,
    pub generate_precompute: bool,

    // Compression args
    pub compressed: bool,
    pub decompress_existing: bool,
    pub compress_existing: bool,
}

impl From<SetupArgs> for SetupConfig {
    fn from(args: SetupArgs) -> Self {
        Self {
            setup_path: args.setup_path,
            precompute_path: args.precompute_path,

            scale: args.scale,
            overwrite: args.overwrite,
            generate_setup: args.generate_setup,
            generate_precompute: args.generate_precompute,
            
            compressed: !args.uncompressed,
            decompress_existing: args.decompress_existing,
            compress_existing: args.compress_existing,
        }
    }
}

impl From<BackendConfig> for SetupConfig {
    fn from(args: BackendConfig) -> Self {
        const DEFAULT_SETUP_PATH: &str = "setup";
        const DEFAULT_PRECOMPUTE_PATH: &str = "precompute";
        // If no path is provided, generate
        // Resort to default path if not provided
        let generate_setup = args.setup_path.is_none();
        let setup_path = args
            .setup_path
            .unwrap_or(DEFAULT_SETUP_PATH.to_string());
        let generate_precompute = args.precompute_path.is_none();
        let precompute_path = args
            .precompute_path
            .unwrap_or(DEFAULT_PRECOMPUTE_PATH.to_string());
        Self {
            setup_path,
            precompute_path,
            scale: args.scale,
            overwrite: false,
            generate_setup,
            generate_precompute,
            compressed: args.compressed,
            decompress_existing: false,
            compress_existing: false,
        }
    }
}

impl Default for SetupConfig {
    fn default() -> Self {
        BackendConfig::default().into()
    }
}

impl SetupConfig {
    pub fn setup_path(&self) -> &str {
        self.setup_path.as_str()
    }

    pub fn precompute_path(&self) -> &str {
        self.precompute_path.as_str()
    }

    pub fn scale(&self) -> usize {
        self.scale
    }

    pub fn overwrite(&self) -> bool {
        self.overwrite
    }

    pub fn generate_setup(&self) -> bool {
        self.generate_setup
    }

    pub fn generate_precompute(&self) -> bool {
        self.generate_precompute
    }

    pub fn compressed(&self) -> bool {
        self.compressed
    }

    pub fn decompress_existing(&self) -> bool {
        self.decompress_existing
    }

    pub fn compress_existing(&self) -> bool {
        self.compress_existing
    }
}

use crate::{RunArgs, SetupArgs};

#[derive(Debug, Clone, Default)]
pub struct BackendConfig {
    pub setup_path: Option<String>,
    pub precompute_path: Option<String>,

    pub scale: usize,
    pub skip_precompute: bool,
}

impl From<RunArgs> for BackendConfig {
    fn from(args: RunArgs) -> Self {
        Self {
            setup_path: args.setup_path,
            precompute_path: args.precompute_path,
            scale: args.scale,
            skip_precompute: false,
        }
    }
}

impl BackendConfig {
    pub fn new(
        setup_path: Option<String>,
        precompute_path: Option<String>,
        scale: usize,
        skip_precompute: bool,
    ) -> Self {
        Self {
            setup_path,
            precompute_path,
            scale,
            skip_precompute,
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
}

#[derive(Debug, Clone)]
pub struct SetupConfig {
    pub setup_path: String,
    pub precompute_path: String,

    pub scale: usize,
    pub overwrite: bool,
    pub generate_setup: bool,
    pub generate_precompute: bool,
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
}

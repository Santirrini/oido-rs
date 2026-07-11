//! CLI parsing del bin `oido`.

use clap::Parser;

/// Estructura de argumentos CLI usando clap.
#[derive(Parser, Debug)]
#[command(name = "oido", version, about = "Local-first cross-platform voice dictation in Rust", long_about = None)]
pub(crate) struct Cli {
    /// Graba interactivamente la tecla de activación y la persiste a disco
    #[arg(long)]
    pub set_hotkey: bool,

    /// Configura el tema persistentemente
    #[arg(long, value_parser = ["dark", "light", "system"])]
    pub theme: Option<String>,

    /// Configura el idioma de UI persistentemente (ej: es, en)
    #[arg(long)]
    pub lang: Option<String>,

    /// Configura el prompt personalizado de whisper.cpp. Si se pasa,
    /// activa automáticamente `prompt_preset = custom`. Vacío = volver
    /// al preset por defecto (bilingüe ES/EN).
    #[arg(long)]
    pub set_prompt: Option<String>,

    /// Realiza un reporte de diagnóstico del sistema y sale
    #[arg(long)]
    pub check: bool,

    /// Muestra el path al archivo de configuración y sale
    #[arg(long)]
    pub config_path: bool,

    /// Busca y aplica actualizaciones de la aplicación (MSI) y sale
    #[arg(long)]
    pub check_update: bool,
}

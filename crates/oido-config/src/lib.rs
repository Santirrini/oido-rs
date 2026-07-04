//! Configuración persistente con escritura atómica cross-platform.
//!
//! Reglas:
//! - Único `Mutex` del workspace (`parking_lot`, regla R3).
//! - Escritura atómica vía tempfile + rename atómico (mismo directorio).
//! - Paths resueltos vía `dirs` (Windows → `%APPDATA%`, macOS →
//!   `~/Library/Application Support`, Linux → `$XDG_CONFIG_HOME` o
//!   `~/.config`).
//!
//! Implementación real entra en Fase 2. Aquí solo el esqueleto.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum ConfigError {
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("json inválido: {0}")]
    Json(#[from] serde_json::Error),
    #[error("config inválida: {0}")]
    Invalid(String),
}

/// Estructura serializable. Los defaults viven en `Config::default()`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    pub hotkey: String,
    pub model: String,
    pub language_ui: String,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            hotkey: "F8".into(),
            model: "ggml-base.bin".into(),
            language_ui: "es".into(),
        }
    }
}

/// Path canónico al directorio de configuración del usuario para Oido.
pub fn config_dir() -> PathBuf {
    dirs::config_dir()
        .unwrap_or_else(|| std::env::temp_dir())
        .join("oido")
}

/// Path al archivo `config.json` final.
pub fn config_file() -> PathBuf {
    config_dir().join("config.json")
}

/// Escribe contenido en `path` atómicamente. Crashtear a mitad de
/// escritura nunca corrompe el archivo destino.
pub fn atomic_write(path: &Path, contents: &[u8]) -> Result<(), ConfigError> {
    use std::io::Write;

    let dir = path.parent().ok_or_else(|| {
        ConfigError::Invalid(format!("path sin padre: {}", path.display()))
    })?;
    std::fs::create_dir_all(dir)?;

    // `NamedTempFile` en el mismo directorio garantiza rename atómico
    // tanto en Windows como en POSIX.
    let mut tmp = tempfile::NamedTempFile::new_in(dir)?;
    tmp.write_all(contents)?;
    tmp.persist(path).map_err(|e| ConfigError::Io(e.error))?;
    Ok(())
}

/// `ConfigStore` (Regla R3). Único mutex del workspace. `oido-core`
/// debe leerlo vía `with`, nunca por dentro de una región crítica
/// durante un callback de hotkey.
pub struct ConfigStore {
    inner: parking_lot::Mutex<Inner>,
}

struct Inner {
    config: Config,
    path: PathBuf,
}

impl ConfigStore {
    /// Carga desde disco si existe; si no, crea con `Config::default()`.
    pub fn load_or_default() -> Result<Self, ConfigError> {
        let path = config_file();
        let cfg = if path.exists() {
            let text = std::fs::read_to_string(&path)?;
            serde_json::from_str(&text)?
        } else {
            Config::default()
        };
        Ok(Self {
            inner: parking_lot::Mutex::new(Inner { config: cfg, path }),
        })
    }

    pub fn snapshot(&self) -> Config {
        self.inner.lock().config.clone()
    }

    /// Reemplaza la config en memoria. La persistencia es responsabilidad
    /// del llamador (`save()`) para permitir batches.
    pub fn replace(&self, new_cfg: Config) {
        self.inner.lock().config = new_cfg;
    }

    pub fn save(&self) -> Result<(), ConfigError> {
        let cfg = self.snapshot();
        let bytes = serde_json::to_vec_pretty(&cfg)?;
        let path = self.inner.lock().path.clone();
        atomic_write(&path, &bytes)
    }
}

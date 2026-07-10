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

/// Tema de color del icono de bandeja.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Theme {
    Dark,
    Light,
    System,
}

fn default_theme() -> Theme {
    Theme::System
}

/// Modo de transcripción de audio (Batch o Streaming).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SttMode {
    Batch,
    Streaming,
}

fn default_stt_mode() -> SttMode {
    SttMode::Batch
}

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
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Config {
    pub hotkey: String,
    pub model: String,
    pub language_ui: String,
    /// Activar aceleración por GPU. Default: `true` si el bin fue
    /// compilado con `--features cuda|metal|vulkan`; `false` en CPU.
    #[serde(default = "default_use_gpu")]
    pub use_gpu: bool,
    /// Número de threads para whisper.cpp. `None` = autodetectar
    /// (`min(cores, 8)`). Algunos valores límites: 1-8.
    #[serde(default = "default_n_threads")]
    pub n_threads: Option<u16>,
    /// Tema de la interfaz de bandeja. Default: `System` (sigue al OS).
    #[serde(default = "default_theme")]
    pub theme: Theme,
    /// Modo de transcripción. Default: `Batch` (por lotes, hold-to-talk clásico).
    #[serde(default = "default_stt_mode")]
    pub stt_mode: SttMode,
}

/// `default_use_gpu` se evalúa en runtime: detecta features compiladas.
pub fn default_use_gpu() -> bool {
    cfg!(any(feature = "cuda", feature = "metal", feature = "vulkan"))
}

fn default_n_threads() -> Option<u16> {
    None
}

impl Default for Config {
    fn default() -> Self {
        Self {
            hotkey: "F8".into(),
            model: "ggml-base.bin".into(),
            language_ui: "es".into(),
            use_gpu: default_use_gpu(),
            n_threads: None,
            theme: default_theme(),
            stt_mode: default_stt_mode(),
        }
    }
}

/// Path canónico al directorio de configuración del usuario para Oido.
pub fn config_dir() -> PathBuf {
    dirs::config_dir()
        .unwrap_or_else(std::env::temp_dir)
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

    let dir = path
        .parent()
        .ok_or_else(|| ConfigError::Invalid(format!("path sin padre: {}", path.display())))?;
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
#[derive(Debug)]
pub struct ConfigStore {
    inner: parking_lot::Mutex<Inner>,
}

#[derive(Debug)]
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

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    impl proptest::arbitrary::Arbitrary for Config {
        type Parameters = ();
        type Strategy = proptest::strategy::BoxedStrategy<Self>;
        fn arbitrary_with(_: Self::Parameters) -> Self::Strategy {
            (
                any::<String>(),
                any::<String>(),
                any::<String>(),
                any::<bool>(),
                proptest::option::of(1u16..=16),
                proptest::sample::select(vec![Theme::Dark, Theme::Light, Theme::System]),
                proptest::sample::select(vec![SttMode::Batch, SttMode::Streaming]),
            )
                .prop_map(
                    |(hotkey, model, language_ui, use_gpu, n_threads, theme, stt_mode)| Self {
                        hotkey,
                        model,
                        language_ui,
                        use_gpu,
                        n_threads,
                        theme,
                        stt_mode,
                    },
                )
                .boxed()
        }
    }

    #[test]
    fn default_config_has_sensible_values() {
        let cfg = Config::default();
        assert_eq!(cfg.hotkey, "F8");
        assert_eq!(cfg.model, "ggml-base.bin");
        assert_eq!(cfg.language_ui, "es");
        assert!(cfg.n_threads.is_none());
        assert_eq!(cfg.stt_mode, SttMode::Batch);
    }

    #[test]
    fn default_use_gpu_matches_compiled_features() {
        let cfg = Config::default();
        let expected = cfg!(any(feature = "cuda", feature = "metal", feature = "vulkan"));
        assert_eq!(cfg.use_gpu, expected);
    }

    /// JSON antiguo (sin los campos nuevos) debe cargar con defaults
    /// aplicados vía `serde(default)`. Esto garantiza retro-compat
    /// cuando el usuario actualiza de Fase 1 a Fase 2.
    #[test]
    fn backward_compat_missing_fields_use_defaults() {
        let json = r#"{"hotkey":"F9","model":"x.bin","language_ui":"en"}"#;
        let cfg: Config = serde_json::from_str(json).expect("JSON antiguo debe parsear");
        assert_eq!(cfg.hotkey, "F9");
        assert_eq!(cfg.use_gpu, default_use_gpu());
        assert!(cfg.n_threads.is_none());
        assert_eq!(cfg.stt_mode, SttMode::Batch);
    }

    #[test]
    fn backward_compat_missing_theme_field_uses_system() {
        let json = r#"{"hotkey":"F9","model":"x.bin","language_ui":"en"}"#;
        let cfg: Config = serde_json::from_str(json).expect("JSON sin theme debe parsear");
        assert_eq!(cfg.theme, Theme::System);
    }

    #[test]
    fn atomic_write_creates_file_with_expected_content() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("config.json");
        atomic_write(&path, b"{\"hotkey\":\"F9\"}").unwrap();
        let read = std::fs::read(&path).unwrap();
        assert_eq!(read, b"{\"hotkey\":\"F9\"}");
    }

    #[test]
    fn atomic_write_replaces_existing_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("config.json");
        atomic_write(&path, b"old").unwrap();
        atomic_write(&path, b"new").unwrap();
        let read = std::fs::read(&path).unwrap();
        assert_eq!(read, b"new");
    }

    #[test]
    fn atomic_write_leaves_no_tmp_leftovers() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("config.json");
        atomic_write(&path, b"hello").unwrap();
        // Sólo el archivo final debe existir; ningún `tmp*` colgado.
        let entries: Vec<_> = std::fs::read_dir(dir.path())
            .unwrap()
            .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
            .collect();
        assert_eq!(entries, vec!["config.json".to_string()]);
    }

    #[test]
    fn atomic_write_fails_when_parent_is_a_file() {
        // Path cuyo "padre" existe como archivo regular, no como
        // directorio. `create_dir_all` rechaza crear un dir donde ya
        // hay un file con ese nombre.
        let dir = tempfile::tempdir().unwrap();
        let blocker = dir.path().join("blocker");
        std::fs::write(&blocker, b"soy un archivo").unwrap();
        let bogus = blocker.join("config.json");
        let res = atomic_write(&bogus, b"x");
        assert!(res.is_err(), "path con padre-archivo debe fallar");
    }

    #[test]
    fn config_store_replace_then_snapshot() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("config.json");
        let store = ConfigStore {
            inner: parking_lot::Mutex::new(Inner {
                config: Config::default(),
                path: path.clone(),
            }),
        };
        let new_cfg = Config {
            hotkey: "Ctrl+Shift+D".into(),
            ..Config::default()
        };
        store.replace(new_cfg.clone());
        assert_eq!(store.snapshot().hotkey, "Ctrl+Shift+D");
    }

    #[test]
    fn config_store_save_then_read_back() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("config.json");
        let store = ConfigStore {
            inner: parking_lot::Mutex::new(Inner {
                config: Config::default(),
                path: path.clone(),
            }),
        };
        store.save().unwrap();
        // Releer manualmente (no usamos load_or_default porque ese mira
        // el path global del usuario).
        let raw = std::fs::read_to_string(&path).unwrap();
        let parsed: Config = serde_json::from_str(&raw).unwrap();
        assert_eq!(parsed, Config::default());
    }

    proptest! {
        /// Cualquier `Config` arbitrario sobrevive un roundtrip
        /// serde_json → bytes → serde_json (Regla: ningún campo se
        /// pierde ni se transforma).
        #[test]
        fn config_serde_roundtrip(cfg in any::<Config>()) {
            let bytes = serde_json::to_vec(&cfg).unwrap();
            let back: Config = serde_json::from_slice(&bytes).unwrap();
            prop_assert_eq!(cfg, back);
        }
    }
}

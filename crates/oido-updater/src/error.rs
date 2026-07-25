//! Errores del updater.
//!
//! Sigue la regla AGENTS.md "Error Handling": crates (`-core`, `-stt`,
//! ...) usan `thiserror` con enums específicos del dominio. `anyhow`
//! está reservado al bin `oido`. Cada variante se mapea a una clase
//! concreta de fallo para que el `?` de las llamadas internas no
//! pierda contexto.

use std::path::PathBuf;

/// Errores del crate `oido-updater`.
#[derive(Debug, thiserror::Error)]
pub enum UpdateError {
    #[error("red/HTTP: {0}")]
    Network(#[from] reqwest::Error),

    #[error("self_update: {0}")]
    SelfUpdate(#[from] self_update::errors::Error),

    #[error("I/O: {0}")]
    Io(#[from] std::io::Error),

    #[error("JSON inválido: {0}")]
    Json(#[from] serde_json::Error),

    #[error("release v{version} no contiene asset {suffix}")]
    AssetNotFound {
        version: String,
        suffix: &'static str,
    },

    #[error("descarga falló (status {status})")]
    DownloadFailed { status: u16 },

    #[error("SHA-256 no coincide (esperado {expected}, calculado {actual})")]
    ChecksumMismatch { expected: String, actual: String },

    #[error("firma Ed25519 inválida o ausente: {detail}")]
    SignatureInvalid { detail: String },

    #[error("instalador falló al ejecutarse: {detail}")]
    InstallFailed { detail: String },

    #[error("plataforma no soportada por este backend: {0}")]
    UnsupportedPlatform(&'static str),

    #[error("updater mal configurado: {0}")]
    Config(String),

    #[error("configuración inválida persistida: {0}")]
    InvalidPersistedConfig(String),

    #[error("otra: {0}")]
    Other(String),
}

impl UpdateError {
    /// ¿Es un fallo que merece notificación visible al usuario?
    /// Las variantes benignas (`UpToDate`, "skipped") NO devuelven
    /// `UpdateError`; esto es para errores donde el scheduler debe
    /// reportar algo al bin (tooltip, log, estado de error).
    pub fn is_user_visible(&self) -> bool {
        matches!(
            self,
            Self::Network(_)
                | Self::DownloadFailed { .. }
                | Self::ChecksumMismatch { .. }
                | Self::SignatureInvalid { .. }
                | Self::InstallFailed { .. }
                | Self::AssetNotFound { .. }
        )
    }

    /// Mensaje corto orientado a tooltip/tray (idioma ES por consistencia
    /// con el resto del menú de bandeja).
    pub fn user_message(&self) -> String {
        match self {
            Self::Network(_) => "Sin conexión a internet".to_string(),
            Self::DownloadFailed { status } => {
                format!("Descarga falló (HTTP {status})")
            }
            Self::ChecksumMismatch { .. } => "Checksum SHA-256 inválido".to_string(),
            Self::SignatureInvalid { .. } => "Firma del update inválida".to_string(),
            Self::InstallFailed { .. } => "Instalador falló".to_string(),
            Self::AssetNotFound { version, suffix } => {
                format!("Release v{version} sin asset {suffix}")
            }
            Self::UnsupportedPlatform(os) => {
                format!("Update no soportado en {os}")
            }
            other => format!("Update: {other}"),
        }
    }

    /// Path implicado, si la variante tiene uno (útil para logs).
    pub fn path(&self) -> Option<&PathBuf> {
        None
    }
}

/// Resultado especializado del crate.
pub type Result<T> = std::result::Result<T, UpdateError>;

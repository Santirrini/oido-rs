//! Tipos de datos del updater: resultados de check, install y eventos
//! hacia el bin (vía `crossbeam_channel`).
//!
//! `Status` se mantiene por compatibilidad (era el único tipo público
//! antes del refactor). `CheckResult` separa el "¿hay update?" del
//! "¿lo instalamos?" para que el scheduler pueda checkear sin instalar.
//! `UpdateEvent` es lo que viaja por el canal de control hacia el
//! binario para que el tray/notificación refleje el resultado.

use serde::{Deserialize, Serialize};

/// Resultado legacy de `check_and_apply` (mantener API para el bin).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Status {
    /// Versión actual ya es >= última publicada.
    UpToDate,
    /// Versión más reciente descargada e instalador en ejecución.
    DownloadedAndInstalling { version: String },
}

/// Resultado de un check sin acción (sólo compara versiones).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct CheckResult {
    /// Versión compilada en el bin (`CARGO_PKG_VERSION`).
    pub current: String,
    /// Última versión publicada en GitHub Releases.
    pub latest: String,
    /// `true` si `latest > current`.
    pub has_update: bool,
}

/// Evento que el scheduler / updater envía al binario vía canal de
/// control. El `match` en el control loop del bin hace `match` sobre
/// estos y dispara tooltip / `TrayState` / log.
///
/// Se serializa como `enum` plano (sin campos binarios) para que sea
/// trivial añadir variantes en el futuro.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum UpdateEvent {
    /// No hay update (current >= latest).
    UpToDate,
    /// Hay update disponible. El bin debe mostrar tooltip persistente.
    /// `version` es la versión nueva (latest).
    UpdateAvailable { version: String, current: String },
    /// Update descargado e instalador lanzado. El proceso debe cerrarse
    /// y el usuario relanzar para que la nueva versión quede activa.
    /// (Elegimos "usuario decide": no forzamos exit; el tooltip avisa.)
    Updated { version: String },
    /// Check saltado porque el usuario marcó "no molestes con vX".
    Skipped { version: String },
    /// Algo falló. `reason` es texto corto orientado a tooltip.
    Failed { reason: String },
}

impl UpdateEvent {
    /// Título corto para logs estructurados.
    pub fn kind(&self) -> &'static str {
        match self {
            Self::UpToDate => "up_to_date",
            Self::UpdateAvailable { .. } => "update_available",
            Self::Updated { .. } => "updated",
            Self::Skipped { .. } => "skipped",
            Self::Failed { .. } => "failed",
        }
    }
}

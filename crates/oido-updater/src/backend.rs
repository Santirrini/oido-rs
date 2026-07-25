//! Trait `InstallerBackend`: la abstracción multiplataforma.
//!
//! Cada OS tiene su propio mecanismo de instalar un binario:
//! - **Windows**: `.msi` instalado con `msiexec`.
//! - **macOS**: `.pkg` (futuro) o `.app` con `open -W` (futuro).
//! - **Linux**: `.deb` con `dpkg` (futuro) o `.AppImage` simplemente
//!   marcado ejecutable (futuro).
//!
//! El trait define el contrato común:
//! - Sufijo de asset a buscar en la release.
//! - Política de firma (hoy: obligatoria; futuro: opcional por canal).
//! - Instalación + resultado (`RequiresRestart` vs `InstalledRunning`).
//!
//! Los stubs macOS / Linux devuelven `UnsupportedPlatform` explícito
//! (no silencioso) para que quede claro en runtime que ese OS no
//! tiene update implementado todavía.

use std::path::Path;

use crate::error::Result;

/// Outcome de la instalación.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InstallOutcome {
    /// El instalador reemplazó archivos; el bin actual NO es la nueva
    /// versión. El usuario debe cerrar y relanzar.
    RequiresRestart,
    /// El bin se reemplazó in-place; el proceso actual ya corre la
    /// nueva versión (raro con installers — `self_update` lo hace así).
    InstalledRunning,
}

/// Backend de instalación por plataforma.
///
/// **Regla R2 AGENTS.md**: este trait debe ser 100% safe Rust. El
/// `unsafe` (si lo hubiera en un backend) vive encapsulado dentro de
/// la implementación; los demás módulos sólo ven este trait.
pub trait InstallerBackend: Send + Sync + 'static {
    /// Nombre legible (logs / tracing).
    fn name(&self) -> &'static str;

    /// Sufijo del asset a buscar en la release (ej. `".msi"`,
    /// `".pkg"`, `".AppImage"`).
    fn asset_suffix(&self) -> &'static str;

    /// ¿La firma Ed25519 es OBLIGATORIA para este backend?
    /// Si `true`, el orquestador rechaza la release si no hay `.minisig`.
    /// Si `false`, la firma se verifica si está presente pero no se exige.
    fn requires_signature(&self) -> bool {
        true
    }

    /// Instala el asset ya descargado y verificado.
    fn install(&self, asset_path: &Path) -> Result<InstallOutcome>;
}

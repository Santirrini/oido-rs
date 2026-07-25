//! `WindowsMsiBackend`: instala un `.msi` vía `msiexec`.
//!
//! Esta implementación es el único backend concreto hoy. macOS y
//! Linux tienen stubs que devuelven `UnsupportedPlatform` (ver
//! [`macos_stub`] / [`linux_stub`]) — no se compilan condicionalmente
//! en este archivo, viven aquí como siempre disponibles y rechazan
//! con error explícito en runtime.

use std::path::Path;

use crate::backend::{InstallOutcome, InstallerBackend};
use crate::error::{Result, UpdateError};

/// Backend Windows: instala un `.msi` con `msiexec /i ... /qb /norestart`.
///
/// Comportamiento:
/// - Flags: `/i` (install), `/qb` (basic UI — muestra progress, no
///   requiere interacción), `/norestart` (no fuerza reinicio del OS;
///   el proceso `oido` sí requiere relanzar, eso lo anuncia el
///   tooltip tras el evento `Updated`).
/// - `spawn` (no `status`) — el instalador corre detached. Si
///   quisiéramos esperar el exit code, bloquearíamos el thread del
///   scheduler. En su lugar, `msiexec` registra el resultado en el
///   Event Log de Windows que el usuario puede consultar.
#[derive(Debug)]
pub struct WindowsMsiBackend;

impl Default for WindowsMsiBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl WindowsMsiBackend {
    pub fn new() -> Self {
        Self
    }
}

impl InstallerBackend for WindowsMsiBackend {
    fn name(&self) -> &'static str {
        "windows-msi"
    }

    fn asset_suffix(&self) -> &'static str {
        ".msi"
    }

    fn install(&self, asset_path: &Path) -> Result<InstallOutcome> {
        let path_str = asset_path
            .to_str()
            .ok_or_else(|| UpdateError::InstallFailed {
                detail: format!("path no UTF-8: {}", asset_path.display()),
            })?;

        let status = std::process::Command::new("msiexec")
            .args(["/i", path_str, "/qb", "/norestart"])
            .spawn()
            .map_err(|e| UpdateError::InstallFailed {
                detail: format!("spawn msiexec: {e}"),
            })?;

        // `spawn` devuelve inmediatamente; el instalador corre en
        // background. El proceso `oido` actual sigue corriendo con la
        // versión vieja hasta que el usuario lo cierre (el tooltip
        // persistente se lo recuerda).
        tracing::info!(pid = ?status.id(), "msiexec lanzado en background");

        Ok(InstallOutcome::RequiresRestart)
    }
}

/// Stub macOS (futuro: `.pkg` con `installer -pkg ... -target /`).
///
/// **No** se compila condicionalmente: queremos que exista el símbolo
/// en el binario cross-compile (al menos en los paths de CI para
/// detectar el uso accidental desde un backend Windows).
#[derive(Debug)]
pub struct MacosPkgBackend;

impl InstallerBackend for MacosPkgBackend {
    fn name(&self) -> &'static str {
        "macos-pkg"
    }

    fn asset_suffix(&self) -> &'static str {
        ".pkg"
    }

    fn install(&self, _asset_path: &Path) -> Result<InstallOutcome> {
        Err(UpdateError::UnsupportedPlatform("macos"))
    }
}

/// Stub Linux (futuro: `.AppImage` con `chmod +x` o `.deb` con `dpkg -i`).
#[derive(Debug)]
pub struct LinuxAppImageBackend;

impl InstallerBackend for LinuxAppImageBackend {
    fn name(&self) -> &'static str {
        "linux-appimage"
    }

    fn asset_suffix(&self) -> &'static str {
        ".AppImage"
    }

    fn install(&self, _asset_path: &Path) -> Result<InstallOutcome> {
        Err(UpdateError::UnsupportedPlatform("linux"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn windows_backend_metadata() {
        let b = WindowsMsiBackend::new();
        assert_eq!(b.name(), "windows-msi");
        assert_eq!(b.asset_suffix(), ".msi");
        assert!(b.requires_signature());
    }

    #[test]
    fn macos_stub_rejects_install() {
        let b = MacosPkgBackend;
        let err = b
            .install(std::path::Path::new("/tmp/oido-0.1.0.pkg"))
            .unwrap_err();
        assert!(matches!(err, UpdateError::UnsupportedPlatform("macos")));
    }

    #[test]
    fn linux_stub_rejects_install() {
        let b = LinuxAppImageBackend;
        let err = b
            .install(std::path::Path::new("/tmp/oido-0.1.0.AppImage"))
            .unwrap_err();
        assert!(matches!(err, UpdateError::UnsupportedPlatform("linux")));
    }
}

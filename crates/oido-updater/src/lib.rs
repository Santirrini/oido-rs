//! Auto-updater del bin `oido`.
//!
//! ## Estructura
//!
//! - [`error`]: tipos `UpdateError` (thiserror) y `Result`.
//! - [`status`]: tipos de datos (`Status` legacy, `CheckResult`,
//!   `UpdateEvent` para el canal de control).
//! - [`github`]: fetch de la release de GitHub (`check_for_update`).
//! - [`verify`]: SHA-256 + firma Ed25519 (minisign) + `download_file`.
//! - [`backend`]: trait `InstallerBackend` (la abstracción
//!   multiplataforma).
//! - [`windows`]: `WindowsMsiBackend`, único backend implementado hoy.
//! - [`scheduler`]: `UpdateScheduler` (thread en background).
//!
//! ## Compatibilidad
//!
//! Este crate sólo expone contenido cuando se compila con la feature
//! `updater` activa. Sin esa feature, `mod` no existe y `use
//! oido_updater::*` da error claro. `check_and_apply` se conserva
//! como API pública para el bin `oido --check-update` (flujo one-shot
//! síncrono). El flujo continuo (auto-check en background) vive en
//! [`scheduler`].

#[cfg(feature = "updater")]
pub mod backend;
#[cfg(feature = "updater")]
pub mod error;
#[cfg(feature = "updater")]
pub mod github;
#[cfg(feature = "updater")]
pub mod scheduler;
#[cfg(feature = "updater")]
pub mod status;
#[cfg(feature = "updater")]
pub mod verify;
#[cfg(feature = "updater")]
pub mod windows;

#[cfg(feature = "updater")]
pub use error::{Result, UpdateError};
#[cfg(feature = "updater")]
pub use github::{check_for_update, find_asset_suffix, BIN_NAME, REPO};
#[cfg(feature = "updater")]
pub use status::{CheckResult, Status, UpdateEvent};
#[cfg(feature = "updater")]
pub use verify::{download_file, verify_minisign, verify_sha256, PUBLIC_KEY};

#[cfg(feature = "updater")]
pub use backend::{InstallOutcome, InstallerBackend};
#[cfg(feature = "updater")]
pub use windows::WindowsMsiBackend;

#[cfg(feature = "updater")]
pub use scheduler::{emit_one_shot_check, UpdateScheduler, UpdateSettings};

#[cfg(feature = "updater")]
use self_update::backends::github::Update;

/// Flujo one-shot usado por `oido --check-update` y por el menú
/// "Buscar actualizaciones" de la bandeja (cuando el usuario lo
/// dispara manualmente). Internamente:
/// 1. `check_for_update` → si no hay, retorna `UpToDate`.
/// 2. Descarga el asset del backend (`.msi` hoy).
/// 3. Descarga y verifica SHA-256.
/// 4. Descarga y verifica firma Ed25519 (si el backend la requiere).
/// 5. `backend.install(asset)` → instala.
/// 6. Devuelve `DownloadedAndInstalling`.
///
/// `backend` se inyecta para que tests puedan usar un mock.
#[cfg(feature = "updater")]
#[tracing::instrument(skip_all, fields(backend = backend.name()))]
pub fn check_and_apply(backend: &dyn InstallerBackend) -> Result<Status> {
    let current = env!("CARGO_PKG_VERSION");
    let result = check_for_update(current)?;
    if !result.has_update {
        return Ok(Status::UpToDate);
    }
    let version = result.latest.clone();

    let updater = Update::configure()
        .build()
        .map_err(|e| UpdateError::Config(format!("self_update Update build: {e}")))?;
    let release = updater.get_latest_release()?;

    // 1. Asset principal.
    let asset = find_asset_suffix(&release, backend.asset_suffix())?;
    // 2. SHA-256 sidecar.
    let sha_asset = find_asset_suffix(&release, ".sha256")?;
    // 3. Minisign sidecar (puede no existir; el backend decide si es
    //    obligatorio o sólo "nice to have").
    let minisig_asset = find_asset_suffix(&release, ".minisig").ok();

    let temp_dir = std::env::temp_dir();
    let asset_path = temp_dir.join(format!("oido-{}{}", version, backend.asset_suffix()));
    let sha_path = temp_dir.join(format!("oido-{}{}.sha256", version, backend.asset_suffix()));
    let minisig_path = temp_dir.join(format!(
        "oido-{}{}.minisig",
        version,
        backend.asset_suffix()
    ));

    verify::download_file(&asset.download_url, &asset_path)?;
    verify::download_file(&sha_asset.download_url, &sha_path)?;
    verify::verify_sha256(&asset_path, &sha_path)?;
    if backend.requires_signature() {
        let sig = minisig_asset.ok_or_else(|| UpdateError::SignatureInvalid {
            detail: format!(
                "backend {} requiere firma pero release no tiene .minisig",
                backend.name()
            ),
        })?;
        verify::download_file(&sig.download_url, &minisig_path)?;
        verify::verify_minisign(&asset_path, &minisig_path)?;
    } else if let Some(sig) = minisig_asset {
        // Verificar igual si está presente (defense in depth).
        let _ = verify::download_file(&sig.download_url, &minisig_path)
            .and_then(|_| verify::verify_minisign(&asset_path, &minisig_path));
    }

    backend.install(&asset_path)?;

    Ok(Status::DownloadedAndInstalling { version })
}

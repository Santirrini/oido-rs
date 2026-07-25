//! Fetch de metadata de releases desde GitHub Releases.
//!
//! Se apoya en `self_update::backends::github` para construir el
//! cliente (que ya tiene la lógica de auth/token y rate-limit
//! consistente con `self_update 0.44`). Sólo usamos el método
//! `get_latest_release`; **no** usamos el `update()` de self_update
//! porque (a) queremos nuestro propio flujo de verify (SHA-256 +
//! Ed25519) y (b) self_update reemplazaría el binario in-place, lo
//! cual no funciona con un .msi en Windows (el instalador debe
//! correr fuera del bin en uso).

use crate::error::{Result, UpdateError};
use crate::status::CheckResult;

/// `owner/repo` en GitHub Releases. Override por env en tests / forks.
pub const REPO: &str = "Santirrini/oido-rs";

/// Nombre del binario (referencial; el asset real es `.msi`).
pub const BIN_NAME: &str = "oido";

/// Compara `current_version` con la última release publicada en
/// GitHub. Devuelve `CheckResult` con la información necesaria para
/// que el caller decida si descargar / notificar / etc.
///
/// `current_version` normalmente es `env!("CARGO_PKG_VERSION")` pero
/// se inyecta para testabilidad.
#[tracing::instrument(skip_all, fields(repo = REPO, current = %current_version))]
pub fn check_for_update(current_version: &str) -> Result<CheckResult> {
    let (owner, repo) = REPO.split_once('/').ok_or_else(|| {
        UpdateError::Config(format!("REPO inválido (esperado owner/repo): {REPO}"))
    })?;

    let updater = self_update::backends::github::Update::configure()
        .repo_owner(owner)
        .repo_name(repo)
        .bin_name(BIN_NAME)
        .current_version(current_version)
        .build()
        .map_err(|e| UpdateError::Config(format!("self_update build: {e}")))?;

    let latest = updater.get_latest_release()?;

    let latest_version = latest.version.clone();
    let has_update =
        self_update::version::bump_is_greater(current_version, &latest_version).unwrap_or(false);

    Ok(CheckResult {
        current: current_version.to_string(),
        latest: latest_version,
        has_update,
    })
}

/// Encuentra el asset cuyo nombre termina en `suffix` dentro de un
/// release. Helper público para los backends instaladores.
pub fn find_asset_suffix<'a>(
    release: &'a self_update::update::Release,
    suffix: &'static str,
) -> Result<&'a self_update::update::ReleaseAsset> {
    release
        .assets
        .iter()
        .find(|a| a.name.ends_with(suffix))
        .ok_or_else(|| UpdateError::AssetNotFound {
            version: release.version.clone(),
            suffix,
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn repo_well_formed() {
        assert!(REPO.contains('/'), "REPO debe ser owner/repo");
        let (owner, repo) = REPO.split_once('/').unwrap();
        assert!(!owner.is_empty());
        assert!(!repo.is_empty());
    }

    #[test]
    fn bin_name_is_oido() {
        assert_eq!(BIN_NAME, "oido");
    }

    /// Versiones iguales → no hay update.
    #[test]
    fn version_equal_no_update() {
        let has = self_update::version::bump_is_greater("0.1.0", "0.1.0").unwrap_or(false);
        assert!(!has);
    }

    #[test]
    fn version_patch_update() {
        assert!(self_update::version::bump_is_greater("0.1.0", "0.1.1").unwrap_or(false));
    }

    #[test]
    fn version_minor_update() {
        assert!(self_update::version::bump_is_greater("0.1.0", "0.2.0").unwrap_or(false));
    }

    #[test]
    fn version_major_update() {
        assert!(self_update::version::bump_is_greater("0.1.0", "1.0.0").unwrap_or(false));
    }

    /// Regresión: la URL de GitHub Releases API que `self_update 0.44`
    /// consulta. Si self_update cambia esta URL, este test falla y
    /// obliga a actualizar `REPO` / el flujo de check.
    #[test]
    fn github_releases_url_is_documented() {
        let (owner, repo) = REPO.split_once('/').unwrap();
        let url = format!("https://api.github.com/repos/{owner}/{repo}/releases/latest");
        assert_eq!(
            url,
            "https://api.github.com/repos/Santirrini/oido-rs/releases/latest"
        );
    }

    /// `find_asset_suffix` devuelve `AssetNotFound` cuando no hay
    /// asset con ese sufijo. Construimos un `Release` directamente
    /// con los campos públicos del struct (`self_update 0.44` no
    /// implementa `Deserialize` para `Release`, sólo `Default`).
    #[test]
    fn find_asset_suffix_missing_returns_error() {
        let release = self_update::update::Release {
            name: "v0.2.0".into(),
            version: "0.2.0".into(),
            date: "2026-01-01".into(),
            body: None,
            assets: vec![self_update::update::ReleaseAsset {
                name: "oido-0.2.0.msi".into(),
                download_url: "https://x/oido-0.2.0.msi".into(),
            }],
        };
        let err = find_asset_suffix(&release, ".minisig").unwrap_err();
        assert!(matches!(err, UpdateError::AssetNotFound { .. }));
    }

    /// Happy path: el asset está, lo devuelve.
    #[test]
    fn find_asset_suffix_present_returns_asset() {
        let release = self_update::update::Release {
            name: "v0.2.0".into(),
            version: "0.2.0".into(),
            date: "2026-01-01".into(),
            body: None,
            assets: vec![
                self_update::update::ReleaseAsset {
                    name: "oido-0.2.0.msi".into(),
                    download_url: "https://x/oido-0.2.0.msi".into(),
                },
                self_update::update::ReleaseAsset {
                    name: "oido-0.2.0.msi.sha256".into(),
                    download_url: "https://x/oido-0.2.0.msi.sha256".into(),
                },
            ],
        };
        let asset = find_asset_suffix(&release, ".sha256").expect("sha256 debe estar");
        assert_eq!(asset.name, "oido-0.2.0.msi.sha256");
        assert_eq!(asset.download_url, "https://x/oido-0.2.0.msi.sha256");
    }
}

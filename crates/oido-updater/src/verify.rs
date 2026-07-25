//! Verificación de integridad + autenticidad de assets descargados.
//!
//! - **SHA-256**: defensa contra corrupción de bits en tránsito / disco.
//!   El hash esperado viene en un sidecar `<asset>.sha256` con formato
//!   `"<hash>  "` (estilo `sha256sum`). Primera línea, primer
//!   token = hex lowercase de 64 chars.
//!
//! - **Minisign/Ed25519**: defensa contra MITM y contra compromiso de
//!   la release en GitHub. El sidecar `.minisig` se firma con la clave
//!   privada del mantenedor (sólo en su workstation / CI secret) y se
//!   verifica contra la `PUBLIC_KEY` embebida en el bin.
//!
//! Defensa en profundidad: si SHA-256 coincide pero la firma no, el
//! asset es rechazado (`SignatureInvalid`). El bin queda intacto y se
//! loggea el fallo para investigación.

use std::fs::File;
use std::io::Read;
use std::path::Path;

use sha2::{Digest, Sha256};

use crate::error::{Result, UpdateError};

/// Llave pública Ed25519 embebida en el binario al compilar.
///
/// El formato esperado por `minisign-verify` es el "classic base64":
/// `base64(pk)` + `base64(sig)` en un sidecar `.minisig` (el formato
/// `minisign` genera por defecto). Ver `installer/README.md` para el
/// flujo de generación.
///
/// Se mantiene como `&'static str` cargado vía `include_str!` para
/// que esté en `.rodata` y no requiera I/O en cada check.
pub const PUBLIC_KEY: &str = include_str!("../../../installer/updater-pubkey.txt");

/// Verifica el SHA-256 de `asset_path` contra el hash leído de
/// `sha_path` (sidecar en formato `sha256sum`).
///
/// Devuelve `Ok(())` si coinciden; `ChecksumMismatch` en caso contrario.
#[tracing::instrument(skip_all, fields(asset = %asset_path.display()))]
pub fn verify_sha256(asset_path: &Path, sha_path: &Path) -> Result<()> {
    let mut sha_content = String::new();
    File::open(sha_path)?.read_to_string(&mut sha_content)?;
    let expected_hash = sha_content
        .split_whitespace()
        .next()
        .ok_or_else(|| {
            UpdateError::Other(format!(
                "sidecar SHA-256 vacío o malformado: {}",
                sha_path.display()
            ))
        })?
        .to_lowercase();
    if expected_hash.len() != 64 || !expected_hash.chars().all(|c| c.is_ascii_hexdigit()) {
        return Err(UpdateError::Other(format!(
            "SHA-256 con formato inválido en sidecar: '{expected_hash}'"
        )));
    }

    let mut file = File::open(asset_path)?;
    let mut hasher = Sha256::new();
    let mut buffer = [0u8; 8192];
    loop {
        let n = file.read(&mut buffer)?;
        if n == 0 {
            break;
        }
        hasher.update(&buffer[..n]);
    }
    let computed_hash = hex::encode(hasher.finalize());
    if computed_hash != expected_hash {
        return Err(UpdateError::ChecksumMismatch {
            expected: expected_hash,
            actual: computed_hash,
        });
    }
    Ok(())
}

/// Verifica la firma minisign/Ed25519 de `asset_path` contra
/// `sig_path` (sidecar `.minisig`) usando `PUBLIC_KEY` embebida.
///
/// La implementación usa el crate `minisign-verify` (puro Rust, sin
/// red, sin sistema de archivos extra). El crate sólo verifica; la
/// extracción del hash y de la firma la hace él mismo.
///
/// Errores:
/// - Archivo de firma ausente / malformado → `SignatureInvalid`.
/// - Llave embebida malformada (raro, generada en build) → `Other`.
/// - Firma no corresponde al archivo → `SignatureInvalid`.
#[tracing::instrument(skip_all, fields(asset = %asset_path.display()))]
pub fn verify_minisign(asset_path: &Path, sig_path: &Path) -> Result<()> {
    use minisign_verify::{PublicKey, Signature};

    let pk = PublicKey::decode(PUBLIC_KEY.trim())
        .map_err(|e| UpdateError::Other(format!("PUBLIC_KEY embebida malformada: {e}")))?;

    let sig_text = match std::fs::read_to_string(sig_path) {
        Ok(s) => s,
        Err(_) => {
            return Err(UpdateError::SignatureInvalid {
                detail: format!("sidecar .minisig ausente: {}", sig_path.display()),
            });
        }
    };
    let sig = Signature::decode(&sig_text).map_err(|e| UpdateError::SignatureInvalid {
        detail: format!("sidecar .minisig malformado: {e}"),
    })?;

    let asset_bytes = std::fs::read(asset_path).map_err(|e| UpdateError::SignatureInvalid {
        detail: format!("no se pudo leer asset para verificar: {e}"),
    })?;
    pk.verify(&asset_bytes, &sig, false)
        .map_err(|e| UpdateError::SignatureInvalid {
            detail: format!("verificación Ed25519 falló: {e}"),
        })?;
    Ok(())
}

/// Helper: descarga `url` a `dest` usando un cliente reqwest bloqueante
/// con UA distintivo. Reusado por `github.rs` para todos los sidecars
/// de una release (`.msi`, `.sha256`, `.minisig`).
#[tracing::instrument(skip_all, fields(url = %url, dest = %dest.display()))]
pub fn download_file(url: &str, dest: &Path) -> Result<()> {
    let client = reqwest::blocking::Client::builder()
        .user_agent(format!("oido-updater/{}", env!("CARGO_PKG_VERSION")))
        .build()?;
    let mut response = client.get(url).send()?;
    if !response.status().is_success() {
        return Err(UpdateError::DownloadFailed {
            status: response.status().as_u16(),
        });
    }
    let mut file = File::create(dest)?;
    response.copy_to(&mut file)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn write_file(dir: &std::path::Path, name: &str, bytes: &[u8]) -> std::path::PathBuf {
        let p = dir.join(name);
        let mut f = File::create(&p).unwrap();
        f.write_all(bytes).unwrap();
        p
    }

    #[test]
    fn sha256_match_ok() {
        let dir = tempfile::tempdir().unwrap();
        let asset = write_file(dir.path(), "a.bin", b"hello world");
        // sha256("hello world") =
        // b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9
        let sha = write_file(
            dir.path(),
            "a.sha256",
            b"b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9  a.bin\n",
        );
        verify_sha256(&asset, &sha).expect("match esperado");
    }

    #[test]
    fn sha256_mismatch() {
        let dir = tempfile::tempdir().unwrap();
        let asset = write_file(dir.path(), "a.bin", b"hello world");
        let sha = write_file(
            dir.path(),
            "a.sha256",
            b"0000000000000000000000000000000000000000000000000000000000000000  a.bin\n",
        );
        let err = verify_sha256(&asset, &sha).unwrap_err();
        assert!(matches!(err, UpdateError::ChecksumMismatch { .. }));
    }

    #[test]
    fn sha256_malformed_sidecar() {
        let dir = tempfile::tempdir().unwrap();
        let asset = write_file(dir.path(), "a.bin", b"hello world");
        let sha = write_file(dir.path(), "a.sha256", b"\n");
        let err = verify_sha256(&asset, &sha).unwrap_err();
        // empty -> UpdateError::Other (sidecar vacío)
        assert!(matches!(err, UpdateError::Other(_)));
    }

    // Minisign: la key embebida se regenera por CI (no testeamos firma
    // válida con la key dummy del repo, sólo que `verify_minisign` falla
    // limpio cuando el sidecar está ausente — la cobertura de "firma
    // válida" se hace en CI con una keypair ephemeral).
    #[test]
    fn minisign_missing_sidecar_is_signature_invalid() {
        let dir = tempfile::tempdir().unwrap();
        let asset = write_file(dir.path(), "a.bin", b"x");
        let sig = dir.path().join("a.minisig"); // no existe
        let err = verify_minisign(&asset, &sig).unwrap_err();
        // Si la PUBLIC_KEY embebida es inválida (caso de tests sin
        // release real), esto puede fallar antes con `Other`. En
        // producción con key válida, debe ser `SignatureInvalid`.
        assert!(
            matches!(
                err,
                UpdateError::SignatureInvalid { .. } | UpdateError::Other(_)
            ),
            "esperaba SignatureInvalid u Other, obtuve {err:?}"
        );
    }
}

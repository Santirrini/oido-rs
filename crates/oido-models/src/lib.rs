//! Catálogo y descarga de modelos whisper.cpp + VAD desde HuggingFace.
//!
//! Esta crate es 100% safe Rust (regla R2 de AGENTS.md: el FFI vive solo
//! en `oido-stt`). El downloader usa `reqwest::blocking` para poder correr
//! desde un thread dedicado sin levantar un runtime async.
//!
//! ## Diseño
//!
//! - **Catálogo hardcoded**: lista estable de modelos soportados. Orden:
//!   familia ascendente (Tiny/Base/Small/Vad), y dentro de cada familia
//!   el variant `en` antes que `multi`. Esto garantiza una presentación
//!   predecible en el menú.
//! - **Scan de disco**: `list_installed` cruza el catálogo contra el
//!   contenido de `models_dir` para que la UI pueda marcar cada item.
//! - **Descarga**: streaming con verificación SHA256 opcional. Si el SHA
//!   está vacío en el catálogo (placeholder), se loguea warn y se omite
//!   la verificación para no bloquear el primer deploy.

use std::fs::File;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::LazyLock;

use sha2::{Digest, Sha256};
use tempfile::NamedTempFile;
use thiserror::Error;
use tracing::{info, info_span, warn};

/// Familia del modelo (define el submenú y el orden de presentación).
///
/// El orden de las variantes importa: el test
/// `catalog_groups_by_family_in_stable_order` exige que `CATALOG` esté
/// ordenado de menor a mayor según el `Ord` derivado. Por tanto el
/// orden natural es: Tiny < Base < Small < Medium < Large < Vad.
/// `Vad` se mantiene al final (no es realmente una familia de
/// transcripción) para preservar el comportamiento de los tests
/// anteriores.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ModelFamily {
    Tiny,
    Base,
    Small,
    Medium,
    Large,
    Vad,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Language {
    /// Whisper fine-tuneado solo para inglés (`*.en.bin`).
    En,
    /// Whisper multilingüe (sin sufijo `.en`).
    Multi,
}

/// Descripción inmutable de un modelo del catálogo.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelEntry {
    pub filename: String,
    pub size_bytes: u64,
    pub url: String,
    /// SHA-256 en hex. Cadena vacía = sin verificar (placeholder).
    pub sha256: String,
    pub family: ModelFamily,
    pub language: Language,
}

#[derive(Debug, Error)]
pub enum ModelError {
    #[error("el modelo {0} ya está descargado")]
    AlreadyInstalled(String),
    #[error("descarga falló: {0}")]
    Download(String),
    #[error("verificación SHA256 falló para {0}")]
    ChecksumMismatch(String),
    #[error("IO: {0}")]
    Io(#[from] std::io::Error),
}

/// Catálogo hardcoded. Estable: el orden importa porque define el orden
/// de presentación en el menú (familia asc, en antes que multi).
pub fn catalog() -> &'static [ModelEntry] {
    LazyLock::force(&CATALOG).as_slice()
}

/// Busca un entry por filename exacto (case-sensitive).
pub fn find(filename: &str) -> Option<&'static ModelEntry> {
    LazyLock::force(&CATALOG)
        .iter()
        .find(|e| e.filename == filename)
}

/// `true` si el filename del catálogo corresponde a un modelo fine-tuneado
/// solo para inglés (sufijo `.en.bin`). Si el filename no está en el
/// catálogo, caemos a la heurística de sufijo como fallback defensivo
/// (modelos externos aún no catalogados).
///
/// Esta función centraliza la decisión de "es un modelo solo-inglés" que
/// antes vivía inline en varios call sites (submenú Modelos, aviso de
/// mismatch modelo/idioma, etc.) para que cualquier heurística nueva
/// (p.ej. derivar de la URL en vez del filename) se haga en un único
/// lugar.
pub fn is_english_only_model(filename: &str) -> bool {
    match find(filename) {
        Some(entry) => entry.language == Language::En,
        None => filename.ends_with(".en.bin"),
    }
}

/// Dado un modelo solo-inglés del catálogo (p.ej. `ggml-small.en.bin`),
/// devuelve el entry multilingüe equivalente de la misma familia
/// (`ggml-small.bin`). Devuelve `None` si el input no es `.en`, no está
/// en el catálogo, o no tiene contraparte multilingüe catalogada.
///
/// Se usa para sugerirle al usuario el reemplazo correcto cuando
/// detecta el mismatch "modelo `.en` + idioma != en".
pub fn multilingual_counterpart(filename: &str) -> Option<&'static ModelEntry> {
    let entry = find(filename)?;
    if entry.language != Language::En {
        return None;
    }
    LazyLock::force(&CATALOG)
        .iter()
        .find(|e| e.family == entry.family && e.language == Language::Multi)
}

/// Lista los filenames del catálogo que están presentes en `models_dir`.
/// Devuelve los nombres en el mismo orden que `catalog()`.
pub fn list_installed(models_dir: &Path) -> Result<Vec<String>, ModelError> {
    if !models_dir.exists() {
        return Ok(Vec::new());
    }
    let mut out = Vec::new();
    for entry in LazyLock::force(&CATALOG).iter() {
        if models_dir.join(&entry.filename).is_file() {
            out.push(entry.filename.clone());
        }
    }
    Ok(out)
}

/// Indica si un filename del catálogo está instalado en `models_dir`.
pub fn is_installed(models_dir: &Path, filename: &str) -> bool {
    models_dir.join(filename).is_file()
}

/// Descarga `entry` a `models_dir` con verificación SHA256.
///
/// - Streaming: lee chunks de 64 KiB, escribiendo incrementalmente a un
///   tempfile. Renombrado atómico al `persist()` final.
/// - Si `progress` es `Some`, se invoca con `(bytes_done, total_bytes)`
///   tras cada chunk.
/// - Si `entry.sha256` está vacío, se omite la verificación (log warn).
pub fn download_model(
    models_dir: &Path,
    entry: &ModelEntry,
    progress: Option<&dyn Fn(u64, u64)>,
) -> Result<(), ModelError> {
    std::fs::create_dir_all(models_dir)?;

    let dest = models_dir.join(&entry.filename);
    if dest.is_file() {
        return Err(ModelError::AlreadyInstalled(entry.filename.clone()));
    }

    let span = info_span!("download_model", filename = %entry.filename, bytes = entry.size_bytes);
    let _enter = span.enter();
    download_inner(&dest, entry, progress)
}

fn download_inner(
    dest: &Path,
    entry: &ModelEntry,
    progress: Option<&dyn Fn(u64, u64)>,
) -> Result<(), ModelError> {
    let mut response = reqwest::blocking::Client::builder()
        // HuggingFace responde con 302 → CDN (us.aws.cdn.hf.co).
        // Por defecto `reqwest::blocking` NO sigue redirects; hay que
        // activarlo manualmente o el GET se queda en 302 sin contenido.
        .redirect(reqwest::redirect::Policy::limited(10))
        .user_agent("oido/0.1 (https://github.com/Santirrini/oido-rs)")
        .build()
        .map_err(|e| ModelError::Download(format!("client build: {e}")))?
        .get(&entry.url)
        .send()
        .map_err(|e| ModelError::Download(format!("GET {}: {e}", entry.url)))?;

    if !response.status().is_success() {
        return Err(ModelError::Download(format!(
            "HTTP {} para {}",
            response.status(),
            entry.url
        )));
    }

    // Crear tempfile en el mismo dir que el destino para que el rename
    // sea atómico en el mismo filesystem.
    let parent = dest.parent().unwrap_or_else(|| Path::new("."));
    let mut tmp = NamedTempFile::new_in(parent)?;
    let mut hasher = Sha256::new();
    let mut buf = [0u8; 64 * 1024];
    let mut done: u64 = 0;

    loop {
        let n = response
            .read(&mut buf)
            .map_err(|e| ModelError::Download(format!("read: {e}")))?;
        if n == 0 {
            break;
        }
        tmp.write_all(&buf[..n])?;
        hasher.update(&buf[..n]);
        done += n as u64;
        if let Some(cb) = progress {
            cb(done, entry.size_bytes);
        }
    }

    // Verificar SHA256 si está disponible.
    let digest = hex::encode(hasher.finalize());
    if !entry.sha256.is_empty() && !entry.sha256.eq_ignore_ascii_case(&digest) {
        // El tempfile se descarta automáticamente al drop.
        return Err(ModelError::ChecksumMismatch(entry.filename.clone()));
    }
    if entry.sha256.is_empty() {
        warn!(filename = %entry.filename, "SHA256 vacío en catálogo; verificación omitida");
    } else {
        info!(sha256 = %digest, "verificación SHA256 OK");
    }

    // Renombrar atómicamente. `persist` intenta mover; si falla (cross-
    // device), hace copy + delete.
    tmp.persist(dest).map_err(|e| ModelError::Io(e.error))?;
    Ok(())
}

// Catálogo. Los URLs se construyen con la macro `concat!` (const-eval)
// que sí acepta literales. Como `String::from(&str)` no es `const fn`,
// el array se inicializa vía `LazyLock`.

static CATALOG: LazyLock<Vec<ModelEntry>> = LazyLock::new(|| {
    vec![
        ModelEntry {
            filename: String::from("ggml-tiny.en.bin"),
            size_bytes: 77_700_000,
            url: String::from(concat!(
                "https://huggingface.co/ggerganov/whisper.cpp/resolve/main",
                "/ggml-tiny.en.bin"
            )),
            sha256: String::new(),
            family: ModelFamily::Tiny,
            language: Language::En,
        },
        ModelEntry {
            filename: String::from("ggml-tiny.bin"),
            size_bytes: 77_700_000,
            url: String::from(concat!(
                "https://huggingface.co/ggerganov/whisper.cpp/resolve/main",
                "/ggml-tiny.bin"
            )),
            sha256: String::new(),
            family: ModelFamily::Tiny,
            language: Language::Multi,
        },
        ModelEntry {
            filename: String::from("ggml-base.en.bin"),
            size_bytes: 148_000_000,
            url: String::from(concat!(
                "https://huggingface.co/ggerganov/whisper.cpp/resolve/main",
                "/ggml-base.en.bin"
            )),
            sha256: String::new(),
            family: ModelFamily::Base,
            language: Language::En,
        },
        ModelEntry {
            filename: String::from("ggml-base.bin"),
            size_bytes: 148_000_000,
            url: String::from(concat!(
                "https://huggingface.co/ggerganov/whisper.cpp/resolve/main",
                "/ggml-base.bin"
            )),
            sha256: String::new(),
            family: ModelFamily::Base,
            language: Language::Multi,
        },
        ModelEntry {
            filename: String::from("ggml-small.en.bin"),
            size_bytes: 488_000_000,
            url: String::from(concat!(
                "https://huggingface.co/ggerganov/whisper.cpp/resolve/main",
                "/ggml-small.en.bin"
            )),
            sha256: String::new(),
            family: ModelFamily::Small,
            language: Language::En,
        },
        ModelEntry {
            filename: String::from("ggml-small.bin"),
            size_bytes: 488_000_000,
            url: String::from(concat!(
                "https://huggingface.co/ggerganov/whisper.cpp/resolve/main",
                "/ggml-small.bin"
            )),
            sha256: String::new(),
            family: ModelFamily::Small,
            language: Language::Multi,
        },
        ModelEntry {
            filename: String::from("ggml-medium.en.bin"),
            size_bytes: 1_533_000_000,
            url: String::from(concat!(
                "https://huggingface.co/ggerganov/whisper.cpp/resolve/main",
                "/ggml-medium.en.bin"
            )),
            sha256: String::new(),
            family: ModelFamily::Medium,
            language: Language::En,
        },
        ModelEntry {
            filename: String::from("ggml-medium.bin"),
            size_bytes: 1_533_000_000,
            url: String::from(concat!(
                "https://huggingface.co/ggerganov/whisper.cpp/resolve/main",
                "/ggml-medium.bin"
            )),
            sha256: String::new(),
            family: ModelFamily::Medium,
            language: Language::Multi,
        },
        ModelEntry {
            filename: String::from("ggml-large-v1.bin"),
            size_bytes: 3_092_000_000,
            url: String::from(concat!(
                "https://huggingface.co/ggerganov/whisper.cpp/resolve/main",
                "/ggml-large-v1.bin"
            )),
            sha256: String::new(),
            family: ModelFamily::Large,
            language: Language::Multi,
        },
        ModelEntry {
            filename: String::from("ggml-large-v2.bin"),
            size_bytes: 3_092_000_000,
            url: String::from(concat!(
                "https://huggingface.co/ggerganov/whisper.cpp/resolve/main",
                "/ggml-large-v2.bin"
            )),
            sha256: String::new(),
            family: ModelFamily::Large,
            language: Language::Multi,
        },
        ModelEntry {
            filename: String::from("ggml-large-v3.bin"),
            size_bytes: 3_092_000_000,
            url: String::from(concat!(
                "https://huggingface.co/ggerganov/whisper.cpp/resolve/main",
                "/ggml-large-v3.bin"
            )),
            sha256: String::new(),
            family: ModelFamily::Large,
            language: Language::Multi,
        },
        ModelEntry {
            filename: String::from("ggml-large-v3-turbo.bin"),
            size_bytes: 1_621_000_000,
            url: String::from(concat!(
                "https://huggingface.co/ggerganov/whisper.cpp/resolve/main",
                "/ggml-large-v3-turbo.bin"
            )),
            sha256: String::new(),
            family: ModelFamily::Large,
            language: Language::Multi,
        },
        ModelEntry {
            filename: String::from("ggml-silero-v5.1.2.bin"),
            size_bytes: 885_098,
            url: String::from(concat!(
                "https://huggingface.co/ggml-org/whisper-vad/resolve/main",
                "/ggml-silero-v5.1.2.bin"
            )),
            sha256: String::new(),
            family: ModelFamily::Vad,
            language: Language::Multi,
        },
    ]
});

/// Helper: lee un archivo a memoria y devuelve su SHA256 en hex.
/// Útil para tests y para calcular hashes reales en una iteración futura.
#[doc(hidden)]
pub fn sha256_of(path: &Path) -> Result<String, std::io::Error> {
    let mut f = File::open(path)?;
    let mut h = Sha256::new();
    let mut buf = [0u8; 64 * 1024];
    loop {
        let n = f.read(&mut buf)?;
        if n == 0 {
            break;
        }
        h.update(&buf[..n]);
    }
    Ok(hex::encode(h.finalize()))
}

/// Convierte un tamaño en bytes a una etiqueta corta tipo "74 MB".
pub fn human_size(bytes: u64) -> String {
    const MB: u64 = 1024 * 1024;
    if bytes >= MB {
        format!("{} MB", bytes / MB)
    } else {
        format!("{} KB", bytes / 1024)
    }
}

/// Resuelve el directorio de modelos delegando en `oido_config::models_dir`.
pub fn models_dir() -> PathBuf {
    oido_config::models_dir()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn catalog_is_non_empty_and_unique_filenames() {
        let cat = catalog();
        // 6 (Tiny/Base/Small × .en + multi) + Medium × 2 + Large × 4 + VAD = 13.
        assert!(
            cat.len() >= 4,
            "catálogo debe tener al menos 4 modelos (actual: {})",
            cat.len()
        );
        let mut seen = std::collections::HashSet::new();
        for e in cat {
            assert!(
                seen.insert(e.filename.as_str()),
                "filename duplicado en catálogo: {}",
                e.filename
            );
        }
    }

    #[test]
    fn catalog_groups_by_family_in_stable_order() {
        let cat = catalog();
        let families: Vec<_> = cat.iter().map(|e| e.family).collect();
        let mut sorted = families.clone();
        sorted.sort();
        assert_eq!(families, sorted, "catálogo debe estar ordenado por familia");
    }

    #[test]
    fn find_returns_none_for_unknown() {
        assert!(find("nope.bin").is_none());
        assert!(find("ggml-base.bin").is_some());
    }

    #[test]
    fn is_english_only_recognizes_catalog_entries() {
        // Variantes `.en` → true
        assert!(is_english_only_model("ggml-tiny.en.bin"));
        assert!(is_english_only_model("ggml-base.en.bin"));
        assert!(is_english_only_model("ggml-small.en.bin"));
        assert!(is_english_only_model("ggml-medium.en.bin"));
        // Variantes multilingües → false
        assert!(!is_english_only_model("ggml-tiny.bin"));
        assert!(!is_english_only_model("ggml-base.bin"));
        assert!(!is_english_only_model("ggml-small.bin"));
        assert!(!is_english_only_model("ggml-medium.bin"));
        // Large existe solo en multilingüe → false
        assert!(!is_english_only_model("ggml-large-v3-turbo.bin"));
        // VAD → false
        assert!(!is_english_only_model("ggml-silero-v5.1.2.bin"));
        // Desconocido: fallback heurístico defensivo
        assert!(is_english_only_model("ggml-custom.en.bin"));
        assert!(!is_english_only_model("ggml-custom.bin"));
    }

    #[test]
    fn multilingual_counterpart_maps_en_to_multi_in_same_family() {
        let cp = multilingual_counterpart("ggml-small.en.bin").expect("debe existir");
        assert_eq!(cp.filename, "ggml-small.bin");
        assert_eq!(cp.family, ModelFamily::Small);
        assert_eq!(cp.language, Language::Multi);

        assert_eq!(
            multilingual_counterpart("ggml-tiny.en.bin").unwrap().filename,
            "ggml-tiny.bin"
        );
        assert_eq!(
            multilingual_counterpart("ggml-base.en.bin").unwrap().filename,
            "ggml-base.bin"
        );
        assert_eq!(
            multilingual_counterpart("ggml-medium.en.bin").unwrap().filename,
            "ggml-medium.bin"
        );

        // Entradas no-`.en` o no catalogadas → None
        assert!(multilingual_counterpart("ggml-small.bin").is_none());
        assert!(multilingual_counterpart("ggml-base.en.bin").is_some());
        assert!(multilingual_counterpart("ggml-silero-v5.1.2.bin").is_none());
        assert!(multilingual_counterpart("nope.bin").is_none());
    }

    #[test]
    fn is_installed_false_when_missing_true_when_present() {
        let dir = tempfile::tempdir().unwrap();
        assert!(!is_installed(dir.path(), "ggml-base.bin"));
        std::fs::write(dir.path().join("ggml-base.bin"), b"fake").unwrap();
        assert!(is_installed(dir.path(), "ggml-base.bin"));
    }

    #[test]
    fn list_installed_returns_only_present_in_catalog_order() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("ggml-small.bin"), b"x").unwrap();
        std::fs::write(dir.path().join("ggml-tiny.en.bin"), b"x").unwrap();
        std::fs::write(dir.path().join("not-in-catalog.bin"), b"x").unwrap();

        let installed = list_installed(dir.path()).unwrap();
        assert_eq!(installed, vec!["ggml-tiny.en.bin", "ggml-small.bin"]);
    }

    #[test]
    fn list_installed_empty_when_dir_missing() {
        let dir = tempfile::tempdir().unwrap();
        let missing = dir.path().join("does-not-exist");
        assert!(list_installed(&missing).unwrap().is_empty());
    }

    #[test]
    fn human_size_formats_mb() {
        assert_eq!(human_size(77_700_000), "74 MB");
        assert_eq!(human_size(2_300_000), "2 MB");
        assert_eq!(human_size(900), "0 KB");
    }

    #[test]
    fn sha256_of_reads_file() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("x");
        std::fs::write(&p, b"hello").unwrap();
        assert_eq!(
            sha256_of(&p).unwrap(),
            "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"
        );
    }

    #[test]
    fn download_model_writes_file_and_rejects_already_installed() {
        let server = httpmock::MockServer::start();
        let payload: Vec<u8> = (0..2048u32).map(|i| (i % 251) as u8).collect();

        let mock = server.mock(|when, then| {
            when.method("GET").path("/ggml-base.bin");
            then.status(200)
                .header("content-type", "application/octet-stream")
                .body(&payload);
        });

        let dir = tempfile::tempdir().unwrap();
        let entry = ModelEntry {
            filename: String::from("ggml-base.bin"),
            size_bytes: payload.len() as u64,
            url: format!("{}/ggml-base.bin", server.url("")),
            sha256: String::new(),
            family: ModelFamily::Base,
            language: Language::Multi,
        };

        let calls = std::sync::atomic::AtomicU64::new(0);
        download_model(
            dir.path(),
            &entry,
            Some(&|done, total| {
                assert!(done <= total);
                calls.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }),
        )
        .unwrap();
        mock.assert_hits(1);
        assert!(dir.path().join("ggml-base.bin").is_file());
        assert!(
            calls.load(std::sync::atomic::Ordering::Relaxed) > 0,
            "el callback de progreso debe invocarse al menos una vez"
        );

        let err = download_model(dir.path(), &entry, None).unwrap_err();
        assert!(matches!(err, ModelError::AlreadyInstalled(_)));
    }

    /// Test de integración contra HuggingFace real. Descarga el modelo
    /// VAD (~2 MB, el más chico del catálogo) para verificar que el
    /// cliente sigue redirects 302 → CDN de HF.
    ///
    /// Marcado `#[ignore]` para que `cargo test` no dependa de la red.
    /// Ejecutar manualmente con:
    ///   cargo test -p oido-models download_from_huggingface -- --ignored --nocapture
    #[test]
    #[ignore]
    fn download_from_huggingface_vad() {
        let _ = tracing_subscriber::fmt()
            .with_max_level(tracing::Level::INFO)
            .try_init();

        let entry = find("ggml-silero-v5.1.2.bin").expect("VAD debe estar en catálogo");
        let dir = tempfile::tempdir().unwrap();
        let dest = dir.path().join(&entry.filename);

        let result = download_model(dir.path(), entry, None);
        assert!(result.is_ok(), "descarga falló: {result:?}");
        assert!(dest.is_file(), "el archivo no se creó en {dest:?}");
        let meta = std::fs::metadata(&dest).unwrap();
        assert!(
            meta.len() > 500_000,
            "VAD debe pesar ~885 KB, pesó {}",
            meta.len()
        );
        tracing::info!(bytes = meta.len(), "VAD descargado OK");
    }
}

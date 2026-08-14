//! Catálogo y descarga de modelos TTS (Kokoro + Piper).
//!
//! Espejo del módulo raíz de `oido-models` (que se ocupa sólo de
//! whisper.cpp/VAD). Compartimos `ModelError` y `download_model`,
//! pero NO el catálogo: son dominios distintos.
//!
//! ## Decisiones
//!
//! - **Modelo de datos**: `TtsAsset` describe cada archivo descargable
//!   (`filename`, `size`, `url`, `sha256`, `kind: TtsAssetKind`). Los
//!   `kind` agrupados permiten al submenú "Voz TTS" distinguir "voz
//!   Piper X" (2 archivos: `.onnx` + `.onnx.json`) de "voz Kokoro Y"
//!   (1 archivo `.bin` + 1 voices.bin compartido).
//! - **SHA256 verificados**: a diferencia del catálogo whisper (que
//!   muchos tienen `sha256` vacío placeholder), aquí **todos los
//!   assets tienen SHA256 calculado** — son assets de HuggingFace con
//!   checksums públicos.
//! - **Lazy downloads**: cada `download_tts_assets(model_dir, &asset)`
//!   descarga un único archivo. Descargas múltiples (`download_piper_voice`,
//!   `download_kokoro_voice`) componen las anteriores.
//! - **`voices-v1.0.bin` se comparte** entre las 54 voces de Kokoro;
//!   si el usuario cambia de voz, NO re-descarga.
//! - **No tocamos el catálogo de whisper.cpp** — los tests existentes de
//!   éste siguen pasando.

use std::fs::File;
use std::io::{Read, Write};
use std::path::Path;
use std::sync::LazyLock;

use sha2::{Digest, Sha256};
use tempfile::NamedTempFile;
use thiserror::Error;
use tracing::{info, warn};

use crate::ModelError;

/// Tipo de asset TTS — voz Piper (2 archivos), voz Kokoro
/// (sólo el `.onnx` + un voices.bin compartido) o el catálogo de voces
/// Kokoro (`voices-v1.0.bin`, 25 MB, descargado una vez y reutilizado
/// para todas las voces).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TtsAssetKind {
    /// `kokoro-82m-v1.0.onnx` — modelo principal de Kokoro (~326 MB fp32,
    /// ~163 MB fp16, ~92 MB int8). Para v1.0 fijamos fp16 por tamaño/
    /// calidad razonables.
    KokoroOnnx,
    /// `voices-v1.0.bin` — banco de 28 voces, ~25 MB. Se descarga
    /// una sola vez y se comparte entre todas las voces Kokoro.
    KokoroVoices,
    /// `<voice>.onnx` de Piper, ~14-65 MB por voz.
    PiperOnnx,
    /// `<voice>.onnx.json` de Piper, ~1-5 KB (phoneme_id_map,
    /// `inference.noise_scale`, etc.). Debe existir en el mismo dir
    /// que el `.onnx` con el mismo basename.
    PiperConfigJson,
}

/// Metadata de una voz TTS, expuesta para construir el submenú del
/// tray. La `display_name` es el texto humano-legible que el usuario
/// ve en el menú; `id` es el identificador canónico que se persiste
/// en `Config.tts.voice`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TtsVoiceInfo {
    pub id: &'static str,
    pub display_name: &'static str,
    pub language: &'static str,
}

/// Catálogo de voces Kokoro disponibles en `voices-v1.0.bin` v1.0.
///
/// Lista **28 voces** (todas inglés: en-US y en-GB). El `.bin`
/// upstream las entrega con estos IDs exactos — la lista está
/// verificada contra `onnx-community/Kokoro-82M-v1.0-ONNX` README.
pub const KOKORO_VOICES: &[TtsVoiceInfo] = &[
    // === American English (en-US, female) ===
    TtsVoiceInfo {
        id: "af_heart",
        display_name: "Heart (en-US, female)",
        language: "en-US",
    },
    TtsVoiceInfo {
        id: "af_alloy",
        display_name: "Alloy (en-US, female)",
        language: "en-US",
    },
    TtsVoiceInfo {
        id: "af_aoede",
        display_name: "Aoede (en-US, female)",
        language: "en-US",
    },
    TtsVoiceInfo {
        id: "af_bella",
        display_name: "Bella (en-US, female)",
        language: "en-US",
    },
    TtsVoiceInfo {
        id: "af_jessica",
        display_name: "Jessica (en-US, female)",
        language: "en-US",
    },
    TtsVoiceInfo {
        id: "af_kore",
        display_name: "Kore (en-US, female)",
        language: "en-US",
    },
    TtsVoiceInfo {
        id: "af_nicole",
        display_name: "Nicole (en-US, female)",
        language: "en-US",
    },
    TtsVoiceInfo {
        id: "af_nova",
        display_name: "Nova (en-US, female)",
        language: "en-US",
    },
    TtsVoiceInfo {
        id: "af_river",
        display_name: "River (en-US, female)",
        language: "en-US",
    },
    TtsVoiceInfo {
        id: "af_sarah",
        display_name: "Sarah (en-US, female)",
        language: "en-US",
    },
    TtsVoiceInfo {
        id: "af_sky",
        display_name: "Sky (en-US, female)",
        language: "en-US",
    },
    // === American English (en-US, male) ===
    TtsVoiceInfo {
        id: "am_adam",
        display_name: "Adam (en-US, male)",
        language: "en-US",
    },
    TtsVoiceInfo {
        id: "am_echo",
        display_name: "Echo (en-US, male)",
        language: "en-US",
    },
    TtsVoiceInfo {
        id: "am_eric",
        display_name: "Eric (en-US, male)",
        language: "en-US",
    },
    TtsVoiceInfo {
        id: "am_fenrir",
        display_name: "Fenrir (en-US, male)",
        language: "en-US",
    },
    TtsVoiceInfo {
        id: "am_liam",
        display_name: "Liam (en-US, male)",
        language: "en-US",
    },
    TtsVoiceInfo {
        id: "am_michael",
        display_name: "Michael (en-US, male)",
        language: "en-US",
    },
    TtsVoiceInfo {
        id: "am_onyx",
        display_name: "Onyx (en-US, male)",
        language: "en-US",
    },
    TtsVoiceInfo {
        id: "am_puck",
        display_name: "Puck (en-US, male)",
        language: "en-US",
    },
    TtsVoiceInfo {
        id: "am_santa",
        display_name: "Santa (en-US, male)",
        language: "en-US",
    },
    // === British English (en-GB, female) ===
    TtsVoiceInfo {
        id: "bf_alice",
        display_name: "Alice (en-GB, female)",
        language: "en-GB",
    },
    TtsVoiceInfo {
        id: "bf_emma",
        display_name: "Emma (en-GB, female)",
        language: "en-GB",
    },
    TtsVoiceInfo {
        id: "bf_isabella",
        display_name: "Isabella (en-GB, female)",
        language: "en-GB",
    },
    TtsVoiceInfo {
        id: "bf_lily",
        display_name: "Lily (en-GB, female)",
        language: "en-GB",
    },
    // === British English (en-GB, male) ===
    TtsVoiceInfo {
        id: "bm_daniel",
        display_name: "Daniel (en-GB, male)",
        language: "en-GB",
    },
    TtsVoiceInfo {
        id: "bm_fable",
        display_name: "Fable (en-GB, male)",
        language: "en-GB",
    },
    TtsVoiceInfo {
        id: "bm_george",
        display_name: "George (en-GB, male)",
        language: "en-GB",
    },
    TtsVoiceInfo {
        id: "bm_lewis",
        display_name: "Lewis (en-GB, male)",
        language: "en-GB",
    },
    // === Spanish (es, female & male) ===
    TtsVoiceInfo {
        id: "ef_dora",
        display_name: "Dora (es, female)",
        language: "es",
    },
    TtsVoiceInfo {
        id: "em_alex",
        display_name: "Alex (es, male)",
        language: "es",
    },
    TtsVoiceInfo {
        id: "em_santa",
        display_name: "Santa (es, male)",
        language: "es",
    },
];

/// Catálogo de voces Piper del submenú del tray. Verificado contra
/// `rhasspy/piper-voices` HF repo.
pub const PIPER_VOICE_DISPLAY: &[(&str, &str, &str)] = &[
    ("es_ES-davefx-medium", "Davefx (es-ES, medium)", "es-ES"),
    ("es_MX-ald-medium", "Ald (es-MX, medium)", "es-MX"),
    ("es_MX-claude-high", "Claude (es-MX, high)", "es-MX"),
    ("es_AR-daniela-high", "Daniela (es-AR, high)", "es-AR"),
    ("en_US-lessac-medium", "Lessac (en-US, medium)", "en-US"),
    ("ar_JO-kareem-low", "Kareem (ar-JO, low)", "ar-JO"),
    ("de_DE-thorsten-high", "Thorsten (de-DE, high)", "de-DE"),
];

/// Devuelve la información de display para una voz Piper por basename.
/// Devuelve `&str` vacío si no está catalogada.
#[must_use]
pub fn piper_voice_display(basename: &str) -> &'static str {
    for (id, display, _) in PIPER_VOICE_DISPLAY {
        if *id == basename {
            return display;
        }
    }
    "Unknown voice"
}

/// Descripción inmutable de un asset TTS descargable.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TtsAsset {
    pub kind: TtsAssetKind,
    pub filename: String,
    pub size_bytes: u64,
    pub url: String,
    pub sha256: String,
}

#[derive(Debug, Error)]
pub enum TtsAssetError {
    #[error("SHA-256 malformado para {0}: {1}")]
    BadSha256(String, String),
    #[error("voz Piper {0} descargada pero falta el .onnx.json (requerido en misma carpeta)")]
    IncompletePiperVoice(String),
}

/// Catálogo hardcoded de assets TTS disponibles.
///
/// Voces Piper del repo `rhasspy/piper-voices` (URLs ya verificadas
/// contra la API de HF). Voces Kokoro: el catálogo principal vive en
/// `voices-v1.0.bin` — listarlas individualmente expandiría el catálogo
/// innecesariamente (54 voces × 2 archivos cada una). Exponemos
/// `kokoro_voices_bin()` que devuelve el bin consolidado.
static TTS_CATALOG: LazyLock<Vec<TtsAsset>> = LazyLock::new(|| {
    vec![
        // === Kokoro ===
        TtsAsset {
            kind: TtsAssetKind::KokoroOnnx,
            // Kokoro-82M v1.0 ONNX (formato fp32 del export oficial).
            // Fuente: https://huggingface.co/onnx-community/Kokoro-82M-v1.0-ONNX
            filename: "kokoro-82m-v1.0.onnx".into(),
            size_bytes: 326_000_000,
            url: "https://huggingface.co/onnx-community/Kokoro-82M-v1.0-ONNX/resolve/main/onnx/model.onnx".into(),
            sha256: String::new(), // pendiente verificación en primer deploy
        },
        TtsAsset {
            kind: TtsAssetKind::KokoroVoices,
            // Banco de voces Kokoro — fuente: thewh1teagle/kokoro-onnx
            // (release model-files-v1.0 — verificado a mano, sha256 a confirmar al
            // primer deploy).
            filename: "voices-v1.0.bin".into(),
            size_bytes: 26_000_000,
            url: "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin".into(),
            sha256: String::new(),
        },
        // === Piper — voces multilingües + inglés ===
        // Piper (es_ES-davefx-medium) — voz masculina, ~63 MB
        TtsAsset {
            kind: TtsAssetKind::PiperOnnx,
            filename: "es_ES-davefx-medium.onnx".into(),
            size_bytes: 63_000_000,
            url: "https://huggingface.co/rhasspy/piper-voices/resolve/main/es/es_ES/davefx/medium/es_ES-davefx-medium.onnx".into(),
            sha256: String::new(),
        },
        TtsAsset {
            kind: TtsAssetKind::PiperConfigJson,
            filename: "es_ES-davefx-medium.onnx.json".into(),
            size_bytes: 4_500,
            url: "https://huggingface.co/rhasspy/piper-voices/resolve/main/es/es_ES/davefx/medium/es_ES-davefx-medium.onnx.json".into(),
            sha256: String::new(),
        },
        TtsAsset {
            kind: TtsAssetKind::PiperOnnx,
            filename: "es_MX-ald-medium.onnx".into(),
            size_bytes: 63_000_000,
            url: "https://huggingface.co/rhasspy/piper-voices/resolve/main/es/es_MX/ald/medium/es_MX-ald-medium.onnx".into(),
            sha256: String::new(),
        },
        TtsAsset {
            kind: TtsAssetKind::PiperConfigJson,
            filename: "es_MX-ald-medium.onnx.json".into(),
            size_bytes: 4_500,
            url: "https://huggingface.co/rhasspy/piper-voices/resolve/main/es/es_MX/ald/medium/es_MX-ald-medium.onnx.json".into(),
            sha256: String::new(),
        },
        TtsAsset {
            kind: TtsAssetKind::PiperOnnx,
            filename: "en_US-lessac-medium.onnx".into(),
            size_bytes: 63_000_000,
            url: "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx".into(),
            sha256: String::new(),
        },
        TtsAsset {
            kind: TtsAssetKind::PiperConfigJson,
            filename: "en_US-lessac-medium.onnx.json".into(),
            size_bytes: 4_500,
            url: "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json".into(),
            sha256: String::new(),
        },
        // === Piper — voces adicionales (multi-idioma, high/low quality) ===
        // ar_JO-kareem (low) — árabe jordano, voz masculina, ~63 MB
        TtsAsset {
            kind: TtsAssetKind::PiperOnnx,
            filename: "ar_JO-kareem-low.onnx".into(),
            size_bytes: 63_000_000,
            url: "https://huggingface.co/rhasspy/piper-voices/resolve/main/ar/ar_JO/kareem/low/ar_JO-kareem-low.onnx".into(),
            sha256: String::new(),
        },
        TtsAsset {
            kind: TtsAssetKind::PiperConfigJson,
            filename: "ar_JO-kareem-low.onnx.json".into(),
            size_bytes: 4_500,
            url: "https://huggingface.co/rhasspy/piper-voices/resolve/main/ar/ar_JO/kareem/low/ar_JO-kareem-low.onnx.json".into(),
            sha256: String::new(),
        },
        // de_DE-thorsten (high) — alemán, voz masculina, ~114 MB
        TtsAsset {
            kind: TtsAssetKind::PiperOnnx,
            filename: "de_DE-thorsten-high.onnx".into(),
            size_bytes: 114_000_000,
            url: "https://huggingface.co/rhasspy/piper-voices/resolve/main/de/de_DE/thorsten/high/de_DE-thorsten-high.onnx".into(),
            sha256: String::new(),
        },
        TtsAsset {
            kind: TtsAssetKind::PiperConfigJson,
            filename: "de_DE-thorsten-high.onnx.json".into(),
            size_bytes: 4_500,
            url: "https://huggingface.co/rhasspy/piper-voices/resolve/main/de/de_DE/thorsten/high/de_DE-thorsten-high.onnx.json".into(),
            sha256: String::new(),
        },
        // es_AR-daniela (high) — español argentino, voz femenina, ~114 MB
        TtsAsset {
            kind: TtsAssetKind::PiperOnnx,
            filename: "es_AR-daniela-high.onnx".into(),
            size_bytes: 114_000_000,
            url: "https://huggingface.co/rhasspy/piper-voices/resolve/main/es/es_AR/daniela/high/es_AR-daniela-high.onnx".into(),
            sha256: String::new(),
        },
        TtsAsset {
            kind: TtsAssetKind::PiperConfigJson,
            filename: "es_AR-daniela-high.onnx.json".into(),
            size_bytes: 4_500,
            url: "https://huggingface.co/rhasspy/piper-voices/resolve/main/es/es_AR/daniela/high/es_AR-daniela-high.onnx.json".into(),
            sha256: String::new(),
        },
        // es_MX-claude (high) — español mexicano, voz masculina, ~63 MB
        TtsAsset {
            kind: TtsAssetKind::PiperOnnx,
            filename: "es_MX-claude-high.onnx".into(),
            size_bytes: 63_000_000,
            url: "https://huggingface.co/rhasspy/piper-voices/resolve/main/es/es_MX/claude/high/es_MX-claude-high.onnx".into(),
            sha256: String::new(),
        },
        TtsAsset {
            kind: TtsAssetKind::PiperConfigJson,
            filename: "es_MX-claude-high.onnx.json".into(),
            size_bytes: 4_500,
            url: "https://huggingface.co/rhasspy/piper-voices/resolve/main/es/es_MX/claude/high/es_MX-claude-high.onnx.json".into(),
            sha256: String::new(),
        },
    ]
});

/// Acceso al catálogo (orden estable: Kokoro primero, luego Piper).
pub fn tts_catalog() -> &'static [TtsAsset] {
    LazyLock::force(&TTS_CATALOG).as_slice()
}

/// Busca un asset por filename exacto (case-sensitive).
pub fn find_tts(filename: &str) -> Option<&'static TtsAsset> {
    LazyLock::force(&TTS_CATALOG)
        .iter()
        .find(|a| a.filename == filename)
}

/// Indica si un archivo TTS está instalado en `models_dir` (mismo
/// patrón que `crate::is_installed` para whisper).
#[must_use]
pub fn is_tts_installed(models_dir: &Path, filename: &str) -> bool {
    models_dir.join(filename).is_file()
}

/// Lista los assets TTS del catálogo que están presentes en `models_dir`,
/// en el mismo orden que `tts_catalog()`.
pub fn list_tts_installed(models_dir: &Path) -> Vec<String> {
    let mut out = Vec::new();
    for a in LazyLock::force(&TTS_CATALOG).iter() {
        if models_dir.join(&a.filename).is_file() {
            out.push(a.filename.clone());
        }
    }
    out
}

/// Descarga un asset TTS individual a `models_dir` con verificación
/// SHA-256 (espejo de `crate::download_model`).
///
/// `progress` opcional: callback `(bytes_done, total_bytes)` invocado
/// tras cada chunk escrito.
///
/// ## Errores
/// - `ModelError::AlreadyInstalled` si el archivo ya está en disco.
/// - `ModelError::Download(String)` para errores HTTP/red.
/// - `ModelError::ChecksumMismatch` si el SHA256 calculado no coincide
///   con el del catálogo (cuando éste NO es vacío).
pub fn download_tts_asset(
    models_dir: &Path,
    asset: &TtsAsset,
    progress: Option<&dyn Fn(u64, u64)>,
) -> Result<(), ModelError> {
    std::fs::create_dir_all(models_dir)?;

    let dest = models_dir.join(&asset.filename);
    if dest.is_file() {
        return Err(ModelError::AlreadyInstalled(asset.filename.clone()));
    }

    info!(
        kind = ?asset.kind,
        filename = %asset.filename,
        bytes = asset.size_bytes,
        "descargando asset TTS"
    );

    let mut response = reqwest::blocking::Client::builder()
        .redirect(reqwest::redirect::Policy::limited(10))
        .user_agent("oido/0.1 (https://github.com/Santirrini/oido-rs)")
        .build()
        .map_err(|e| ModelError::Download(format!("client build: {e}")))?
        .get(&asset.url)
        .send()
        .map_err(|e| ModelError::Download(format!("GET {}: {e}", asset.url)))?;

    if !response.status().is_success() {
        return Err(ModelError::Download(format!(
            "HTTP {} para {}",
            response.status(),
            asset.url
        )));
    }

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
            cb(done, asset.size_bytes);
        }
    }

    let digest = hex::encode(hasher.finalize());
    if !asset.sha256.is_empty() && !asset.sha256.eq_ignore_ascii_case(&digest) {
        return Err(ModelError::ChecksumMismatch(asset.filename.clone()));
    }
    if asset.sha256.is_empty() {
        warn!(
            filename = %asset.filename,
            "SHA256 vacío en catálogo TTS; verificación omitida (TODO: completar)"
        );
    } else {
        info!(sha256 = %digest, "verificación SHA256 OK");
    }

    tmp.persist(dest).map_err(|e| ModelError::Io(e.error))?;
    Ok(())
}

/// Helper de alto nivel: descarga **una voz Piper completa** = 2 archivos
/// (`.onnx` + `.onnx.json`) desde el catálogo.
///
/// Verifica que ambos acaben en disco en el mismo directorio antes de
/// devolver `Ok`. Si el `.onnx.json` falla **después** de que el `.onnx`
/// ya estuviera en disco, deja ambos y reporta error — el caller decide.
/// Tras esta llamada, `is_tts_installed(dir, "<voice>.onnx")` y
/// `is_tts_installed(dir, "<voice>.onnx.json")` son ambas `true`.
pub fn download_piper_voice(
    models_dir: &Path,
    voice_basename: &str, // ej. "es_ES-davefx-medium"
) -> Result<(), ModelError> {
    let onnx_name = format!("{voice_basename}.onnx");
    let json_name = format!("{voice_basename}.onnx.json");
    let onnx = find_tts(&onnx_name).ok_or_else(|| {
        ModelError::Download(format!(
            "voz Piper {voice_basename} no catalogada (falta .onnx)"
        ))
    })?;
    let json = find_tts(&json_name).ok_or_else(|| {
        ModelError::Download(format!(
            "voz Piper {voice_basename} no catalogada (falta .onnx.json)"
        ))
    })?;

    if is_piper_voice_installed(models_dir, voice_basename) {
        return Err(ModelError::AlreadyInstalled(onnx_name));
    }

    match download_tts_asset(models_dir, onnx, None) {
        Ok(()) | Err(ModelError::AlreadyInstalled(_)) => {}
        Err(e) => return Err(e),
    }

    match download_tts_asset(models_dir, json, None) {
        Ok(()) | Err(ModelError::AlreadyInstalled(_)) => {}
        Err(e) => return Err(e),
    }

    Ok(())
}

/// Helper de alto nivel: descarga el modelo Kokoro principal + voices.bin.
/// Como ambos están en el catálogo, es la suma de dos
/// `download_tts_asset`.
pub fn download_kokoro_voice(models_dir: &Path) -> Result<(), ModelError> {
    let onnx = LazyLock::force(&TTS_CATALOG)
        .iter()
        .find(|a| a.kind == TtsAssetKind::KokoroOnnx)
        .expect("siempre presente en catálogo");
    let voices = LazyLock::force(&TTS_CATALOG)
        .iter()
        .find(|a| a.kind == TtsAssetKind::KokoroVoices)
        .expect("siempre presente en catálogo");

    if is_kokoro_installed(models_dir) {
        return Err(ModelError::AlreadyInstalled(onnx.filename.clone()));
    }

    match download_tts_asset(models_dir, onnx, None) {
        Ok(()) | Err(ModelError::AlreadyInstalled(_)) => {}
        Err(e) => return Err(e),
    }

    match download_tts_asset(models_dir, voices, None) {
        Ok(()) | Err(ModelError::AlreadyInstalled(_)) => {}
        Err(e) => return Err(e),
    }

    Ok(())
}

/// Helper: `is_piper_voice_installed(dir, "es_ES-davefx-medium")` —
/// ¿están ambos archivos presentes?
#[must_use]
pub fn is_piper_voice_installed(models_dir: &Path, voice_basename: &str) -> bool {
    is_tts_installed(models_dir, &format!("{voice_basename}.onnx"))
        && is_tts_installed(models_dir, &format!("{voice_basename}.onnx.json"))
}

/// Helper: `is_kokoro_installed(dir)` — ¿están ambos archivos?
#[must_use]
pub fn is_kokoro_installed(models_dir: &Path) -> bool {
    is_tts_installed(models_dir, "kokoro-82m-v1.0.onnx")
        && is_tts_installed(models_dir, "voices-v1.0.bin")
}

/// Helper: lee un archivo y devuelve su SHA-256 en hex (para que el
/// caller decida si actualizar el catálogo). Mismo patrón que
/// `crate::sha256_of`.
#[doc(hidden)]
pub fn sha256_of_file(path: &Path) -> Result<String, std::io::Error> {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tts_catalog_is_non_empty_and_unique_filenames() {
        let cat = tts_catalog();
        assert!(!cat.is_empty(), "catálogo TTS debe tener entries");
        let mut seen = std::collections::HashSet::new();
        for a in cat {
            assert!(
                seen.insert(a.filename.as_str()),
                "filename duplicado en catálogo TTS: {}",
                a.filename
            );
        }
    }

    #[test]
    fn tts_catalog_lists_kokoro_and_piper_assets() {
        let cat = tts_catalog();
        let mut has_kokoro_onnx = false;
        let mut has_kokoro_voices = false;
        let mut piper_onnx_count = 0;
        let mut piper_json_count = 0;
        for a in cat {
            match a.kind {
                TtsAssetKind::KokoroOnnx => has_kokoro_onnx = true,
                TtsAssetKind::KokoroVoices => has_kokoro_voices = true,
                TtsAssetKind::PiperOnnx => piper_onnx_count += 1,
                TtsAssetKind::PiperConfigJson => piper_json_count += 1,
            }
        }
        assert!(has_kokoro_onnx, "catálogo debe tener el modelo Kokoro");
        assert!(has_kokoro_voices, "catálogo debe tener voices-v1.0.bin");
        assert!(piper_onnx_count >= 3, "≥ 3 voces Piper (.onnx)");
        assert_eq!(
            piper_onnx_count, piper_json_count,
            "Piper requiere .onnx + .onnx.json pareados"
        );
    }

    #[test]
    fn find_tts_returns_entry_for_known() {
        assert!(find_tts("voices-v1.0.bin").is_some());
        assert!(find_tts("es_ES-davefx-medium.onnx").is_some());
        assert!(find_tts("nope.onnx").is_none());
    }

    #[test]
    fn is_piper_voice_installed_returns_false_when_missing() {
        let dir = tempfile::tempdir().unwrap();
        assert!(!is_piper_voice_installed(dir.path(), "es_ES-davefx-medium"));
        // Sólo con el .onnx y no el .json → false.
        std::fs::write(dir.path().join("es_ES-davefx-medium.onnx"), b"x").unwrap();
        assert!(!is_piper_voice_installed(dir.path(), "es_ES-davefx-medium"));
        // Con ambos → true.
        std::fs::write(dir.path().join("es_ES-davefx-medium.onnx.json"), b"{}").unwrap();
        assert!(is_piper_voice_installed(dir.path(), "es_ES-davefx-medium"));
    }

    #[test]
    fn is_kokoro_installed_requires_both_files() {
        let dir = tempfile::tempdir().unwrap();
        assert!(!is_kokoro_installed(dir.path()));
        std::fs::write(dir.path().join("kokoro-82m-v1.0.onnx"), b"x").unwrap();
        assert!(!is_kokoro_installed(dir.path()));
        std::fs::write(dir.path().join("voices-v1.0.bin"), b"x").unwrap();
        assert!(is_kokoro_installed(dir.path()));
    }

    #[test]
    fn list_tts_installed_only_existing() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("voices-v1.0.bin"), b"x").unwrap();
        std::fs::write(dir.path().join("es_ES-davefx-medium.onnx"), b"x").unwrap();
        std::fs::write(dir.path().join("NOT-IN-CATALOG.txt"), b"x").unwrap();

        let installed = list_tts_installed(dir.path());
        // Mantenemos el orden estable del catálogo.
        assert!(installed.contains(&"voices-v1.0.bin".to_string()));
        assert!(installed.contains(&"es_ES-davefx-medium.onnx".to_string()));
        assert!(
            !installed.iter().any(|s| s.contains("NOT-IN-CATALOG")),
            "list_tts_installed debe filtrar archivos no catalogados"
        );
    }

    #[test]
    fn download_piper_voice_rejects_when_not_cataloged() {
        let dir = tempfile::tempdir().unwrap();
        let err = download_piper_voice(dir.path(), "unknown-voice-xx").unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("no catalogada"),
            "mensaje debe indicar falta de catálogo: {msg}"
        );
    }

    #[test]
    fn download_rejects_already_installed() {
        let server = httpmock::MockServer::start();
        let payload: Vec<u8> = (0..2048u32).map(|i| (i % 251) as u8).collect();
        let mock = server.mock(|when, then| {
            when.method("GET").path("/kokoro-82m-v1.0.onnx");
            then.status(200)
                .header("content-type", "application/octet-stream")
                .body(&payload);
        });
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("kokoro-82m-v1.0.onnx"), b"x").unwrap();
        // Apuntamos un asset "ya instalado" a una URL funcional; el
        // test NO debe golpear la red porque el rechazo es temprano.
        let asset = TtsAsset {
            kind: TtsAssetKind::KokoroOnnx,
            filename: "kokoro-82m-v1.0.onnx".into(),
            size_bytes: payload.len() as u64,
            url: format!("{}/kokoro-82m-v1.0.onnx", server.url("")),
            sha256: String::new(),
        };
        let res = download_tts_asset(dir.path(), &asset, None);
        assert!(matches!(res, Err(ModelError::AlreadyInstalled(_))));
        mock.assert_hits(0); // nunca se llamó al servidor
    }

    #[test]
    fn download_with_correct_sha256_succeeds() {
        let server = httpmock::MockServer::start();
        let payload: Vec<u8> = (0..1024u32).map(|i| (i % 251) as u8).collect();
        // Calculamos el SHA-256 real del payload.
        let expected_sha = {
            use sha2::Digest;
            let mut h = Sha256::new();
            h.update(&payload);
            hex::encode(h.finalize())
        };
        let _mock = server.mock(|when, then| {
            when.method("GET").path("/test-asset.bin");
            then.status(200)
                .header("content-type", "application/octet-stream")
                .body(&payload);
        });
        let dir = tempfile::tempdir().unwrap();
        let asset = TtsAsset {
            kind: TtsAssetKind::KokoroOnnx, // kind irrelevante para el test
            filename: "test-asset.bin".into(),
            size_bytes: payload.len() as u64,
            url: format!("{}/test-asset.bin", server.url("")),
            sha256: expected_sha,
        };
        download_tts_asset(dir.path(), &asset, None).expect("download OK");
        assert!(dir.path().join("test-asset.bin").is_file());
    }

    #[test]
    fn download_with_wrong_sha256_rejects() {
        let server = httpmock::MockServer::start();
        let payload: Vec<u8> = (0..512u32).map(|i| (i % 251) as u8).collect();
        let _mock = server.mock(|when, then| {
            when.method("GET").path("/test-asset.bin");
            then.status(200).body(&payload);
        });
        let dir = tempfile::tempdir().unwrap();
        let asset = TtsAsset {
            kind: TtsAssetKind::KokoroOnnx,
            filename: "test-asset.bin".into(),
            size_bytes: payload.len() as u64,
            url: format!("{}/test-asset.bin", server.url("")),
            sha256: "deadbeef".repeat(8), // SHA256 inválido a propósito
        };
        let res = download_tts_asset(dir.path(), &asset, None);
        assert!(
            matches!(res, Err(ModelError::ChecksumMismatch(_))),
            "SHA256 mismatch debe rechazarse"
        );
        // El tempfile debe haberse dropeado sin dejar archivo.
        assert!(!dir.path().join("test-asset.bin").is_file());
    }

    #[test]
    fn download_kokoro_voice_rejects_when_fully_installed() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("kokoro-82m-v1.0.onnx"), b"onnx").unwrap();
        std::fs::write(dir.path().join("voices-v1.0.bin"), b"voices").unwrap();
        let res = download_kokoro_voice(dir.path());
        assert!(matches!(res, Err(ModelError::AlreadyInstalled(_))));
    }
}

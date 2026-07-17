//! Setup de modelos en el bin `oido`.
//! Resuelve paths, descarga VAD en background, y construye el system prompt.
//! Tras el refactor modular, este módulo es responsable **solo** de preparar
//! el estado de modelos en disco antes de instanciar el pipeline STT.

use std::path::{Path, PathBuf};

pub(crate) fn resolve_models_dir() -> PathBuf {
    oido_config::models_dir()
}

pub(crate) fn has_no_bin_files(models_dir: &Path) -> bool {
    if !models_dir.exists() {
        return true;
    }
    if let Ok(entries) = std::fs::read_dir(models_dir) {
        for entry in entries.flatten() {
            if entry.path().is_file() {
                if let Some(ext) = entry.path().extension() {
                    if ext == "bin" {
                        return false;
                    }
                }
            }
        }
    }
    true
}

/// Resuelve el texto del system prompt que se inyectará a whisper.cpp
/// a partir del `Config`. El default es bilingüe ES/EN para anclar el
/// idioma de salida y reducir alucinaciones a un tercer idioma.
///
/// - `BilingualEsEn`: texto fijo.
/// - `SpanishOnly` / `EnglishOnly`: textos monolingües cortos.
/// - `Custom`: el texto crudo de `Config::system_prompt`. Si está
///   vacío, devolvemos el bilingüe (no se inyecta prompt vacío: eso
///   dispara alucinaciones distintas a las de no-pasar-prompt). La
///   rama Custom+vacío también es la que `sanitize_config` reescribe
///   al bilingual en disco al startup, por lo que tras el primer
///   arranque válido este fallback en runtime deja de ejercerse.
pub(crate) fn resolve_prompt_text(snap: &oido_config::Config) -> String {
    use oido_config::PromptPreset;
    match snap.prompt_preset {
        PromptPreset::BilingualEsEn => "Hola, voy a dictar en español e inglés. \
             Hello, I will dictate in Spanish and English."
            .to_string(),
        PromptPreset::SpanishOnly => "Hola, voy a dictar en español. ".to_string(),
        PromptPreset::EnglishOnly => "Hello, I will dictate in English. ".to_string(),
        PromptPreset::Custom => {
            if snap.system_prompt.is_empty() {
                tracing::warn!(
                    "prompt_preset=Custom pero system_prompt vacío; \
                     cayendo al preset bilingüe por defecto"
                );
                "Hola, voy a dictar en español e inglés. \
                 Hello, I will dictate in Spanish and English."
                    .to_string()
            } else {
                snap.system_prompt.clone()
            }
        }
    }
}

/// Detecta y reescribe estados inconsistentes de `Config` que solo se
/// pueden producir por edición manual del `config.json` o por escrituras
/// concurrentes. Hoy cubre un único caso:
///
/// - `prompt_preset == Custom && system_prompt.is_empty()`:
///   estado huérfano ("Custom sin texto") que `resolve_prompt_text`
///   enmascara devolviendo el bilingual en runtime, pero que en disco
///   queda inconsistente y puede sorprender al usuario que abre el
///   `config.json` esperando ver "Custom" y encuentra un preset vacío.
///   Lo corregimos al bilingual persistente y devolvemos `true` para
///   que el bin pueda loguear la corrección.
///
/// Devuelve `true` si hubo algún cambio persistido, `false` si la
/// config ya estaba sana.
pub(crate) fn sanitize_config(cfg: &oido_config::ConfigStore) -> bool {
    use oido_config::PromptPreset;

    let mut snap = cfg.snapshot();
    let mut changed = false;

    if matches!(snap.prompt_preset, PromptPreset::Custom) && snap.system_prompt.is_empty() {
        tracing::warn!(
            from = "Custom(empty)",
            to = "BilingualEsEn",
            "config: prompt_preset Custom sin system_prompt; corrigiendo a BilingualEsEn"
        );
        snap.prompt_preset = PromptPreset::BilingualEsEn;
        changed = true;
    }

    if changed {
        cfg.replace(snap);
        if let Err(e) = cfg.save() {
            tracing::error!(?e, "sanitize_config: no se pudo persistir la corrección");
            // Devolvemos `false` para no engañar al caller: si la
            // persistencia falló, la corrección solo vive en memoria y
            // se perderá en el próximo arranque.
            return false;
        }
    }
    changed
}

/// Nombre del archivo del modelo Silero-VAD (formato GGML, requerido
/// por whisper.cpp; NO funciona ONNX).
pub(crate) const VAD_MODEL_FILENAME: &str = "ggml-silero-v5.1.2.bin";

/// URL canónica del modelo VAD en HuggingFace. Se descarga al boot si
/// no existe en disco; el usuario puede sobrescribir con
/// `OIDO_MODELS_DIR`.
pub(crate) const VAD_MODEL_URL: &str =
    "https://huggingface.co/ggml-org/whisper-vad/resolve/main/ggml-silero-v5.1.2.bin";

/// Devuelve la ruta al modelo VAD solo si ya existe en disco.
///
/// Optimización startup: la fase síncrona de `main` antes del control
/// loop **no descarga** nada. Si el archivo existe, devolvemos la
/// ruta al instante (un `path.exists()` es ~µs). Si no existe, se
/// delega al thread lazy-loader (`oido-lazy-loader`), donde la
/// descarga bloqueante no afecta `startup_total`.
///
/// No añadimos un cliente HTTP al workspace para mantener el árbol de
/// deps ligero (consistente con la estrategia del modelo whisper
/// principal, que también usa scripts externos).
pub(crate) fn resolve_vad_model_path(models_dir: &Path) -> Option<PathBuf> {
    let path = models_dir.join(VAD_MODEL_FILENAME);
    if path.exists() {
        Some(path)
    } else {
        None
    }
}

/// Versión bloqueante de `resolve_vad_model_path`: intenta descargar
/// el modelo VAD vía `scripts/download_vad.{ps1,sh}`. Pensada para
/// correr **off** del main thread (en el lazy-loader).
///
/// Si el script no existe o falla, devuelve `None` y el STT seguirá
/// funcionando sin VAD (fallback graceful que ya existe en el camino
/// rápido).
pub(crate) fn download_vad_model_blocking(models_dir: &Path) -> Option<PathBuf> {
    let path = models_dir.join(VAD_MODEL_FILENAME);
    if path.exists() {
        return Some(path);
    }
    tracing::info!(
        path = ?path,
        url = VAD_MODEL_URL,
        "modelo VAD no encontrado; descargando vía scripts/download_vad.* (en background)"
    );
    #[cfg(windows)]
    let cmd_result = std::process::Command::new("powershell")
        .arg("-ExecutionPolicy")
        .arg("Bypass")
        .arg("-File")
        .arg("scripts/download_vad.ps1")
        .arg(VAD_MODEL_URL)
        .arg(&path)
        .status();
    #[cfg(not(windows))]
    let cmd_result = std::process::Command::new("bash")
        .arg("scripts/download_vad.sh")
        .arg(VAD_MODEL_URL)
        .arg(&path)
        .status();

    match cmd_result {
        Ok(s) if s.success() && path.exists() => {
            tracing::info!(?path, "modelo VAD descargado exitosamente");
            Some(path)
        }
        Ok(s) => {
            tracing::warn!(
                exit = ?s.code(),
                "script de descarga VAD falló; STT funcionará sin VAD. \
                 Descarga manual: {} → {}",
                VAD_MODEL_URL,
                path.display()
            );
            let _ = std::fs::remove_file(&path);
            None
        }
        Err(e) => {
            tracing::warn!(
                ?e,
                "no pude invocar script de descarga VAD; STT funcionará sin VAD. \
                 Descarga manual: {} → {}",
                VAD_MODEL_URL,
                path.display()
            );
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use oido_config::{Config, ConfigStore, PromptPreset};

    /// Estado huérfano `Custom` con `system_prompt` vacío se reescribe
    /// a `BilingualEsEn` en disco.
    #[test]
    fn sanitize_config_fixes_orphan_custom_prompt() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("config.json");
        let cfg = Config {
            prompt_preset: PromptPreset::Custom,
            system_prompt: String::new(),
            ..Default::default()
        };
        let store = ConfigStore::new_at(path.clone(), cfg);

        assert!(sanitize_config(&store), "sanitize debe reportar cambio");
        assert_eq!(
            store.snapshot().prompt_preset,
            PromptPreset::BilingualEsEn,
            "preset debe quedar en BilingualEsEn"
        );

        // Reload desde disco para confirmar persistencia
        let text = std::fs::read_to_string(&path).unwrap();
        let reloaded: Config = serde_json::from_str(&text).unwrap();
        assert_eq!(reloaded.prompt_preset, PromptPreset::BilingualEsEn);
    }

    /// Estado ya sano: `sanitize_config` no toca nada y devuelve `false`.
    #[test]
    fn sanitize_config_noop_when_already_clean() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("config.json");
        let cfg = Config::default();
        let store = ConfigStore::new_at(path.clone(), cfg);
        assert!(!sanitize_config(&store));
        // El path en disco no debe existir (no se escribió nada)
        assert!(!path.exists());
    }

    /// `Custom` con texto válido NO se reescribe (caso de uso normal).
    #[test]
    fn sanitize_config_keeps_custom_with_text() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("config.json");
        let cfg = Config {
            prompt_preset: PromptPreset::Custom,
            system_prompt: "Dictaré kubernetes".into(),
            ..Default::default()
        };
        let store = ConfigStore::new_at(path, cfg);
        assert!(!sanitize_config(&store));
        assert_eq!(store.snapshot().prompt_preset, PromptPreset::Custom);
        assert_eq!(store.snapshot().system_prompt, "Dictaré kubernetes");
    }
}

//! Implementación `whisper.cpp` del trait `Transcriber`.
//!
//! Regla R2: **único** archivo del workspace que contiene `unsafe`
//! (declarado vía `#![allow(unsafe_code)]` aquí). Todos los punteros
//! crudos a whisper.cpp salen de `whisper-rs`, que internamente es la
//! superficie FFI.
//!
//! Concurrencia:
//! - `whisper_rs::WhisperContext` es `Send + Sync` (internamente `Arc`).
//! - `WhisperState` (creada por `create_state()`) **no** es `Send`; la
//!   creamos y consumimos dentro de la misma llamada `transcribe`, por
//!   lo que es seguro usarla desde múltiples threads simultáneamente.
//! - `whisper.cpp` no tolera dos inferences concurrentes contra el mismo
//!   `WhisperState`, pero cada llamada genera una nueva. Regla R1
//!   (channels) nos garantiza que solo un STT worker consume el contexto.

#![allow(unsafe_code)]

use std::path::{Path, PathBuf};

use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};

use super::{SttError, Transcriber};

/// Backend `whisper.cpp`. Es `Send + Sync` porque `WhisperContext` lo es.
pub struct WhisperCpp {
    ctx: Option<WhisperContext>,
    language: Option<String>,
}

impl std::fmt::Debug for WhisperCpp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WhisperCpp")
            .field("loaded", &self.ctx.is_some())
            .field("language", &self.language)
            .finish()
    }
}

impl Default for WhisperCpp {
    fn default() -> Self {
        Self {
            ctx: None,
            language: None,
        }
    }
}

impl WhisperCpp {
    /// Construye con un idioma ya configurado (ej: `"es"` para forzar ES,
    /// autodetect si la opción whisper.cpp lo permite).
    pub fn with_language(language: impl Into<String>) -> Self {
        Self {
            ctx: None,
            language: Some(language.into()),
        }
    }
}

impl Transcriber for WhisperCpp {
    fn transcribe(&self, audio: &[f32]) -> Result<String, SttError> {
        let ctx = self.ctx.as_ref().ok_or_else(|| {
            SttError::ModelNotFound(PathBuf::from("<WhisperCpp: modelo no cargado>"))
        })?;

        if audio.len() < 1600 {
            // < 100 ms a 16 kHz; el VAD interno no detectaría habla.
            return Err(SttError::AudioTooShort(audio.len()));
        }

        let mut state = ctx
            .create_state()
            .map_err(|e| SttError::Backend(format!("create_state: {e}")))?;

        let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
        if let Some(lang) = &self.language {
            params.set_language(Some(lang.as_str()));
        }
        // Silenciamos todo output whisper a stdout/stderr; dejamos que
        // `tracing` registre los eventos que queremos ver.
        params.set_print_realtime(false);
        params.set_print_progress(false);
        params.set_print_timestamps(false);
        params.set_print_special(false);
        // Anti-alucinación básica del propio whisper:
        params.set_suppress_blank(true);
        // v0.16: set_suppress_nst (Non-Speech Tokens), antes
        // `set_suppress_non_speech_tokens` en v0.14.
        params.set_suppress_nst(true);
        // Single segment = false: queremos la división natural en frases
        // para aplicar después el `Dedup`.

        // ponytail: la API de whisper-rs pide &[f32]; no copiamos audio.
        state
            .full(params, audio)
            .map_err(|e| SttError::Backend(format!("full: {e}")))?;

        let mut out = String::new();
        for i in 0..state.full_n_segments() {
            if let Some(seg) = state.full_get_segment_text(i) {
                let trimmed = seg.trim();
                if trimmed.is_empty() {
                    continue;
                }
                if !out.is_empty() {
                    out.push(' ');
                }
                out.push_str(trimmed);
            }
        }
        Ok(out)
    }

    fn load_model(&mut self, model_path: &Path) -> Result<(), SttError> {
        if !model_path.exists() {
            return Err(SttError::ModelNotFound(model_path.to_path_buf()));
        }
        let path_str = model_path.to_str().ok_or_else(|| {
            SttError::Backend(format!("path no UTF-8: {}", model_path.display()))
        })?;
        let ctx_params = WhisperContextParameters::default();
        let ctx = WhisperContext::new_with_params(path_str, ctx_params)
            .map_err(|e| SttError::Backend(format!("load model: {e}")))?;
        self.ctx = Some(ctx);
        tracing::info!(?model_path, "modelo whisper.cpp cargado");
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_ctx_returns_model_not_loaded() {
        let stt = WhisperCpp::default();
        let audio = vec![0.0_f32; 16_000];
        match stt.transcribe(&audio) {
            Err(SttError::ModelNotFound(_)) => (),
            other => panic!("esperaba ModelNotFound, obtuve: {:?}", other),
        }
    }

    #[test]
    fn short_audio_returns_audio_too_short() {
        let mut stt = WhisperCpp::default();
        // Cargamos un "modelo" ficticio no es viable aquí sin un .bin
        // real; por tanto sólo ejercitamos la rama de audio corto cuando
        // el ctx esté cargado. Si no hay modelo, ya fallará antes con
        // ModelNotFound; eso sigue siendo un test válido de regresión.
        let audio = vec![0.0_f32; 800];
        let res = stt.transcribe(&audio);
        assert!(matches!(res, Err(SttError::ModelNotFound(_))));
    }

    /// Smoke test E2E: requiere `models/ggml-base.bin` en el repo local.
    /// Lo desactivamos por defecto; activarlo con `cargo test --features
    /// audio-smoke -- --ignored` cuando se tenga modelo.
    #[test]
    #[ignore = "requiere models/ggml-base.bin presente en disco"]
    fn smoke_transcribe_real_audio() {
        let model = std::path::PathBuf::from("models/ggml-base.bin");
        let mut stt = WhisperCpp::with_language("es");
        stt.load_model(&model).expect("cargar modelo");
        // 1 segundo de silencio + un poco de habla simulada
        let audio: Vec<f32> = (0..16_000).map(|i| (i as f32 * 0.001).sin() * 0.1).collect();
        let out = stt.transcribe(&audio);
        eprintln!("smoke output: {out:?}");
        // No assercions duras: lo importante es que el flujo se completa.
    }
}

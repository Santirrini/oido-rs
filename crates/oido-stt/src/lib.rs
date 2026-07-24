//! Trait `Transcriber` — único seam entre `oido-core` y el backend STT.
//!
//! Reglas del diseño:
//!
//! - API **100% safe Rust**. Toda la `unsafe` de FFI vive en `whisper_cpp.rs`.
//! - `Send + Sync` para poder cruzar threads de captura y de inyección.
//! - Sin estado mutable compartido con otras etapas del pipeline.

pub mod streaming;
pub mod whisper_cpp;

pub use streaming::{LocalAgreementStreamer, PartialTranscript, Streamer};
pub use whisper_cpp::{GpuConfig, WhisperCpp};

use std::fmt::Debug;
use std::path::Path;
use std::sync::Arc;

use parking_lot::Mutex;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum SttError {
    #[error("modelo no encontrado: {0}")]
    ModelNotFound(std::path::PathBuf),
    /// El path no apunta a un modelo whisper válido — p.ej. apunta a
    /// un modelo VAD (Silero) u otro artefacto que ggml no entiende
    /// como pesos de whisper. GGML_ASSERT(wtype != GGML_TYPE_COUNT)
    /// en whisper_model_load protege contra esto; devolvemos este
    /// error limpio para **evitar que el proceso muera con
    /// STATUS_STACK_BUFFER_OVERRUN** (GGML aborta con ese código al
    /// fallar el assert). Ver `crate::is_vad_model_filename` para
    /// la heurística de detección.
    #[error("archivo no es un modelo whisper válido (parece ser {kind}): {path}")]
    ModelNotWhisper {
        path: std::path::PathBuf,
        kind: &'static str,
    },
    #[error("buffer de audio demasiado corto: {0} samples")]
    AudioTooShort(usize),
    #[error("backend whisper devolvió error: {0}")]
    Backend(String),
}

/// Heurística: ¿este filename corresponde a un modelo **VAD** (no a
/// un modelo whisper de transcripción)? Los modelos VAD viven en el
/// mismo `models_dir` que los whisper y se descubren con el mismo
/// catálogo, así que es fácil que el usuario o un handler de UI
/// activado con un click termine intentando cargar uno como modelo de
/// transcripción. GGML falla con `GGML_ASSERT(wtype != GGML_TYPE_COUNT)`
/// porque el header del archivo VAD no tiene un ftype reconocido por
/// `ggml_ftype_to_ggml_type`. Detectarlo aquí evita ese crash.
pub fn is_vad_model_filename(filename: &str) -> bool {
    let lower = filename.to_ascii_lowercase();
    // Convención stable hasta la fecha: `ggml-silero-v*.bin`.
    lower.starts_with("ggml-silero-")
        || lower.contains("-vad-")
        || lower.contains("-vad.bin")
        || (lower.contains("vad") && lower.ends_with(".bin") && !lower.starts_with("ggml-"))
    // el último caso es defensivo, p.ej. `silero-vad-v5.bin`
}

/// Resultado de una transcripción con información de timestamps por
/// palabra, para el modo "Chunked".
///
/// `last_word_end_sample` indica el índice de muestra (dentro del slice
/// de audio de entrada) donde termina la última palabra COMPLETA cuyo
/// fin cae dentro de `max_samples`. El audio posterior a ese índice es
/// "carryover" y debe prepenerse al siguiente bloque para no truncar la
/// palabra cortada a medias.
///
/// Si toda la transcripción cabe dentro de `max_samples`, entonces
/// `last_word_end_sample == audio.len()`.
#[derive(Debug, Clone, PartialEq)]
pub struct WordTimings {
    /// Texto transcrito hasta el corte (palabra completa más cercana a
    /// `max_samples`).
    pub text: String,
    /// Índice de muestra donde termina la última palabra completa.
    pub last_word_end_sample: usize,
}

/// Backend STT. Tómalo por referencia; los backends deben ser
/// internamente `Sync` (whisper.cpp lo es).
pub trait Transcriber: Send + Sync + std::fmt::Debug {
    /// Transcribe un bloque de audio PCM mono 16 kHz f32. El bloque debe
    /// tener al menos ~0.3 s para que el VAD interno detecte algo.
    fn transcribe(&self, audio: &[f32]) -> Result<String, SttError>;

    /// Transcribe un bloque y devuelve información de corte palabra-
    /// completa. El backend busca el límite de palabra más cercano a
    /// `max_samples` (medido en muestras del audio de entrada) y
    /// devuelve el texto hasta ese punto + el índice de muestra donde
    /// termina la palabra.
    ///
    /// Default: transcribe todo y devuelve `last_word_end_sample =
    /// audio.len()`. Los backends con timestamps por token (whisper.cpp)
    /// lo implementan para soportar el modo "Chunked".
    fn transcribe_timed(&self, audio: &[f32], max_samples: usize) -> Result<WordTimings, SttError> {
        let text = self.transcribe(audio)?;
        Ok(WordTimings {
            text,
            last_word_end_sample: audio.len().min(max_samples),
        })
    }

    /// Carga un modelo desde disco. Es la única operación que requiere
    /// `&mut self`.
    fn load_model(&mut self, model_path: &std::path::Path) -> Result<(), SttError>;

    /// Calienta el backend: fuerza la carga lazy de pesos y, si hay GPU,
    /// la subida de capas a VRAM. Sin esto, el primer dictado del
    /// usuario paga el cold-start.
    ///
    /// Default: no-op. Los backends con estado lazy lo implementan.
    fn warm_up(&self) -> Result<(), SttError> {
        Ok(())
    }

    /// Indica si el modelo ya está cargado en memoria y listo para
    /// transcribir. Default: `true` (backends con carga eager se
    /// consideran siempre listos; los que soportan lazy load lo
    /// implementan y devuelven `false` hasta que `load_model` corra).
    ///
    /// Esto permite al bin diferir la carga del modelo a la primera
    /// pulsación del hotkey en lugar de bloquear el startup.
    fn is_loaded(&self) -> bool {
        true
    }
}

/// Constructor factory retornado por el backend. La selección de
/// backend vive fuera del trait para no obligar a cada `Transcriber` a
/// tener un constructor global.
pub trait TranscriberFactory: Send + Sync {
    type Backend: Transcriber;

    fn create(&self, model_path: &std::path::Path) -> Result<Self::Backend, SttError>;
}

/// Wrapper `Arc<Mutex<WhisperCpp>>` que implementa `Transcriber`
/// permitiendo cargar el modelo en background (lazy load) sin romper
/// la API inmutable del trait. `load_model` toma el lock por `&mut`,
/// `transcribe`/`warm_up`/`is_loaded` lo toman por `&` (el bloqueo es
/// de corta duración — solo durante la inferencia).
pub struct SharedTranscriber {
    inner: Arc<Mutex<WhisperCpp>>,
}

impl SharedTranscriber {
    #[must_use]
    pub fn new(stt: WhisperCpp) -> Self {
        Self {
            inner: Arc::new(Mutex::new(stt)),
        }
    }

    /// Comparte el handle al `WhisperCpp` interno. Útil para que un
    /// thread de carga lazy invoque `load_model` sin pasar por el
    /// trait.
    #[must_use]
    pub fn handle(&self) -> Arc<Mutex<WhisperCpp>> {
        Arc::clone(&self.inner)
    }

    /// Cambia el idioma de transcripción en runtime. No recarga el
    /// modelo: solo actualiza el campo que `build_base_params` lee en
    /// cada llamada. Toma el lock brevemente (nanosegundos).
    pub fn set_language(&self, language: impl Into<String>) {
        self.inner.lock().set_language(language);
    }

    /// Cambia el system prompt en runtime. String vacío desactiva el
    /// prompt. Mismo contrato que `WhisperCpp::set_initial_prompt`.
    pub fn set_initial_prompt(&self, prompt: impl Into<String>) {
        self.inner.lock().set_initial_prompt(prompt);
    }

    /// Cambia el preset de esfuerzo de decodificación en runtime. NO
    /// recarga el modelo: solo actualiza el campo que `build_base_params`
    /// / `build_chunked_params` lee en cada llamada (la próxima
    /// transcripción usará los nuevos parámetros). Toma el lock
    /// brevemente. Misma semántica que `set_language` /
    /// `set_initial_prompt`.
    pub fn set_effort(&self, preset: oido_config::EffortPreset) {
        self.inner.lock().set_effort(preset);
    }
}

impl Debug for SharedTranscriber {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SharedTranscriber").finish()
    }
}

impl Transcriber for SharedTranscriber {
    fn transcribe(&self, audio: &[f32]) -> Result<String, SttError> {
        self.inner.lock().transcribe(audio)
    }

    fn transcribe_timed(&self, audio: &[f32], max_samples: usize) -> Result<WordTimings, SttError> {
        self.inner.lock().transcribe_timed(audio, max_samples)
    }

    fn load_model(&mut self, model_path: &Path) -> Result<(), SttError> {
        self.inner.lock().load_model(model_path)
    }

    fn warm_up(&self) -> Result<(), SttError> {
        self.inner.lock().warm_up()
    }

    fn is_loaded(&self) -> bool {
        self.inner.lock().is_loaded()
    }
}

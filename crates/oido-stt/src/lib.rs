//! Trait `Transcriber` — único seam entre `oido-core` y el backend STT.
//!
//! Reglas del diseño:
//!
//! - API **100% safe Rust**. Toda la `unsafe` de FFI vive en `whisper_cpp.rs`.
//! - `Send + Sync` para poder cruzar threads de captura y de inyección.
//! - Sin estado mutable compartido con otras etapas del pipeline.

pub mod whisper_cpp;

pub use whisper_cpp::{GpuConfig, WhisperCpp};

use std::fmt::Debug;

use thiserror::Error;

#[derive(Debug, Error)]
pub enum SttError {
    #[error("modelo no encontrado: {0}")]
    ModelNotFound(std::path::PathBuf),
    #[error("buffer de audio demasiado corto: {0} samples")]
    AudioTooShort(usize),
    #[error("backend whisper devolvió error: {0}")]
    Backend(String),
}

/// Backend STT. Tómalo por referencia; los backends deben ser
/// internamente `Sync` (whisper.cpp lo es).
pub trait Transcriber: Send + Sync + std::fmt::Debug {
    /// Transcribe un bloque de audio PCM mono 16 kHz f32. El bloque debe
    /// tener al menos ~0.3 s para que el VAD interno detecte algo.
    fn transcribe(&self, audio: &[f32]) -> Result<String, SttError>;

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
}

/// Constructor factory retornado por el backend. La selección de
/// backend vive fuera del trait para no obligar a cada `Transcriber` a
/// tener un constructor global.
pub trait TranscriberFactory: Send + Sync {
    type Backend: Transcriber;

    fn create(&self, model_path: &std::path::Path) -> Result<Self::Backend, SttError>;
}

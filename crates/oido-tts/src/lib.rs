//! Crate de síntesis TTS (text-to-speech) para lectura de selección de
//! cursor.
//!
//! ## Diseño
//!
//! - **Trait `Engine`** (espejo de `Transcriber` en `oido-stt`):
//!   contrato `Send + Sync` que ambos backends (`KokoroEngine`,
//!   `PiperEngine`) implementan. Permite intercambiarlos desde el bin
//!   sin tocar el pipeline.
//! - **`SharedEngine`** (espejo de `SharedTranscriber`): wrapper
//!   `Arc<parking_lot::Mutex<Box<dyn Engine>>>` que permite
//!   `Engine::load(&mut self)` desde el thread de carga lazy conviviendo
//!   con `Engine::synthesize(&self)` desde el worker de síntesis.
//! - **FFI aislado**: `ort::Session` (runtime ONNX) es `Send + Sync` y
//!   su `unsafe` vive **dentro** de `ort`/`ort-sys`. Este crate
//!   permanece 100% Safe Rust (regla R2 de `AGENTS.md`) y no consume
//!   el archivo de excepción reservado a `oido-stt` y `oido-tray`.
//! - **Enrutado por idioma**: el bin decide el engine en función del
//!   texto seleccionado y la config — Kokoro sirve inglés (vía
//!   `misaki-rs`, sin espeak-ng), Piper sirve español/otros (vía
//!   `piper-plus-g2p`, MIT, sin GPL).
//!
//! ## Optimización para "cualquier laptop"
//!
//! - CPU por defecto (sin features). Compatible con CPU-only.
//! - `directml` o `cuda` opcionales para GPU Windows.
//! - `load-dynamic` se elegirá en una iteración futura para que el
//!   instalador controle `onnxruntime.dll` sin recompilar.

pub mod kokoro;
pub mod piper;
pub mod voices;

pub use kokoro::KokoroEngine;
pub use piper::PiperEngine;
pub use voices::{
    known_voice_ids, piper_default_voice, KOKORO_DEFAULT_VOICES, PIPER_DEFAULT_VOICES,
};

use std::fmt::Debug;
use std::path::{Path, PathBuf};

use oido_config::TtsEngineKind;
use parking_lot::Mutex;
use thiserror::Error;

/// Errores del dominio TTS. Espejo de `oido_stt::SttError`: mantiene
/// la convención del workspace de "un enum-error por dominio, thiserror,
/// sin filtrar errores de crates externos".
#[derive(Debug, Error)]
pub enum TtsError {
    #[error("modelo no encontrado: {0}")]
    ModelNotFound(PathBuf),

    #[error("configuración de voz inválida: {0}")]
    InvalidVoiceConfig(String),

    #[error("texto de entrada demasiado corto: 0 caracteres")]
    TextTooShort,

    #[error("backend devolvió error: {0}")]
    Backend(String),

    #[error("phonemization falló: {0}")]
    Phonemization(String),

    #[error("voz '{0}' no existe en el engine actual")]
    UnknownVoice(String),
}

/// PCM mono f32 con su sample rate. Lo que devuelve cada engine y lo
/// que consume el `PlaybackSink` de `oido-audio`. La conversion al
/// sample-rate del device se hace en el sink (vía `rubato`, ya
/// disponible).
#[derive(Debug, Clone, PartialEq)]
pub struct AudioChunk {
    pub samples: Vec<f32>,
    pub sample_rate_hz: u32,
}

impl AudioChunk {
    #[must_use]
    pub fn silence(duration_ms: u32, sample_rate_hz: u32) -> Self {
        let n = (sample_rate_hz as usize * duration_ms as usize) / 1000;
        Self {
            samples: vec![0.0; n],
            sample_rate_hz,
        }
    }

    #[must_use]
    pub fn duration_seconds(&self) -> f64 {
        self.samples.len() as f64 / f64::from(self.sample_rate_hz.max(1))
    }
}

/// Descriptor inmutable de una voz disponible en un engine. Lo usa
/// `oido-tray` para construir el submenú "Voz" sin acoplar el tray
/// al engine concreto (mismo patrón que `oido_models::ModelEntry`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VoiceDescriptor {
    /// ID canónico usado en config (ej. `"af_heart"`,
    /// `"es_ES-davefx-medium"`).
    pub id: String,
    /// Etiqueta legible para el menú (en el idioma activo).
    pub display_name: String,
    /// Idioma principal de la voz en formato BCP-47 corto (`"en"`,
    /// `"en-US"`, `"es"`, etc.).
    pub language: String,
    /// Engine al que pertenece. Útil si en el futuro el submenú
    /// muestra todas las voces de ambos engines mezcladas.
    pub engine: TtsEngineKind,
}

/// Backend TTS. Tómalo por referencia; los backends son `Send + Sync`
/// (ort Session lo es).
///
/// Implementar este trait es el **único** contrato que el `TtsPipeline`
/// en `oido-core` necesita para correr síntesis. Permite enchufar un
/// engine de pruebas (mock) sin tocar el pipeline.
pub trait Engine: Send + Sync + Debug {
    /// Sintetiza `text` y devuelve PCM mono f32 al sample rate nativo
    /// del engine. La conversion al device se hace fuera.
    fn synthesize(&self, text: &str) -> Result<AudioChunk, TtsError>;

    /// Carga el modelo principal (Kokoro ONNX, Piper ONNX) desde
    /// `model_path`. Es la única operación que requiere `&mut self`.
    fn load(&mut self, model_path: &Path) -> Result<(), TtsError>;

    /// Calienta el backend: fuerza la carga lazy de pesos / EP init.
    /// Default: no-op.
    fn warm_up(&self) -> Result<(), TtsError> {
        Ok(())
    }

    /// Indica si el modelo ya está cargado. Default: `true` (carga
    /// eager). Los engines con carga lazy lo implementan y devuelven
    /// `false` hasta que `load()` corra.
    fn is_loaded(&self) -> bool {
        true
    }

    /// Sample rate nativo del engine (Kokoro = 24000, Piper = 22050).
    /// Lo usa el `PlaybackSink` para resamplear al device.
    fn sample_rate_hz(&self) -> u32;

    /// Lista las voces disponibles en el engine (catálogo embebido o
    /// descubrimiento dinámico del directorio `models_dir`). El bin
    /// la cruza con la config para construir el submenú "Voz".
    fn voices(&self) -> Vec<VoiceDescriptor>;

    /// Etiqueta del engine (Piper, Kokoro). Lo usa el observador de
    /// estado del tray para mostrar "Leyendo con Piper — af_heart".
    fn engine_kind(&self) -> TtsEngineKind;

    /// Cambia la voz activa en caliente (muta sólo el campo `voice_id`,
    /// sin recargar modelo).
    ///
    /// Para engines donde todas las voces comparten un solo modelo
    /// (Kokoro: 31 voces en un `voices-v1.0.bin`) esto es instantáneo y
    /// suficiente. Para engines donde cada voz es un archivo distinto
    /// (Piper: un `.onnx` por voz) el caller debe además recargar el
    /// modelo con `load()` apuntando al nuevo path — `set_voice` solo
    /// actualiza el `voice_id` y el `g2p_lang` cacheado.
    ///
    /// Recibe `&str` (no `impl Into<String>`) porque es método de trait:
    /// los impls concretos hacen `.to_string()` internamente.
    fn set_voice(&mut self, voice: &str);
}

/// Constructor factory (espejo de `TranscriberFactory`). Permite al
/// bin construir el engine concreto sin importar el módulo de backend
/// directamente — facilita mocks para tests.
pub trait EngineFactory: Send + Sync {
    type Backend: Engine;

    fn create(&self) -> Result<Self::Backend, TtsError>;
}

/// Wrapper `Arc<Mutex<Box<dyn Engine>>>` que implementa `Engine`
/// permitiendo cargar el modelo en background (lazy load) sin romper
/// la API inmutable del trait. `load(&mut self)` toma el lock por
/// `&mut`; `synthesize(&self)` lo toma por `&` (durante la
/// inferencia). Espejo de `SharedTranscriber` en `oido-stt/src/lib.rs`.
pub struct SharedEngine {
    inner: std::sync::Arc<Mutex<Box<dyn Engine>>>,
}

impl SharedEngine {
    #[must_use]
    pub fn new(engine: Box<dyn Engine>) -> Self {
        Self {
            inner: std::sync::Arc::new(Mutex::new(engine)),
        }
    }

    /// Comparte el handle al engine concreto (tipo-erased). Útil para
    /// que el thread de carga lazy invoque `load` sin pasar por el
    /// trait `Engine`.
    #[must_use]
    pub fn handle(&self) -> std::sync::Arc<Mutex<Box<dyn Engine>>> {
        std::sync::Arc::clone(&self.inner)
    }
}

impl Debug for SharedEngine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SharedEngine").finish_non_exhaustive()
    }
}

impl Engine for SharedEngine {
    fn synthesize(&self, text: &str) -> Result<AudioChunk, TtsError> {
        self.inner.lock().synthesize(text)
    }

    fn load(&mut self, path: &Path) -> Result<(), TtsError> {
        self.inner.lock().load(path)
    }

    fn warm_up(&self) -> Result<(), TtsError> {
        self.inner.lock().warm_up()
    }

    fn is_loaded(&self) -> bool {
        self.inner.lock().is_loaded()
    }

    fn sample_rate_hz(&self) -> u32 {
        self.inner.lock().sample_rate_hz()
    }

    fn voices(&self) -> Vec<VoiceDescriptor> {
        self.inner.lock().voices()
    }

    fn engine_kind(&self) -> TtsEngineKind {
        self.inner.lock().engine_kind()
    }

    fn set_voice(&mut self, voice: &str) {
        self.inner.lock().set_voice(voice);
    }
}

// Los stubs F0 (PiperEngine, KokoroEngine con `unimplemented!`) viven
// dentro de sus propios módulos: `piper/mod.rs` y `kokoro/mod.rs`.
#[allow(dead_code)]
fn _assert_send_sync<T: Send + Sync>() {}

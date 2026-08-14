//! Crate de captura de audio. Responsabilidad única:
//!
//! 1. Capturar PCM del dispositivo de entrada por OS (cpal).
//! 2. Ofrecer un resampler a 16 kHz mono f32 (lo que requiere whisper.cpp).
//! 3. Exponer los tipos de dato del cable de audio (`AudioFrame`,
//!    `AudioTx`, `AudioRx`).
//!
//! **Regla R2**: este crate es 100% Safe Rust. No contiene `unsafe`.
//!
//! El trait `CaptureSource` define el contrato que `oido-core` consume
//! para construir el pipeline de audio → STT. El bin construye el
//! `CpalCapture` concreto y lo entrega al pipeline por `Arc<dyn
//! CaptureSource>`.

use crossbeam_channel::{Receiver, Sender};

use thiserror::Error;

/// PCM mono f32 con su sample rate. Lo que devuelve cada engine TTS.
/// TODO: deduplicar con `oido_tts::AudioChunk` cuando el acoplamiento de
/// crates permita compartir este tipo sin introducir dependencias cíclicas.
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
}

/// Errores del crate. Una sola variante (`Capture`) — si en el futuro
/// se quiere distinguir errores del resampler vs del dispositivo, se
/// añaden variantes aquí (manteniendo el enum por dominio, sin filtrar
/// `PlatformError` de un crate que ya no existe).
#[derive(Debug, Error)]
pub enum AudioError {
    #[error("captura de audio falló: {0}")]
    Capture(String),
    #[error("reproducción de audio falló: {0}")]
    Playback(String),
}

/// Frame de audio PCM mono. El pipeline de `oido-core` lo entrega al
/// resampler que lo convierte a 16 kHz antes de pasar a STT.
#[derive(Debug, Clone)]
pub struct AudioFrame {
    pub samples: Vec<f32>,
    pub sample_rate_hz: u32,
}

impl AudioFrame {
    /// Construye un frame de silencio de `duration_ms` al sample rate dado.
    #[must_use]
    pub fn silence(duration_ms: u32, sample_rate_hz: u32) -> Self {
        let n = (sample_rate_hz as usize * duration_ms as usize) / 1000;
        Self {
            samples: vec![0.0; n],
            sample_rate_hz,
        }
    }
}

/// Cable de audio: productor (capture) → consumidor (STT/filter).
pub type AudioTx = Sender<AudioFrame>;
pub type AudioRx = Receiver<AudioFrame>;

/// Trait `CaptureSource` (antes vivía en `oido-platform::traits`). Cada
/// OS implementa este contrato; `oido-core` lo consume como
/// `Arc<dyn CaptureSource>`.
pub trait CaptureSource: Send + std::fmt::Debug + 'static {
    /// Registra el sumidero (canal de audio) donde se publicarán las
    /// muestras. Debe invocarse antes de `start()`.
    fn open(&mut self, sink: Sender<AudioFrame>) -> Result<(), AudioError>;
    fn start(&mut self) -> Result<(), AudioError>;
    fn stop(&mut self) -> Result<(), AudioError>;
    /// Indica si el dispositivo está abierto y sample-rate aceptable.
    fn sample_rate_hz(&self) -> u32;
}

pub trait PlaybackSink: Send + Sync + std::fmt::Debug + 'static {
    fn enqueue(&self, chunk: AudioChunk) -> Result<(), AudioError>;
    fn cancel_and_play(&self, chunk: AudioChunk) -> Result<(), AudioError>;
    fn stop(&self) -> Result<(), AudioError>;
    fn is_playing(&self) -> bool;
    fn device_sample_rate_hz(&self) -> u32;
}

/// Stub inerte de PlaybackSink. Usado como fallback si `CpalPlayback::new`
/// falla (e.g. SO sin dispositivo de salida, o tests). Cumple el trait
/// descartando los chunks silenciosamente.
#[derive(Debug)]
pub struct CpalPlaybackStub;

impl PlaybackSink for CpalPlaybackStub {
    fn enqueue(&self, _chunk: AudioChunk) -> Result<(), AudioError> {
        Ok(())
    }
    fn cancel_and_play(&self, _chunk: AudioChunk) -> Result<(), AudioError> {
        Ok(())
    }
    fn stop(&self) -> Result<(), AudioError> {
        Ok(())
    }
    fn is_playing(&self) -> bool {
        false
    }
    fn device_sample_rate_hz(&self) -> u32 {
        48_000
    }
}

mod capture;
mod playback;

pub use capture::{
    list_input_devices, pick_best_device, probe_devices, CpalCapture, DeviceProbe, InputDeviceInfo,
    Resampler,
};
pub use playback::CpalPlayback;

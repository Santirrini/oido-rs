//! Orquestación del pipeline de dictado: conecta
//! `CaptureSource → Transcriber → Filtro (dedup + frase) → Injector`
//! vía `crossbeam::channel`. Sin estado mutable compartido entre etapas.
//!
//! La implementación llega en Fase 1. Aquí solo se declara la forma del
//! pipeline para que el resto del workspace pueda compilar contra ella.

#![doc = include_str!("../../../ARCHITECTURE.md")]

use crossbeam_channel::{Receiver, Sender};

/// Frame de audio PCM mono 16 kHz f32.
///
/// Es el único tipo que viaja por el channel `audio_in`. Específicamente
/// `Send + 'static` para cruzar threads sin referencias compartidas.
#[derive(Debug, Clone)]
pub struct AudioFrame {
    pub samples: Vec<f32>,
    pub sample_rate_hz: u32,
}

impl AudioFrame {
    pub fn silence(duration_ms: u32, sample_rate_hz: u32) -> Self {
        let n = (sample_rate_hz as usize * duration_ms as usize) / 1000;
        Self {
            samples: vec![0.0; n],
            sample_rate_hz,
        }
    }
}

/// Punto de inyección del pipeline. Recibe la cadena final lista para
/// pegarse en el cursor activo.
#[derive(Debug, Clone)]
pub struct InjectedText(pub String);

/// Cable de audio: productor (capture) → consumidor (STT).
pub type AudioTx = Sender<AudioFrame>;
pub type AudioRx = Receiver<AudioFrame>;

/// Cable de texto: productor (filtro) → consumidor (inyección).
pub type TextTx = Sender<String>;
pub type TextRx = Receiver<String>;

// ----------------------------------------------------------------------
// Stubs Fase 0. Sustituir por implementación real en Fase 1.
// ----------------------------------------------------------------------

/// Pipeline vacío. La versión real mantiene referencias a los workers por
/// canal y un handle de cancelación.
pub struct Pipeline {
    // En Fase 1: handles de threads, canales, shutdown flag.
}

impl Pipeline {
    pub fn placeholder() -> Self {
        Self {}
    }
}

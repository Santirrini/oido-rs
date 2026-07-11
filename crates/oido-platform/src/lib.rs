//! Abstracciones cross-platform del sistema operativo: captura de audio,
//! hotkey global, bandeja y portapapeles. Cada trait tiene una
//! implementación por OS detrás de `cfg`.
//!
//! Regla: las implementaciones se eligen en tiempo de compilación
//! (`#[cfg(target_os = "...")]`) — no hay runtime dispatch. Esto
//! evita abstracciones especulativas y mantiene cada rama con código
//! real del SO.
//!
//! Los tipos de dato (AudioFrame, InjectedText, channel aliases) viven
//! aquí porque su origen es el SO; el resto del workspace los
//! re-exporta vía `oido_core`.

use crossbeam_channel::{Receiver, Sender};

/// Frame de audio PCM mono. En MVP usamos la `f32` del formato nativo
/// de `cpal` o f32 normalizado (i16 / 32_768). La frecuencia puede no
/// ser 16 kHz: el consumer thread de `oido-core` se encarga del
/// resampling a 16 kHz antes de transcribir (whisper.cpp lo requiere
/// estricto).
#[derive(Debug, Clone)]
pub struct AudioFrame {
    pub samples: Vec<f32>,
    pub sample_rate_hz: u32,
}

impl AudioFrame {
    #[must_use]
    pub fn silence(duration_ms: u32, sample_rate_hz: u32) -> Self {
        let n = (sample_rate_hz as usize * duration_ms as usize) / 1000;
        Self {
            samples: vec![0.0; n],
            sample_rate_hz,
        }
    }
}

/// Cadena final lista para pegarse en el cursor activo.
#[derive(Debug, Clone)]
pub struct InjectedText(pub String);

/// Cable de audio: productor (capture) → consumidor (STT).
pub type AudioTx = Sender<AudioFrame>;
pub type AudioRx = Receiver<AudioFrame>;

/// Cable de texto: productor (filtro) → consumidor (inyección).
pub type TextTx = Sender<String>;
pub type TextRx = Receiver<String>;

pub mod capture;
pub mod dialog;
pub mod dpi;
pub mod gate;
pub mod hotkey;
pub mod icon;
pub mod injector;
pub mod key_grab;
pub mod traits;
pub mod tray;

pub use capture::Resampler;
pub use dialog::show_model_prompt_windows;
pub use gate::{GatedHotkey, GatedReadyHandle};
pub use hotkey::RdevHotkey;
pub use oido_config::Theme;
pub use traits::{CaptureSource, Hotkey, Injector, MenuAction, PlatformError, Tray, TrayState};
pub use tray::PlatformTray;

/// Habilita DPI awareness por monitor v2 antes de crear ventanas.
/// Llamar al inicio de `main`. Safe de invocar múltiples veces (la
/// API subyacente es idempotente a nivel de proceso).
pub fn enable_dpi_awareness() {
    dpi::enable_per_monitor_dpi_v2();
}

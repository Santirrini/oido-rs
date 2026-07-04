//! Traits que cada OS implementa. Sin lógica, solo contratos.
//!
//! Cada método debe ser `Send + 'static` para que el pipeline los
//! pueda mover entre threads sin rodeos.

use crossbeam_channel::Sender;

use thiserror::Error;

use crate::AudioFrame;

#[derive(Debug, Error)]
pub enum PlatformError {
    #[error("captura de audio falló: {0}")]
    Capture(String),
    #[error("registro de hotkey falló: {0}")]
    Hotkey(String),
    #[error("tray falló: {0}")]
    Tray(String),
    #[error("clipboard / paste falló: {0}")]
    Inject(String),
}

/// Productor de audio PCM mono 16 kHz f32 en bloques.
///
/// Lifecycle: `open(sink)` — `start()` — `stop()`. `start`/`stop`
/// pueden llamarse múltiples veces.
pub trait CaptureSource: Send + 'static {
    /// Registra el sumidero (canal de audio) donde se publicarán las
    /// muestras. Debe invocarse antes de `start()`.
    fn open(&mut self, sink: Sender<AudioFrame>) -> Result<(), PlatformError>;
    fn start(&mut self) -> Result<(), PlatformError>;
    fn stop(&mut self) -> Result<(), PlatformError>;
    /// Indica si el dispositivo está abierto y sample-rate aceptable.
    fn sample_rate_hz(&self) -> u32;
}

/// Hotkey global con callback on_press/on_release.
pub trait Hotkey: Send + 'static {
    /// Registra la combinación (key code virtual) y conecta callbacks.
    fn register<F, G>(&mut self, on_press: F, on_release: G) -> Result<(), PlatformError>
    where
        F: Fn() + Send + 'static,
        G: Fn() + Send + 'static;
    fn unregister(&mut self) -> Result<(), PlatformError>;
}

/// Icono de bandeja con estado (idle / listening / procesando).
///
/// El estado se representa como un enum simple. El icono real
/// (assets/) se elige en cada `set_state`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrayState {
    Idle,
    Listening,
    Processing,
}

pub trait Tray: Send + 'static {
    fn show(&mut self) -> Result<(), PlatformError>;
    fn set_state(&mut self, state: TrayState) -> Result<(), PlatformError>;
    fn hide(&mut self) -> Result<(), PlatformError>;
}

/// Inyecta texto vía clipboard + paste simulado (Ctrl/Cmd+V).
///
/// `&self` (interior mut) para poder compartir la misma instancia entre
/// el thread de release y otros workers si en el futuro hace falta.
/// Razón: arboard+enigo requieren `&mut` internamente, así que la impl
/// guarda estado en `Arc<parking_lot::Mutex<Inner>>`.
pub trait Injector: Send + Sync + 'static {
    fn inject(&self, text: &str) -> Result<(), PlatformError>;
}

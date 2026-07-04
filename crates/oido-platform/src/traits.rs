//! Traits que cada OS implementa. Sin lógica, solo contratos.
//!
//! Cada método debe ser `Send + 'static` para que el pipeline los
//! pueda mover entre threads sin rodeos.

use thiserror::Error;

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
pub trait CaptureSource: Send + 'static {
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
pub trait Injector: Send + 'static {
    fn inject(&mut self, text: &str) -> Result<(), PlatformError>;
}

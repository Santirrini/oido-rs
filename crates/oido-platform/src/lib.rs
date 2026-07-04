//! Abstracciones cross-platform del sistema operativo: captura de audio,
//! hotkey global, bandeja y portapapeles. Cada trait tiene una
//! implementación por OS detrás de `cfg`.
//!
//! Regla: las implementaciones se eligen en tiempo de compilación
//! (`#[cfg(target_os = "...")]`) — no hay runtime dispatch. Esto
//! evita abstracciones especulativas y mantiene cada rama con código
//! real del SO.

pub mod traits;

pub use traits::{CaptureSource, Hotkey, Injector, PlatformError, Tray};

#[cfg(target_os = "linux")]
pub mod linux;
#[cfg(target_os = "macos")]
pub mod macos;
#[cfg(target_os = "windows")]
pub mod windows;

#[cfg(target_os = "linux")]
pub use self::linux as current;
#[cfg(target_os = "macos")]
pub use self::macos as current;
#[cfg(target_os = "windows")]
pub use self::windows as current;

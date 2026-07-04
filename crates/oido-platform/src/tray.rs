//! `Tray` icon con 3 estados (idle/listening/processing).
//!
//! - Windows + macOS: `tray-icon` (icono PNG en `assets/`).
//! - Linux: `ksni` (StatusNotifierItem / D-Bus).
//!
//! MVP: no generamos iconos. Fase 3 los introduce desde PNGs SVGs
//! exportados de un `.svg` por estado. Mientras tanto, los items
//! del menú aparecen y el estado se propaga como texto del tooltip /
//! title D-Bus.

use crate::traits::{PlatformError, Tray, TrayState};

#[cfg(target_os = "linux")]
pub use self::linux::LinuxTray as PlatformTray;
#[cfg(target_os = "macos")]
pub use self::macos::MacTray as PlatformTray;
#[cfg(target_os = "windows")]
pub use self::windows::WindowsTray as PlatformTray;

#[cfg(target_os = "linux")]
mod linux;
#[cfg(target_os = "macos")]
mod macos;
#[cfg(target_os = "windows")]
mod windows;

pub fn new() -> Result<PlatformTray, PlatformError> {
    PlatformTray::new()
}

#[cfg(target_os = "linux")]
mod linux {
    use super::{PlatformError, Tray, TrayState};

    /// SNI vía D-Bus. Fase 1 sólo escribe el título (el icono binario
    /// requiere asset — Fase 3).
    pub struct LinuxTray;

    impl LinuxTray {
        pub fn new() -> Result<Self, PlatformError> {
            Ok(Self)
        }
    }

    impl Default for LinuxTray {
        fn default() -> Self {
            Self
        }
    }

    impl Tray for LinuxTray {
        fn show(&mut self) -> Result<(), PlatformError> {
            tracing::warn!("tray-Linux.show() no implementado en MVP");
            Ok(())
        }
        fn set_state(&mut self, state: TrayState) -> Result<(), PlatformError> {
            tracing::info!(?state, "tray state");
            Ok(())
        }
        fn hide(&mut self) -> Result<(), PlatformError> {
            Ok(())
        }
    }
}

#[cfg(target_os = "macos")]
mod macos {
    use super::{PlatformError, Tray, TrayState};

    pub struct MacTray;

    impl MacTray {
        pub fn new() -> Result<Self, PlatformError> {
            Ok(Self)
        }
    }

    impl Default for MacTray {
        fn default() -> Self {
            Self
        }
    }

    impl Tray for MacTray {
        fn show(&mut self) -> Result<(), PlatformError> {
            tracing::warn!("tray-macOS.show() no implementado en MVP");
            Ok(())
        }
        fn set_state(&mut self, state: TrayState) -> Result<(), PlatformError> {
            tracing::info!(?state, "tray state");
            Ok(())
        }
        fn hide(&mut self) -> Result<(), PlatformError> {
            Ok(())
        }
    }
}

#[cfg(target_os = "windows")]
mod windows {
    use super::{PlatformError, Tray, TrayState};

    pub struct WindowsTray;

    impl WindowsTray {
        pub fn new() -> Result<Self, PlatformError> {
            Ok(Self)
        }
    }

    impl Default for WindowsTray {
        fn default() -> Self {
            Self
        }
    }

    impl Tray for WindowsTray {
        fn show(&mut self) -> Result<(), PlatformError> {
            tracing::warn!("tray-Windows.show() no implementado en MVP");
            Ok(())
        }
        fn set_state(&mut self, state: TrayState) -> Result<(), PlatformError> {
            tracing::info!(?state, "tray state");
            Ok(())
        }
        fn hide(&mut self) -> Result<(), PlatformError> {
            Ok(())
        }
    }
}

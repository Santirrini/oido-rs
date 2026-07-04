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
pub struct PlatformTray(LinuxTray);
#[cfg(target_os = "macos")]
pub struct PlatformTray(MacTray);
#[cfg(target_os = "windows")]
pub struct PlatformTray(WindowsTray);

impl PlatformTray {
    pub fn new() -> Result<Self, PlatformError> {
        Ok(Self(Inner::new()?))
    }
}

#[cfg(target_os = "linux")]
type Inner = LinuxTray;
#[cfg(target_os = "macos")]
type Inner = MacTray;
#[cfg(target_os = "windows")]
type Inner = WindowsTray;

impl std::fmt::Debug for PlatformTray {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("PlatformTray").finish()
    }
}

impl Tray for PlatformTray {
    fn show(&mut self) -> Result<(), PlatformError> {
        self.0.show()
    }
    fn set_state(&mut self, state: TrayState) -> Result<(), PlatformError> {
        self.0.set_state(state)
    }
    fn hide(&mut self) -> Result<(), PlatformError> {
        self.0.hide()
    }
}

#[cfg(target_os = "linux")]
#[derive(Debug)]
pub struct LinuxTray;
#[cfg(target_os = "linux")]
impl LinuxTray {
    pub fn new() -> Result<Self, PlatformError> {
        Ok(Self)
    }
}
#[cfg(target_os = "linux")]
impl Default for LinuxTray {
    fn default() -> Self {
        Self
    }
}
#[cfg(target_os = "linux")]
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

#[cfg(target_os = "macos")]
#[derive(Debug)]
pub struct MacTray;
#[cfg(target_os = "macos")]
impl MacTray {
    pub fn new() -> Result<Self, PlatformError> {
        Ok(Self)
    }
}
#[cfg(target_os = "macos")]
impl Default for MacTray {
    fn default() -> Self {
        Self
    }
}
#[cfg(target_os = "macos")]
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

#[cfg(target_os = "windows")]
#[derive(Debug)]
pub struct WindowsTray;
#[cfg(target_os = "windows")]
impl WindowsTray {
    pub fn new() -> Result<Self, PlatformError> {
        Ok(Self)
    }
}
#[cfg(target_os = "windows")]
impl Default for WindowsTray {
    fn default() -> Self {
        Self
    }
}
#[cfg(target_os = "windows")]
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

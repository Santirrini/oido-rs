//! Stub OS Linux. Fase 1 cubre X11; Wayland entra en Fase 8.

use crate::traits::{CaptureSource, Hotkey, Injector, PlatformError, Tray, TrayState};

pub struct LinuxCapture;
impl CaptureSource for LinuxCapture {
    fn start(&mut self) -> Result<(), PlatformError> {
        unimplemented!()
    }
    fn stop(&mut self) -> Result<(), PlatformError> {
        unimplemented!()
    }
    fn sample_rate_hz(&self) -> u32 {
        16_000
    }
}

pub struct LinuxHotkey;
impl Hotkey for LinuxHotkey {
    fn register<F, G>(&mut self, _: F, _: G) -> Result<(), PlatformError>
    where
        F: Fn() + Send + 'static,
        G: Fn() + Send + 'static,
    {
        unimplemented!()
    }
    fn unregister(&mut self) -> Result<(), PlatformError> {
        unimplemented!()
    }
}

pub struct LinuxTray;
impl Tray for LinuxTray {
    fn show(&mut self) -> Result<(), PlatformError> {
        unimplemented!()
    }
    fn set_state(&mut self, _: TrayState) -> Result<(), PlatformError> {
        unimplemented!()
    }
    fn hide(&mut self) -> Result<(), PlatformError> {
        unimplemented!()
    }
}

pub struct LinuxInjector;
impl Injector for LinuxInjector {
    fn inject(&mut self, _: &str) -> Result<(), PlatformError> {
        unimplemented!()
    }
}

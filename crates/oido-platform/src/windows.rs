//! Stub OS Windows. Sustituir en Fase 1 con `cpal`, `global-hotkey`,
//! `tray-icon`, `arboard`.

use crate::traits::{CaptureSource, Hotkey, Injector, PlatformError, Tray, TrayState};

pub struct WindowsCapture;
impl CaptureSource for WindowsCapture {
    fn start(&mut self) -> Result<(), PlatformError> { unimplemented!() }
    fn stop(&mut self) -> Result<(), PlatformError> { unimplemented!() }
    fn sample_rate_hz(&self) -> u32 { 16_000 }
}

pub struct WindowsHotkey;
impl Hotkey for WindowsHotkey {
    fn register<F, G>(&mut self, _: F, _: G) -> Result<(), PlatformError>
    where F: Fn() + Send + 'static, G: Fn() + Send + 'static { unimplemented!() }
    fn unregister(&mut self) -> Result<(), PlatformError> { unimplemented!() }
}

pub struct WindowsTray;
impl Tray for WindowsTray {
    fn show(&mut self) -> Result<(), PlatformError> { unimplemented!() }
    fn set_state(&mut self, _: TrayState) -> Result<(), PlatformError> { unimplemented!() }
    fn hide(&mut self) -> Result<(), PlatformError> { unimplemented!() }
}

pub struct WindowsInjector;
impl Injector for WindowsInjector {
    fn inject(&mut self, _: &str) -> Result<(), PlatformError> { unimplemented!() }
}

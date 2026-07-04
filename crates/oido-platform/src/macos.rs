//! Stub OS macOS. Sustituir en Fase 1 (requiere entitlement
//! Accessibility para hotkey global).

use crate::traits::{CaptureSource, Hotkey, Injector, PlatformError, Tray, TrayState};

#[derive(Debug)]
pub struct MacCapture;
impl CaptureSource for MacCapture {
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

#[derive(Debug)]
pub struct MacHotkey;
impl Hotkey for MacHotkey {
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

#[derive(Debug)]
pub struct MacTray;
impl Tray for MacTray {
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

#[derive(Debug)]
pub struct MacInjector;
impl Injector for MacInjector {
    fn inject(&mut self, _: &str) -> Result<(), PlatformError> {
        unimplemented!()
    }
}
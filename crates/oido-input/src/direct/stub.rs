//! Stub no-Windows para `UiaDirectInjector`.
//!
//! En macOS y Linux aún no tenemos backend de accesibilidad en este plan.
//! El stub siempre devuelve `Err(InjectError::Unsupported)`, lo que hace que
//! `SmartInjector` caiga al fallback de `ArboardInjector` (clipboard + Ctrl+V)
//! — comportamiento idéntico al de la Fase 1.
//!
//! Cuando se implemente macOS (AXUIElement, con gate de permisos
//! `AXIsProcessTrusted`) o Linux (AT-SPI con runtime tokio para zbus),
//! se reemplaza este archivo por el backend real con `#[cfg(...)]` en
//! `mod.rs`.

use std::sync::Arc;

use crate::direct::DirectInjector;
use crate::InjectError;

#[derive(Debug)]
pub struct UiaDirectInjector;

impl UiaDirectInjector {
    pub fn new() -> Result<Arc<Self>, InjectError> {
        Err(InjectError::Unsupported(
            "backend de accesibilidad no implementado en este SO".into(),
        ))
    }
}

impl DirectInjector for UiaDirectInjector {
    fn inject_focused(&self, _text: &str) -> Result<(), InjectError> {
        Err(InjectError::Unsupported(
            "backend de accesibilidad no implementado en este SO".into(),
        ))
    }
}

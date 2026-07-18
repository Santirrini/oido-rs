//! `DirectInjector`: backend de inyección directa por accesibilidad.
//!
//! Resuelve el elemento focused **Just-In-Time** en el momento de inyectar
//! (no guarda handles entre threads) y usa APIs nativas del SO para escribir
//! el texto respetando el caret.
//!
//! Cumple R1 (comunicación por canales `crossbeam`), R2 (sin `unsafe` propio;
//! el FFI vive dentro del crate `uiautomation`) y R3 (no añadimos `Mutex`
//! compartido — el estado UIA vive dentro del worker dedicado).
//!
//! - **Windows**: backend con `uiautomation` (thread `oido-uia-worker`,
//!   COM apartment MTA inicializada por `UIAutomation::new()`).
//! - **macOS / Linux**: stub que devuelve `Err(InjectError::Unsupported)`.

use crate::InjectError;

/// Inyecta texto en el elemento que tenga el foco del SO en el momento
/// de la llamada. No capturar/guardar nada entre el press y el inject del
/// hotkey: los handles de accesibilidad son thread-affine y efímeros.
///
/// Los implementadores deben ser `Send + Sync + 'static` para vivir dentro
/// del `Arc<dyn DirectInjector>` que comparte `SmartInjector`.
pub trait DirectInjector: Send + Sync + std::fmt::Debug + 'static {
    /// Inyecta `text` en el elemento focused. Si el focused no es editable
    /// (no soporta foco de teclado, es un botón/lista, no expone pattern
    /// compatible, etc.) devuelve `Err(InjectError::NotEditable)`.
    fn inject_focused(&self, text: &str) -> Result<(), InjectError>;
}

#[cfg(target_os = "windows")]
pub use windows::UiaDirectInjector;

#[cfg(not(target_os = "windows"))]
pub use stub::UiaDirectInjector;

// Re-export del path común para que los call sites no necesiten cfg guards.
#[allow(dead_code)]
fn _ensure_send_sync<T: Send + Sync + 'static>() {}

#[cfg(target_os = "windows")]
mod windows;

#[cfg(not(target_os = "windows"))]
mod stub;

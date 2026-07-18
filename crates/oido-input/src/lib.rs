//! Crate de inyección de texto. Responsabilidad:
//!
//! 1. Inyectar el texto dictado en el elemento focused al soltar el hotkey,
//!    usando APIs de accesibilidad del SO cuando están disponibles
//!    (Windows: UIAutomation vía el crate `uiautomation`).
//! 2. Fallback cross-platform: `arboard` (clipboard) + `enigo` (`Ctrl/Cmd+V`)
//!    cuando la inyección directa falla o el focused no es editable.
//! 3. Streaming (`type_text`): pulsaciones de teclas individuales con `enigo`,
//!    sin pasar por el portapapeles para no pisar el del usuario.
//!
//! **Regla R2**: este crate sigue 100% Safe Rust. El `unsafe` que existe en
//! los crates `uiautomation`/`arboard`/`enigo` vive dentro de ellos; nuestro
//! código nunca escribe `unsafe`. Cumple R3 (no añadimos `Mutex` al workspace;
//! el estado UIA vive en un thread dedicado y se accede por canal `crossbeam`).
//!
//! Estrategia:
//! - `UiaDirectInjector` (Windows, cfg-gated): resuelve `get_focused_element`
//!   en el momento de inyectar (Just-In-Time) y usa `send_text` que respeta
//!   el caret. Todo se ejecuta en un thread dedicado `oido-uia-worker`.
//! - `ArboardInjector` (cross-platform): `Arc<parking_lot::Mutex<Inner>>`
//!   para permitir `inject`/`type_text` concurrentes con `&self`.
//! - `SmartInjector`: encadena ambos con fallback y se entrega al pipeline
//!   como `Arc<dyn Injector>` desde `main.rs`.

use thiserror::Error;

/// Errores del crate.
#[derive(Debug, Error)]
pub enum InjectError {
    #[error("clipboard / paste falló: {0}")]
    Inject(String),

    /// El elemento actualmente focused no es editable o no soporta
    /// `send_text` (e.g. es un botón, una lista, o un Edit sin focus
    /// de teclado). El llamador debería caer a `ArboardInjector::inject`.
    #[error("elemento focused no es editable")]
    NotEditable,

    /// La inyección directa por accesibilidad no está disponible en
    /// esta plataforma o no pudo inicializarse. El llamador debería
    /// usar el fallback de clipboard.
    #[error("inyección directa no disponible: {0}")]
    Unsupported(String),
}

/// Trait `Injector` (antes en `oido-platform::traits`). El bin / pipeline
/// lo consume para inyectar el texto transcrito al final del pipeline.
///
/// `&self` (interior mut) para poder compartir la misma instancia entre
/// el thread de release y otros workers. Razón: arboard + enigo
/// requieren `&mut` internamente, así que la impl guarda estado en
/// `Arc<parking_lot::Mutex<Inner>>`.
pub trait Injector: Send + Sync + std::fmt::Debug + 'static {
    fn inject(&self, text: &str) -> Result<(), InjectError>;

    /// Escribe texto simulando pulsaciones de teclas individuales (enigo text),
    /// ideal para streaming incremental para no pisar el clipboard.
    ///
    /// Implementación por defecto: delega a `inject`.
    fn type_text(&self, text: &str) -> Result<(), InjectError> {
        self.inject(text)
    }
}

mod direct;
mod injector;
mod smart;

pub use direct::{DirectInjector, UiaDirectInjector};
pub use injector::ArboardInjector;
pub use smart::SmartInjector;

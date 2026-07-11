//! Crate de inyección de texto. Responsabilidad única:
//!
//! 1. Escribir el texto dictado en el portapapeles del SO.
//! 2. Simular `Ctrl/Cmd+V` para pegarlo en la app con foco.
//!
//! **Regla R2**: este crate es 100% Safe Rust. No contiene `unsafe`.
//!
//! Estrategia: `arboard` (cross-platform) + `enigo` (cross-platform
//! keyboard sim). La impl usa `Arc<parking_lot::Mutex<Inner>>` para
//! que `inject` pueda llamarse desde varios threads con `&self`.

use thiserror::Error;

/// Errores del crate.
#[derive(Debug, Error)]
pub enum InjectError {
    #[error("clipboard / paste falló: {0}")]
    Inject(String),
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

mod injector;

pub use injector::ArboardInjector;

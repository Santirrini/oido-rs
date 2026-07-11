//! Crate de hotkeys globales. Responsabilidad única:
//!
//! 1. Registrar una combinación de teclas hold-to-talk (`rdev::grab` con
//!    supresión selectiva).
//! 2. Ofrecer un wrapper `GatedHotkey` que descarta callbacks hasta
//!    que el modelo STT termina de cargar (carga lazy).
//! 3. Permitir al usuario grabar interactivamente su próxima tecla
//!    (`key_grab`).
//!
//! **Regla R2**: este crate es 100% Safe Rust. No contiene `unsafe`.
//!
//! Tras el refactor modular este crate ya **no depende de `oido-stt`**
//! (se eliminó la llamada muerta a `get_current_win32_thread_id`).

use thiserror::Error;

/// Errores del crate.
#[derive(Debug, Error)]
pub enum HotkeyError {
    #[error("registro de hotkey falló: {0}")]
    Hotkey(String),
}

/// Trait `Hotkey` (antes en `oido-platform::traits`). El bin lo consume
/// para construir el pipeline de hotkey-gated.
pub trait Hotkey: std::fmt::Debug + 'static {
    /// Registra la combinación apuntada por `binding` y conecta los
    /// callbacks boxed. El binding debe estar en formato canónico
    /// aceptado por `parse`.
    fn register(
        &mut self,
        binding: &str,
        on_press: Box<dyn Fn() + Send + 'static>,
        on_release: Box<dyn Fn() + Send + 'static>,
    ) -> Result<(), HotkeyError>;
    fn unregister(&mut self) -> Result<(), HotkeyError>;
}

mod gate;
mod hotkey;
mod key_grab;

pub use gate::{GatedHotkey, GatedReadyHandle};
pub use hotkey::{parse, serialize, RdevHotkey};
pub use key_grab::grab_next_key;

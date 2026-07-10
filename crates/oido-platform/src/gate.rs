//! `GatedHotkey` — wrapper sobre `RdevHotkey` que **suprime** los
//! callbacks de `on_press` / `on_release` hasta que se llame
//! `mark_ready()`. Diseñado para implementar carga lazy del modelo
//! whisper: el bin registra el hotkey de inmediato (la API lo
//! requiere), pero el primer press no llega al pipeline hasta que
//! `load_model` + `warm_up` hayan terminado.
//!
//! Antes de `mark_ready`, los press/release se **descartan** silenciosa-
//! mente. Una vez listo, el comportamiento es idéntico al del
//! `RdevHotkey` envuelto. Si el usuario aprieta el hotkey antes de
//! que el modelo esté listo, la pulsación se pierde (la UI del bin
//! debe mostrar `TrayState::Loading` durante ese intervalo para que
//! el feedback sea claro).
//!
//! Concurrencia: usa `AtomicBool` para el flag, así que `mark_ready`
//! puede llamarse desde un thread de carga en background y los
//! demux threads de `RdevHotkey` lo observan en cada iteración.
//! Memoria: `Ordering::SeqCst` es suficiente (un cambio de ready
//! ocurre una vez en la vida del proceso).

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use crate::hotkey::RdevHotkey;
use crate::traits::{Hotkey, PlatformError};

/// Hotkey con compuerta: los callbacks no se invocan hasta `mark_ready`.
#[derive(Debug)]
pub struct GatedHotkey {
    inner: RdevHotkey,
    ready: Arc<AtomicBool>,
}

impl Default for GatedHotkey {
    fn default() -> Self {
        Self::new()
    }
}

impl GatedHotkey {
    #[must_use]
    pub fn new() -> Self {
        Self {
            inner: RdevHotkey::new(),
            ready: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Comparte el handle al flag de ready con el thread de carga
    /// lazy. El thread de carga llama `handle.mark_ready()` cuando
    /// `load_model` + `warm_up` terminan.
    #[must_use]
    pub fn ready_handle(&self) -> GatedReadyHandle {
        GatedReadyHandle {
            ready: Arc::clone(&self.ready),
        }
    }

    /// Acceso de solo lectura al flag de ready.
    #[must_use]
    pub fn is_ready(&self) -> bool {
        self.ready.load(Ordering::SeqCst)
    }
}

impl Hotkey for GatedHotkey {
    fn register(
        &mut self,
        binding: &str,
        on_press: Box<dyn Fn() + Send + 'static>,
        on_release: Box<dyn Fn() + Send + 'static>,
    ) -> Result<(), PlatformError> {
        // Envolvemos los callbacks del usuario en un check del flag.
        // Antes de ready, el press/release se ignora silenciosamente.
        let ready_press = Arc::clone(&self.ready);
        let on_press = Box::new(move || {
            if ready_press.load(Ordering::SeqCst) {
                (on_press)();
            } else {
                tracing::debug!("hotkey press descartado: modelo aún no listo");
            }
        });
        let ready_release = Arc::clone(&self.ready);
        let on_release = Box::new(move || {
            if ready_release.load(Ordering::SeqCst) {
                (on_release)();
            } else {
                tracing::debug!("hotkey release descartado: modelo aún no listo");
            }
        });
        self.inner.register(binding, on_press, on_release)
    }

    fn unregister(&mut self) -> Result<(), PlatformError> {
        self.inner.unregister()
    }
}

/// Handle liviano para señalar a un `GatedHotkey` que el modelo ya
/// está listo. Se clona y se mueve al thread de carga lazy.
#[derive(Debug, Clone)]
pub struct GatedReadyHandle {
    ready: Arc<AtomicBool>,
}

impl GatedReadyHandle {
    /// Marca el hotkey como listo. Llamar varias veces es idempotente.
    pub fn mark_ready(&self) {
        self.ready.store(true, Ordering::SeqCst);
    }
}

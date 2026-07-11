//! `Injector`: clipboard + paste simulado.
//!
//! - `arboard` para el portapapeles.
//! - `enigo` para la pulsación sintética de Ctrl/Cmd+V en la ventana
//!   activa (el SO ya controla el foco).
//!
//! macOS aviso: en sandbox el pegado sintético bloquea. Hay que añadir
//! `enable_transient_cdevent_for_tracking_area` o habilitar el bundle
//! `osascript` (documentado Fase 5 cuando se firme).
//!
//! La impl usa `Arc<parking_lot::Mutex<Inner>>` para que se pueda
//! llamar a `inject` desde varios threads simultáneamente con `&self`,
//! respetando la firma del trait `Injector`.

use std::sync::Arc;

use arboard::Clipboard;
use enigo::{Direction, Enigo, Key, Keyboard, Settings};
use parking_lot::Mutex;

use crate::{InjectError, Injector};

#[cfg(target_os = "macos")]
const MODIFIER: Key = Key::Meta;
#[cfg(not(target_os = "macos"))]
const MODIFIER: Key = Key::Control;

struct Inner {
    clipboard: Clipboard,
    enigo: Enigo,
}

pub struct ArboardInjector {
    inner: Arc<Mutex<Inner>>,
}

impl std::fmt::Debug for ArboardInjector {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ArboardInjector").finish_non_exhaustive()
    }
}

impl ArboardInjector {
    pub fn new() -> Result<Arc<Self>, InjectError> {
        let clipboard =
            Clipboard::new().map_err(|e| InjectError::Inject(format!("clipboard: {e}")))?;
        let enigo = Enigo::new(&Settings::default())
            .map_err(|e| InjectError::Inject(format!("enigo: {e}")))?;
        Ok(Arc::new(Self {
            inner: Arc::new(Mutex::new(Inner { clipboard, enigo })),
        }))
    }
}

impl Injector for ArboardInjector {
    fn inject(&self, text: &str) -> Result<(), InjectError> {
        let mut inner = self.inner.lock();
        inner
            .clipboard
            .set_text(text.to_owned())
            .map_err(|e| InjectError::Inject(format!("clipboard set: {e}")))?;

        // Pequeño respiro para que la ventana destino procese el cambio
        // antes de mandar la pulsación; en Win especialmente ahorra
        // races con apps de bajo overhead. Por defecto 5 ms, configurable
        // mediante la variable de entorno OIDO_INJECT_GUARD_MS.
        let guard_ms = std::env::var("OIDO_INJECT_GUARD_MS")
            .ok()
            .and_then(|val| val.parse::<u64>().ok())
            .unwrap_or(5);
        if guard_ms > 0 {
            std::thread::sleep(std::time::Duration::from_millis(guard_ms));
        }

        inner
            .enigo
            .key(MODIFIER, Direction::Press)
            .map_err(|e| InjectError::Inject(format!("enigo mod press: {e}")))?;
        inner
            .enigo
            .key(Key::Unicode('v'), Direction::Press)
            .map_err(|e| InjectError::Inject(format!("enigo V press: {e}")))?;
        inner
            .enigo
            .key(Key::Unicode('v'), Direction::Release)
            .map_err(|e| InjectError::Inject(format!("enigo V release: {e}")))?;
        inner
            .enigo
            .key(MODIFIER, Direction::Release)
            .map_err(|e| InjectError::Inject(format!("enigo mod release: {e}")))?;
        // Re-press el modificador a "ninguna" semántica: el SO ya está
        // esperando. Si la app destino ignoró V por estado de foco,
        // Fase 1 no lo reintenta (YAGNI retry).
        Ok(())
    }

    fn type_text(&self, text: &str) -> Result<(), InjectError> {
        let mut inner = self.inner.lock();
        inner
            .enigo
            .text(text)
            .map_err(|e| InjectError::Inject(format!("enigo text: {e}")))?;
        Ok(())
    }
}

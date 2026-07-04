//! `Injector`: clipboard + paste simulado.
//!
//! - `arboard` para el portapapeles.
//! - `enigo` para la pulsación sintética de Ctrl/Cmd+V en la ventana
//!   activa (el SO ya controla el foco).
//!
//!
//! macOS aviso: en sandbox el pegado sintético bloquea. Hay que añadir
//! `enable_transient_cdevent_for_tracking_area` o habilitar el bundle
//! `osascript` (documentado Fase 5 cuando se firme).

use arboard::Clipboard;
use enigo::{Direction, Enigo, Key, Keyboard, Settings};

use crate::traits::{Injector, PlatformError};

#[cfg(target_os = "macos")]
const MODIFIER: Key = Key::Meta;
#[cfg(not(target_os = "macos"))]
const MODIFIER: Key = Key::Control;

pub struct ArboardInjector {
    clipboard: Clipboard,
    enigo: Enigo,
}

impl std::fmt::Debug for ArboardInjector {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ArboardInjector").finish_non_exhaustive()
    }
}

impl Default for ArboardInjector {
    fn default() -> Self {
        Self::new().expect("init arboard+enigo")
    }
}

impl ArboardInjector {
    pub fn new() -> Result<Self, PlatformError> {
        let clipboard =
            Clipboard::new().map_err(|e| PlatformError::Inject(format!("clipboard: {e}")))?;
        let enigo = Enigo::new(&Settings::default())
            .map_err(|e| PlatformError::Inject(format!("enigo: {e}")))?;
        Ok(Self { clipboard, enigo })
    }
}

impl Injector for ArboardInjector {
    fn inject(&mut self, text: &str) -> Result<(), PlatformError> {
        self.clipboard
            .set_text(text.to_owned())
            .map_err(|e| PlatformError::Inject(format!("clipboard set: {e}")))?;

        // Pequeño respiro para que la ventana destino procese el cambio
        // antes de mandar la pulsación; en Win especialmente ahorra
        // races con apps de bajo overhead.
        std::thread::sleep(std::time::Duration::from_millis(20));

        self.enigo
            .key(Key::Unicode('v'), Direction::Press)
            .map_err(|e| PlatformError::Inject(format!("enigo V press: {e}")))?;
        self.enigo
            .key(Key::Unicode('v'), Direction::Release)
            .map_err(|e| PlatformError::Inject(format!("enigo V release: {e}")))?;
        self.enigo
            .key(MODIFIER, Direction::Release)
            .map_err(|e| PlatformError::Inject(format!("enigo mod release: {e}")))?;

        // Re-press el modificador a "ninguna" semántica: el SO ya está
        // esperando. Si la app destino ignoró V por estado de foco,
        // Fase 1 no lo reintenta (YAGNI retry).
        Ok(())
    }
}

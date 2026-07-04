//! `Hotkey` global via `global-hotkey` (cross-platform: usa `xdo` en X11,
//! `winapi` en Windows, `CGEvent` en macOS).
//!
//! MVP: tecla F8 fija. Fase 2 lo hace configurable desde `Config`.
//!
//! Limitacion conocida en macOS: la primera vez el usuario debe dar
//! permiso Accessibility al bin desde Preferencias → Privacidad y
//! Seguridad. Esto requiere entitlement `NSAppleEventsUsageDescription`
//! en Info.plist (lo añadimos en Fase 5 cuando llegue el instalador).

use global_hotkey::hotkey::{Code, HotKey as GHKey, Modifiers};
use global_hotkey::{GlobalHotKeyEvent, GlobalHotKeyManager, HotKeyState};

use crate::traits::{Hotkey, PlatformError};

const DEFAULT_HOTKEY_CODE: Code = Code::F8;

pub struct GhHotkey {
    manager: GlobalHotKeyManager,
    registered: Option<GHKey>,
}

impl std::fmt::Debug for GhHotkey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GhHotkey")
            .field("registered", &self.registered.is_some())
            .finish()
    }
}

impl Default for GhHotkey {
    fn default() -> Self {
        Self::new()
    }
}

impl GhHotkey {
    #[must_use]
    pub fn new() -> Self {
        Self {
            manager: GlobalHotKeyManager::new().expect("init global-hotkey"),
            registered: None,
        }
    }
}

impl Hotkey for GhHotkey {
    fn register(
        &mut self,
        on_press: Box<dyn Fn() + Send + 'static>,
        on_release: Box<dyn Fn() + Send + 'static>,
    ) -> Result<(), PlatformError> {
        // El usuario quiere procesar eventos en su thread; pero
        // `GlobalHotKeyManager::event_handler()` se invoca desde un
        // thread interno del crate, y ese closure debe poder moverse
        // ahí. Usamos channels para no bloquear ese thread.
        let (press_tx, press_rx) = crossbeam_channel::unbounded::<()>();
        let (release_tx, release_rx) = crossbeam_channel::unbounded::<()>();
        let hotkey = GHKey::new(Some(Modifiers::empty()), DEFAULT_HOTKEY_CODE);
        self.manager
            .register(hotkey)
            .map_err(|e| PlatformError::Hotkey(format!("register: {e}")))?;
        self.registered = Some(hotkey);

        // Drena los eventos del receiver global y los reparte a los
        // callbacks del usuario. El thread vive hasta el final del
        // proceso; en Fase 2 gestionamos shutdown explícito.
        std::thread::Builder::new()
            .name("oido-hotkey".into())
            .spawn(move || {
                loop {
                    let evt = match GlobalHotKeyEvent::receiver().recv() {
                        Ok(e) => e,
                        Err(_) => return,
                    };
                    if evt.state() == HotKeyState::Pressed {
                        let _ = press_tx.send(());
                    } else {
                        let _ = release_tx.send(());
                    }
                }
            })
            .map_err(|e| PlatformError::Hotkey(format!("spawn hotkey thread: {e}")))?;

        std::thread::Builder::new()
            .name("oido-hotkey-press".into())
            .spawn(move || {
                while press_rx.recv().is_ok() {
                    on_press();
                }
            })
            .map_err(|e| PlatformError::Hotkey(format!("spawn press thread: {e}")))?;

        std::thread::Builder::new()
            .name("oido-hotkey-release".into())
            .spawn(move || {
                while release_rx.recv().is_ok() {
                    on_release();
                }
            })
            .map_err(|e| PlatformError::Hotkey(format!("spawn release thread: {e}")))?;

        Ok(())
    }

    fn unregister(&mut self) -> Result<(), PlatformError> {
        if let Some(h) = self.registered.take() {
            self.manager
                .unregister(h)
                .map_err(|e| PlatformError::Hotkey(format!("unregister: {e}")))?;
        }
        Ok(())
    }
}

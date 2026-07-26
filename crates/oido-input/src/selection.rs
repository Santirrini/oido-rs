//! Lector de selección de cursor con fallback de clipboard.
//!
//! Estrategia:
//! 1. **Primario** (Windows opt-in, vía UIA): `DirectInjector::read_selection`
//!    que ya implementa `oido-uia-worker`. Si devuelve
//!    `Ok(text)`, listo.
//! 2. **Fallback** (cross-platform, sin permisos especiales): preservamos
//!    el clipboard del usuario → enviamos `Ctrl+C` (o `Cmd+C` en macOS)
//!    vía `enigo` → leemos el clipboard → restauramos.
//!
//! **Por qué el fallback es seguro pero imperfecto**: pisar el clipboard
//! del usuario (con copia + Ctrl+C) normalmente es invisible, pero:
//!
//! - El usuario puede tener contenido binario (e.g. screenshots) que al
//!   "Ctrl+C" no genera texto y produce un clipboard vacío o error.
//! - El usuario puede estar interactuando con un control que ignora
//!   Ctrl+C (terminal, juego, navegador en fullscreen).
//!
//! Por eso UIA es el path preferido cuando está disponible, y el bin
//! avisa al usuario via `tracing::warn!` cuando activa el fallback
//! reiteradamente (indicador de "configura accesos para esta app").

use std::sync::Arc;

use arboard::Clipboard;
use enigo::{Direction, Enigo, Key, Keyboard, Settings};

use crate::direct::DirectInjector;
use crate::InjectError;

/// Trait abstracto sobre el wrapper de inyector para que el bin y los
/// tests puedan componer. Es exactamente igual que el comportamiento que
/// queríamos: primero UIA, después fallback de clipboard.
/// Ver `SelectionReader` más abajo.
///
/// Diseño: el `SelectionReader` tiene un handle al `DirectInjector`
/// (que encapsula el worker UIA) y un `Clipboard` cacheado.
///
/// **Restricción**: NO usar directamente desde threads concurrentes — la
/// API de `Clipboard::set_text` y `Enigo::key` toman locks
/// implícitos. Componemos con `Send + Sync` por clonación de `Arc`.
pub struct SelectionReader {
    direct: Option<Arc<dyn DirectInjector>>,
    /// Plataforma donde corre este bin. Windows usa `Control+C`;
    /// macOS usa `Meta+C`. Linux: `Control+C` (estándar X11/Wayland).
    platform: Platform,
    /// Clipboard handle. Se inicializa lazy en `read()`.
    clipboard: Option<Clipboard>,
}

/// Plataformas Soportadas por el fallback de clipboard. Las variantes
/// determinan el modificador de copia (`Ctrl` vs `Meta`). `pub` para
/// que el bin pueda hacer `Platform::current()`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Platform {
    Windows,
    Macos,
    Linux,
}

impl Platform {
    /// El modificador primario: Ctrl en Win/Linux, Meta (Cmd) en macOS.
    /// `enigo` lo abstrae vía `Key::Control` / `Key::Meta`.
    fn copy_modifier(&self) -> Key {
        match self {
            Platform::Macos => Key::Meta,
            _ => Key::Control,
        }
    }

    /// Detecta la plataforma del bin actual.
    #[must_use]
    pub fn current() -> Self {
        if cfg!(target_os = "macos") {
            Platform::Macos
        } else if cfg!(target_os = "linux") {
            Platform::Linux
        } else {
            Platform::Windows
        }
    }
}

impl std::fmt::Debug for SelectionReader {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SelectionReader")
            .field("direct", &self.direct.as_ref().map(|d| d as &dyn std::fmt::Debug))
            .field("platform", &self.platform)
            .finish()
    }
}

impl SelectionReader {
    /// Construye el reader. Si `direct` es `Some`, intenta UIA primero;
    /// si es `None`, va directo al fallback de clipboard (e.g. macOS,
    /// Linux, o Windows con UIA deshabilitado por env).
    #[must_use]
    pub fn new(direct: Option<Arc<dyn DirectInjector>>, platform: Platform) -> Self {
        Self {
            direct,
            platform,
            clipboard: None,
        }
    }

    /// Variante conveniente que detecta la plataforma del bin.
    /// Llamar en `main()` (donde el target está disponible).
    #[must_use]
    pub fn with_current_platform(
        direct: Option<Arc<dyn DirectInjector>>,
    ) -> Self {
        Self::new(direct, Platform::current())
    }

    /// Lee el texto actualmente seleccionado en el control con foco.
    ///
    /// **Nota sobre timing**: el envío de `Ctrl+C` y el `Clipboard::get_text`
    /// tienen que correr en threads distintos de los callbacks de
    /// hotkey (`rdev`) para evitar races — el callback de hotkey
    /// debe soltar el control lo antes posible. Esta función está
    /// diseñada para llamarse desde un thread dedicado (e.g. el
    /// worker del `TtsPipeline` que ya tengo en `oido-core`).
    ///
    /// Estrategia:
    /// 1. Intenta UIA si hay `direct`.
    /// 2. Si UIA falla o no existe, intenta clipboard:
    ///    a. Guarda clipboard actual.
    ///    b. `Enigo::key_down` modificador + `Key::C` + soltar.
    ///    c. Lee clipboard.
    ///    d. Restaura clipboard original.
    /// 3. Devuelve `Err(Unsupported(_))` si todo lo anterior falla.
    pub fn read(&mut self) -> Result<String, InjectError> {
        // 1) Intentar UIA si está disponible.
        if let Some(d) = &self.direct {
            match d.read_selection() {
                Ok(text) => return Ok(text),
                Err(e) => {
                    tracing::debug!(
                        ?e,
                        "UIA read_selection falló; intentando fallback clipboard"
                    );
                }
            }
        }

        // 2) Fallback clipboard + Ctrl/Cmd+C.
        self.read_via_clipboard()
    }

    /// Path clipboard-only (puede llamarse si queremos saltarnos UIA).
    pub fn read_via_clipboard(&mut self) -> Result<String, InjectError> {
        // Lazy-init del clipboard (puede fallar si el SO no provee uno,
        // e.g. Linux sin X11).
        if self.clipboard.is_none() {
            let c = Clipboard::new().map_err(|e| {
                InjectError::Unsupported(format!("clipboard init: {e}"))
            })?;
            self.clipboard = Some(c);
        }
        let clipboard = self.clipboard.as_mut().expect("just initialized");

        // a) Guarda contenido actual (puede ser texto, imagen u otros —
        // `get_text` extrae el texto si lo hay).
        let saved: Option<String> = clipboard.get_text().ok();

        // b) Envía Ctrl/Cmd+C.
        // Importante: `Enigo::new` necesita configuración de plataforma
        // (especialmente macOS). `Settings::default()` elige el backend
        // correcto.
        send_copy_shortcut(self.platform).map_err(|e| {
            InjectError::Unsupported(format!("send_copy_shortcut: {e}"))
        })?;

        // c) Lee clipboard nuevo. Si falla o devuelve vacío, no hay
        // selección textual (puede ser imagen, archivo, control sin
        // selección, etc.).
        let result_text = clipboard.get_text().map_err(|e| {
            InjectError::Unsupported(format!("clipboard get_text: {e}"))
        })?;

        // d) Restaurar. Si el guardado no era texto, se omite la
        // restauración para evitar pisar el clipboard con tipos
        // incompatibles (imagen, etc.) — `set_text` lo rechazaría y
        // haríamos daño.
        if let Some(saved_text) = saved {
            // set_text puede fallar (e.g. el owner del clipboard cambió
            // entre save/restore). Logueamos pero no es fatal.
            if let Err(e) = clipboard.set_text(saved_text) {
                tracing::warn!(?e, "no se pudo restaurar clipboard");
            }
        }

        Ok(result_text)
    }
}

/// Envía Ctrl+C (o Cmd+C en macOS) y vuelve al estado normal.
///
/// Envolvemos `Enigo` para que el resto del módulo `oido-input` no
/// importe `enigo` directamente (mantiene el grafo de dependencias
/// limpio y facilita cambiar la impl en tests).
fn send_copy_shortcut(platform: Platform) -> Result<(), String> {
    let modifier = platform.copy_modifier();
    let mut enigo = Enigo::new(&Settings::default())
        .map_err(|e| format!("enigo init: {e}"))?;

    enigo
        .key(modifier, Direction::Press)
        .map_err(|e| format!("enigo press(modifier): {e}"))?;
    enigo
        .key(Key::Unicode('c'), Direction::Click)
        .map_err(|e| format!("enigo click('c'): {e}"))?;
    enigo
        .key(modifier, Direction::Release)
        .map_err(|e| format!("enigo release(modifier): {e}"))?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Mock de `DirectInjector` programable.
    #[derive(Debug, Default)]
    struct MockDirect {
        next_selection: std::sync::Mutex<Option<Result<String, InjectError>>>,
        calls: std::sync::Mutex<u32>,
    }
    impl MockDirect {
        fn new(next: Result<String, InjectError>) -> Arc<Self> {
            Arc::new(Self {
                next_selection: std::sync::Mutex::new(Some(next)),
                calls: std::sync::Mutex::new(0),
            })
        }
    }
    impl DirectInjector for MockDirect {
        fn inject_focused(&self, _text: &str) -> Result<(), InjectError> {
            unimplemented!()
        }
        fn read_selection(&self) -> Result<String, InjectError> {
            *self.calls.lock().unwrap() += 1;
            self.next_selection
                .lock()
                .unwrap()
                .take()
                .unwrap_or(Err(InjectError::Unsupported("no stub answer".into())))
        }
    }

    /// Sin direct: directo al fallback de clipboard. En test el
    /// fallback usa clipboard real — lo excluimos para no depender del
    /// entorno (CI headless fallaría).
    #[test]
    fn with_direct_ok_returns_direct_result() {
        let direct = MockDirect::new(Ok("hola mundo".into()));
        let mut reader = SelectionReader::new(Some(direct.clone()), Platform::Windows);
        let got = reader.read().expect("read_selection Ok debe propagarse");
        assert_eq!(got, "hola mundo");
        assert_eq!(*direct.calls.lock().unwrap(), 1, "debe intentar 1 vez el direct");
    }

    /// Con direct devolviendo error: el reader debe intentar el fallback
    /// de clipboard. Marcado `#[ignore]` porque requiere un display
    /// gráfico funcional (CI headless o máquinas sin sesión activa
    /// abortan al instanciar `Enigo`). Para correr localmente:
    ///   `cargo test -p oido-input --lib selection::tests::with_direct_err_falls_back_to_clipboard -- --ignored`
    #[test]
    #[ignore = "requiere display gráfico (Enigo + Clipboard hacen syscall al SO)"]
    fn with_direct_err_falls_back_to_clipboard() {
        let direct = MockDirect::new(Err(InjectError::Unsupported("mock".into())));
        let mut reader = SelectionReader::new(Some(direct.clone()), Platform::Windows);
        let _ = reader.read();
        assert!(
            *direct.calls.lock().unwrap() >= 1,
            "direct debe llamarse al menos una vez"
        );
    }

    /// Sin direct: salta directo al fallback.
    /// Mismo motivo que el test anterior (display gráfico requerido).
    #[test]
    #[ignore = "requiere display gráfico"]
    fn without_direct_goes_straight_to_clipboard_fallback() {
        let mut reader = SelectionReader::new(None, Platform::Windows);
        let _ = reader.read();
    }

    /// El modificador de macOS es Meta (Cmd), no Control. Test puro
    /// (no toca enigo), corre en cualquier sitio.
    #[test]
    fn platform_copy_modifier_per_os() {
        assert_eq!(Platform::Macos.copy_modifier(), Key::Meta);
        assert_eq!(Platform::Windows.copy_modifier(), Key::Control);
        assert_eq!(Platform::Linux.copy_modifier(), Key::Control);
    }

    /// Plataforma detectada por cfg macro.
    #[test]
    fn platform_current_is_a_supported_variant() {
        match Platform::current() {
            Platform::Windows | Platform::Macos | Platform::Linux => {}
        }
    }

    /// Constructor respeta el orden de prioridad: direct → clipboard.
    #[test]
    fn new_keeps_direct_arc() {
        let direct = MockDirect::new(Ok("x".into()));
        let reader = SelectionReader::new(Some(direct.clone()), Platform::Linux);
        // Sólo verificamos que compila y que no panicea el constructor.
        assert!(matches!(reader.platform, Platform::Linux));
    }
}

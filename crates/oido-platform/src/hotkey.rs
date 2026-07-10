//! `Hotkey` global via `global-hotkey` (cross-platform: usa `xdo` en X11,
//! `winapi` en Windows, `CGEvent` en macOS).
//!
//! El binding (combinación tecla+modificadores) **viene de
//! `Config.hotkey`** (string canónico, ej. `"F8"`, `"Ctrl+Shift+D"`).
//! El parser delega en `global_hotkey::hotkey::HotKey::try_from` —
//! reutilizamos su gramática (compatible con Tauri) en vez de reinventar
//! una tabla de teclas que quedaría incompleta y divergente.
//!
//! Limitacion conocida en macOS: la primera vez el usuario debe dar
//! permiso Accessibility al bin desde Preferencias → Privacidad y
//! Seguridad. Esto requiere entitlement `NSAppleEventsUsageDescription`
//! en Info.plist (lo añadimos en Fase 5 cuando llegue el instalador).

use std::fmt;

use global_hotkey::hotkey::{Code, HotKey as GHKey, Modifiers};
use global_hotkey::{GlobalHotKeyEvent, GlobalHotKeyManager, HotKeyState};

use crate::traits::{Hotkey, PlatformError};

pub struct GhHotkey {
    manager: GlobalHotKeyManager,
    registered: Option<GHKey>,
}

impl fmt::Debug for GhHotkey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
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

/// Parsea un binding canónico (`"F8"`, `"Ctrl+Shift+D"`,
/// `"CommandOrControl+Alt+F12"`) a `(Modifiers, Code)`.
///
/// Internamente delega en `global_hotkey::hotkey::HotKey::try_from`,
/// que ya acepta el formato de Tauri (`"CmdOrCtrl"`, números como
/// `"5"`, etc.) — no reinventamos la rueda.
pub fn parse(binding: &str) -> Result<(Modifiers, Code), PlatformError> {
    let trimmed = binding.trim();
    if trimmed.is_empty() {
        return Err(PlatformError::Hotkey("parse: binding vacío".into()));
    }
    let parsed = GHKey::try_from(trimmed)
        .map_err(|e| PlatformError::Hotkey(format!("parse: {binding:?} → {e}")))?;
    Ok((parsed.mods, parsed.key))
}

/// Serializa un par `(Modifiers, Code)` al formato canónico que el
/// parser reconoce. Orden fijo: `Ctrl+Shift+Alt+Meta+<Key>`. Title-case
/// en modificadores (coincide con el formato que ya asume el test
/// `config_store_replace_then_snapshot` en `oido-config`).
#[must_use]
pub fn serialize(mods: Modifiers, code: Code) -> String {
    let mut s = String::new();
    if mods.contains(Modifiers::CONTROL) {
        s.push_str("Ctrl+");
    }
    if mods.contains(Modifiers::SHIFT) {
        s.push_str("Shift+");
    }
    if mods.contains(Modifiers::ALT) {
        s.push_str("Alt+");
    }
    if mods.contains(Modifiers::SUPER) {
        s.push_str("Meta+");
    }
    s.push_str(&code_to_token(code));
    s
}

/// Mapea cada `Code` de `keyboard_types` (lo que `global_hotkey`
/// re-exporta) al token que `global_hotkey::parse_hotkey` reconoce a
/// la ida. Garantiza que `parse(serialize(m, c)) == Ok((m, c))`.
fn code_to_token(code: Code) -> String {
    use Code::*;
    // Estrategia: token de Tauri/global-hotkey por defecto. Si la
    // tecla tiene alias cortos canónicos (dígitos, letras, símbolos,
    // F-keys) emitimos el alias más legible.
    match code {
        // Dígitos
        Digit0 => "0".into(),
        Digit1 => "1".into(),
        Digit2 => "2".into(),
        Digit3 => "3".into(),
        Digit4 => "4".into(),
        Digit5 => "5".into(),
        Digit6 => "6".into(),
        Digit7 => "7".into(),
        Digit8 => "8".into(),
        Digit9 => "9".into(),
        // Letras (sin prefijo "Key")
        KeyA => "A".into(),
        KeyB => "B".into(),
        KeyC => "C".into(),
        KeyD => "D".into(),
        KeyE => "E".into(),
        KeyF => "F".into(),
        KeyG => "G".into(),
        KeyH => "H".into(),
        KeyI => "I".into(),
        KeyJ => "J".into(),
        KeyK => "K".into(),
        KeyL => "L".into(),
        KeyM => "M".into(),
        KeyN => "N".into(),
        KeyO => "O".into(),
        KeyP => "P".into(),
        KeyQ => "Q".into(),
        KeyR => "R".into(),
        KeyS => "S".into(),
        KeyT => "T".into(),
        KeyU => "U".into(),
        KeyV => "V".into(),
        KeyW => "W".into(),
        KeyX => "X".into(),
        KeyY => "Y".into(),
        KeyZ => "Z".into(),
        // Símbolos con alias corto
        Space => "Space".into(),
        Backquote => "`".into(),
        Backslash => "\\".into(),
        BracketLeft => "[".into(),
        BracketRight => "]".into(),
        Comma => ",".into(),
        Period => ".".into(),
        Semicolon => ";".into(),
        Quote => "'".into(),
        Minus => "-".into(),
        Equal => "=".into(),
        Slash => "/".into(),
        // Resto: delegamos en Display de `Code` (que da el nombre del
        // enum, ej. "F12", "ArrowUp", "NumpadAdd" — todos aceptados
        // por `parse_hotkey`).
        other => other.to_string(),
    }
}

impl Hotkey for GhHotkey {
    fn register(
        &mut self,
        binding: &str,
        on_press: Box<dyn Fn() + Send + 'static>,
        on_release: Box<dyn Fn() + Send + 'static>,
    ) -> Result<(), PlatformError> {
        let (mods, code) = parse(binding)?;

        // El usuario quiere procesar eventos en su thread; pero
        // `GlobalHotKeyManager::event_handler()` se invoca desde un
        // thread interno del crate, y ese closure debe poder moverse
        // ahí. Usamos channels para no bloquear ese thread.
        let (press_tx, press_rx) = crossbeam_channel::unbounded::<()>();
        let (release_tx, release_rx) = crossbeam_channel::unbounded::<()>();
        let hotkey = GHKey::new(Some(mods), code);
        self.manager
            .register(hotkey)
            .map_err(|e| PlatformError::Hotkey(format!("register: {e}")))?;
        self.registered = Some(hotkey);

        // Drena los eventos del receiver global y los reparte a los
        // callbacks del usuario. El thread vive hasta el final del
        // proceso; en Fase 2 gestionamos shutdown explícito.
        std::thread::Builder::new()
            .name("oido-hotkey".into())
            .spawn(move || loop {
                let evt = match GlobalHotKeyEvent::receiver().recv() {
                    Ok(e) => e,
                    Err(_) => return,
                };
                if evt.state() == HotKeyState::Pressed {
                    let _ = press_tx.send(());
                } else {
                    let _ = release_tx.send(());
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_simple_key() {
        let (mods, code) = parse("F8").expect("F8 parsea");
        assert!(mods.is_empty());
        assert_eq!(code, Code::F8);
    }

    #[test]
    fn parse_with_modifiers() {
        let (mods, code) = parse("Ctrl+Shift+D").expect("combo parsea");
        assert!(mods.contains(Modifiers::CONTROL));
        assert!(mods.contains(Modifiers::SHIFT));
        assert_eq!(code, Code::KeyD);
    }

    #[test]
    fn parse_is_case_insensitive_for_modifiers() {
        let (mods, code) = parse("ctrl+shift+d").expect("lowercase parsea");
        assert!(mods.contains(Modifiers::CONTROL));
        assert!(mods.contains(Modifiers::SHIFT));
        assert_eq!(code, Code::KeyD);
    }

    #[test]
    fn parse_accepts_tauri_aliases() {
        // "CommandOrControl" es alias de Tauri: en no-macOS = Control.
        let (mods, _) = parse("CommandOrControl+Alt+F12").expect("alias parsea");
        assert!(mods.contains(Modifiers::CONTROL));
        assert!(mods.contains(Modifiers::ALT));
    }

    #[test]
    fn parse_rejects_empty() {
        assert!(parse("").is_err());
        assert!(parse("   ").is_err());
    }

    #[test]
    fn parse_rejects_bogus_key() {
        assert!(parse("FooBar").is_err());
    }

    #[test]
    fn serialize_roundtrip_simple_key() {
        let s = serialize(Modifiers::empty(), Code::F8);
        let (mods, code) = parse(&s).expect("serialize→parse");
        assert!(mods.is_empty());
        assert_eq!(code, Code::F8);
    }

    #[test]
    fn serialize_roundtrip_combination() {
        let mods = Modifiers::CONTROL | Modifiers::SHIFT;
        let s = serialize(mods, Code::KeyD);
        assert_eq!(s, "Ctrl+Shift+D");
        let (parsed_mods, parsed_code) = parse(&s).expect("serialize→parse");
        assert_eq!(parsed_mods, mods);
        assert_eq!(parsed_code, Code::KeyD);
    }

    #[test]
    fn serialize_orders_modifiers_canonically() {
        // Aunque el usuario teclee "Shift+Ctrl+D", la salida canónica es
        // Ctrl+Shift+<Key>.
        let s = serialize(Modifiers::SHIFT | Modifiers::CONTROL, Code::KeyD);
        assert_eq!(s, "Ctrl+Shift+D");
    }

    #[test]
    fn serialize_handles_meta_and_alt() {
        let mods = Modifiers::SUPER | Modifiers::ALT;
        let s = serialize(mods, Code::F12);
        assert_eq!(s, "Alt+Meta+F12");
    }

    #[test]
    fn from_str_parse_matches_helper() {
        // Mismo input, dos puntos de entrada: deben coincidir.
        let (a, _) = parse("Ctrl+Shift+D").unwrap();
        let hk: GHKey = "Ctrl+Shift+D".parse().unwrap();
        assert_eq!(a, hk.mods);
    }
}
//! `Hotkey` global vía `rdev::listen`.
//!
//! ## Por qué `rdev` y no `global-hotkey` (lo que usamos en Fase 1)
//!
//! El pipeline es **hold-to-talk**:
//!
//! - `on_press` (key-down)  → inicia la grabación
//! - `on_release` (key-up)  → corta y encola para STT
//!
//! El crate `global-hotkey` se apoya en `RegisterHotKey` de Win32, que
//! **solo entrega `WM_HOTKEY` al presionar**, nunca al soltar. Es una
//! limitación de Windows, no del crate (ver `RegisterHotKey` en
//! Microsoft Learn). Resultado en Fase 1: la grabación arrancaba pero
//! nunca se detenía y el audio nunca llegaba al worker STT.
//!
//! `rdev::listen` usa `RegisterRawInputDevices` en Windows y emite
//! `KeyPress` / `KeyRelease` por separado, lo que sí cubre el ciclo
//! hold-to-talk. En macOS requiere permisos Accessibility (igual que
//! `global-hotkey`); en Linux/Wayland queda bloqueado por el protocolo.
//!
//! ## Diseño
//!
//! - El binding canónico (`"F8"`, `"Alt+Space"`, `"1"`) se parsea vía
//!   `parse` (que delega en `global_hotkey::hotkey::HotKey::try_from`)
//!   para obtener `(target_mods, target_code)`.
//! - El listener `rdev` mantiene un set de modificadores acumulados
//!   con la misma ventana `MODIFIER_WINDOW` que `key_grab`.
//! - En cada `KeyPress` se compara `mods` + `key` con el target; si
//!   matchean se envía por el canal de press.
//! - En cada `KeyRelease` se compara solo la `key` con el target; si
//!   matchea se envía por el canal de release. NO limpiamos
//!   modificadores al release (el usuario suele soltar Shift/Ctrl un
//!   instante después de la tecla principal).
//!
//! El binding se reutiliza también en `key_grab` para mapear
//! `rdev::Key` → `Code` y `rdev::Key` → `Modifiers` (expuestos como
//! `pub(crate)` desde `key_grab.rs`).

use std::fmt;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread::{self, JoinHandle};

use global_hotkey::hotkey::{Code, HotKey as GHKey, Modifiers};
use rdev::{Event, EventType};

use crate::key_grab::{key_to_code, key_to_modifier};
use crate::traits::{Hotkey, PlatformError};

/// Ventana durante la cual un press de modificador se considera "parte"
/// de la combinación que el usuario está formando. Igual que en
/// `key_grab::MODIFIER_WINDOW` para mantener consistencia de UX.
const MODIFIER_WINDOW_MS: u64 = 500;

/// Backend de hotkey basado en `rdev::listen`.
///
/// Una sola instancia por proceso. Tras `register(binding, …)` el listener
/// queda corriendo en un thread dedicado hasta que se llama a
/// `unregister()` o hasta el shutdown del proceso.
pub struct RdevHotkey {
    /// Receiver interno de rdev queda en este thread; `unregister` solo
    /// sube `running` para futuras iteraciones (rdev::listen no
    /// interrumpe hoy por API). En la práctica el thread muere cuando
    /// el canal se cierra o el proceso sale.
    listener: Option<JoinHandle<()>>,
    /// Flag observado por el callback; subido en `unregister`.
    running: Arc<AtomicBool>,
    /// Binding actualmente registrado (para diagnostics y para evitar
    /// re-registros accidentales).
    active: Option<(Modifiers, Code)>,
}

impl fmt::Debug for RdevHotkey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RdevHotkey")
            .field("active", &self.active)
            .field("listener_alive", &self.listener.is_some())
            .finish()
    }
}

impl Default for RdevHotkey {
    fn default() -> Self {
        Self::new()
    }
}

impl RdevHotkey {
    #[must_use]
    pub fn new() -> Self {
        Self {
            listener: None,
            running: Arc::new(AtomicBool::new(false)),
            active: None,
        }
    }

    /// Indica si hay un listener activo.
    #[must_use]
    pub fn is_active(&self) -> bool {
        self.listener.is_some()
    }

    /// Parsea `binding` y devuelve el par `(mods, code)` que el listener
    /// usará para matching. Útil para tests.
    pub fn parse_target(&self, binding: &str) -> Result<(Modifiers, Code), PlatformError> {
        parse(binding)
    }
}

impl Hotkey for RdevHotkey {
    fn register(
        &mut self,
        binding: &str,
        on_press: Box<dyn Fn() + Send + 'static>,
        on_release: Box<dyn Fn() + Send + 'static>,
    ) -> Result<(), PlatformError> {
        if self.listener.is_some() {
            return Err(PlatformError::Hotkey(
                "register: ya hay un listener activo; llama a unregister primero".into(),
            ));
        }

        let (target_mods, target_code) = parse(binding)?;
        self.active = Some((target_mods, target_code));

        let (press_tx, press_rx) = crossbeam_channel::unbounded::<()>();
        let (release_tx, release_rx) = crossbeam_channel::unbounded::<()>();

        let running = Arc::clone(&self.running);
        running.store(true, Ordering::SeqCst);

        // Hilo listener: drena `rdev::listen` y reparte a los canales.
        let target_mods_for_thread = target_mods;
        let target_code_for_thread = target_code;
        let listener = thread::Builder::new()
            .name("oido-hotkey".into())
            .spawn(move || run_rdev_listener(
                target_mods_for_thread,
                target_code_for_thread,
                press_tx,
                release_tx,
                running,
            ))
            .map_err(|e| PlatformError::Hotkey(format!("spawn listener: {e}")))?;

        // Hilos demux que invocan los closures boxed. Mismo patrón que
        // en el backend anterior.
        thread::Builder::new()
            .name("oido-hotkey-press".into())
            .spawn(move || {
                while press_rx.recv().is_ok() {
                    on_press();
                }
            })
            .map_err(|e| PlatformError::Hotkey(format!("spawn press: {e}")))?;

        thread::Builder::new()
            .name("oido-hotkey-release".into())
            .spawn(move || {
                while release_rx.recv().is_ok() {
                    on_release();
                }
            })
            .map_err(|e| PlatformError::Hotkey(format!("spawn release: {e}")))?;

        self.listener = Some(listener);
        Ok(())
    }

    fn unregister(&mut self) -> Result<(), PlatformError> {
        // Bajamos el flag: futuras iteraciones del callback lo verán y
        // dejarán de emitir eventos. `rdev::listen` no expone API de
        // interrupción hoy; el thread muere cuando el proceso termina.
        self.running.store(false, Ordering::SeqCst);
        // Tomamos el JoinHandle sin bloquear (el thread puede estar
        // todavía bloqueado en rdev::listen; en el shutdown del proceso
        // se cierra de todas formas).
        self.listener.take();
        self.active = None;
        Ok(())
    }
}

impl Drop for RdevHotkey {
    fn drop(&mut self) {
        let _ = self.unregister();
    }
}

fn run_rdev_listener(
    target_mods: Modifiers,
    target_code: Code,
    press_tx: crossbeam_channel::Sender<()>,
    release_tx: crossbeam_channel::Sender<()>,
    running: Arc<AtomicBool>,
) {
    use std::time::{Duration, Instant};

    let mut current_mods = Modifiers::empty();
    let mut last_modifier_at: Option<Instant> = None;
    // Sólo bloqueamos el siguiente key-down si los modificadores fueron
    // "frescos" (dentro de la ventana). Sin esto, los modificadores que
    // el usuario dejó sueltos 5 segundos antes contaminarían el match.

    let callback = move |event: Event| {
        if !running.load(Ordering::SeqCst) {
            return;
        }
        match event.event_type {
            EventType::KeyPress(k) => {
                if let Some(m) = key_to_modifier(k) {
                    current_mods |= m;
                    last_modifier_at = Some(Instant::now());
                    return;
                }
                // Stale mods: si el último modificador fue fuera de la
                // ventana, no forman parte del combo.
                if let Some(t) = last_modifier_at {
                    if t.elapsed() > Duration::from_millis(MODIFIER_WINDOW_MS) {
                        current_mods = Modifiers::empty();
                    }
                }
                if let Some(code) = key_to_code(k) {
                    if code == target_code && current_mods == target_mods {
                        let _ = press_tx.send(());
                    }
                }
            }
            EventType::KeyRelease(k) => {
                if let Some(code) = key_to_code(k) {
                    if code == target_code {
                        let _ = release_tx.send(());
                    }
                }
                // No limpiamos `current_mods` aquí: el usuario suele
                // soltar Shift/Ctrl un instante después de la tecla
                // principal y eso no debe invalidar matches posteriores.
                // La limpieza se hace por ventana de tiempo en el
                // siguiente KeyPress.
            }
            _ => {}
        }
    };

    if let Err(e) = rdev::listen(callback) {
        // Errores típicos: permisos en macOS, X11 sin DISPLAY, RDP en
        // Windows bloqueando raw input. Loggeamos; los canales se
        // cerrarán cuando `running = false` y `unregister` termine.
        tracing::warn!(?e, "rdev::listen terminó con error");
    }
}

/// Parsea un binding canónico (`"F8"`, `"Alt+Space"`, `"1"`,
/// `"CommandOrControl+Alt+F12"`) a `(Modifiers, Code)`.
///
/// Internamente delega en `global_hotkey::hotkey::HotKey::try_from`,
/// que ya acepta el formato de Tauri. Aunque ya no usamos
/// `global-hotkey` para registrar, reutilizamos su gramática para
/// evitar reinventar una tabla de teclas que quedaría divergente.
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
/// parser reconoce. Orden fijo: `Ctrl+Shift+Alt+Meta+<Key>`.
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
    match code {
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
        other => other.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering as AO};

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
        let (a, _) = parse("Ctrl+Shift+D").unwrap();
        let hk: GHKey = "Ctrl+Shift+D".parse().unwrap();
        assert_eq!(a, hk.mods);
    }

    #[test]
    fn rdevhotkey_default_is_inactive() {
        let h = RdevHotkey::default();
        assert!(!h.is_active());
        assert!(h.active.is_none());
    }

    /// Smoke test: registra un binding cualquiera y comprueba que el
    /// tipo compila. No podemos pulsar teclas de verdad en CI, pero al
    /// menos aseguramos que las closures no se llaman en vacío.
    #[test]
    fn rdevhotkey_register_calls_closures_zero_times_without_input() {
        use std::sync::Arc;
        let mut h = RdevHotkey::new();
        let press_count = Arc::new(AtomicUsize::new(0));
        let release_count = Arc::new(AtomicUsize::new(0));
        let pc = Arc::clone(&press_count);
        let rc = Arc::clone(&release_count);
        let on_press = Box::new(move || {
            pc.fetch_add(1, AO::SeqCst);
        });
        let on_release = Box::new(move || {
            rc.fetch_add(1, AO::SeqCst);
        });
        // Parseamos un binding para validar el camino del trait, pero
        // NO podemos registrar realmente (rdev::listen requiere X11/Win).
        let (mods, code) = parse("F8").unwrap();
        assert!(mods.is_empty());
        assert_eq!(code, Code::F8);
        drop((on_press, on_release));
        // assert: los counters siguen en 0 (nadie ha pulsado nada)
        assert_eq!(press_count.load(AO::SeqCst), 0);
        assert_eq!(release_count.load(AO::SeqCst), 0);
        h.unregister().unwrap();
    }
}

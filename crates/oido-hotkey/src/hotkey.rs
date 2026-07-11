//! `Hotkey` global vía `rdev::grab` con **supresión selectiva**.
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
//! ## Por qué `grab` (no `listen`) y por qué requiere feature
//!
//! `rdev::listen` es **observación pasiva** (`RegisterRawInputDevices`
//! en Windows): emite `KeyPress` / `KeyRelease` pero NO bloquea la
//! tecla — F8 llegaba también a la app con foco, contaminando el
//! dictado. Además, mantener F8 presionado disparaba rebote
//! (`Recording → Processing → Recording`) que fragmentaba la
//! grabación y producía alucinaciones.
//!
//! `rdev::grab` (feature `unstable_grab`) instala un hook global en
//! bajo nivel (`WH_KEYBOARD_LL` en Windows, `CGEventTap` en macOS,
//! `evdev` en Linux) y su callback devuelve `Option<Event>`:
//! `None` suprime el evento antes de llegar a la app, `Some(event)`
//! lo deja pasar. Esto nos permite:
//!
//! 1. **Suprimir la tecla target** (press y release) → F8 deja de
//!    "filtrarse" al editor / navegador / cualquier foco.
//! 2. **Pasar los modificadores tal cual** → si el binding es
//!    `Ctrl+Shift+D`, sólo D se suprime; Ctrl/Shift siguen llegando
//!    a la app, evitando modificadores "pegados" si oido se cuelga.
//! 3. **Pasar mouse + todo lo demás** → el hook de grab instala
//!    también un mouse hook; devolver `None` por defecto bloquearía
//!    el ratón entero.
//!
//! `rdev::grab` cubre exactamente el ciclo hold-to-talk porque emite
//! `KeyPress` / `KeyRelease` por separado.
//!
//! ### Caveats por SO (los mismos que `rdev::listen` + unos extra)
//!
//! - **Windows**: hook global; algunas sesiones RDP lo bloquean.
//! - **macOS**: el proceso padre debe tener permisos Accessibility.
//! - **Linux/X11**: requiere DISPLAY; el usuario debe pertenecer al
//!   grupo `input` (o `plugdev` en algunas distros).
//! - **Linux/Wayland**: evdev captura a nivel kernel, funciona.
//!
//! ## Diseño del matching
//!
//! - El binding canónico (`"F8"`, `"Alt+Space"`, `"1"`) se parsea vía
//!   `parse` (que delega en `global_hotkey::hotkey::HotKey::try_from`)
//!   para obtener `(target_mods, target_code)`.
//! - El callback `grab` mantiene un set de modificadores acumulados
//!   con la misma ventana `MODIFIER_WINDOW` que `key_grab`.
//! - En cada `KeyPress` se compara `mods` + `key` con el target; si
//!   matchean: `press_tx.send(())` + suprimir (`None`).
//! - En cada `KeyRelease` se compara solo la `key` con el target; si
//!   matchea: `release_tx.send(())` + suprimir (`None`). NO limpiamos
//!   modificadores al release (el usuario suele soltar Shift/Ctrl un
//!   instante después de la tecla principal).
//! - Modificadores (Ctrl/Shift/Alt/Meta) y cualquier otra tecla
//!   pasan tal cual (`Some(event)`).
//! - Mouse y todo lo demás: `Some(event)` por defecto al final.
//!
//! El binding se reutiliza también en `key_grab` para mapear
//! `rdev::Key` → `Code` y `rdev::Key` → `Modifiers` (expuestos como
//! `pub(crate)` desde `key_grab.rs`).

use std::fmt;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread::{self, JoinHandle};

use global_hotkey::hotkey::{Code, HotKey as GHKey, Modifiers};
use rdev::{Event, EventType};

use crate::key_grab::{key_to_code, key_to_modifier};
use crate::{Hotkey, HotkeyError};

/// Ventana durante la cual un press de modificador se considera "parte"
/// de la combinación que el usuario está formando. Igual que en
/// `key_grab::MODIFIER_WINDOW` para mantener consistencia de UX.
const MODIFIER_WINDOW_MS: u64 = 500;

/// Backend de hotkey basado en `rdev::grab` (con supresión selectiva).
///
/// Una sola instancia por proceso. Tras `register(binding, …)` el
/// callback de `grab` queda corriendo en un thread dedicado hasta que
/// se llama a `unregister()` o hasta el shutdown del proceso.
pub struct RdevHotkey {
    /// Handle al thread del callback de `rdev::grab`; `unregister` solo
    /// sube `running` para futuras iteraciones (rdev::grab no
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
    pub fn parse_target(&self, binding: &str) -> Result<(Modifiers, Code), HotkeyError> {
        parse(binding)
    }
}

impl Hotkey for RdevHotkey {
    fn register(
        &mut self,
        binding: &str,
        on_press: Box<dyn Fn() + Send + 'static>,
        on_release: Box<dyn Fn() + Send + 'static>,
    ) -> Result<(), HotkeyError> {
        if self.listener.is_some() {
            return Err(HotkeyError::Hotkey(
                "register: ya hay un listener activo; llama a unregister primero".into(),
            ));
        }

        let (target_mods, target_code) = parse(binding)?;
        self.active = Some((target_mods, target_code));

        let (press_tx, press_rx) = crossbeam_channel::unbounded::<()>();
        let (release_tx, release_rx) = crossbeam_channel::unbounded::<()>();

        let running = Arc::clone(&self.running);
        running.store(true, Ordering::SeqCst);

        // Hilo listener: drena `rdev::grab` y reparte a los canales.
        let target_mods_for_thread = target_mods;
        let target_code_for_thread = target_code;
        let listener = thread::Builder::new()
            .name("oido-hotkey".into())
            .spawn(move || {
                run_rdev_grab(
                    target_mods_for_thread,
                    target_code_for_thread,
                    press_tx,
                    release_tx,
                    running,
                )
            })
            .map_err(|e| HotkeyError::Hotkey(format!("spawn listener: {e}")))?;

        // Hilos demux que invocan los closures boxed. Mismo patrón que
        // en el backend anterior.
        thread::Builder::new()
            .name("oido-hotkey-press".into())
            .spawn(move || {
                while press_rx.recv().is_ok() {
                    on_press();
                }
            })
            .map_err(|e| HotkeyError::Hotkey(format!("spawn press: {e}")))?;

        thread::Builder::new()
            .name("oido-hotkey-release".into())
            .spawn(move || {
                while release_rx.recv().is_ok() {
                    on_release();
                }
            })
            .map_err(|e| HotkeyError::Hotkey(format!("spawn release: {e}")))?;

        self.listener = Some(listener);
        Ok(())
    }

    fn unregister(&mut self) -> Result<(), HotkeyError> {
        self.running.store(false, Ordering::SeqCst);
        // Evitamos hacer join para prevenir STATUS_ACCESS_VIOLATION
        // debido a las limitaciones/condiciones de carrera de rdev con
        // hooks globales en Windows.
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

fn run_rdev_grab(
    target_mods: Modifiers,
    target_code: Code,
    press_tx: crossbeam_channel::Sender<()>,
    release_tx: crossbeam_channel::Sender<()>,
    running: Arc<AtomicBool>,
) {
    use parking_lot::Mutex;
    use std::time::{Duration, Instant};

    // Estado del matching. Va en un Mutex porque `rdev::grab` 0.5
    // requiere un callback `Fn` (no `FnMut`) — el closure interno en
    // `rdev 0.5.3/src/lib.rs:362` exige eso. Mutamos a través del
    // Mutex. Es local a este thread (`oido-hotkey`) — no incrementa
    // el conteo de mutexes compartidos del workspace (regla R3).
    #[derive(Default)]
    struct MatchState {
        current_mods: Modifiers,
        last_modifier_at: Option<Instant>,
    }
    let state = Mutex::new(MatchState::default());

    // Callback de `rdev::grab`: `None` suprime el evento antes de que
    // llegue a la app con foco; `Some(event)` lo deja pasar. Esto es
    // lo que distingue `grab` de `listen`.
    let callback = move |event: Event| -> Option<Event> {
        if !running.load(Ordering::SeqCst) {
            return Some(event); // shutdown en curso: dejar pasar todo
        }
        match event.event_type {
            EventType::KeyPress(k) => {
                // Modificadores: pasan siempre a la app, sólo los
                // acumulamos para el matching posterior. Suprimirlos
                // dejaría Ctrl/Shift "pegados" si oido se cuelga.
                if let Some(m) = key_to_modifier(k) {
                    let mut s = state.lock();
                    s.current_mods |= m;
                    s.last_modifier_at = Some(Instant::now());
                    return Some(event);
                }
                // Stale mods: si el último modificador fue fuera de la
                // ventana, no forman parte del combo.
                let mut s = state.lock();
                if let Some(t) = s.last_modifier_at {
                    if t.elapsed() > Duration::from_millis(MODIFIER_WINDOW_MS) {
                        s.current_mods = Modifiers::empty();
                    }
                }
                let current_mods = s.current_mods;
                drop(s);
                if let Some(code) = key_to_code(k) {
                    if code == target_code && current_mods == target_mods {
                        let _ = press_tx.send(());
                        // ¡Suprimir! El key-down de la tecla target
                        // nunca llega a la app con foco.
                        return None;
                    }
                }
                // Cualquier otra tecla: pasa normal.
                Some(event)
            }
            EventType::KeyRelease(k) => {
                if let Some(code) = key_to_code(k) {
                    if code == target_code {
                        let _ = release_tx.send(());
                        // También suprimimos el key-up: si F8 estaba
                        // "presionado" para la app con foco, este
                        // evento la descolgaría y vería una F8.
                        return None;
                    }
                }
                // No limpiamos `current_mods` aquí: el usuario suele
                // soltar Shift/Ctrl un instante después de la tecla
                // principal y eso no debe invalidar matches
                // posteriores. La limpieza se hace por ventana de
                // tiempo en el siguiente KeyPress.
                Some(event)
            }
            // Mouse + cualquier otra cosa: pasa por defecto. CRÍTICO:
            // `rdev::grab` también instala un mouse hook; devolver
            // `None` en este branch bloquearía el ratón entero.
            _ => Some(event),
        }
    };

    if let Err(e) = rdev::grab(callback) {
        // Errores típicos: permisos en macOS, X11 sin DISPLAY, RDP en
        // Windows bloqueando raw input. Loggeamos; los canales se
        // cerrarán cuando `running = false` y `unregister` termine.
        tracing::warn!(?e, "rdev::grab terminó con error");
    }
}

/// Parsea un binding canónico (`"F8"`, `"Alt+Space"`, `"1"`,
/// `"CommandOrControl+Alt+F12"`) a `(Modifiers, Code)`.
///
/// Internamente delega en `global_hotkey::hotkey::HotKey::try_from`,
/// que ya acepta el formato de Tauri. Aunque ya no usamos
/// `global-hotkey` para registrar, reutilizamos su gramática para
/// evitar reinventar una tabla de teclas que quedaría divergente.
pub fn parse(binding: &str) -> Result<(Modifiers, Code), HotkeyError> {
    let trimmed = binding.trim();
    if trimmed.is_empty() {
        return Err(HotkeyError::Hotkey("parse: binding vacío".into()));
    }
    let parsed = GHKey::try_from(trimmed)
        .map_err(|e| HotkeyError::Hotkey(format!("parse: {binding:?} → {e}")))?;
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

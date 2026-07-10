//! Captura global interactiva de la próxima tecla que el usuario
//! pulse. Usado por el bin `oido --set-hotkey` para que el usuario
//! "grabe" la tecla de activación sin editar `config.json` a mano.
//!
//! Mecánica:
//! 1. Spawneamos un hilo que llama `rdev::listen` (bloqueante, event-
//!    driven, OS-level).
//! 2. Acumulamos presses de modificadores (Ctrl/Shift/Alt/Meta) que
//!    ocurran en una ventana corta antes de la primera tecla "real".
//! 3. Cuando llega un press de una tecla no-modificadora, devolvemos
//!    `(Modifiers, Code)` por canal bounded 1 y dejamos que el listener
//!    termine.
//!
//!
//! Limitaciones:
//! - macOS: misma restricción de Accessibility que `global-hotkey`.
//! - Linux/X11: requiere un display accesible; en Wayland puro el
//!   acceso a teclas globales está bloqueado por el protocolo (es un
//!   problema del SO, no de este crate).
//! - Windows: `rdev::listen` usa `RegisterRawInputDevices`; algunas
//!   sesiones RDP lo bloquean. Si falla devolvemos `PlatformError`.

use std::time::{Duration, Instant};

use crossbeam_channel::{bounded, select, tick, Receiver};
use global_hotkey::hotkey::{Code, Modifiers};
use rdev::{Event, EventType, Key};

use crate::traits::PlatformError;

/// Timeout total del grab. Pasado ese tiempo sin pulsación útil,
/// devolvemos error en vez de colgar el bin.
const GRAB_TIMEOUT: Duration = Duration::from_secs(15);

/// Ventana durante la cual un press de modificador se considera "parte"
/// de la combinación que el usuario está formando.
const MODIFIER_WINDOW: Duration = Duration::from_millis(500);

/// Captura la siguiente tecla pulsada por el usuario a nivel global y
/// devuelve `(modificadores, tecla)` listos para pasarse a
/// `oido_platform::hotkey::register` o serializarse con
/// `oido_platform::hotkey::serialize`.
///
/// `Escape` cancela el grab y devuelve `PlatformError::Hotkey(...)`.
pub fn grab_next_key() -> Result<(Modifiers, Code), PlatformError> {
    let (tx, rx) = bounded::<(Modifiers, Code)>(1);

    // El hilo de `rdev::listen` se apropia de `tx`. Si termina
    // (canal cerrado o error) el `select!` macro lo detecta.
    std::thread::Builder::new()
        .name("oido-key-grab".into())
        .spawn(move || run_listener(tx))
        .map_err(|e| PlatformError::Hotkey(format!("grab: spawn thread: {e}")))?;

    // Timeout duro para no colgar el bin si nadie pulsa nada.
    let ticker = tick(GRAB_TIMEOUT);

    select! {
        recv(rx) -> msg => match msg {
            Ok(pair) => Ok(pair),
            Err(_) => Err(PlatformError::Hotkey("grab: listener terminó sin resultado".into())),
        },
        recv(ticker) -> _ => Err(PlatformError::Hotkey(format!(
            "grab: timeout {GRAB_TIMEOUT:?} sin pulsación"
        ))),
    }
}

fn run_listener(tx: crossbeam_channel::Sender<(Modifiers, Code)>) {
    // Movemos el sender a un Option para poder "tomarlo" (y soltar el
    // canal) cuando llega Escape. Esto mantiene el closure como
    // `FnMut` (no consume `tx` por valor en la rama normal).
    let mut tx = Some(tx);
    let mut modifiers = Modifiers::empty();
    let mut last_modifier_at: Option<Instant> = None;

    let callback = move |event: Event| match event.event_type {
        EventType::KeyPress(k) => {
            if let Some(m) = key_to_modifier(k) {
                modifiers |= m;
                last_modifier_at = Some(Instant::now());
                return;
            }
            if matches!(k, Key::Escape) {
                // Escape cancela: soltar el sender cierra el canal y el
                // `recv` del caller retorna con error.
                let _ = tx.take();
                return;
            }
            let code = match key_to_code(k) {
                Some(c) => c,
                None => return, // tecla sin Code mapeada: ignorar
            };
            // Si el último modificador es muy viejo, limpiamos (es de
            // una sesión anterior).
            if let Some(t) = last_modifier_at {
                if t.elapsed() > MODIFIER_WINDOW {
                    modifiers = Modifiers::empty();
                }
            }
            if let Some(sender) = tx.as_ref() {
                let _ = sender.send((modifiers, code));
            }
            // Una vez enviada la tecla principal, soltamos el sender
            // para que el siguiente callback no consuma CPU en un
            // canal ya entregado.
            let _ = tx.take();
        }
        EventType::KeyRelease(_) => {
            // No limpiamos modificadores al release: el usuario suele
            // soltar Shift/Ctrl justo después de la tecla principal,
            // y eso no debe invalidar el binding capturado.
        }
        _ => {}
    };

    // `rdev::listen` retorna `Result` con errores del backend. Si
    // falla (p.ej. permisos en macOS, X11 sin DISPLAY), loggeamos y
    // dejamos que el canal se cierre por timeout.
    if let Err(e) = rdev::listen(callback) {
        // No podemos devolver este error por el canal (ya tenemos un
        // resultado potencial), así que solo lo loggeamos y dejamos
        // que el `recv` del main observe el cierre.
        tracing::warn!(?e, "rdev::listen terminó con error");
    }
}

fn key_to_modifier(k: Key) -> Option<Modifiers> {
    match k {
        Key::ControlLeft | Key::ControlRight => Some(Modifiers::CONTROL),
        Key::ShiftLeft | Key::ShiftRight => Some(Modifiers::SHIFT),
        Key::Alt | Key::AltGr => Some(Modifiers::ALT),
        Key::MetaLeft | Key::MetaRight => Some(Modifiers::SUPER),
        _ => None,
    }
}

/// Mapea un `rdev::Key` a su equivalente en `keyboard_types::Code`
/// (el mismo tipo que `global_hotkey` consume). Cubre teclas comunes;
/// teclas no mapeadas se ignoran silenciosamente.
fn key_to_code(k: Key) -> Option<Code> {
    use Code::*;
    let mapped = match k {
        // Letras
        Key::KeyA => KeyA,
        Key::KeyB => KeyB,
        Key::KeyC => KeyC,
        Key::KeyD => KeyD,
        Key::KeyE => KeyE,
        Key::KeyF => KeyF,
        Key::KeyG => KeyG,
        Key::KeyH => KeyH,
        Key::KeyI => KeyI,
        Key::KeyJ => KeyJ,
        Key::KeyK => KeyK,
        Key::KeyL => KeyL,
        Key::KeyM => KeyM,
        Key::KeyN => KeyN,
        Key::KeyO => KeyO,
        Key::KeyP => KeyP,
        Key::KeyQ => KeyQ,
        Key::KeyR => KeyR,
        Key::KeyS => KeyS,
        Key::KeyT => KeyT,
        Key::KeyU => KeyU,
        Key::KeyV => KeyV,
        Key::KeyW => KeyW,
        Key::KeyX => KeyX,
        Key::KeyY => KeyY,
        Key::KeyZ => KeyZ,
        // Dígitos
        Key::Num0 => Digit0,
        Key::Num1 => Digit1,
        Key::Num2 => Digit2,
        Key::Num3 => Digit3,
        Key::Num4 => Digit4,
        Key::Num5 => Digit5,
        Key::Num6 => Digit6,
        Key::Num7 => Digit7,
        Key::Num8 => Digit8,
        Key::Num9 => Digit9,
        // F-keys
        Key::F1 => F1,
        Key::F2 => F2,
        Key::F3 => F3,
        Key::F4 => F4,
        Key::F5 => F5,
        Key::F6 => F6,
        Key::F7 => F7,
        Key::F8 => F8,
        Key::F9 => F9,
        Key::F10 => F10,
        Key::F11 => F11,
        Key::F12 => F12,
        // Símbolos y modificadores standalone
        Key::Space => Space,
        Key::Tab => Tab,
        Key::Return => Enter,
        Key::KpReturn => NumpadEnter,
        Key::Backspace => Backspace,
        Key::Delete => Delete,
        Key::Insert => Insert,
        Key::Home => Home,
        Key::End => End,
        Key::PageUp => PageUp,
        Key::PageDown => PageDown,
        Key::UpArrow => ArrowUp,
        Key::DownArrow => ArrowDown,
        Key::LeftArrow => ArrowLeft,
        Key::RightArrow => ArrowRight,
        Key::BackQuote => Backquote,
        Key::Minus => Minus,
        Key::Equal => Equal,
        Key::LeftBracket => BracketLeft,
        Key::RightBracket => BracketRight,
        Key::BackSlash => Backslash,
        Key::SemiColon => Semicolon,
        Key::Quote => Quote,
        Key::Comma => Comma,
        Key::Dot => Period,
        Key::Slash => Slash,
        Key::KpMinus => NumpadSubtract,
        Key::KpPlus => NumpadAdd,
        Key::KpMultiply => NumpadMultiply,
        Key::KpDivide => NumpadDivide,
        Key::Kp0 => Numpad0,
        Key::Kp1 => Numpad1,
        Key::Kp2 => Numpad2,
        Key::Kp3 => Numpad3,
        Key::Kp4 => Numpad4,
        Key::Kp5 => Numpad5,
        Key::Kp6 => Numpad6,
        Key::Kp7 => Numpad7,
        Key::Kp8 => Numpad8,
        Key::Kp9 => Numpad9,
        Key::KpDelete => NumpadDecimal, // equivalente razonable
        Key::CapsLock => CapsLock,
        Key::NumLock => NumLock,
        Key::ScrollLock => ScrollLock,
        Key::PrintScreen => PrintScreen,
        Key::Pause => Pause,
        // Estos llegan al grab pero ya se filtran arriba:
        Key::Escape | Key::Alt | Key::AltGr | Key::MetaLeft | Key::MetaRight
        | Key::ShiftLeft | Key::ShiftRight | Key::ControlLeft | Key::ControlRight
        | Key::Function | Key::IntlBackslash | Key::Unknown(_) => return None,
    };
    Some(mapped)
}

/// Suprime el warning de variable no usada cuando se compila sin tests.
#[allow(dead_code)]
fn _ensure_receiver_in_scope(rx: &Receiver<(Modifiers, Code)>) {
    let _ = rx;
}
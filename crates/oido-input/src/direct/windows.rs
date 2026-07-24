//! Backend Windows: UIAutomation vía el crate `uiautomation`.
//!
//! `UIAutomation` y `UIElement` son `!Send + !Sync` (atadura al COM apartment
//! que inicializa `UIAutomation::new()`). Por seguridad, todo el estado UIA
//! vive **exclusivamente** dentro del thread `oido-uia-worker`; el tipo
//! público `UiaDirectInjector` sólo guarda un `crossbeam::Sender` (que es
//! `Send + Sync`) y delega por canal.
//!
//! El worker drena los `Job { text, reply }` que llegan por el canal, llama
//! `get_focused_element` + `send_text` (respeta el caret; `set_value`
//! reemplazaría el contenido del campo y rompería el caso de uso de añadir
//! texto a un Edit ya rellenado), y devuelve el resultado por `reply`.
//!
//! Variables de entorno:
//! - `OIDO_UIA_SEND_INTERVAL_MS` (default 1): ms entre pulsaciones simuladas
//!   por `send_text`. Con 0 puede perder caracteres en textos largos.
//! - `OIDO_UIA_ENABLED` (default "0", **opt-in**): el backend UIA está
//!   **desactivado por defecto**. La inicialización de COM/UIAutomation
//!   interfiere con la captura WASAPI de cpal en este sistema
//!   (causa raíz no aislada todavía — candidatos: audio ducking de
//!   accesibilidad al cargar `UIAutomationCore.dll`, cambio del
//!   COM apartment a nivel proceso, u otro efecto de la carga de
//!   DLLs UIA). Con UIA desactivado se usa el camino stable
//!   `ArboardInjector` (clipboard + Ctrl+V).
//!   - Para activar UIA explícitamente (opt-in, experimentación):
//!     setea `OIDO_UIA_ENABLED=1` antes de arrancar `oido`.
//!   - Sigue el plan de seguimiento para aislar la causa de la
//!     interferencia y reactivar UIA de forma no disruptiva.
//! - `OIDO_UIA_INIT_MODE` (default `full`, sólo aplica si
//!   `OIDO_UIA_ENABLED=1`): bisección diagnóstica de la interferencia.
//!   - `park`: spawnea el worker thread pero **NO** inicializa COM ni
//!     UIA; todos los jobs devuelven `Unsupported` y caen al fallback
//!     clipboard. Aísla "¿interfiere solo tener el thread + canal?".
//!   - `full`: comportamiento normal (`UIAutomation::new()` → COM MTA
//!     + `CoCreateInstance(CUIAutomation)` → carga `UIAutomationCore.dll`).
//!   - Nota: no existe un modo `com`-solo (solo `CoInitializeEx` sin
//!     `CoCreateInstance`) porque requiere `unsafe` directo en este
//!     crate, prohibido por R2. Si `park` sale limpio y `full` sucio,
//!     el culpable es la DLL/UIA in-process → fix: aislamiento en
//!     proceso hijo (sin necesidad de desambiguar COM-init vs DLL).

use std::sync::Arc;
use std::thread;

use crossbeam_channel::{bounded, Receiver, Sender};
use uiautomation::core::UIAutomation;

use crate::direct::DirectInjector;
use crate::InjectError;

const WORKER_QUEUE_CAP: usize = 8;
const ENV_ENABLED: &str = "OIDO_UIA_ENABLED";
const ENV_INTERVAL_MS: &str = "OIDO_UIA_SEND_INTERVAL_MS";
const ENV_INIT_MODE: &str = "OIDO_UIA_INIT_MODE";
const DEFAULT_INTERVAL_MS: u64 = 1;
const WORKER_THREAD_NAME: &str = "oido-uia-worker";

/// Modo de inicialización del worker (ver `OIDO_UIA_INIT_MODE`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum InitMode {
    /// Worker thread vivo, sin COM/UIA. Jobs devuelven `Unsupported`.
    /// Diagnóstico: aísla la interferencia del thread + canal solos.
    Park,
    /// `UIAutomation::new()` completo (comportamiento normal).
    Full,
}

fn env_init_mode() -> InitMode {
    match std::env::var(ENV_INIT_MODE).as_deref() {
        Ok("park") => InitMode::Park,
        _ => InitMode::Full,
    }
}

/// Wrapper `Send + Sync` que sólo guarda el extremo emisor del canal.
/// El estado UIA real vive en el thread `oido-uia-worker`.
pub struct UiaDirectInjector {
    tx: Sender<Job>,
}

impl std::fmt::Debug for UiaDirectInjector {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("UiaDirectInjector").finish_non_exhaustive()
    }
}

struct Job {
    text: String,
    reply: Sender<Result<(), InjectError>>,
}

impl UiaDirectInjector {
    /// Lanza el worker thread y devuelve un handle listo para inyectar.
    /// Si `OIDO_UIA_ENABLED=0` o el worker no se pudo spawnear, devuelve
    /// `Err(InjectError::Unsupported)` para que `SmartInjector` haga
    /// fallback transparente a `ArboardInjector`.
    pub fn new() -> Result<Arc<Self>, InjectError> {
        if env_enabled() == Some(false) {
            return Err(InjectError::Unsupported("UIA deshabilitado por env".into()));
        }

        let (tx, rx) = bounded::<Job>(WORKER_QUEUE_CAP);
        let spawn_result = thread::Builder::new()
            .name(WORKER_THREAD_NAME.into())
            .spawn(move || run_worker(rx));

        match spawn_result {
            Ok(_) => Ok(Arc::new(Self { tx })),
            Err(e) => Err(InjectError::Unsupported(format!("thread spawn: {e}"))),
        }
    }
}

impl DirectInjector for UiaDirectInjector {
    fn inject_focused(&self, text: &str) -> Result<(), InjectError> {
        let (reply_tx, reply_rx) = bounded::<Result<(), InjectError>>(1);
        let job = Job {
            text: text.to_owned(),
            reply: reply_tx,
        };

        // Si el canal está saturado (8 jobs encolados), bloqueamos: preferimos
        // esperar antes que descartar transcripciones del usuario.
        self.tx
            .send(job)
            .map_err(|_| InjectError::Unsupported("worker UIA muerto".into()))?;

        reply_rx
            .recv()
            .map_err(|_| InjectError::Unsupported("worker UIA murió antes de responder".into()))?
    }
}

// ---------------- worker internals ----------------

fn run_worker(rx: Receiver<Job>) {
    let mode = env_init_mode();
    tracing::info!(?mode, "arrancando worker UIA");

    if mode == InitMode::Park {
        // Diagnóstico: NO inicializamos COM ni UIA. El thread vive y
        // drena el canal devolviendo `Unsupported` a cada job, que
        // `SmartInjector` degradea a clipboard. Así medimos si la mera
        // presencia del thread + canal interfiere con WASAPI.
        tracing::warn!(
            "INIT_MODE=park: COM/UIA NO inicializado; todos los jobs \
             caen a clipboard (modo diagnóstico)"
        );
        drain_with_unavailable(&rx);
        return;
    }

    let automation = match UIAutomation::new() {
        Ok(a) => a,
        Err(e) => {
            tracing::warn!(
                ?e,
                "UIAutomation::new falló; drenando canal con Unsupported"
            );
            drain_with_unavailable(&rx);
            return;
        }
    };

    loop {
        match rx.recv() {
            Ok(job) => {
                let result = inject_via_uia(&automation, &job.text);
                // Si el receptor ya cayó (caller soltó la tx), no hacer nada:
                // el resultado se descarta y el siguiente job quizás también
                // falle con `Unsupported`.
                let _ = job.reply.send(result);
            }
            Err(_) => {
                tracing::info!("canal cerrado; saliendo del worker UIA");
                return;
            }
        }
    }
}

fn drain_with_unavailable(rx: &Receiver<Job>) {
    while let Ok(job) = rx.recv() {
        let _ = job
            .reply
            .send(Err(InjectError::Unsupported("UIA no inicializado".into())));
    }
}

fn inject_via_uia(automation: &UIAutomation, text: &str) -> Result<(), InjectError> {
    let element = automation
        .get_focused_element()
        .map_err(|e| InjectError::Unsupported(format!("get_focused_element: {e}")))?;

    // Filtro de "editable": pregunta explícita al SO. Si el método
    // falla (timeout COM, fallo del backend), devolvemos `Unsupported`
    // para que el caller NO degrade a NotEditable (que sería incorrecto
    // — no sabemos nada sobre el elemento si el check falla).
    match element.is_keyboard_focusable() {
        Ok(true) => { /* proceed */ }
        Ok(false) => {
            tracing::debug!(
                "get_focused_element devolvió algo no keyboard-focusable; \
                 fallback a clipboard",
            );
            return Err(InjectError::NotEditable);
        }
        Err(e) => {
            // Falso positivo o timeout. No podemos afirmar que el focused
            // no es editable: caemos al clipboard (mejor que perder el
            // texto) pero advertimos para diagnóstico.
            tracing::warn!(?e, "is_keyboard_focusable falló; fallback a clipboard",);
            return Err(InjectError::Unsupported(format!(
                "is_keyboard_focusable: {e}"
            )));
        }
    }

    // `send_text` ya hace `set_focus` internamente (verificado en el
    // crate uiautomation core.rs:send_text). Llamarlo pre-cría un
    // curso de race con el hotkey grab. Omitimos el pre-set_focus y
    // delegamos: si send_text falla, devolvemos Unsupported y el
    // fallback a clipboard absorbe la transición de foco.

    let interval = env_interval_ms();
    element
        .send_text(text, interval)
        .map_err(|e| InjectError::Unsupported(format!("send_text: {e}")))
}

// ---------------- env helpers ----------------

fn env_enabled() -> Option<bool> {
    match std::env::var(ENV_ENABLED) {
        Ok(v) if v == "0" || v.eq_ignore_ascii_case("false") => Some(false),
        Ok(v) if v == "1" || v.eq_ignore_ascii_case("true") => Some(true),
        Ok(_) => None,         // valor no reconocido: dejar default
        Err(_) => Some(false), // no seteada: DESACTIVADO (default opt-in)
    }
}

fn env_interval_ms() -> u64 {
    std::env::var(ENV_INTERVAL_MS)
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(DEFAULT_INTERVAL_MS)
}

// Asegura que el campo `tx: Sender<Job>` cumple `Send` (requerido por el
// trait bound `Send + Sync` de DirectInjector). Si Job deja de ser Send,
// este fn no compila, Atrapa regresiones en tiempo de compilación.
#[allow(dead_code)]
fn _assert_send() {
    fn assert_send<T: Send>() {}
    assert_send::<Sender<Job>>();
    assert_send::<&UiaDirectInjector>();
}

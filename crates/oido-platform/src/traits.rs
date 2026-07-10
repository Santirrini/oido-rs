//! Traits que cada OS implementa. Sin lógica, solo contratos.
//!
//! Cada método debe ser `Send + 'static` para que el pipeline los
//! pueda mover entre threads sin rodeos.

use std::fmt::Debug;

use crossbeam_channel::{Receiver, Sender};

use thiserror::Error;

use oido_config::Theme;

use crate::AudioFrame;

#[derive(Debug, Error)]
pub enum PlatformError {
    #[error("captura de audio falló: {0}")]
    Capture(String),
    #[error("registro de hotkey falló: {0}")]
    Hotkey(String),
    #[error("tray falló: {0}")]
    Tray(String),
    #[error("clipboard / paste falló: {0}")]
    Inject(String),
}

/// Productor de audio PCM mono 16 kHz f32 en bloques.
///
/// Lifecycle: `open(sink)` — `start()` — `stop()`. `start`/`stop`
/// pueden llamarse múltiples veces.
pub trait CaptureSource: Send + Debug + 'static {
    /// Registra el sumidero (canal de audio) donde se publicarán las
    /// muestras. Debe invocarse antes de `start()`.
    fn open(&mut self, sink: Sender<AudioFrame>) -> Result<(), PlatformError>;
    fn start(&mut self) -> Result<(), PlatformError>;
    fn stop(&mut self) -> Result<(), PlatformError>;
    /// Indica si el dispositivo está abierto y sample-rate aceptable.
    fn sample_rate_hz(&self) -> u32;
}

/// Hotkey global con callback on_press/on_release.
///
/// `Box<dyn Fn() + Send>` (no generic) para ser dyn-compatible.
///
/// Ponytail: el requisito de `Send` en el trait crea fricción porque
/// `global_hotkey::GlobalHotKeyManager` contiene un `*mut c_void`
/// internamente (!Send en Windows). Como el Hotkey solo se usa
/// dentro del thread principal del bin, no necesitamos `Send` en el
/// trait. `Register/inject` ocurre antes de spawnar workers de audio.
///
/// El binding (canónico, p.ej. `"F8"` o `"Ctrl+Shift+D"`) se pasa en
/// cada `register` para que el bin pueda propagar el valor de
/// `Config.hotkey` sin acoplar el trait a `Config`.
pub trait Hotkey: Debug + 'static {
    /// Registra la combinación apuntada por `binding` y conecta los
    /// callbacks boxed. El binding debe estar en formato canónico
    /// aceptado por `oido_platform::hotkey::parse`.
    fn register(
        &mut self,
        binding: &str,
        on_press: Box<dyn Fn() + Send + 'static>,
        on_release: Box<dyn Fn() + Send + 'static>,
    ) -> Result<(), PlatformError>;
    fn unregister(&mut self) -> Result<(), PlatformError>;
}

/// Icono de bandeja con estado (idle / listening / procesando / pausado / error).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrayState {
    Idle,
    Listening,
    Processing,
    Paused,
    /// Modelo whisper cargando de forma lazy. El hotkey queda
    /// registrado pero el primer dictado se difiere hasta que termine
    /// la carga + warm-up.
    Loading,
    Error,
}

/// Acciones que puede disparar el menú nativo de bandeja.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MenuAction {
    ChangeHotkey,
    SetTheme(Theme),
    SetSttMode(oido_config::SttMode),
    OpenModelsDir,
    CheckUpdates,
    TogglePause,
    Exit,
}

pub trait Tray: 'static {
    fn show(&mut self) -> Result<(), PlatformError>;
    /// Actualiza el icono y el tooltip según el estado y el tema activo.
    fn set_state(&mut self, state: TrayState, theme: Theme) -> Result<(), PlatformError>;
    fn hide(&mut self) -> Result<(), PlatformError>;
    /// Devuelve el receptor de acciones de menú (solo la primera llamada devuelve `Some`).
    fn take_menu_events(&mut self) -> Option<Receiver<MenuAction>>;
}

/// Inyecta texto vía clipboard + paste simulado (Ctrl/Cmd+V).
///
/// `&self` (interior mut) para poder compartir la misma instancia entre
/// el thread de release y otros workers si en el futuro hace falta.
/// Razón: arboard+enigo requieren `&mut` internamente, así que la impl
/// guarda estado en `Arc<parking_lot::Mutex<Inner>>`.
pub trait Injector: Send + Sync + Debug + 'static {
    fn inject(&self, text: &str) -> Result<(), PlatformError>;

    /// Escribe texto simulando pulsaciones de teclas individuales (enigo text),
    /// ideal para streaming incremental para no pisar el clipboard.
    ///
    /// Implementación por defecto: delega a `inject`.
    fn type_text(&self, text: &str) -> Result<(), PlatformError> {
        self.inject(text)
    }
}

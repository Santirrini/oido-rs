//! Traits y tipos específicos del tray que viven en `oido-tray`.
//!
//! Antes del refactor estos tipos estaban en `oido-platform/src/traits.rs`
//! junto con `CaptureSource`, `Hotkey`, `Injector`, etc. Tras la
//! división modular cada crate expone sus propios traits y un enum
//! error por dominio (`TrayError` aquí, `AudioError` en `oido-audio`,
//! `HotkeyError` en `oido-hotkey`, `InjectError` en `oido-input`).

use std::fmt::Debug;

use crossbeam_channel::Receiver;

use thiserror::Error;

use oido_config::{PromptPreset, SttMode, Theme, UiLanguage};

use crate::tray::sections::MenuSection;

/// Errores del crate tray (dominio: UI / bandeja / popup).
#[derive(Debug, Error)]
pub enum TrayError {
    #[error("tray falló: {0}")]
    Tray(String),
}

/// Estado del icono de bandeja con semántica visual.
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
    SetSttMode(SttMode),
    /// Click sobre uno de los items del submenú "Idioma de la
    /// interfaz". Provoca un rebuild_menu con los strings del nuevo
    /// idioma.
    SetUiLanguage(UiLanguage),
    /// Click sobre uno de los items del submenú "Prompt del sistema".
    /// El bin decide qué texto concreto se inyecta a whisper.cpp
    /// (preset vs. custom).
    SetPromptPreset(PromptPreset),
    OpenModelsDir,
    CheckUpdates,
    TogglePause,
    Exit,
    /// Click sobre un item del submenú "Modelos". El `String` es el
    /// filename del modelo (p.ej. `"ggml-base.bin"`). El dispatch en
    /// el bin decide si el modelo está instalado (activar) o no
    /// (descargar).
    ModelItem(String),
}

/// Trait `Tray` (antes en `oido-platform::traits`).
pub trait Tray: 'static {
    fn show(&mut self) -> Result<(), TrayError>;
    /// Actualiza el icono y el tooltip según el estado y el tema activo.
    fn set_state(&mut self, state: TrayState, theme: Theme) -> Result<(), TrayError>;
    fn hide(&mut self) -> Result<(), TrayError>;
    /// Devuelve el receptor de acciones de menú (solo la primera llamada
    /// devuelve `Some`).
    fn take_menu_events(&mut self) -> Option<Receiver<MenuAction>>;
    /// Reconstruye el árbol de menú con las secciones dadas y lo
    /// re-adjunta al icono. Necesario para refrescar el submenú
    /// "Modelos" después de una descarga o cambio de modelo activo.
    fn rebuild_menu(&mut self, sections: Vec<Box<dyn MenuSection>>) -> Result<(), TrayError>;
}

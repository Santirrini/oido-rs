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

use oido_config::{EffortPreset, PromptPreset, SttMode, Theme, UiLanguage};

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
    /// Click sobre uno de los items del submenú "Esfuerzo". Mapea un
    /// preset de calidad de decodificación a los parámetros concretos
    /// de `whisper_full_params`. Se propaga en caliente vía
    /// `SharedTranscriber::set_effort` (no requiere recargar modelo).
    SetEffort(EffortPreset),
    /// Click sobre "Editar texto personalizado…" dentro del submenú de
    /// prompt. El bin abre un diálogo nativo con un campo de texto
    /// (Win32 InputBox en Windows; stub en otros OS), toma el valor
    /// y lo persiste vía `ControlMessage::SetPromptText`.
    EditPrompt,
    OpenModelsDir,
    CheckUpdates,
    TogglePause,
    Exit,
    /// Click sobre un item del submenú "Modelos". El `String` es el
    /// filename del modelo (p.ej. `"ggml-base.bin"`). El dispatch en
    /// el bin decide si el modelo está instalado (activar) o no
    /// (descargar).
    ModelItem(String),
    /// Click sobre el item de aviso "⚠ Cambiar a modelo multilingüe…"
    /// visible solo cuando hay mismatch entre modelo `.en` y un idioma
    /// distinto a inglés. El `String` es el filename del modelo
    /// multilingüe sugerido (p.ej. `"ggml-small.bin"`). El bin delega
    /// en el mismo handler que un click sobre el submenú Modelos: si
    /// está instalado se activa, si no se descarga y luego se activa.
    FixModelLanguage(String),
}

/// Trait `Tray` (antes en `oido-platform::traits`).
pub trait Tray: 'static {
    fn show(&mut self) -> Result<(), TrayError>;
    /// Actualiza el icono y el tooltip según el estado y el tema activo.
    fn set_state(&mut self, state: TrayState, theme: Theme) -> Result<(), TrayError>;
    /// Sobrescribe el tooltip del icono de bandeja. Usado por el aviso
    /// de mismatch modelo/idioma para mostrar un mensaje persistente
    /// que coexiste con el icono de estado (el icono refleja el estado
    /// operativo vía `set_state`; el tooltip es ortogonal).
    ///
    /// Solo implementado en Win/macOS (los backends con `tray-icon`).
    /// En Linux el stub es un no-op.
    fn set_tooltip(&mut self, _tooltip: &str) -> Result<(), TrayError> {
        Ok(())
    }
    fn hide(&mut self) -> Result<(), TrayError>;
    /// Devuelve el receptor de acciones de menú (solo la primera llamada
    /// devuelve `Some`).
    fn take_menu_events(&mut self) -> Option<Receiver<MenuAction>>;
    /// Reconstruye el árbol de menú con las secciones dadas y lo
    /// re-adjunta al icono. Necesario para refrescar el submenú
    /// "Modelos" después de una descarga o cambio de modelo activo.
    fn rebuild_menu(&mut self, sections: Vec<Box<dyn MenuSection>>) -> Result<(), TrayError>;
}

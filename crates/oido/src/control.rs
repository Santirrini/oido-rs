//! Mensajes de control internos para el ciclo de vida del hilo principal.
//!
//! Se envían por un `crossbeam_channel` desde el listener de menú
//! (`oido-menu-listener`) o desde handlers de flags one-shot al loop
//! principal de `main`.

use oido_config::Theme;
use oido_tray::TrayState;

/// Mensajes de control. El loop principal en `main` hace `match` y
/// dispatcha a las funciones del runtime.
#[allow(dead_code)] // ActivateModel se construye desde el thread `oido-downloader`.
#[derive(Debug)]
pub(crate) enum ControlMessage {
    ChangeHotkey,
    HotkeyChanged(Result<String, String>),
    SetTrayState(TrayState),
    SetTheme(Theme),
    SetSttMode(oido_config::SttMode),
    /// Click sobre el submenú "Idioma de la interfaz". Provoca un
    /// `rebuild_menu` con los nuevos strings.
    SetUiLanguage(oido_config::UiLanguage),
    /// Click sobre el submenú "Prompt del sistema". El bin decide qué
    /// texto concreto se inyecta a whisper.cpp (preset vs. custom).
    SetPromptPreset(oido_config::PromptPreset),
    /// Click sobre el submenú "Esfuerzo". Mapea a `FullParams` de
    /// whisper.cpp y se propaga en caliente a transcriber/streamer
    /// (no requiere recargar modelo ni reiniciar pipeline).
    SetEffort(oido_config::EffortPreset),
    Exit,
    /// Reconstruye el submenú "Modelos" con el estado actual del disco.
    /// Se envía tras una descarga o tras activar un modelo distinto,
    /// para que las marcas ✓/↓ y ← activo reflejen la realidad.
    RefreshMenu,
    /// Activa un modelo descargado (filename) en el transcriber activo.
    /// Idempotente; el bin ya reemplaza el modelo en el SharedTranscriber.
    ActivateModel(String),
    /// Click sobre un dispositivo del submenú "Micrófono" o un valor
    /// del flag `--set-mic`. `None` = modo automático (default del OS);
    /// `Some(name)` = fijado al dispositivo con ese nombre exacto.
    /// El handler reconstruye el `CaptureSource` (shut down + start
    /// pipeline, sin recargar modelo).
    SetInputDevice(Option<String>),
    /// Click sobre el item "Re-probar micrófonos" del submenú. El
    /// handler lanza el sondeo de calidad en un thread dedicado
    /// (`oido-mic-probe`) y, si encuentra un dispositivo con mejor
    /// señal, envía un `SetInputDevice` por el canal de control.
    ProbeMicrophones,
}

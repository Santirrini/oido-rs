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
    /// Evento del updater (`oido-updater::UpdateEvent`): el scheduler
    /// en background notifica que hay update disponible, falló, etc.
    /// El handler del control loop traduce el evento a tooltip del
    /// tray y/o `TrayState`. El `UpdateEvent` sólo se construye cuando
    /// el bin se compila con `--features updater`.
    #[cfg(feature = "updater")]
    UpdateEvent(oido_updater::UpdateEvent),
    /// Cambio explícito de tooltip persistente (sin tocar el icono).
    /// Usado por el handler de `UpdateEvent` para mostrar
    /// "Nueva versión vX — reinicia para aplicar". Vacío = limpiar.
    ///
    /// El tray tiene un tooltip "transitorio" coexistiendo con el
    /// icono de estado (set via `set_state`). El tooltip persistente
    /// es ortogonal y se setea con `set_tooltip`. Vacío = limpiar.
    SetUpdateTooltip(String),

    // ============================================================
    // TTS (lectura de selección de cursor)
    // ============================================================
    /// Toggle del sistema TTS (`Config::tts.enabled`).
    ToggleTts,
    /// Cambio de motor TTS (`Config::tts.engine`).
    SetTtsEngine(oido_config::TtsEngineKind),
    /// Cambio de voz TTS (`Config::tts.voice`).
    SetTtsVoice(String),
    /// Sincroniza el TTS runtime con la config actual tras un cambio de
    /// voz desde el submenú "Voces TTS". Decide entre `set_voice` barato
    /// (Kokoro: todas las voces comparten el mismo `.onnx` + `voices.bin`)
    /// o `rebuild_tts_runtime` (engine switch, o voz Piper que requiere
    /// recargar un `.onnx` distinto).
    ///
    /// A diferencia de `SetTtsVoice`, este mensaje NO persiste config —
    /// el caller (`handle_tts_model_click`) ya la mutó antes de enviarlo.
    /// Sólo propaga el cambio al engine vivo.
    SyncTtsRuntime,
    /// Cambio de velocidad TTS (`Config::tts.speed_milli`).
    SetTtsSpeed(u16),
    /// El usuario disparó "leer selección ahora" (vía menú o hotkey).
    /// El bin llama a `SelectionReader::read()` y envía el texto al
    /// `TtsPipeline`. Sin payload porque el texto se obtiene en el
    /// handler (no en el callback de hotkey, que debe ser rápido).
    TtsReadSelection,
}

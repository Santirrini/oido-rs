//! Bin CLI `oido`. Arranca logger + carga config + levanta pipeline +
//! espera Ctrl+C. La UI Tauri llega en Fase 3.

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result};
use oido_config::ConfigStore;
use oido_core::{Pipeline, PipelineConfig, PipelineEvent, PipelineState};
use oido_platform::{
    capture::CpalCapture, hotkey::GhHotkey, injector::ArboardInjector, PlatformTray, Tray,
    TrayState,
};
use oido_stt::{Transcriber, WhisperCpp};
use parking_lot::Mutex;
use tracing_subscriber::EnvFilter;

/// Resuelve el directorio donde vive el modelo whisper.
///
/// Prioridad:
/// 1. `OIDO_MODELS_DIR` env var (escape hatch para tests / packaging).
/// 2. `dirs::data_dir()/oido/models` (estándar cross-platform; lo crea
///    si no existe).
/// 3. Relativo `models/` (último recurso; sólo útil corriendo desde la
///    raíz del repo).
fn resolve_models_dir() -> PathBuf {
    if let Ok(dir) = std::env::var("OIDO_MODELS_DIR") {
        return PathBuf::from(dir);
    }
    let data_dir = dirs::data_dir()
        .unwrap_or_else(std::env::temp_dir)
        .join("oido")
        .join("models");
    if let Err(e) = std::fs::create_dir_all(&data_dir) {
        tracing::warn!(
            ?e,
            ?data_dir,
            "no pude crear dir de modelos bajo data_dir; fallback a relativo `models/`"
        );
        return PathBuf::from("models");
    }
    data_dir
}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| EnvFilter::new("info,whisper_rs=warn,whisper_rs_sys=warn")),
        )
        .with_target(true)
        .init();

    tracing::info!("oido 2.0 starting (Fase 1: dicta-y-pega)");

    // 1) Config (defaults aplican si no hay config.json).
    let cfg = ConfigStore::load_or_default().context("loading config")?;
    let snap = cfg.snapshot();
    tracing::info!(
        hotkey = %snap.hotkey,
        model = %snap.model,
        language = %snap.language_ui,
        "config cargada"
    );

    // 2) Componer implementaciones por OS.
    let capture = Box::new(CpalCapture::new().context("init captura audio")?);
    let hotkey = Box::new(GhHotkey::new());
    let injector = ArboardInjector::new().context("init injector clipboard")?;

    // 3) Tray (MVP: stub en los 3 OS, sólo loggea el estado). Si falla
    //    a futuro seguimos sin tray — el log sigue mostrando el estado.
    let tray = PlatformTray::new()
        .map(|t| Arc::new(Mutex::new(Some(t))))
        .unwrap_or_else(|e| {
            tracing::warn!(?e, "tray no disponible; el estado se loggea solamente");
            Arc::new(Mutex::new(None))
        });

    // 4) Cargar modelo whisper.
    let models_dir = resolve_models_dir();
    let model_path = models_dir.join(&snap.model);
    tracing::info!(?model_path, "cargando modelo whisper");
    let mut stt = WhisperCpp::with_language(&snap.language_ui);
    if let Err(e) = stt.load_model(&model_path) {
        tracing::warn!(
            ?e,
            path = ?model_path,
            "modelo whisper no encontrado; STT devolverá error hasta que descargues uno. \
             Tip: scripts/download_model.ps1 (Win) o scripts/download_model.sh (Unix)"
        );
    }
    let transcriber: Arc<dyn Transcriber> = Arc::new(stt);

    // 5) Construir y arrancar pipeline.
    let pipeline_cfg = PipelineConfig {
        capture,
        transcriber,
        injector,
        hotkey,
    };
    let mut pipeline = Pipeline::new(pipeline_cfg);

    // 6) Ctrl+C → apagado limpio. Handler sólo setea un flag; el loop
    //    principal lo observa y llama a `shutdown` antes de salir.
    let shutdown_requested = Arc::new(AtomicBool::new(false));
    ctrlc::set_handler({
        let flag = Arc::clone(&shutdown_requested);
        move || {
            tracing::info!("Ctrl+C recibido, terminando");
            flag.store(true, Ordering::SeqCst);
        }
    })
    .context("instalando handler Ctrl+C")?;

    // 7) Observer: traduce eventos de estado a log + tray.
    let events = pipeline.events();
    let tray_for_observer = Arc::clone(&tray);
    std::thread::Builder::new()
        .name("oido-event-observer".into())
        .spawn(move || {
            while let Ok(evt) = events.recv() {
                let PipelineEvent::State(state) = evt;
                tracing::info!(?state, "pipeline state");
                let tray_state = match state {
                    PipelineState::Idle => TrayState::Idle,
                    PipelineState::Recording => TrayState::Listening,
                    PipelineState::Processing => TrayState::Processing,
                };
                let mut guard = tray_for_observer.lock();
                if let Some(t) = guard.as_mut() {
                    let _ = t.set_state(tray_state);
                }
            }
        })?;

    pipeline.start().context("arrancando pipeline")?;

    tracing::info!("hold F8, dicta, suelta. Ctrl+C para salir.");

    // Loop principal: poll del flag de shutdown. 200 ms es suficientemente
    // fino para sentirse "instantáneo" al cancelar pero suficientemente
    // gordo para no quemar CPU.
    while !shutdown_requested.load(Ordering::SeqCst) {
        std::thread::sleep(Duration::from_millis(200));
    }

    let _ = pipeline.shutdown();
    tracing::info!("oido 2.0 cerrado");
    Ok(())
}

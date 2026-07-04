//! Bin CLI `oido`. Por ahora arranca logger + carga config + levanta
//! pipeline + espera Ctrl+C. La UI Tauri llega en Fase 3.

use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result};
use oido_config::ConfigStore;
use oido_core::{Pipeline, PipelineConfig, PipelineEvent, PipelineState};
use oido_platform::{capture::CpalCapture, hotkey::GhHotkey, injector::ArboardInjector};
use oido_stt::{Transcriber, WhisperCpp};
use tracing_subscriber::EnvFilter;

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

    // 3) Cargar modelo whisper. La ruta por defecto es `models/<config.model>`.
    let model_path = std::path::PathBuf::from("models").join(&snap.model);
    let mut stt = WhisperCpp::with_language(&snap.language_ui);
    if let Err(e) = stt.load_model(&model_path) {
        tracing::warn!(
            ?e,
            path = ?model_path,
            "modelo whisper no encontrado; STT devolverá error hasta que descargues uno"
        );
    }
    let transcriber: Arc<dyn oido_stt::Transcriber> = Arc::new(stt);

    // 4) Construir y arrancar pipeline.
    let pipeline_cfg = PipelineConfig {
        capture,
        transcriber,
        injector,
        hotkey,
    };
    let mut pipeline = Pipeline::new(pipeline_cfg);

    // 5) Observador: traduce eventos de estado a log (tray real -> Fase 3).
    let events = pipeline.events();
    std::thread::Builder::new()
        .name("oido-event-observer".into())
        .spawn(move || {
            while let Ok(evt) = events.recv() {
                if let PipelineEvent::State(state) = evt {
                    tracing::info!(?state, "pipeline state");
                }
            }
        })?;

    pipeline.start().context("arrancando pipeline")?;

    tracing::info!("hold F8, dicta, suelta. Ctrl+C para salir.");
    // Loop simple: heartbeat periódica para que el proceso viva y
    // mantenga log-alive para diagnóstico.
    loop {
        std::thread::sleep(Duration::from_secs(60));
    }
}

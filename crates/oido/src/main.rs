//! Bin CLI `oido`. Por ahora solo inicia el logger y verifica que el
//! workspace carga su config. El hotkey + audio + tray pipeline llega
//! en Fase 1.

use anyhow::{Context, Result};
use oido_config::ConfigStore;

fn main() -> Result<()> {
    // Logger estructurado vía tracing. Nivel vía env (debug, info, warn).
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .with_target(true)
        .init();

    tracing::info!("oido 2.0 starting (Fase 0 scaffold)");

    let cfg = ConfigStore::load_or_default().context("loading config")?;
    let snap = cfg.snapshot();
    tracing::info!(
        hotkey = %snap.hotkey,
        model = %snap.model,
        language = %snap.language_ui,
        "config cargada"
    );

    // Fase 1: levantar pipeline, registrar hotkey, mostrar tray.
    let _pipeline = oido_core::Pipeline::placeholder();
    tracing::info!("pipeline placeholder listo. Cerrando hasta Fase 1.");

    Ok(())
}

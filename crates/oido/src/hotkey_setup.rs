//! Sub-comando `--set-hotkey`: graba la tecla de activación interactivamente.

use std::sync::Arc;

use anyhow::{Context, Result};
use oido_config::ConfigStore;
use oido_hotkey::{grab_next_key, parse as parse_hotkey, serialize as serialize_hotkey};

pub(crate) fn run_set_hotkey(cfg: Arc<ConfigStore>) -> Result<String> {
    tracing::info!(
        "pulsa la tecla que quieras usar como activador (Esc para cancelar, Ctrl+C para abortar)…"
    );

    let (mods, code) = grab_next_key().context("capturando tecla")?;
    let new_binding = serialize_hotkey(mods, code);

    // Validar binding generado
    parse_hotkey(&new_binding)
        .with_context(|| format!("binding generado inválido: {new_binding:?}"))?;

    let mut new_cfg = cfg.snapshot();
    let previous = std::mem::replace(&mut new_cfg.hotkey, new_binding.clone());
    cfg.replace(new_cfg);
    cfg.save().context("guardando config")?;

    tracing::info!(
        previous = %previous,
        new = %new_binding,
        path = ?oido_config::config_file(),
        "hotkey actualizado"
    );
    Ok(new_binding)
}

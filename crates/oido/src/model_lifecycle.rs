//! Maneja el ciclo de vida del modelo activo: clicks en el menú,
//! descarga lazy y activación post-descarga.

use std::path::Path;
use std::sync::Arc;
use std::thread;

#[allow(unused_imports)]
use crossbeam_channel::Sender;
#[allow(unused_imports)]
use oido_config::ConfigStore;
#[allow(unused_imports)]
use oido_stt::SharedTranscriber;
use oido_stt::Transcriber;
use oido_tray::TrayState;

use crate::{resolve_models_dir, ControlMessage};
/// Maneja el click sobre un item del submenú "Modelos".
///
/// - Si el modelo ya está instalado: lo activa (snapshot → replace → save →
///   load_model + warm_up sobre el SharedTranscriber) y refresca el menú
///   para que aparezca `← activo`.
/// - Si no está instalado: lanza un thread dedicado `oido-downloader` que
///   descarga, verifica, y al terminar envía `RefreshMenu` +
///   `ActivateModel(filename)` + `SetTrayState(Idle)`.
///
/// Esta función es la única autorizada a tocar el transcriber desde el
/// lado del menú (regla R1: comunicación por canales).
pub(crate) fn handle_model_click(
    filename: &str,
    control_tx: &crossbeam_channel::Sender<ControlMessage>,
    cfg: &Arc<oido_config::ConfigStore>,
    shared: Option<&Arc<oido_stt::SharedTranscriber>>,
) {
    let models_dir = resolve_models_dir();
    let mp = models_dir.join(filename);

    if mp.is_file() {
        // Activar directamente.
        let mut snap = cfg.snapshot();
        if snap.model != filename {
            snap.model = filename.to_string();
            cfg.replace(snap.clone());
            if let Err(e) = cfg.save() {
                tracing::error!(?e, "no se pudo guardar config tras activar modelo");
                return;
            }
            tracing::info!(model = %filename, "modelo activo actualizado");
        }
        // Recargar el modelo en el SharedTranscriber (si existe).
        if let Some(shared) = shared {
            let handle = shared.handle();
            let load_res = handle.lock().load_model(&mp);
            match load_res {
                Ok(()) => {
                    let _ = handle.lock().warm_up();
                }
                Err(e) => {
                    tracing::error!(?e, "load_model falló al activar {filename}");
                    let _ = control_tx.send(ControlMessage::SetTrayState(TrayState::Error));
                    return;
                }
            }
        }
        // Refrescar el menú para mover la marca ← activo.
        let _ = control_tx.send(ControlMessage::RefreshMenu);
    } else {
        // Buscar el entry en el catálogo para obtener URL/tamaño.
        let entry = oido_models::find(filename).cloned();
        let Some(entry) = entry else {
            tracing::warn!(filename, "click sobre modelo no presente en catálogo");
            return;
        };
        // Marcar "descargando" en la bandeja.
        let _ = control_tx.send(ControlMessage::SetTrayState(TrayState::Loading));
        let tx = control_tx.clone();
        let dir = models_dir.clone();
        let shared_for_dl = shared.map(Arc::clone);
        let cfg_for_dl = Arc::clone(cfg);
        let span = tracing::info_span!("download_model_user", filename = %filename);
        let _ = thread::Builder::new()
            .name("oido-downloader".into())
            .spawn(move || {
                let _enter = span.enter();
                match oido_models::download_model(&dir, &entry, None) {
                    Ok(()) => {
                        tracing::info!(filename = %entry.filename, "descarga completa");
                        // Refrescar menú para reflejar ✓ en el item.
                        let _ = tx.send(ControlMessage::RefreshMenu);
                        // Activar el modelo recién descargado.
                        activate_after_download(
                            &entry.filename,
                            &dir,
                            &cfg_for_dl,
                            shared_for_dl.as_ref(),
                            &tx,
                        );
                    }
                    Err(e) => {
                        tracing::error!(
                            ?e,
                            filename = %entry.filename,
                            "descarga falló"
                        );
                        let _ = tx.send(ControlMessage::SetTrayState(TrayState::Error));
                    }
                }
            });
    }
}

/// Helper: activa un modelo recién descargado (espejo de la rama "ya
/// instalado" de `handle_model_click`, pero separado para mantener el
/// thread del downloader minimalista).
pub(crate) fn activate_after_download(
    filename: &str,
    models_dir: &Path,
    cfg: &Arc<oido_config::ConfigStore>,
    shared: Option<&Arc<oido_stt::SharedTranscriber>>,
    control_tx: &crossbeam_channel::Sender<ControlMessage>,
) {
    let mut snap = cfg.snapshot();
    snap.model = filename.to_string();
    cfg.replace(snap.clone());
    if let Err(e) = cfg.save() {
        tracing::error!(?e, "no se pudo guardar config tras descarga");
    }
    let mp = models_dir.join(filename);
    if let Some(shared) = shared {
        let handle = shared.handle();
        if let Err(e) = handle.lock().load_model(&mp) {
            tracing::error!(?e, "load_model falló tras descarga");
            let _ = control_tx.send(ControlMessage::SetTrayState(TrayState::Error));
            return;
        }
        let _ = handle.lock().warm_up();
    }
    let _ = control_tx.send(ControlMessage::SetTrayState(TrayState::Idle));
    tracing::info!(filename, "modelo activado tras descarga");
}

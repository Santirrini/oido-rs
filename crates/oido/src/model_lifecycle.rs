//! Maneja el ciclo de vida del modelo activo: clicks en el menú,
//! descarga lazy y activación post-descarga.

use std::path::Path;
use std::sync::Arc;
use std::thread;

#[allow(unused_imports)]
use crossbeam_channel::Sender;
#[allow(unused_imports)]
use oido_config::ConfigStore;
use oido_stt::is_vad_model_filename;
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

    // Guardia: el submenú "Modelos" del tray mezcla modelos de
    // transcripción con el VAD (mismo `models_dir`, mismo catálogo).
    // Si el usuario hace clic en el VAD lo activamos como modelo de
    // transcripción, GGML_ASSERT(wtype != GGML_TYPE_COUNT) revienta
    // el proceso. Rechazamos el click directamente y reseteamos la
    // `Config.model` al modelo anterior si ya apuntaba al VAD.
    if is_vad_model_filename(filename) {
        tracing::error!(
            filename,
            "click sobre el modelo VAD; ignorado para evitar crash de GGML. \
             El VAD se activa vía `WhisperCpp::with_vad(...)`, no como modelo principal."
        );
        // Si por error la config apuntaba ya al VAD, la limpiamos al
        // fallback (ggml-base.bin si está, si no el primero que
        // encontremos en disco).
        let mut snap = cfg.snapshot();
        if is_vad_model_filename(&snap.model) || !models_dir.join(&snap.model).is_file() {
            if let Some(fallback) = whisper_fallback_filename(&models_dir) {
                tracing::warn!(
                    from = %snap.model,
                    to = %fallback,
                    "Config.model apuntaba a un modelo inválido; corrigiendo"
                );
                snap.model = fallback;
                cfg.replace(snap.clone());
                let _ = cfg.save();
                let _ = control_tx.send(ControlMessage::RefreshMenu);
            }
        }
        return;
    }

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
    // Guardia idéntica a la de `handle_model_click`: rechazar la
    // activación si el filename del catálogo es VAD.
    if is_vad_model_filename(filename) {
        tracing::error!(
            filename,
            "activate_after_download rechazó VAD; no se persiste como modelo principal"
        );
        return;
    }
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

/// Resuelve un filename de modelo seguro (no-VAD, no-vacío, que esté en
/// el catálogo de `oido_models`) para usar como fallback. Orden de
/// preferencia:
///   1. `ggml-base.bin` si está instalado (default histórico).
///   2. Otro `ggml-*.bin` (no VAD) que esté instalado, priorizando
///      `Small > Base > Tiny > Medium > Large` por tamaño (los más
///      chicos primero → carga rápida en fallback).
///   3. `None` si no hay nada: el caller decidirá entre descargar
///      `ggml-base.bin` o seguir sin transcripción.
pub(crate) fn whisper_fallback_filename(models_dir: &Path) -> Option<String> {
    use oido_models::ModelFamily;

    // 1) Default preferido.
    if models_dir.join("ggml-base.bin").is_file() {
        return Some("ggml-base.bin".to_string());
    }
    // 2) Cualquier ggml-* instalado que NO sea VAD.
    let mut installed = oido_models::list_installed(models_dir).unwrap_or_default();
    installed.retain(|f| !is_vad_model_filename(f));
    if installed.is_empty() {
        return None;
    }
    // Orden estable: familias más chicas primero (Tiny > Base > Small >
    // Medium > Large > Vad). Vad ya está excluido por el retain. Large
    // puede tardar en cargar; preferimos Tiny/Base/Small como fallback.
    let priority = [
        ModelFamily::Tiny,
        ModelFamily::Base,
        ModelFamily::Small,
        ModelFamily::Medium,
        ModelFamily::Large,
    ];
    for fam in priority {
        if let Some(entry) = oido_models::catalog()
            .iter()
            .find(|e| e.family == fam && installed.contains(&e.filename))
        {
            return Some(entry.filename.clone());
        }
    }
    // Si ninguno estaba catalogado (no debería pasar), devolver el primero.
    installed.into_iter().next()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// El helper `is_vad_model_filename` rechaza el filename canónico
    /// del VAD de Silero (el bug que provocaba el crash GGML_ASSERT).
    #[test]
    fn is_vad_rejects_silero_filename() {
        assert!(is_vad_model_filename("ggml-silero-v5.1.2.bin"));
        assert!(is_vad_model_filename("ggml-silero-v6.bin"));
    }

    #[test]
    fn is_vad_accepts_whisper_filenames() {
        assert!(!is_vad_model_filename("ggml-base.bin"));
        assert!(!is_vad_model_filename("ggml-tiny.en.bin"));
        assert!(!is_vad_model_filename("ggml-large-v3-turbo.bin"));
    }

    /// El fallback prefiere `ggml-base.bin` si está instalado, luego
    /// cualquier `ggml-*.bin` que no sea VAD.
    #[test]
    fn fallback_prefers_base_then_smallest() {
        let dir = tempfile::tempdir().unwrap();
        // Solo tiny instalado → debe elegir tiny.
        std::fs::write(dir.path().join("ggml-tiny.bin"), b"x").unwrap();
        std::fs::write(dir.path().join("ggml-silero-v5.1.2.bin"), b"x").unwrap();
        let pick = whisper_fallback_filename(dir.path()).unwrap();
        assert_eq!(pick, "ggml-tiny.bin", "debe preferir Tiny si no hay base");
    }

    #[test]
    fn fallback_prefers_base_when_both_present() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("ggml-base.bin"), b"x").unwrap();
        std::fs::write(dir.path().join("ggml-tiny.bin"), b"x").unwrap();
        let pick = whisper_fallback_filename(dir.path()).unwrap();
        assert_eq!(pick, "ggml-base.bin", "default histórico: base");
    }

    #[test]
    fn fallback_returns_none_when_no_whisper_only_vad() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("ggml-silero-v5.1.2.bin"), b"x").unwrap();
        assert!(whisper_fallback_filename(dir.path()).is_none());
    }
}

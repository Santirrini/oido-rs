//! Bin CLI `oido`. Punto de entrada delgado: tras el refactor
//! modular profundo este archivo solo orquesta. La lógica vive en
//! módulos hermanos (cli, control, models_setup, model_lifecycle,
//! diagnostics, runtime, hotkey_setup).

#[cfg(feature = "updater")]
use oido_updater as updater;

use anyhow::{Context, Result};
use clap::Parser;
use oido_audio::CpalCapture;
use oido_config::{Config, ConfigStore, Theme};
use oido_core::{Pipeline, PipelineConfig, PipelineEvent, PipelineState};
use oido_hotkey::{parse as parse_hotkey, GatedHotkey, Hotkey};
use oido_input::{ArboardInjector, DirectInjector, SmartInjector};
use oido_stt::{GpuConfig, SharedTranscriber, Transcriber, WhisperCpp};
use oido_tray::{
    default_sections, enable_dpi_awareness, BuildContext, MenuAction, PlatformTray, Tray, TrayState,
};
// `show_model_prompt_windows` sólo existe en la rama windows de oido-tray.
#[cfg(target_os = "windows")]
use oido_tray::show_model_prompt_windows;
#[cfg(any(target_os = "windows", target_os = "macos"))]
use oido_tray::mismatch_tooltip;
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::Instant;
use tracing_subscriber::EnvFilter;

mod cli;
mod control;
mod diagnostics;
mod hotkey_setup;
mod model_lifecycle;
mod models_setup;
mod runtime;

use cli::Cli;
use control::ControlMessage;
use diagnostics::{format_config_table, run_check};
use hotkey_setup::run_set_hotkey;
use model_lifecycle::{activate_after_download, handle_model_click};
use models_setup::{
    download_vad_model_blocking, has_no_bin_files, resolve_models_dir, resolve_prompt_text,
    resolve_vad_model_path, sanitize_config, VAD_MODEL_FILENAME,
};
use runtime::ActivePipeline;

/// Detecta mismatch "modelo solo-inglés activo con idioma distinto a en"
/// y devuelve el filename del modelo multilingüe equivalente que se le
/// debería sugerir al usuario (p.ej. `ggml-small.bin` para `ggml-small.
/// en.bin`). Devuelve `None` si no hay mismatch o si no hay
/// contraparte catalogada.
fn detect_model_lang_mismatch(snap: &Config) -> Option<String> {
    if !oido_models::is_english_only_model(&snap.model) {
        return None;
    }
    if snap.language_ui.eq_ignore_ascii_case("en") {
        return None;
    }
    oido_models::multilingual_counterpart(&snap.model).map(|e| e.filename.to_string())
}

/// Construye el `BuildContext` del menú a partir del snapshot actual de
/// la config. Centraliza el cálculo de `model_lang_mismatch` para que
/// todos los call sites (initial startup, `SetTheme`, `SetSttMode`,
/// `SetUiLanguage`, `SetPromptPreset`, `RefreshMenu`) reflejen el
/// mismo estado sin duplicar lógica.
fn build_ctx(snap: &Config) -> BuildContext {
    BuildContext {
        models_dir: resolve_models_dir(),
        active_model: snap.model.clone(),
        ui_language: snap.ui_language,
        theme: snap.theme,
        stt_mode: snap.stt_mode,
        prompt_preset: snap.prompt_preset,
        prompt_custom_text: snap.system_prompt.clone(),
        effort: snap.effort,
        model_lang_mismatch: detect_model_lang_mismatch(snap),
    }
}

fn main() -> Result<()> {
    // Cronómetro raíz del proceso: se usa para reportar el tiempo total
    // hasta que el pipeline queda listo y escuchando. El objetivo (Fase 2)
    // es bajar este número drásticamente con carga diferida del modelo.
    let startup_total = Instant::now();

    // DPI awareness PER-MONITOR V2: debe ser lo **primero** que el
    // proceso hace en Windows. Sin esto, en pantallas HiDPI/4K el
    // icono y los textos salen borrosos (Windows estira el bitmap).
    // No-op fuera de Windows.
    enable_dpi_awareness();

    // Inicializar logger con soporte para colores ANSI y salida a stderr
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| EnvFilter::new("info,whisper_rs=warn,whisper_rs_sys=warn")),
        )
        .with_target(true)
        .with_ansi(true)
        .with_writer(std::io::stderr)
        .init();

    tracing::info!("oido 2.0 starting (Fase 1: dicta-y-pega optimizado)");
    tracing::info!(
        "Tip: Para el mejor rendimiento en CPU, se recomienda usar el modelo large-v3-turbo-q5_0."
    );

    // Parsear argumentos CLI
    let cli = Cli::parse();

    // 1) Config (defaults aplican si no hay config.json).
    let t_config = Instant::now();
    let cfg = Arc::new(ConfigStore::load_or_default().context("loading config")?);
    tracing::info!(
        phase = "config_load",
        elapsed_ms = t_config.elapsed().as_millis() as u64,
        "startup phase completa"
    );

    // Si se pasa --config-path, se muestra y sale
    if cli.config_path {
        println!("{}", oido_config::config_file().display());
        return Ok(());
    }

    // Manejar flag `--check-update`: buscar actualizaciones y salir
    if cli.check_update {
        #[cfg(feature = "updater")]
        {
            tracing::info!("Buscando actualizaciones...");
            match updater::check_and_apply() {
                Ok(updater::Status::UpToDate) => {
                    tracing::info!("La aplicación ya está actualizada.");
                }
                Ok(updater::Status::DownloadedAndInstalling { version }) => {
                    tracing::info!(
                        "Nueva versión v{} descargada. Iniciando instalador...",
                        version
                    );
                }
                Err(e) => {
                    tracing::error!("Error al buscar o aplicar actualización: {:?}", e);
                    std::process::exit(1);
                }
            }
            return Ok(());
        }
        #[cfg(not(feature = "updater"))]
        {
            tracing::error!("El actualizador automático no está disponible en esta compilación.");
            std::process::exit(1);
        }
    }

    // Manejar modificación persistente de tema/idioma vía CLI si se solicitaron
    let mut snap = cfg.snapshot();
    let mut changed = false;

    // Sanitización one-shot: reescribe estados inconsistentes de la
    // config (típicamente `prompt_preset=Custom` con `system_prompt`
    // vacío por edición manual del config.json o por una versión vieja
    // del bin). Si hubo cambio, refrescamos `snap` para que el resto
    // del startup vea el valor ya corregido.
    if sanitize_config(&cfg) {
        snap = cfg.snapshot();
        tracing::info!("config sanitizada en disco; recargada para esta sesión");
    }

    // Auto-recovery: si `Config.model` apunta a un archivo inválido
    // — VAD, archivo inexistente, o filename no catalogado — caemos
    // al primer whisper válido disponible. Sin esto, GGML_ASSERT en
    // `whisper_model_load` aborta el proceso con
    // STATUS_STACK_BUFFER_OVERRUN al primer press del hotkey (o al
    // startup si el modelo se carga eager). El fallback es tolerante:
    // si no hay nada, deja el valor como está y el flow posterior
    // (lazy-loader + first-run dialog en Windows) se encargará de
    // descargar `ggml-base.bin`.
    let models_dir_for_recovery = crate::resolve_models_dir();
    {
        let configured = &snap.model;
        let on_disk = models_dir_for_recovery.join(configured).is_file();
        let is_vad = oido_stt::is_vad_model_filename(configured);
        let needs_recovery = is_vad || !on_disk;
        if needs_recovery {
            tracing::warn!(
                model = %configured,
                on_disk,
                is_vad,
                "Config.model apunta a un modelo inválido; buscando fallback"
            );
            if let Some(fallback) =
                crate::model_lifecycle::whisper_fallback_filename(&models_dir_for_recovery)
            {
                if fallback != *configured {
                    tracing::info!(
                        from = %configured,
                        to = %fallback,
                        "auto-recovery: corrigiendo Config.model al primer whisper instalado"
                    );
                    snap.model = fallback.clone();
                    cfg.replace(snap.clone());
                    if let Err(e) = cfg.save() {
                        tracing::error!(?e, "no se pudo persistir auto-recovery de model");
                    }
                }
            } else if is_vad {
                // Caso patológico: la config apuntaba a un VAD y no
                // hay NINGÚN modelo whisper instalado. Forzamos el
                // default y dejamos que el rest del flujo (first-run
                // prompt, comando `oido --set-model`, etc.) lo
                // descargue.
                tracing::warn!(
                    "no hay whisper instalado; revirtiendo Config.model al default (ggml-base.bin)"
                );
                snap.model = "ggml-base.bin".into();
                cfg.replace(snap.clone());
                let _ = cfg.save();
            }
        }
    }

    if let Some(ref theme_str) = cli.theme {
        let theme = match theme_str.as_str() {
            "dark" => Theme::Dark,
            "light" => Theme::Light,
            _ => Theme::System,
        };
        snap.theme = theme;
        changed = true;
    }

    if let Some(ref lang) = cli.lang {
        snap.language_ui = lang.clone();
        changed = true;
    }

    if let Some(ref prompt) = cli.set_prompt {
        if prompt.is_empty() {
            // `--set-prompt ""` resetea al preset por defecto (bilingüe).
            snap.prompt_preset = oido_config::PromptPreset::BilingualEsEn;
            snap.system_prompt.clear();
        } else {
            snap.prompt_preset = oido_config::PromptPreset::Custom;
            snap.system_prompt = prompt.clone();
        }
        changed = true;
        tracing::info!(?snap.prompt_preset, "system prompt actualizado por CLI");
    }

    if changed {
        cfg.replace(snap.clone());
        cfg.save().context("guardando cambios de config CLI")?;
        tracing::info!("Configuración actualizada persistentemente.");
    }

    // Manejar sub-comando `--set-hotkey`: graba interactivamente la tecla de activación
    if cli.set_hotkey {
        let _ = run_set_hotkey(Arc::clone(&cfg))?;
        return Ok(());
    }

    // Manejar flag `--check`: reporte y salir
    if cli.check {
        return run_check(&snap);
    }

    // Si el binding de la config no parsea, warn y caemos al default
    let binding = match parse_hotkey(&snap.hotkey) {
        Ok(_) => snap.hotkey.clone(),
        Err(e) => {
            tracing::warn!(
                binding = %snap.hotkey,
                ?e,
                "binding de config inválido; usando default"
            );
            Config::default().hotkey
        }
    };

    // Imprimir tabla resumen de configuración activa
    let models_dir = resolve_models_dir();
    println!("{}", format_config_table(&snap, &models_dir));

    // Configurar tema preferido de menús nativos en Windows
    #[cfg(target_os = "windows")]
    oido_tray::set_windows_menu_theme(snap.theme);

    // 2) Inicializar Tray nativo
    let t_tray = Instant::now();
    // El árbol del menú se compone declarativamente a partir de
    // `MenuSection`s. Aquí usamos el set canónico (`default_sections`
    // produce el mismo árbol que el menú legacy). Si en el futuro hay
    // que añadir una sección, basta con sumar un struct que implemente
    // `MenuSection` y agregarlo a la lista — `tray.rs::build_menu`
    // deja de tocar.
    let mut tray = PlatformTray::new(
        models_dir.clone(),
        snap.model.clone(),
        snap.ui_language,
        snap.prompt_preset,
    )
    .ok();
    let menu_rx = tray.as_mut().and_then(|t| t.take_menu_events());
    tracing::info!(
        phase = "tray_init",
        elapsed_ms = t_tray.elapsed().as_millis() as u64,
        "startup phase completa"
    );

    // Reconstruir el árbol inicial con el `BuildContext` completo,
    // incluyendo la detección de mismatch modelo/idioma. Esto es
    // necesario porque `PlatformTray::new` solo conoce el subconjunto
    // mínimo y siempre construye `BuildContext::initial` (con
    // `model_lang_mismatch: None`). Sin este rebuild, el ítem de aviso
    // ⚠ nunca aparecería en el primer arranque.
    if let Some(ref mut t) = tray {
        let sections = default_sections(&build_ctx(&snap));
        if let Err(e) = t.rebuild_menu(sections) {
            tracing::error!(?e, "no se pudo reconstruir el menú inicial");
        }
        // Si hay mismatch, sobrescribe el tooltip con el aviso
        // persistente. El icono sigue reflejando el estado operativo
        // vía `set_state` más abajo.
        if detect_model_lang_mismatch(&snap).is_some() {
            #[cfg(any(target_os = "windows", target_os = "macos"))]
            {
                let _ = t.set_tooltip(&mismatch_tooltip(snap.ui_language));
            }
        }
    }

    // 3) Cargar modelo whisper. GPU + threads vienen de Config.
    let model_path = models_dir.join(&snap.model);
    tracing::info!(?model_path, "cargando modelo whisper");
    let gpu_config = if snap.use_gpu {
        GpuConfig {
            use_gpu: true,
            flash_attn: true,
        }
    } else {
        GpuConfig::default()
    };

    // n_threads por worker: con N workers en paralelo, dividir el total
    // entre ellos evita oversubscription. `STT_WORKERS` viene de
    // oido-core (única fuente de verdad) para evitar que el cálculo
    // se desincronice si cambia el tamaño del pool.
    let n_threads_per_worker = snap.n_threads.unwrap_or_else(|| {
        let total = std::thread::available_parallelism()
            .map(|n| n.get() as u16)
            .unwrap_or(4)
            .min(8);
        // Piso de 2 para que whisper.cpp pueda paralelizar el decoder.
        (total / oido_core::STT_WORKERS).max(2)
    });

    // VAD nativo: si el modelo GGML existe, activamos el recorte de
    // silencios que hace whisper.cpp antes del encoder. Si NO existe,
    // el lazy-loader intentará descargarlo **off** del main thread
    // (ver `download_vad_model_blocking`). Esto evita que un primer
    // arranque sin VAD pague los segundos de red en `startup_total`.
    // Si la descarga falla, fallback graceful (STT sigue funcionando).
    let t_vad = Instant::now();
    let vad_path = resolve_vad_model_path(&models_dir);
    tracing::info!(
        phase = "vad_resolve",
        elapsed_ms = t_vad.elapsed().as_millis() as u64,
        vad_available = vad_path.is_some(),
        defer_download_if_missing = vad_path.is_none(),
        "startup phase completa"
    );
    if vad_path.is_none() {
        tracing::warn!(
            "VAD desactivado (modelo no disponible). Para mejorar latencia en \
             audios 10-30s con pausas, descarga ggml-silero-v5.1.2.bin al \
             directorio de modelos."
        );
    }

    let stt_mode = snap.stt_mode;
    let mut transcriber: Option<Arc<dyn Transcriber>> = None;
    // Handle fuerte al `SharedTranscriber` (modo Batch) para que el
    // thread de carga lazy pueda invocar `load_model` sin pasar por
    // el trait `Transcriber`. `None` en modo Streaming.
    let mut shared_transcriber: Option<Arc<SharedTranscriber>> = None;
    let mut streamer: Option<oido_stt::LocalAgreementStreamer> = None;

    if stt_mode == oido_config::SttMode::Batch || stt_mode == oido_config::SttMode::Chunked {
        // System prompt resuelto una sola vez al startup; si cambia
        // en runtime via menú/CLI, se propaga via `set_initial_prompt`
        // sobre el `SharedTranscriber` (no requiere recargar el modelo).
        let prompt = resolve_prompt_text(&snap);
        let mut stt_builder = WhisperCpp::with_language(&snap.language_ui)
            .with_initial_prompt(&prompt)
            .with_runtime(gpu_config, n_threads_per_worker)
            .with_effort(snap.effort);
        let stt = if let Some(ref vp) = vad_path {
            stt_builder = stt_builder.with_vad(vp.clone());
            stt_builder
        } else {
            stt_builder
        };
        // **Lazy load**: el modelo se carga en background la primera
        // vez que el usuario aprieta el hotkey (ver `start_pipeline`).
        // `SharedTranscriber` envuelve el WhisperCpp en un Mutex para
        // que `load_model` (&mut) pueda convivir con `transcribe` (&self)
        // a través del trait.
        tracing::info!(
            path = ?model_path,
            prompt_chars = prompt.chars().count(),
            effort = ?snap.effort,
            "modelo whisper NO se carga al startup (lazy); se cargará en el primer press del hotkey"
        );
        let shared = SharedTranscriber::new(stt);
        let shared_arc = Arc::new(shared);
        shared_transcriber = Some(Arc::clone(&shared_arc));
        transcriber = Some(shared_arc as Arc<dyn Transcriber>);
    } else {
        let prompt = resolve_prompt_text(&snap);
        let st = oido_stt::LocalAgreementStreamer::new(
            Some(snap.language_ui.clone()),
            gpu_config,
            n_threads_per_worker,
        )
        .with_initial_prompt(&prompt)
        .with_effort(snap.effort);
        // **Lazy load**: ver comentario en la rama Batch. El modelo
        // streaming se carga en la primera pulsación del hotkey.
        tracing::info!(
            path = ?model_path,
            "modelo whisper streaming NO se carga al startup (lazy); se cargará en el primer press"
        );
        streamer = Some(st);
    }

    // Canal de control para el loop de eventos en el hilo principal
    let (control_tx, control_rx) = crossbeam_channel::bounded::<ControlMessage>(16);

    // Hilo oido-menu-listener
    if let Some(rx) = menu_rx {
        let control_tx_for_menu = control_tx.clone();
        let cfg_for_menu = Arc::clone(&cfg);
        let shared_for_menu = shared_transcriber.as_ref().map(Arc::clone);
        thread::Builder::new()
            .name("oido-menu-listener".into())
            .spawn(move || {
                while let Ok(action) = rx.recv() {
                    match action {
                        MenuAction::ChangeHotkey => {
                            let _ = control_tx_for_menu.send(ControlMessage::ChangeHotkey);
                        }
                        MenuAction::SetTheme(theme) => {
                            let _ = control_tx_for_menu.send(ControlMessage::SetTheme(theme));
                        }
                        MenuAction::SetSttMode(mode) => {
                            let _ = control_tx_for_menu.send(ControlMessage::SetSttMode(mode));
                        }
                        MenuAction::SetUiLanguage(lang) => {
                            let _ = control_tx_for_menu.send(ControlMessage::SetUiLanguage(lang));
                        }
                        MenuAction::SetPromptPreset(preset) => {
                            let _ =
                                control_tx_for_menu.send(ControlMessage::SetPromptPreset(preset));
                        }
                        MenuAction::SetEffort(preset) => {
                            let _ =
                                control_tx_for_menu.send(ControlMessage::SetEffort(preset));
                        }
                        MenuAction::EditPrompt => {
                            // Abrir el config.json del usuario en el editor
                            // nativo del OS (Bloc de Notas en Windows, etc.)
                            // para editar el campo `system_prompt` con todas
                            // las comodidades de un editor de verdad.
                            //
                            // Como `show_prompt_editor_windows` bloquea el
                            // hilo del listener mientras el usuario edita,
                            // el hotkey queda momentáneamente en cola; es un
                            // trade-off aceptable (la edición es ocasional)
                            // y mantiene el `unsafe` Win32 confinado al crate.
                            let config_path = oido_config::config_file();
                            let res = oido_tray::show_prompt_editor_windows(&config_path);
                            if res.is_none() {
                                tracing::error!(
                                    path = ?config_path,
                                    "no se pudo abrir el editor de texto para el prompt"
                                );
                            }
                            // El reload de config y la propagación al STT
                            // ocurren en el siguiente arranque (cambio de
                            // hotkey, mode-switch, o restart). No es un
                            // reload hot porque el bin no observa el mtime
                            // de config.json (ver ConfigStore::load_or_default).
                        }
                        MenuAction::OpenModelsDir => {
                            let models_dir = resolve_models_dir();
                            if let Err(e) = open::that(&models_dir) {
                                tracing::error!(?e, "no se pudo abrir el directorio de modelos");
                            }
                        }
                        MenuAction::ModelItem(filename) => {
                            handle_model_click(
                                &filename,
                                &control_tx_for_menu,
                                &cfg_for_menu,
                                shared_for_menu.as_ref(),
                            );
                        }
                        MenuAction::FixModelLanguage(_suggested) => {
                            // El id canónico no viaja el filename (la
                            // sección compone el label, no el id) —
                            // resolvemos aquí el contraparte
                            // multilingüe desde la config actual para
                            // no arrastrar estado a través del canal.
                            // Si por algún motivo ya no hay mismatch
                            // (carrera con un SetSttMode o cambio de
                            // modelo), no hacemos nada.
                            let snap = cfg_for_menu.snapshot();
                            if let Some(target) = detect_model_lang_mismatch(&snap) {
                                tracing::info!(
                                    from = %snap.model,
                                    to = %target,
                                    "FixModelLanguage: usuario aceptó el cambio sugerido"
                                );
                                handle_model_click(
                                    &target,
                                    &control_tx_for_menu,
                                    &cfg_for_menu,
                                    shared_for_menu.as_ref(),
                                );
                            } else {
                                tracing::debug!(
                                    "FixModelLanguage click sin mismatch activo; ignorado"
                                );
                            }
                        }
                        MenuAction::CheckUpdates => {
                            #[cfg(feature = "updater")]
                            {
                                tracing::info!("Iniciando búsqueda de actualizaciones en background...");
                                let control_tx_clone = control_tx_for_menu.clone();
                                let _ = thread::Builder::new()
                                    .name("oido-updater-bg".into())
                                    .spawn(move || {
                                        let _ = control_tx_clone.send(ControlMessage::SetTrayState(TrayState::Loading));
                                        match updater::check_and_apply() {
                                            Ok(updater::Status::UpToDate) => {
                                                tracing::info!("La aplicación ya está actualizada.");
                                                let _ = control_tx_clone.send(ControlMessage::SetTrayState(TrayState::Idle));
                                            }
                                            Ok(updater::Status::DownloadedAndInstalling { version }) => {
                                                tracing::info!("Nueva versión v{} descargada e instalando en background.", version);
                                                let _ = control_tx_clone.send(ControlMessage::SetTrayState(TrayState::Idle));
                                            }
                                            Err(e) => {
                                                tracing::error!("Error buscando/aplicando actualizaciones en background: {:?}", e);
                                                let _ = control_tx_clone.send(ControlMessage::SetTrayState(TrayState::Error));
                                            }
                                        }
                                    });
                            }
                            #[cfg(not(feature = "updater"))]
                            {
                                tracing::warn!("El actualizador automático no está habilitado en esta build.");
                            }
                        }
                        MenuAction::TogglePause => {
                            tracing::info!("Pausa/Reanudar placeholder — Fase 2");
                        }
                        MenuAction::Exit => {
                            let _ = control_tx_for_menu.send(ControlMessage::Exit);
                        }
                    }
                }
            })?;
    }

    // 4) Loop de ejecución y orquestación del pipeline
    let mut current_binding = binding.clone();
    let mut pipeline_opt: Option<ActivePipeline> = None;
    let mut observer_handle: Option<JoinHandle<()>> = None;

    let control_tx_for_start = control_tx.clone();
    // `model_path` se clona aquí para que la closure del lazy-loader
    // (que captura por movimiento) pueda usarlo, y los call sites
    // posteriores (cambio de hotkey / modo) también.
    let model_path_for_pipeline = model_path.clone();

    // Función auxiliar para iniciar el pipeline
    let start_pipeline = move |binding_str: &str,
                               mode: oido_config::SttMode,
                               tr_opt: &Option<Arc<dyn Transcriber>>,
                               st_opt: &Option<oido_stt::LocalAgreementStreamer>,
                               shared_opt: &Option<Arc<SharedTranscriber>>,
                               is_downloading: bool|
          -> Result<(ActivePipeline, JoinHandle<()>)> {
        let capture = Box::new(CpalCapture::new().context("init captura audio")?);
        // `GatedHotkey` envuelve el `RdevHotkey` y suprime los callbacks
        // de press/release hasta que se llame `mark_ready()` en el handle
        // compartido. Esto implementa la carga lazy: el pipeline se
        // arranca YA (con `is_loaded() == false`), pero el primer press
        // no llega al STT hasta que el modelo esté cargado.
        // El `mut` es necesario: el `register` interno que
        // `Pipeline::start` invoca después toma `&mut self`. El
        // compilador no lo ve porque el `&mut` se materializa a través
        // del `Box<dyn Hotkey>`, así que silenciamos el falso positivo.
        #[allow(unused_mut)]
        let mut gated = GatedHotkey::new();
        let ready_handle = gated.ready_handle();
        // `gated` se mueve al `Box` y se registrará con `&mut self`
        // dentro del pipeline. La `mut` es necesaria para el `register`
        // interno que `Pipeline::start` invocará después.
        let hotkey: Box<dyn Hotkey> = Box::new(gated);
        let clipboard = ArboardInjector::new().context("init injector clipboard")?;
        // Cadena de inyección: UIAutomation (Windows-only, respeta el caret)
        // con fallback transparente a clipboard+Ctrl+V. En macOS/Linux el
        // backend stub devuelve Unsupported y caemos directo al fallback.
        let direct: Option<Arc<dyn DirectInjector>> =
            match oido_input::UiaDirectInjector::new() {
                Ok(d) => Some(d),
                Err(e) => {
                    tracing::warn!(
                        ?e,
                        "UIA direct injector no inicializado; solo clipboard",
                    );
                    None
                }
            };
        let injector: Arc<dyn oido_input::Injector> =
            Arc::new(SmartInjector::new(direct, clipboard));

        let mut active_pipe = match mode {
            oido_config::SttMode::Batch => {
                let tr = tr_opt
                    .clone()
                    .context("transcriber no cargado en modo batch")?;
                let pipeline_cfg = PipelineConfig {
                    capture,
                    transcriber: tr,
                    injector,
                    hotkey,
                    hotkey_binding: binding_str.to_string(),
                };
                ActivePipeline::Batch(Pipeline::new(pipeline_cfg))
            }
            oido_config::SttMode::Streaming => {
                tracing::warn!(
                    mode = "streaming",
                    "Modo de dictado en prueba (no estable). Solo Batch está recomendado para uso normal."
                );
                let st = st_opt
                    .clone()
                    .context("streamer no cargado en modo streaming")?;
                let pipeline_cfg = oido_core::StreamingPipelineConfig {
                    capture,
                    streamer: Box::new(st),
                    injector,
                    hotkey,
                    hotkey_binding: binding_str.to_string(),
                };
                ActivePipeline::Streaming(oido_core::StreamingPipeline::new(pipeline_cfg))
            }
            oido_config::SttMode::Chunked => {
                tracing::warn!(
                    mode = "chunked",
                    "Modo de dictado en prueba (no estable). Solo Batch está recomendado para uso normal."
                );
                // Chunked reusa el mismo backend WhisperCpp que Batch
                // (transcribe_timed vive en el mismo struct). Requiere
                // `chunk_duration_secs`; 5.0s es el default recomendado.
                let tr = tr_opt
                    .clone()
                    .context("transcriber no cargado en modo chunked")?;
                let pipeline_cfg = oido_core::ChunkedPipelineConfig {
                    capture,
                    transcriber: tr,
                    injector,
                    hotkey,
                    hotkey_binding: binding_str.to_string(),
                    chunk_duration_secs: 5.0,
                };
                ActivePipeline::Chunked(oido_core::ChunkedPipeline::new(pipeline_cfg))
            }
        };

        let events = active_pipe.events();
        let control_tx_for_obs = control_tx_for_start.clone();

        let obs = thread::Builder::new()
            .name("oido-event-observer".into())
            .spawn(move || {
                while let Ok(evt) = events.recv() {
                    match evt {
                        PipelineEvent::State(state) => {
                            tracing::info!(?state, "pipeline state");
                            let ts = match state {
                                PipelineState::Idle => TrayState::Idle,
                                PipelineState::Recording => TrayState::Listening,
                                PipelineState::Processing => TrayState::Processing,
                                PipelineState::Error => TrayState::Error,
                            };
                            let _ = control_tx_for_obs.send(ControlMessage::SetTrayState(ts));
                        }
                        PipelineEvent::Shutdown => {
                            tracing::info!("Observer recibiendo evento de apagado.");
                            break;
                        }
                    }
                }
            })?;

        active_pipe.start().context("arrancando pipeline")?;

        // **Lazy load trigger**: lanzamos la carga del modelo + warm-up
        // en un hilo dedicado. Mientras carga, el `GatedHotkey` descarta
        // silenciosamente cualquier press del usuario (que verá el
        // estado `Loading` por la señal de control_tx). Al terminar,
        // marcamos el hotkey como listo y la próxima pulsación del
        // usuario funcionará normalmente.
        let control_tx_for_lazy = control_tx_for_start.clone();
        let model_path_for_lazy = model_path_for_pipeline.clone();
        let shared_for_lazy: Option<Arc<SharedTranscriber>> = shared_opt.as_ref().map(Arc::clone);
        // Si el modelo VAD no estaba en disco al startup, intentamos
        // descargarlo off del thread principal. No bloquea `startup_total`.
        let models_dir_for_lazy = resolve_models_dir();
        let did_vad_exist_at_startup = models_dir_for_lazy.join(VAD_MODEL_FILENAME).exists();
        thread::Builder::new()
            .name("oido-lazy-loader".into())
            .spawn(move || {
                if is_downloading {
                    // Si ya se está descargando el modelo en background, dejamos que el
                    // descargador lo active y cambie el estado al terminar.
                    return;
                }
                // Notificamos al bin que estamos cargando.
                let _ = control_tx_for_lazy.send(ControlMessage::SetTrayState(TrayState::Loading));

                let t = Instant::now();
                let load_result: anyhow::Result<()> = if mode == oido_config::SttMode::Batch
                    || mode == oido_config::SttMode::Chunked
                {
                    if let Some(shared) = shared_for_lazy.as_ref() {
                        let handle = shared.handle();
                        {
                            let mut guard = handle.lock();
                            if let Err(e) = guard.load_model(&model_path_for_lazy) {
                                tracing::error!(
                                    ?e,
                                    path = ?model_path_for_lazy,
                                    "lazy load: modelo no encontrado"
                                );
                                drop(guard);
                                let _ = control_tx_for_lazy
                                    .send(ControlMessage::SetTrayState(TrayState::Error));
                                return;
                            }
                            tracing::info!(
                                size_mib = %format!(
                                    "{:.1}",
                                    std::fs::metadata(&model_path_for_lazy)
                                        .map(|m| m.len() as f64 / 1024.0 / 1024.0)
                                        .unwrap_or(0.0)
                                ),
                                "modelo whisper cargado (lazy batch)"
                            );
                            // guard se libera al salir del bloque
                        }
                        // Warm-up: nuevo lock en un lock_api::Mutex.
                        let started = Instant::now();
                        let warmup_result = handle.lock().warm_up();
                        match warmup_result {
                            Ok(()) => tracing::info!(
                                warmup_ms = started.elapsed().as_millis() as u64,
                                "warm-up STT lazy completado"
                            ),
                            Err(e) => tracing::warn!(
                                ?e,
                                "warm-up lazy falló; el primer dictado será más lento"
                            ),
                        }
                    } else {
                        // `shared_for_lazy == None` ocurre en dos casos:
                        //
                        // 1. **Modo source = Streaming en el startup**. En
                        //    este branch solo se construyó el
                        //    `LocalAgreementStreamer` y `shared_transcriber`
                        //    quedó `None`. Pero ahora `mode == Batch/Chunked`,
                        //    lo cual es incongruente con el startup: solo
                        //    posible si el handler `SetSttMode` ya
                        //    reemplazó `transcriber` con un WhisperCpp
                        //    eager-cargado (ver líneas ~880-900). En ese
                        //    path el modelo **ya está en memoria**, no hay
                        //    nada que hacer aquí.
                        //
                        // 2. **Defensivo**: cualquier otro edge-case donde
                        //    el `SharedTranscriber` no esté disponible. Lo
                        //    tratamos como un "skip" informativo (no error)
                        //    para no abortar el binario si el modelo ya
                        //    está cargado por otra vía. Si realmente NO
                        //    hay modelo cargado, el primer press fallará
                        //    en el pipeline y se diagnosticará allí.
                        tracing::info!(
                            "lazy load no requerido: modelo whisper presumiblemente ya \
                             cargado por otra vía (cambio de modo eager). Abriendo hotkey."
                        );
                    }
                    Ok(())
                } else {
                    // Streaming: el `LocalAgreementStreamer` no es
                    // Sync; el `Box<dyn Streamer>` no se puede mutar
                    // desde aquí. Para mantener el alcance del refactor
                    // conservador, NO implementamos lazy load para el
                    // modo streaming en esta fase (queda como TODO).
                    tracing::warn!(
                        "lazy load no implementado para modo streaming; el modo streaming \
                         cargará el modelo eagerly (sin lazy)"
                    );
                    Ok(())
                };

                if let Err(e) = load_result {
                    tracing::error!(?e, "lazy load falló");
                    let _ =
                        control_tx_for_lazy.send(ControlMessage::SetTrayState(TrayState::Error));
                    return;
                }

                // Si el VAD no estaba al startup, intentamos descargarlo
                // ahora (off del main thread). Esto no bloquea
                // `startup_total`. Si falla, el STT sigue sin VAD
                // (el STT ya se construyó sin vad_path al inicio).
                if !did_vad_exist_at_startup
                    && (mode == oido_config::SttMode::Batch
                        || mode == oido_config::SttMode::Chunked)
                {
                    let t_vad_dl = Instant::now();
                    let vad_downloaded = download_vad_model_blocking(&models_dir_for_lazy);
                    tracing::info!(
                        phase = "vad_download_lazy",
                        elapsed_ms = t_vad_dl.elapsed().as_millis() as u64,
                        downloaded = vad_downloaded.is_some(),
                        "intento de descarga VAD post-startup (en lazy-loader)"
                    );
                    // NOTA: si la descarga tuvo éxito, recargar el modelo
                    // con VAD requeriría reiniciar el loader. Por ahora
                    // solo aplicaría al siguiente `SetSttMode`. Queda
                    // como optim opcional (`refresh_vad_post_download`).
                }

                tracing::info!(
                    phase = "lazy_load_total",
                    elapsed_ms = t.elapsed().as_millis() as u64,
                    "lazy load completo; hotkey ahora operativo"
                );

                // Abrimos la compuerta: el siguiente press del usuario
                // llegará al pipeline.
                ready_handle.mark_ready();
                let _ = control_tx_for_lazy.send(ControlMessage::SetTrayState(TrayState::Idle));
            })?;

        Ok((active_pipe, obs))
    };

    // `is_downloading_at_startup` se asigna a `true` únicamente en la
    // rama windows (vía el bloque cfg-gated más abajo). En linux/macos
    // permanece `false` y un `let mut` sería unused en -D warnings.
    //
    // Para evitar esto declaramos la variable en cada rama con su
    // propio `let`. En windows: `let mut ... = false` y se muta
    // dentro del bloque. En linux/macos: `let ... = false` (sin
    // mut). El bloque de "scope" compartido (`model_missing` y
    // `has_no_bins`) y el cómputo del bool se hacen con un match
    // cfg-gated que produce el mismo nombre.
    let model_missing = !models_dir.join(&snap.model).exists();
    let has_no_bins = has_no_bin_files(&models_dir);

    #[cfg(target_os = "windows")]
    let mut is_downloading_at_startup = false;
    #[cfg(not(target_os = "windows"))]
    let is_downloading_at_startup = false;

    #[cfg(target_os = "windows")]
    {
        if (has_no_bins || model_missing) && show_model_prompt_windows() {
            is_downloading_at_startup = true;
            let entry = oido_models::find("ggml-base.bin").cloned();
            if let Some(entry) = entry {
                let tx = control_tx.clone();
                let dir = models_dir.clone();
                let shared_for_dl = shared_transcriber.as_ref().map(Arc::clone);
                let cfg_for_dl = Arc::clone(&cfg);
                let span =
                    tracing::info_span!("download_model_startup", filename = "ggml-base.bin");
                let _ = thread::Builder::new()
                    .name("oido-downloader".into())
                    .spawn(move || {
                        let _enter = span.enter();
                        tracing::info!("Descargando ggml-base.bin desde el inicio...");
                        match oido_models::download_model(&dir, &entry, None) {
                            Ok(()) => {
                                tracing::info!("descarga de inicio completa");
                                let _ = tx.send(ControlMessage::RefreshMenu);
                                activate_after_download(
                                    &entry.filename,
                                    &dir,
                                    &cfg_for_dl,
                                    shared_for_dl.as_ref(),
                                    &tx,
                                );
                            }
                            Err(e) => {
                                tracing::error!(?e, "descarga de inicio falló");
                                let _ = tx.send(ControlMessage::SetTrayState(TrayState::Error));
                            }
                        }
                    });
            }
        }
    }

    // Arranque inicial
    match start_pipeline(
        &current_binding,
        stt_mode,
        &transcriber,
        &streamer,
        &shared_transcriber,
        is_downloading_at_startup,
    ) {
        Ok((pipe, obs)) => {
            pipeline_opt = Some(pipe);
            observer_handle = Some(obs);
            tracing::info!(
                hotkey = %current_binding,
                "hold {current_binding}, dicta, suelta. Ctrl+C para salir."
            );
            if let Some(ref mut t) = tray {
                let theme = cfg.snapshot().theme;
                let initial_state = if is_downloading_at_startup {
                    TrayState::Loading
                } else if model_missing || has_no_bins {
                    TrayState::Error
                } else {
                    TrayState::Idle
                };
                let _ = t.set_state(initial_state, theme);
            }
        }
        Err(e) => {
            tracing::error!(?e, "no se pudo arrancar el pipeline inicial");
        }
    }

    // Resumen final del startup: cuanto tardó desde `fn main()` hasta
    // pipeline armado + tray visible. Métrica objetivo para Fase 2.
    tracing::info!(
        phase = "startup_total",
        elapsed_ms = startup_total.elapsed().as_millis() as u64,
        "startup completo; bin listo para dictar"
    );

    // Instalar handler de Ctrl+C redirigiéndolo a nuestro canal de control
    let control_tx_ctrlc = control_tx.clone();
    ctrlc::set_handler(move || {
        tracing::info!("Ctrl+C recibido, terminando");
        let _ = control_tx_ctrlc.send(ControlMessage::Exit);
    })
    .context("instalando handler Ctrl+C")?;

    let mut current_tray_state = TrayState::Idle;
    let mut is_changing_hotkey = false;

    // Ciclo de control del hilo principal (no bloqueante en Windows)
    loop {
        // En Windows, es imperativo procesar el bucle de mensajes Win32
        // para que tray-icon reciba los clics del mouse y levante el menú.
        #[cfg(target_os = "windows")]
        oido_tray::pump_windows_message_loop();

        // Procesar todos los mensajes de control listos en la cola
        let mut should_exit = false;
        while let Ok(msg) = control_rx.try_recv() {
            match msg {
                ControlMessage::ChangeHotkey => {
                    if is_changing_hotkey {
                        tracing::warn!("Cambio de hotkey ya está en curso, ignorando.");
                        continue;
                    }
                    tracing::info!("Iniciando cambio de hotkey desde el menú...");
                    is_changing_hotkey = true;

                    // Feedback visual en la bandeja de que estamos esperando input
                    if let Some(ref mut t) = tray {
                        let theme = cfg.snapshot().theme;
                        let _ = t.set_state(TrayState::Paused, theme);
                    }

                    // 1) Detener el pipeline actual para liberar el gancho global de teclado
                    if let Some(mut pipe) = pipeline_opt.take() {
                        let _ = pipe.shutdown();
                    }
                    if let Some(obs) = observer_handle.take() {
                        let _ = obs.join();
                    }

                    // 2) Grabar tecla en background
                    let cfg_clone = Arc::clone(&cfg);
                    let control_tx_clone = control_tx.clone();
                    std::thread::Builder::new()
                        .name("oido-hotkey-changer".into())
                        .spawn(move || {
                            let res = run_set_hotkey(cfg_clone).map_err(|e| e.to_string());
                            let _ = control_tx_clone.send(ControlMessage::HotkeyChanged(res));
                        })
                        .expect("spawn hotkey changer thread");
                }
                ControlMessage::HotkeyChanged(res) => {
                    is_changing_hotkey = false;
                    match res {
                        Ok(new_binding) => {
                            current_binding = new_binding;
                            tracing::info!("Nuevo hotkey grabado: {}", current_binding);
                        }
                        Err(e) => {
                            tracing::error!(
                                error = %e,
                                "falló la grabación del hotkey, volviendo al anterior"
                            );
                        }
                    }

                    // 3) Reiniciar pipeline
                    let snap = cfg.snapshot();
                    match start_pipeline(
                        &current_binding,
                        snap.stt_mode,
                        &transcriber,
                        &streamer,
                        &shared_transcriber,
                        false,
                    ) {
                        Ok((pipe, obs)) => {
                            pipeline_opt = Some(pipe);
                            observer_handle = Some(obs);
                            tracing::info!("Pipeline reiniciado con hotkey {}", current_binding);
                            if let Some(ref mut t) = tray {
                                let theme = snap.theme;
                                let _ = t.set_state(TrayState::Idle, theme);
                            }
                        }
                        Err(e) => {
                            tracing::error!(?e, "no se pudo reiniciar el pipeline");
                        }
                    }
                }
                ControlMessage::SetTrayState(state) => {
                    current_tray_state = state;
                    let theme = cfg.snapshot().theme;
                    if let Some(ref mut t) = tray {
                        let _ = t.set_state(state, theme);
                    }
                }
                ControlMessage::SetTheme(theme) => {
                    let mut snap = cfg.snapshot();
                    snap.theme = theme;
                    cfg.replace(snap.clone());
                    if let Err(e) = cfg.save() {
                        tracing::error!(?e, "no se pudo guardar la configuración");
                    } else {
                        tracing::info!("Tema actualizado a {:?}", theme);
                    }
                    #[cfg(target_os = "windows")]
                    oido_tray::set_windows_menu_theme(theme);
                    if let Some(ref mut t) = tray {
                        let _ = t.set_state(current_tray_state, theme);
                        // Reconstruir el árbol del menú para refrescar
                        // la marca ✓ del tema activo. Sin esta llamada,
                        // el ✓ del tema anterior permanece visible
                        // aunque la config ya esté actualizada.
                        let sections = default_sections(&build_ctx(&snap));
                        if let Err(e) = t.rebuild_menu(sections) {
                            tracing::error!(
                                ?e,
                                "no se pudo reconstruir el menú tras cambio de tema"
                            );
                        }
                    }
                }
                ControlMessage::SetSttMode(mode) => {
                    let mut snap = cfg.snapshot();
                    if snap.stt_mode != mode {
                        snap.stt_mode = mode;
                        cfg.replace(snap.clone());
                        if let Err(e) = cfg.save() {
                            tracing::error!(?e, "no se pudo guardar la configuración");
                        } else {
                            tracing::info!("Modo STT actualizado persistentemente a {:?}", mode);
                        }

                        // 1) Detener el pipeline actual para liberar el gancho global de teclado
                        if let Some(mut pipe) = pipeline_opt.take() {
                            let _ = pipe.shutdown();
                        }
                        if let Some(obs) = observer_handle.take() {
                            let _ = obs.join();
                        }

                        // 2) Liberar backend inactivo y cargar el nuevo.
                        // Batch y Chunked comparten el mismo backend
                        // (WhisperCpp + SharedTranscriber); solo difieren
                        // en el pipeline que los consume. Streaming usa
                        // LocalAgreementStreamer.
                        if mode == oido_config::SttMode::Batch
                            || mode == oido_config::SttMode::Chunked
                        {
                            streamer = None; // Liberar memoria de streaming
                            if transcriber.is_none() {
                                tracing::info!("Cargando modelo whisper en modo {:?}...", mode);
                                let prompt = resolve_prompt_text(&snap);
                                let mut stt_builder = WhisperCpp::with_language(&snap.language_ui)
                                    .with_initial_prompt(&prompt)
                                    .with_runtime(gpu_config, n_threads_per_worker)
                                    .with_effort(snap.effort);
                                if let Some(vp) = vad_path.as_ref() {
                                    stt_builder = stt_builder.with_vad(vp.clone());
                                }
                                let mut stt = stt_builder;
                                if let Err(e) = stt.load_model(&model_path) {
                                    tracing::warn!(
                                        ?e,
                                        "no se pudo cargar el modelo en modo {:?}",
                                        mode
                                    );
                                } else {
                                    let _ = stt.warm_up();
                                    transcriber = Some(Arc::new(stt));
                                }
                            }
                        } else {
                            transcriber = None; // Liberar memoria de batch
                            if streamer.is_none() {
                                tracing::info!("Cargando modelo whisper en modo Streaming...");
                                let prompt = resolve_prompt_text(&snap);
                                let mut st = oido_stt::LocalAgreementStreamer::new(
                                    Some(snap.language_ui.clone()),
                                    gpu_config,
                                    n_threads_per_worker,
                                )
                                .with_initial_prompt(&prompt)
                                .with_effort(snap.effort);
                                if let Err(e) = st.load_model(&model_path) {
                                    tracing::warn!(
                                        ?e,
                                        "no se pudo cargar el modelo en modo Streaming"
                                    );
                                } else {
                                    let _ = st.warm_up();
                                    streamer = Some(st);
                                }
                            }
                        }

                        // 3) Iniciar el pipeline con el nuevo modo y transcriptor
                        match start_pipeline(
                            &current_binding,
                            mode,
                            &transcriber,
                            &streamer,
                            &shared_transcriber,
                            false,
                        ) {
                            Ok((pipe, obs)) => {
                                pipeline_opt = Some(pipe);
                                observer_handle = Some(obs);
                                tracing::info!("Pipeline iniciado en modo {:?}", mode);
                                if let Some(ref mut t) = tray {
                                    let theme = snap.theme;
                                    let _ = t.set_state(TrayState::Idle, theme);
                                    // Reconstruir el árbol del menú para
                                    // refrescar la marca ✓ del modo activo.
                                    // Sin esta llamada, el menú conserva
                                    // el ✓ del modo anterior aunque la
                                    // config ya esté actualizada y el
                                    // pipeline corra en el nuevo modo.
                                    let sections = default_sections(&build_ctx(&snap));
                                    if let Err(e) = t.rebuild_menu(sections) {
                                        tracing::error!(
                                            ?e,
                                            "no se pudo reconstruir el menú tras cambio de modo de dictado"
                                        );
                                    }
                                }
                            }
                            Err(e) => {
                                tracing::error!(
                                    ?e,
                                    "no se pudo arrancar el pipeline en modo {:?}",
                                    mode
                                );
                            }
                        }
                    }
                }
                ControlMessage::Exit => {
                    should_exit = true;
                }
                ControlMessage::SetUiLanguage(lang) => {
                    let mut snap = cfg.snapshot();
                    snap.ui_language = lang;
                    cfg.replace(snap);
                    if let Err(e) = cfg.save() {
                        tracing::error!(?e, "no se pudo guardar la configuración de idioma");
                    } else {
                        tracing::info!(?lang, "idioma de UI actualizado");
                    }
                    // Reconstruir el árbol del menú con los nuevos
                    // strings. Mantenemos el tema, modo, prompt, etc. del
                    // snapshot actual.
                    let snap = cfg.snapshot();
                    if let Some(ref mut t) = tray {
                        let sections = default_sections(&build_ctx(&snap));
                        if let Err(e) = t.rebuild_menu(sections) {
                            tracing::error!(
                                ?e,
                                "no se pudo reconstruir el menú tras cambio de idioma"
                            );
                        }
                    }
                }
                ControlMessage::SetPromptPreset(preset) => {
                    let mut snap = cfg.snapshot();
                    snap.prompt_preset = preset;
                    // Si el usuario entra en Custom sin texto, conservamos
                    // el system_prompt previo (no se borra). La edición del
                    // texto solo es posible vía `--set-prompt`.
                    cfg.replace(snap);
                    if let Err(e) = cfg.save() {
                        tracing::error!(?e, "no se pudo guardar la configuración de prompt");
                    } else {
                        tracing::info!(?preset, "preset de prompt actualizado");
                    }
                    // Propagar al STT en runtime (Batch y/o Streaming)
                    // sin recargar el modelo. Toma el lock brevemente.
                    let snap = cfg.snapshot();
                    let prompt = resolve_prompt_text(&snap);
                    if let Some(shared) = &shared_transcriber {
                        shared.set_initial_prompt(prompt.clone());
                        tracing::info!("system prompt propagado a SharedTranscriber");
                    }
                    if let Some(stream) = streamer.as_mut() {
                        stream.set_initial_prompt(prompt.clone());
                        tracing::info!("system prompt propagado a LocalAgreementStreamer");
                    }
                    // Reconstruir el menú para refrescar la marca ✓ del
                    // preset activo y el preview del texto custom.
                    if let Some(ref mut t) = tray {
                        let sections = default_sections(&build_ctx(&snap));
                        if let Err(e) = t.rebuild_menu(sections) {
                            tracing::error!(
                                ?e,
                                "no se pudo reconstruir el menú tras cambio de prompt"
                            );
                        }
                    }
                }
                ControlMessage::SetEffort(preset) => {
                    // El esfuerzo solo afecta a los `FullParams` que se
                    // construyen en cada transcripción; no requiere
                    // recargar el modelo ni reiniciar el pipeline (hot
                    // reload), igual que `SetPromptPreset`.
                    let mut snap = cfg.snapshot();
                    if snap.effort == preset {
                        // No-op: re-emitir el menú refresca marcas ✓ sin
                        // trabajo real, pero evita la propagación inútil.
                    } else {
                        snap.effort = preset;
                        cfg.replace(snap.clone());
                        if let Err(e) = cfg.save() {
                            tracing::error!(
                                ?e,
                                "no se pudo guardar la configuración de esfuerzo"
                            );
                        } else {
                            tracing::info!(?preset, "esfuerzo de decodificación actualizado");
                        }
                        // Propagar al STT en runtime (Batch y Streaming)
                        // sin recargar el modelo. Toma el lock brevemente.
                        if let Some(shared) = &shared_transcriber {
                            shared.set_effort(preset);
                            tracing::info!("esfuerzo propagado a SharedTranscriber");
                        }
                        if let Some(stream) = streamer.as_mut() {
                            stream.set_effort(preset);
                            tracing::info!("esfuerzo propagado a LocalAgreementStreamer");
                        }
                    }
                    // Reconstruir el menú para refrescar la marca ✓ del
                    // preset activo.
                    if let Some(ref mut t) = tray {
                        let sections = default_sections(&build_ctx(&snap));
                        if let Err(e) = t.rebuild_menu(sections) {
                            tracing::error!(
                                ?e,
                                "no se pudo reconstruir el menú tras cambio de esfuerzo"
                            );
                        }
                    }
                }
                ControlMessage::RefreshMenu => {
                    let snap = cfg.snapshot();
                    let sections = default_sections(&build_ctx(&snap));
                    if let Some(ref mut t) = tray {
                        if let Err(e) = t.rebuild_menu(sections) {
                            tracing::error!(?e, "no se pudo reconstruir el menú");
                        }
                    }
                }
                ControlMessage::ActivateModel(filename) => {
                    activate_after_download(
                        &filename,
                        &resolve_models_dir(),
                        &cfg,
                        shared_transcriber.as_ref(),
                        &control_tx,
                    );
                }
            }
        }

        if should_exit {
            break;
        }

        // Evitar quemar 100% de CPU
        std::thread::sleep(std::time::Duration::from_millis(16));
    }

    // Apagado ordenado
    if let Some(mut pipe) = pipeline_opt.take() {
        let _ = pipe.shutdown();
    }
    if let Some(obs) = observer_handle.take() {
        let _ = obs.join();
    }

    tracing::info!("oido 2.0 cerrado");
    Ok(())
}

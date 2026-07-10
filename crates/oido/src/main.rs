//! Bin CLI `oido`. Arranca logger + carga config + levanta pipeline +
//! espera Ctrl+C. La UI Tauri llega en Fase 3.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::Instant;

use anyhow::{Context, Result};
use clap::Parser;
use oido_config::{Config, ConfigStore, Theme};
use oido_core::{Pipeline, PipelineConfig, PipelineEvent, PipelineState};
use oido_platform::{
    capture::CpalCapture,
    hotkey::{self as hotkey_mod, RdevHotkey},
    injector::ArboardInjector,
    key_grab, MenuAction, PlatformTray, Tray, TrayState,
};
use oido_stt::{GpuConfig, Transcriber, WhisperCpp};
use tracing_subscriber::EnvFilter;

/// Estructura de argumentos CLI usando clap.
#[derive(Parser, Debug)]
#[command(name = "oido", version, about = "Local-first cross-platform voice dictation in Rust", long_about = None)]
struct Cli {
    /// Graba interactivamente la tecla de activación y la persiste a disco
    #[arg(long)]
    set_hotkey: bool,

    /// Configura el tema persistentemente
    #[arg(long, value_parser = ["dark", "light", "system"])]
    theme: Option<String>,

    /// Configura el idioma de UI persistentemente (ej: es, en)
    #[arg(long)]
    lang: Option<String>,

    /// Realiza un reporte de diagnóstico del sistema y sale
    #[arg(long)]
    check: bool,

    /// Muestra el path al archivo de configuración y sale
    #[arg(long)]
    config_path: bool,
}

/// Mensajes de control internos para el ciclo de vida del hilo principal.
enum ControlMessage {
    ChangeHotkey,
    HotkeyChanged(Result<String, String>),
    SetTrayState(TrayState),
    SetTheme(Theme),
    Exit,
}

/// Resuelve el directorio donde vive el modelo whisper.
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

/// Nombre del archivo del modelo Silero-VAD (formato GGML, requerido
/// por whisper.cpp; NO funciona ONNX).
const VAD_MODEL_FILENAME: &str = "ggml-silero-v5.1.2.bin";

/// URL canónica del modelo VAD en HuggingFace. Se descarga al boot si
/// no existe en disco; el usuario puede sobrescribir con
/// `OIDO_MODELS_DIR`.
const VAD_MODEL_URL: &str =
    "https://huggingface.co/ggml-org/whisper-vad/resolve/main/ggml-silero-v5.1.2.bin";

/// Devuelve la ruta al modelo VAD si existe en disco. Si no, intenta
/// descargarlo vía `scripts/download_vad.{ps1,sh}`; si la descarga
/// falla, devuelve `None` y el STT funcionará sin VAD (fallback graceful).
///
/// No añadimos un cliente HTTP al workspace para mantener el árbol de
/// deps ligero (consistente con la estrategia del modelo whisper
/// principal, que también usa scripts externos).
fn resolve_vad_model_path(models_dir: &Path) -> Option<PathBuf> {
    let path = models_dir.join(VAD_MODEL_FILENAME);
    if path.exists() {
        return Some(path);
    }
    tracing::info!(
        path = ?path,
        url = VAD_MODEL_URL,
        "modelo VAD no encontrado; intentando descarga vía scripts/download_vad.*"
    );
    // Construye el comando según el SO. Mantenemos esto simple: en
    // Windows invocamos PowerShell con un script .ps1, en Unix bash con
    // un script .sh. Si el script no existe o falla, devolvemos None
    // (STT funcionará sin VAD).
    #[cfg(windows)]
    let cmd_result = std::process::Command::new("powershell")
        .arg("-ExecutionPolicy")
        .arg("Bypass")
        .arg("-File")
        .arg("scripts/download_vad.ps1")
        .arg(VAD_MODEL_URL)
        .arg(&path)
        .status();
    #[cfg(not(windows))]
    let cmd_result = std::process::Command::new("bash")
        .arg("scripts/download_vad.sh")
        .arg(VAD_MODEL_URL)
        .arg(&path)
        .status();

    match cmd_result {
        Ok(s) if s.success() && path.exists() => {
            tracing::info!(?path, "modelo VAD descargado exitosamente");
            Some(path)
        }
        Ok(s) => {
            tracing::warn!(
                exit = ?s.code(),
                "script de descarga VAD falló; STT funcionará sin VAD. \
                 Descarga manual: {} → {}",
                VAD_MODEL_URL,
                path.display()
            );
            let _ = std::fs::remove_file(&path);
            None
        }
        Err(e) => {
            tracing::warn!(
                ?e,
                "no pude invocar script de descarga VAD; STT funcionará sin VAD. \
                 Descarga manual: {} → {}",
                VAD_MODEL_URL,
                path.display()
            );
            None
        }
    }
}

/// Formatea la configuración activa en una tabla elegante usando caracteres Unicode.
fn format_config_table(config: &Config, models_dir: &Path) -> String {
    let mut s = String::new();
    s.push_str("┌──────────────────────────────────────────────────────────┐\n");
    s.push_str("│                  oido 2.0 Configuración                  │\n");
    s.push_str("├──────────────────────────┬───────────────────────────────┤\n");
    s.push_str(&format!(
        "│ Tecla de Activación      │ {:<29} │\n",
        config.hotkey
    ));
    s.push_str(&format!(
        "│ Modelo Whisper           │ {:<29} │\n",
        config.model
    ));
    s.push_str(&format!(
        "│ Idioma de UI             │ {:<29} │\n",
        config.language_ui
    ));
    s.push_str(&format!(
        "│ Usar GPU                 │ {:<29} │\n",
        if config.use_gpu { "Sí" } else { "No" }
    ));
    let threads_str = match config.n_threads {
        Some(t) => t.to_string(),
        None => "Auto".to_string(),
    };
    s.push_str(&format!(
        "│ Hilos Whisper            │ {:<29} │\n",
        threads_str
    ));
    let theme_str = match config.theme {
        Theme::Dark => "dark",
        Theme::Light => "light",
        Theme::System => "system",
    };
    s.push_str(&format!(
        "│ Tema                     │ {:<29} │\n",
        theme_str
    ));
    s.push_str("├──────────────────────────┼───────────────────────────────┤\n");
    let models_dir_str = models_dir.to_string_lossy();
    let truncated_dir = if models_dir_str.len() > 29 {
        format!("...{}", &models_dir_str[models_dir_str.len() - 26..])
    } else {
        models_dir_str.to_string()
    };
    s.push_str(&format!(
        "│ Directorio de Modelos    │ {:<29} │\n",
        truncated_dir
    ));
    s.push_str("└──────────────────────────┴───────────────────────────────┘");
    s
}

/// Ejecuta el diagnóstico del sistema para el flag --check.
fn run_check(config: &Config) -> Result<()> {
    let models_dir = resolve_models_dir();
    let table = format_config_table(config, &models_dir);
    println!("{}", table);

    println!("\n--- Diagnóstico de Sistema ---");
    println!("Versión oido: {}", env!("CARGO_PKG_VERSION"));

    let cfg_file = oido_config::config_file();
    println!(
        "Archivo de Configuración: {} ({})",
        cfg_file.display(),
        if cfg_file.exists() {
            "Presente"
        } else {
            "No encontrado, usando defaults"
        }
    );

    println!("Directorio de Modelos: {}", models_dir.display());
    let model_path = models_dir.join(&config.model);
    println!(
        "Modelo Whisper: {} ({})",
        model_path.display(),
        if model_path.exists() {
            "Descargado"
        } else {
            "Falta / No encontrado"
        }
    );

    let has_gpu_support = oido_config::default_use_gpu();
    println!(
        "GPU Compilada: {}",
        if has_gpu_support { "Sí" } else { "No" }
    );

    Ok(())
}

fn main() -> Result<()> {
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
    let cfg = Arc::new(ConfigStore::load_or_default().context("loading config")?);

    // Si se pasa --config-path, se muestra y sale
    if cli.config_path {
        println!("{}", oido_config::config_file().display());
        return Ok(());
    }

    // Manejar modificación persistente de tema/idioma vía CLI si se solicitaron
    let mut snap = cfg.snapshot();
    let mut changed = false;

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
    let binding = match hotkey_mod::parse(&snap.hotkey) {
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
    oido_stt::set_windows_menu_theme(snap.theme);

    // 2) Inicializar Tray nativo
    let mut tray = PlatformTray::new().ok();
    let menu_rx = tray.as_mut().and_then(|t| t.take_menu_events());

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
    // entre ellos evita oversubscription. El STT_WORKERS es 2 (ver
    // oido-core/src/pipeline.rs); reflejamos ese número aquí para
    // evitar acoplamiento (cambio local si varía el pool).
    const STT_WORKERS: u16 = 2;
    let n_threads_per_worker = snap.n_threads.unwrap_or_else(|| {
        let total = std::thread::available_parallelism()
            .map(|n| n.get() as u16)
            .unwrap_or(4)
            .min(8);
        // Piso de 2 para que whisper.cpp pueda paralelizar el decoder.
        (total / STT_WORKERS).max(2)
    });

    // VAD nativo: si el modelo GGML existe (o se descarga), activamos
    // el recorte de silencios que hace whisper.cpp antes del encoder.
    // Si no, fallback graceful (STT sigue funcionando).
    let vad_path = resolve_vad_model_path(&models_dir);
    if vad_path.is_none() {
        tracing::warn!(
            "VAD desactivado (modelo no disponible). Para mejorar latencia en \
             audios 10-30s con pausas, descarga ggml-silero-v5.1.2.bin al \
             directorio de modelos."
        );
    }

    let mut stt_builder =
        WhisperCpp::with_language(&snap.language_ui).with_runtime(gpu_config, n_threads_per_worker);
    let stt = if let Some(vp) = vad_path {
        stt_builder = stt_builder.with_vad(vp);
        stt_builder
    } else {
        stt_builder
    };
    let mut stt = stt;
    if let Err(e) = stt.load_model(&model_path) {
        tracing::warn!(
            ?e,
            path = ?model_path,
            "modelo whisper no encontrado; STT devolverá error hasta que descargues uno. \
             Tip: scripts/download_model.ps1 (Win) o scripts/download_model.sh (Unix)"
        );
    } else {
        let size_mib = std::fs::metadata(&model_path)
            .map(|m| m.len() as f64 / 1024.0 / 1024.0)
            .unwrap_or(0.0);
        tracing::info!(
            path = ?model_path,
            size_mib = %format!("{:.1}", size_mib),
            "modelo whisper cargado exitosamente"
        );

        let model_name = model_path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("");
        if model_name.contains("large-v3")
            && !model_name.contains("turbo")
            && !model_name.contains("q")
            && !snap.use_gpu
        {
            tracing::warn!(
                "Estás usando un modelo large-v3 FP16 en CPU sin GPU. Esto causará una latencia extremadamente alta (~2-3x más lento). \
                 Se recomienda usar large-v3-turbo o una versión cuantizada (e.g. large-v3-turbo-q5_0) para ejecución en CPU."
            );
        }

        // Warm-up
        let started = Instant::now();
        match stt.warm_up() {
            Ok(()) => tracing::info!(
                warmup_ms = started.elapsed().as_millis() as u64,
                "warm-up STT completado"
            ),
            Err(e) => tracing::warn!(?e, "warm-up STT falló; el primer dictado será más lento"),
        }
    }
    let transcriber: Arc<dyn Transcriber> = Arc::new(stt);

    // Canal de control para el loop de eventos en el hilo principal
    let (control_tx, control_rx) = crossbeam_channel::bounded::<ControlMessage>(16);

    // Hilo oido-menu-listener
    if let Some(rx) = menu_rx {
        let control_tx_for_menu = control_tx.clone();
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
                        MenuAction::OpenModelsDir => {
                            let models_dir = resolve_models_dir();
                            if let Err(e) = open::that(&models_dir) {
                                tracing::error!(?e, "no se pudo abrir el directorio de modelos");
                            }
                        }
                        MenuAction::CheckUpdates => {
                            tracing::info!("Buscar actualizaciones placeholder — Fase 5");
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
    let mut pipeline_opt: Option<Pipeline> = None;
    let mut observer_handle: Option<JoinHandle<()>> = None;

    let control_tx_for_start = control_tx.clone();
    let transcriber_for_start = Arc::clone(&transcriber);

    // Función auxiliar para iniciar el pipeline
    let start_pipeline = move |binding_str: &str| -> Result<(Pipeline, JoinHandle<()>)> {
        let capture = Box::new(CpalCapture::new().context("init captura audio")?);
        let hotkey: Box<dyn oido_platform::Hotkey> = Box::new(RdevHotkey::new());
        let injector = ArboardInjector::new().context("init injector clipboard")?;

        let pipeline_cfg = PipelineConfig {
            capture,
            transcriber: Arc::clone(&transcriber_for_start),
            injector,
            hotkey,
            hotkey_binding: binding_str.to_string(),
        };
        let mut pipe = Pipeline::new(pipeline_cfg);

        let events = pipe.events();
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

        pipe.start().context("arrancando pipeline")?;

        Ok((pipe, obs))
    };

    // Arranque inicial
    match start_pipeline(&current_binding) {
        Ok((pipe, obs)) => {
            pipeline_opt = Some(pipe);
            observer_handle = Some(obs);
            tracing::info!(
                hotkey = %current_binding,
                "hold {current_binding}, dicta, suelta. Ctrl+C para salir."
            );
            if let Some(ref mut t) = tray {
                let theme = cfg.snapshot().theme;
                let _ = t.set_state(TrayState::Idle, theme);
            }
        }
        Err(e) => {
            tracing::error!(?e, "no se pudo arrancar el pipeline inicial");
        }
    }

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
        oido_stt::pump_windows_message_loop();

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
                    match start_pipeline(&current_binding) {
                        Ok((pipe, obs)) => {
                            pipeline_opt = Some(pipe);
                            observer_handle = Some(obs);
                            tracing::info!("Pipeline reiniciado con hotkey {}", current_binding);
                            if let Some(ref mut t) = tray {
                                let theme = cfg.snapshot().theme;
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
                    cfg.replace(snap);
                    if let Err(e) = cfg.save() {
                        tracing::error!(?e, "no se pudo guardar la configuración");
                    } else {
                        tracing::info!("Tema actualizado a {:?}", theme);
                    }
                    #[cfg(target_os = "windows")]
                    oido_stt::set_windows_menu_theme(theme);
                    if let Some(ref mut t) = tray {
                        let _ = t.set_state(current_tray_state, theme);
                    }
                }
                ControlMessage::Exit => {
                    should_exit = true;
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

/// Sub-comando `--set-hotkey`: graba la tecla de activación interactivamente.
fn run_set_hotkey(cfg: Arc<ConfigStore>) -> Result<String> {
    tracing::info!(
        "pulsa la tecla que quieras usar como activador (Esc para cancelar, Ctrl+C para abortar)…"
    );

    let (mods, code) = key_grab::grab_next_key().context("capturando tecla")?;
    let new_binding = hotkey_mod::serialize(mods, code);

    // Validar binding generado
    hotkey_mod::parse(&new_binding)
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

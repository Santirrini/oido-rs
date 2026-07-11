//! Bin CLI `oido`. Arranca logger + carga config + levanta pipeline +
//! espera Ctrl+C. La UI Tauri llega en Fase 3.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::Instant;

#[cfg(feature = "updater")]
mod updater;

use anyhow::{Context, Result};
use clap::Parser;
use oido_config::{Config, ConfigStore, Theme};
use oido_core::{Pipeline, PipelineConfig, PipelineEvent, PipelineState};
use oido_platform::{
    capture::CpalCapture,
    hotkey::{self as hotkey_mod},
    injector::ArboardInjector,
    key_grab, GatedHotkey, MenuAction, PlatformTray, Tray, TrayState,
};
use oido_stt::{GpuConfig, SharedTranscriber, Transcriber, WhisperCpp};
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

    /// Configura el prompt personalizado de whisper.cpp. Si se pasa,
    /// activa automáticamente `prompt_preset = custom`. Vacío = volver
    /// al preset por defecto (bilingüe ES/EN).
    #[arg(long)]
    set_prompt: Option<String>,

    /// Realiza un reporte de diagnóstico del sistema y sale
    #[arg(long)]
    check: bool,

    /// Muestra el path al archivo de configuración y sale
    #[arg(long)]
    config_path: bool,

    /// Busca y aplica actualizaciones de la aplicación (MSI) y sale
    #[arg(long)]
    check_update: bool,
}

/// Mensajes de control internos para el ciclo de vida del hilo principal.
#[allow(dead_code)] // ActivateModel se construye desde el thread `oido-downloader`.
enum ControlMessage {
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
    Exit,
    /// Reconstruye el submenú "Modelos" con el estado actual del disco.
    /// Se envía tras una descarga o tras activar un modelo distinto,
    /// para que las marcas ✓/↓ y ← activo reflejen la realidad.
    RefreshMenu,
    /// Activa un modelo descargado (filename) en el transcriber activo.
    /// Idempotente; el bin ya reemplaza el modelo en el SharedTranscriber.
    ActivateModel(String),
}

/// Resuelve el directorio donde vive el modelo whisper.
/// Delegado a `oido_config::models_dir` (única fuente de verdad del
/// workspace; `oido_models` también la consume).
fn resolve_models_dir() -> PathBuf {
    oido_config::models_dir()
}

fn has_no_bin_files(models_dir: &Path) -> bool {
    if !models_dir.exists() {
        return true;
    }
    if let Ok(entries) = std::fs::read_dir(models_dir) {
        for entry in entries.flatten() {
            if entry.path().is_file() {
                if let Some(ext) = entry.path().extension() {
                    if ext == "bin" {
                        return false;
                    }
                }
            }
        }
    }
    true
}

/// Resuelve el texto del system prompt que se inyectará a whisper.cpp
/// a partir del `Config`. El default es bilingüe ES/EN para anclar el
/// idioma de salida y reducir alucinaciones a un tercer idioma.
///
/// - `BilingualEsEn`: texto fijo.
/// - `SpanishOnly` / `EnglishOnly`: textos monolingües cortos.
/// - `Custom`: el texto crudo de `Config::system_prompt`. Si está
///   vacío, devolvemos el bilingüe (no se inyecta prompt vacío: eso
///   dispara alucinaciones distintas a las de no-pasar-prompt).
fn resolve_prompt_text(snap: &oido_config::Config) -> String {
    use oido_config::PromptPreset;
    match snap.prompt_preset {
        PromptPreset::BilingualEsEn => "Hola, voy a dictar en español e inglés. \
             Hello, I will dictate in Spanish and English."
            .to_string(),
        PromptPreset::SpanishOnly => "Hola, voy a dictar en español. ".to_string(),
        PromptPreset::EnglishOnly => "Hello, I will dictate in English. ".to_string(),
        PromptPreset::Custom => {
            if snap.system_prompt.is_empty() {
                tracing::warn!(
                    "prompt_preset=Custom pero system_prompt vacío; \
                     cayendo al preset bilingüe por defecto"
                );
                "Hola, voy a dictar en español e inglés. \
                 Hello, I will dictate in Spanish and English."
                    .to_string()
            } else {
                snap.system_prompt.clone()
            }
        }
    }
}

/// Nombre del archivo del modelo Silero-VAD (formato GGML, requerido
/// por whisper.cpp; NO funciona ONNX).
const VAD_MODEL_FILENAME: &str = "ggml-silero-v5.1.2.bin";

/// URL canónica del modelo VAD en HuggingFace. Se descarga al boot si
/// no existe en disco; el usuario puede sobrescribir con
/// `OIDO_MODELS_DIR`.
const VAD_MODEL_URL: &str =
    "https://huggingface.co/ggml-org/whisper-vad/resolve/main/ggml-silero-v5.1.2.bin";

/// Devuelve la ruta al modelo VAD solo si ya existe en disco.
///
/// Optimización startup: la fase síncrona de `main` antes del control
/// loop **no descarga** nada. Si el archivo existe, devolvemos la
/// ruta al instante (un `path.exists()` es ~µs). Si no existe, se
/// delega al thread lazy-loader (`oido-lazy-loader`), donde la
/// descarga bloqueante no afecta `startup_total`.
///
/// No añadimos un cliente HTTP al workspace para mantener el árbol de
/// deps ligero (consistente con la estrategia del modelo whisper
/// principal, que también usa scripts externos).
fn resolve_vad_model_path(models_dir: &Path) -> Option<PathBuf> {
    let path = models_dir.join(VAD_MODEL_FILENAME);
    if path.exists() {
        Some(path)
    } else {
        None
    }
}

/// Versión bloqueante de `resolve_vad_model_path`: intenta descargar
/// el modelo VAD vía `scripts/download_vad.{ps1,sh}`. Pensada para
/// correr **off** del main thread (en el lazy-loader).
///
/// Si el script no existe o falla, devuelve `None` y el STT seguirá
/// funcionando sin VAD (fallback graceful que ya existe en el camino
/// rápido).
fn download_vad_model_blocking(models_dir: &Path) -> Option<PathBuf> {
    let path = models_dir.join(VAD_MODEL_FILENAME);
    if path.exists() {
        return Some(path);
    }
    tracing::info!(
        path = ?path,
        url = VAD_MODEL_URL,
        "modelo VAD no encontrado; descargando vía scripts/download_vad.* (en background)"
    );
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
fn handle_model_click(
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
fn activate_after_download(
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
    let ui_lang_str = match config.ui_language {
        oido_config::UiLanguage::Es => "es",
        oido_config::UiLanguage::En => "en",
        oido_config::UiLanguage::Bilingual => "bilingual",
    };
    s.push_str(&format!(
        "│ Idioma del Menú          │ {:<29} │\n",
        ui_lang_str
    ));
    let prompt_str = match config.prompt_preset {
        oido_config::PromptPreset::BilingualEsEn => "Bilingüe ES+EN",
        oido_config::PromptPreset::SpanishOnly => "Solo español",
        oido_config::PromptPreset::EnglishOnly => "Solo inglés",
        oido_config::PromptPreset::Custom => "Personalizado",
    };
    s.push_str(&format!(
        "│ System Prompt            │ {:<29} │\n",
        prompt_str
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

enum ActivePipeline {
    Batch(Pipeline),
    Streaming(oido_core::StreamingPipeline),
}

impl ActivePipeline {
    fn events(&self) -> crossbeam_channel::Receiver<PipelineEvent> {
        match self {
            ActivePipeline::Batch(p) => p.events(),
            ActivePipeline::Streaming(p) => p.events(),
        }
    }

    fn start(&mut self) -> anyhow::Result<()> {
        match self {
            ActivePipeline::Batch(p) => p.start(),
            ActivePipeline::Streaming(p) => p.start(),
        }
    }

    fn shutdown(&mut self) -> anyhow::Result<()> {
        match self {
            ActivePipeline::Batch(p) => p.shutdown(),
            ActivePipeline::Streaming(p) => p.shutdown(),
        }
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
    oido_platform::enable_dpi_awareness();

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
                    tracing::info!("Nueva versión v{} descargada. Iniciando instalador...", version);
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

    if stt_mode == oido_config::SttMode::Batch {
        // System prompt resuelto una sola vez al startup; si cambia
        // en runtime via menú/CLI, se propaga via `set_initial_prompt`
        // sobre el `SharedTranscriber` (no requiere recargar el modelo).
        let prompt = resolve_prompt_text(&snap);
        let mut stt_builder = WhisperCpp::with_language(&snap.language_ui)
            .with_initial_prompt(&prompt)
            .with_runtime(gpu_config, n_threads_per_worker);
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
        .with_initial_prompt(&prompt);
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
        let hotkey: Box<dyn oido_platform::Hotkey> = Box::new(gated);
        let injector = ArboardInjector::new().context("init injector clipboard")?;

        let mut active_pipe = if mode == oido_config::SttMode::Batch {
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
        } else {
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
                let load_result: anyhow::Result<()> = if mode == oido_config::SttMode::Batch {
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
                        tracing::error!("lazy load Batch: shared_transcriber no disponible");
                        let _ = control_tx_for_lazy
                            .send(ControlMessage::SetTrayState(TrayState::Error));
                        return;
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
                if !did_vad_exist_at_startup && mode == oido_config::SttMode::Batch {
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

    let mut is_downloading_at_startup = false;
    let model_missing = !models_dir.join(&snap.model).exists();
    let has_no_bins = has_no_bin_files(&models_dir);

    #[cfg(target_os = "windows")]
    {
        if (has_no_bins || model_missing) && oido_platform::show_model_prompt_windows() {
            is_downloading_at_startup = true;
            let entry = oido_models::find("ggml-base.bin").cloned();
            if let Some(entry) = entry {
                let tx = control_tx.clone();
                let dir = models_dir.clone();
                let shared_for_dl = shared_transcriber.as_ref().map(Arc::clone);
                let cfg_for_dl = Arc::clone(&cfg);
                let span = tracing::info_span!("download_model_startup", filename = "ggml-base.bin");
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

                        // 2) Liberar backend inactivo y cargar el nuevo
                        if mode == oido_config::SttMode::Batch {
                            streamer = None; // Liberar memoria de streaming
                            if transcriber.is_none() {
                                tracing::info!("Cargando modelo whisper en modo Batch...");
                                let prompt = resolve_prompt_text(&snap);
                                let mut stt_builder = WhisperCpp::with_language(&snap.language_ui)
                                    .with_initial_prompt(&prompt)
                                    .with_runtime(gpu_config, n_threads_per_worker);
                                if let Some(vp) = vad_path.as_ref() {
                                    stt_builder = stt_builder.with_vad(vp.clone());
                                }
                                let mut stt = stt_builder;
                                if let Err(e) = stt.load_model(&model_path) {
                                    tracing::warn!(?e, "no se pudo cargar el modelo en modo Batch");
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
                                .with_initial_prompt(&prompt);
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
                    let ctx = oido_platform::tray::sections::BuildContext {
                        models_dir: resolve_models_dir(),
                        active_model: snap.model.clone(),
                        ui_language: snap.ui_language,
                        theme: snap.theme,
                        stt_mode: snap.stt_mode,
                        prompt_preset: snap.prompt_preset,
                        prompt_custom_text: snap.system_prompt.clone(),
                    };
                    if let Some(ref mut t) = tray {
                        let sections = oido_platform::tray::sections::default_sections(&ctx);
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
                    let ctx = oido_platform::tray::sections::BuildContext {
                        models_dir: resolve_models_dir(),
                        active_model: snap.model.clone(),
                        ui_language: snap.ui_language,
                        theme: snap.theme,
                        stt_mode: snap.stt_mode,
                        prompt_preset: snap.prompt_preset,
                        prompt_custom_text: snap.system_prompt.clone(),
                    };
                    if let Some(ref mut t) = tray {
                        let sections = oido_platform::tray::sections::default_sections(&ctx);
                        if let Err(e) = t.rebuild_menu(sections) {
                            tracing::error!(
                                ?e,
                                "no se pudo reconstruir el menú tras cambio de prompt"
                            );
                        }
                    }
                }
                ControlMessage::RefreshMenu => {
                    let snap = cfg.snapshot();
                    let ctx = oido_platform::tray::sections::BuildContext {
                        models_dir: resolve_models_dir(),
                        active_model: snap.model.clone(),
                        ui_language: snap.ui_language,
                        theme: snap.theme,
                        stt_mode: snap.stt_mode,
                        prompt_preset: snap.prompt_preset,
                        prompt_custom_text: snap.system_prompt.clone(),
                    };
                    let sections = oido_platform::tray::sections::default_sections(&ctx);
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

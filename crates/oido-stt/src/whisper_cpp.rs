//! Implementación `whisper.cpp` del trait `Transcriber`.
//!
//! Regla R2: **único** archivo del workspace que contiene `unsafe`
//! (declarado vía `#![allow(unsafe_code)]` aquí). Todos los punteros
//! crudos a whisper.cpp salen de `whisper-rs`, que internamente es la
//! superficie FFI.
//!
//! Concurrencia:
//! - `whisper_rs::WhisperContext` es `Send + Sync` (internamente `Arc`).
//! - `WhisperState` (creada por `create_state()`) **no** es `Send`; la
//!   creamos y consumimos dentro de la misma llamada `transcribe`, por
//!   lo que es seguro usarla desde múltiples threads simultáneamente.
//! - `whisper.cpp` no tolera dos inferences concurrentes contra el mismo
//!   `WhisperState`, pero cada llamada genera una nueva. Regla R1
//!   (channels) nos garantiza que solo un STT worker consume el contexto.
//!
//! Parámetros optimizados para dictado corto hold-to-talk (<30s): ver
//! `Transcriber::transcribe` abajo.

#![allow(unsafe_code)]

use std::path::{Path, PathBuf};

use whisper_rs::{
    FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters, WhisperVadParams,
};

use super::{SttError, Transcriber};

/// Configuración de aceleración por GPU.
///
/// En `whisper-rs` 0.16, `WhisperContextParameters` sólo expone `use_gpu`
/// (no `n_gpu_layers`); el offload total lo gestiona internamente la
/// feature de compilación (`cuda`/`metal`/`vulkan`).
///
/// *Nota de rendimiento:* Compilar con `--features vulkan` (o la feature homónima)
/// habilita la aceleración en GPUs integradas e integradas/discretas mediante Vulkan,
/// una alternativa multiplataforma y de menor fricción que no requiere SDKs propietarios.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GpuConfig {
    pub use_gpu: bool,
    pub flash_attn: bool,
}

impl GpuConfig {
    /// Auto-detecta según las features compiladas. Si el bin se compiló
    /// con `cuda`/`metal`/`vulkan`, devuelve GPU activado; en caso
    /// contrario, CPU.
    pub fn auto_detect() -> Self {
        if cfg!(any(feature = "cuda", feature = "metal", feature = "vulkan")) {
            Self {
                use_gpu: true,
                flash_attn: true,
            }
        } else {
            Self {
                use_gpu: false,
                flash_attn: false,
            }
        }
    }
}

impl Default for GpuConfig {
    fn default() -> Self {
        Self::auto_detect()
    }
}

/// Backend `whisper.cpp`. Es `Send + Sync` porque `WhisperContext` lo es.
#[derive(Debug)]
pub struct WhisperCpp {
    ctx: Option<WhisperContext>,
    language: Option<String>,
    gpu_config: GpuConfig,
    n_threads: u16,
    /// Ruta al modelo Silero-VAD en formato GGML (`ggml-silero-v*.bin`).
    /// Si es `None`, el VAD nativo de whisper.cpp queda desactivado y el
    /// audio se procesa completo (comportamiento legacy).
    vad_model_path: Option<PathBuf>,
}

impl Default for WhisperCpp {
    fn default() -> Self {
        Self {
            ctx: None,
            language: None,
            gpu_config: GpuConfig::default(),
            n_threads: detect_n_threads(),
            vad_model_path: None,
        }
    }
}

impl WhisperCpp {
    /// Construye con un idioma ya configurado (ej: `"es"` para forzar ES,
    /// autodetect si la opción whisper.cpp lo permite).
    pub fn with_language(language: impl Into<String>) -> Self {
        Self {
            ctx: None,
            language: Some(language.into()),
            gpu_config: GpuConfig::default(),
            n_threads: detect_n_threads(),
            vad_model_path: None,
        }
    }

    /// Configura GPU + número de threads explícitamente. Útil para el
    /// bin que lee `Config::use_gpu` / `Config::n_threads`.
    #[must_use]
    pub fn with_runtime(mut self, gpu: GpuConfig, n_threads: u16) -> Self {
        self.gpu_config = gpu;
        self.n_threads = n_threads;
        self
    }

    /// Activa el VAD nativo de whisper.cpp con el modelo Silero en
    /// formato GGML. Si el archivo no existe, whisper.cpp devolverá
    /// error al primer `transcribe()`; el bin debe descargar el modelo
    /// antes de llamar a `load_model`.
    ///
    /// Con VAD activo, whisper.cpp **recorta físicamente** las regiones
    /// sin voz del audio ANTES de pasarlo al encoder, lo que reduce la
    /// latencia en dictados con pausas largas.
    #[must_use]
    pub fn with_vad(mut self, vad_model_path: PathBuf) -> Self {
        self.vad_model_path = Some(vad_model_path);
        self
    }
}

/// Devuelve el número de threads óptimo para whisper.cpp.
/// `min(cores, 8)` porque whisper.cpp no escala bien más allá.
fn detect_n_threads() -> u16 {
    std::thread::available_parallelism()
        .map(|n| n.get() as u16)
        .unwrap_or(4)
        .min(8)
}

impl Transcriber for WhisperCpp {
    fn transcribe(&self, audio: &[f32]) -> Result<String, SttError> {
        // whisper.cpp requiere un mínimo de audio (ej. 300 ms = 4800 muestras @ 16kHz) para
        // producir algo útil. Si entra menos, devuelve audio demasiado corto en lugar de alucinar.
        if audio.len() < 4_800 {
            return Err(SttError::AudioTooShort(audio.len()));
        }

        let ctx = self.ctx.as_ref().ok_or_else(|| {
            SttError::ModelNotFound(PathBuf::from("<WhisperCpp: modelo no cargado>"))
        })?;

        let mut state = ctx
            .create_state()
            .map_err(|e| SttError::Backend(format!("create_state: {e}")))?;

        // Greedy determinista: el más rápido y el más repetible. Para
        // dictado interactivo no necesitamos beam search (5× más lento).
        let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });

        // === Throughput / paralelismo ===
        params.set_n_threads(self.n_threads as i32);

        // === Output ===
        if let Some(lang) = &self.language {
            params.set_language(Some(lang.as_str()));
        }
        params.set_translate(false);
        params.set_print_realtime(false);
        params.set_print_progress(false);
        params.set_print_timestamps(false);
        params.set_print_special(false);

        // === Anti-alucinación ===
        params.set_suppress_blank(true);
        params.set_suppress_nst(true);
        params.set_temperature(0.0);
        params.set_temperature_inc(0.0); // sin fallback de temperatura
                                         // NOTA: `set_max_len` mide CARACTERES por segmento (no tokens).
                                         // Activarlo fragmenta palabras a la mitad y degrada la salida.
                                         // La anti-alucinación real ya está cubierta por
                                         // `entropy_thold`/`logprob_thold`/`suppress_blank`/`suppress_nst`.
                                         // Por eso NO llamamos set_max_len: el default (0 = sin límite) es
                                         // el correcto para hold-to-talk.

        // === Optimización para dictado corto (<30s) ===
        // `single_segment` solo afecta al DECODER (segmentación de
        // marcas de tiempo dentro de la ventana); NO ahorra trabajo del
        // encoder. Para audio <30s, el bucle de whisper.cpp itera una
        // sola vez sin importar este flag. Lo dejamos en true porque
        // produce una salida limpia sin timestamps intermedios, que es
        // lo que queremos para inyectar texto.
        params.set_single_segment(true);
        // No usar el resultado de la transcripción anterior como prompt;
        // en dictado puntual causa que frases se "peguen" entre
        // activaciones.
        params.set_no_context(true);
        // Sin timestamps a nivel de token (overhead innecesario).
        params.set_token_timestamps(false);

        // `audio_ctx` son mel-frames reducidos por 2, donde
        // 1500 = 30 s = 50 frames/segundo. La fórmula correcta es
        // `audio_secs * 50`, con un padding de 2 s para evitar truncar
        // el último segundo por redondeo, piso de 100 (2 s) y techo de
        // 1500 (30 s, capacidad máxima del encoder).
        let audio_secs = audio.len() as f32 / 16_000.0;
        let ctx_frames = (((audio_secs + 2.0) * 50.0) as i32).clamp(100, 1500);
        params.set_audio_ctx(ctx_frames);

        // Umbrales de confianza para mitigar alucinaciones y falsas detecciones en silencios
        params.set_entropy_thold(2.4);
        params.set_logprob_thold(-1.0);
        // Nota: no_speech_thold puede no estar implementado completamente según la versión de whisper.cpp, pero se setea preventivamente.
        params.set_no_speech_thold(0.6);

        // === VAD nativo de whisper.cpp ===
        // Si se configuró un modelo Silero-VAD en formato GGML, lo
        // activamos para que whisper.cpp recorte las pausas antes del
        // encoder. Esto reduce la latencia en dictados con silencios
        // largos (caso típico 10-30 s con pausas entre frases).
        //
        // ORDEN OBLIGATORIO (verificado contra whisper-rs 0.16):
        //   1. set_vad_model_path (si no, enable_vad PANIQUEA)
        //   2. set_vad_params (opcional; usa defaults razonables)
        //   3. enable_vad(true)
        if let Some(vad_path) = &self.vad_model_path {
            let vad_path_str = vad_path.to_str().ok_or_else(|| {
                SttError::Backend(format!("VAD path no UTF-8: {}", vad_path.display()))
            })?;
            params.set_vad_model_path(Some(vad_path_str));

            let mut vad_params = WhisperVadParams::new();
            // 0.5 = balance estándar Silero (default).
            vad_params.set_threshold(0.5);
            // 250 ms mínimo de habla para considerar un segmento (evita
            // disparar por ruidos cortos tipo clic).
            vad_params.set_min_speech_duration(250);
            // 500 ms de silencio = split point. Pausas más largas se
            // recortan; más cortas se preservan para mantener la
            // cadencia natural del habla.
            vad_params.set_min_silence_duration(500);
            // Padding de 30 ms alrededor de cada segmento de voz para
            // evitar cortar el inicio/fin de palabras.
            vad_params.set_speech_pad(30);
            params.set_vad_params(vad_params);

            params.enable_vad(true);
            tracing::debug!(vad_path = ?vad_path, "VAD nativo activado");
        }

        state
            .full(params, audio)
            .map_err(|e| SttError::Backend(format!("full: {e}")))?;

        let mut out = String::new();
        for i in 0..state.full_n_segments() {
            let Some(seg) = state.get_segment(i) else {
                continue;
            };
            let Ok(text) = seg.to_str_lossy() else {
                continue;
            };
            let trimmed = text.trim();
            if trimmed.is_empty() {
                continue;
            }
            if !out.is_empty() {
                out.push(' ');
            }
            out.push_str(trimmed);
        }
        Ok(out)
    }

    fn load_model(&mut self, model_path: &Path) -> Result<(), SttError> {
        if !model_path.exists() {
            return Err(SttError::ModelNotFound(model_path.to_path_buf()));
        }
        let path_str = model_path
            .to_str()
            .ok_or_else(|| SttError::Backend(format!("path no UTF-8: {}", model_path.display())))?;
        let ctx_params = WhisperContextParameters {
            use_gpu: self.gpu_config.use_gpu,
            flash_attn: self.gpu_config.flash_attn,
            ..Default::default()
        };
        let ctx = WhisperContext::new_with_params(path_str, ctx_params)
            .map_err(|e| SttError::Backend(format!("load model: {e}")))?;
        self.ctx = Some(ctx);
        tracing::info!(
            ?model_path,
            use_gpu = self.gpu_config.use_gpu,
            flash_attn = self.gpu_config.flash_attn,
            n_threads = self.n_threads,
            "modelo whisper.cpp cargado"
        );
        Ok(())
    }

    fn warm_up(&self) -> Result<(), SttError> {
        // Una inferencia de 30 s de silencio (no 1 s) fuerza:
        // 1. La materialización de capas en memoria (CPU) o VRAM (GPU).
        // 2. La compilación lazy de kernels CUDA/Metal si aplica.
        // 3. Las rutas de decoder multi-segmento y el VAD (si está
        //    activo) hasta la capacidad máxima del encoder (1500 frames
        //    = 30 s). Sin esto, el primer dictado de 20 s paga el
        //    cold-path completo, añadiendo 1-3 s extra.
        //
        // El coste es ~5-10 s de arranque adicional; el beneficio es
        // que el primer dictado del usuario corre "caliente" sin
        // cold-start. Logueamos la duración para detectar regresiones.
        let silence: Vec<f32> = vec![0.0; 16_000 * 30];
        let started = std::time::Instant::now();
        let _ = self.transcribe(&silence)?;
        tracing::info!(
            warmup_ms = started.elapsed().as_millis() as u64,
            "warm-up STT completado (30s)"
        );
        Ok(())
    }
}

/// Procesa de forma no bloqueante la cola de mensajes de Win32 para bombear los
/// eventos del tray icon (como clics y menú contextual).
#[cfg(target_os = "windows")]
pub fn pump_windows_message_loop() {
    use windows_sys::Win32::UI::WindowsAndMessaging::{
        DispatchMessageW, PeekMessageW, TranslateMessage, MSG, PM_REMOVE,
    };
    unsafe {
        let mut msg: MSG = std::mem::zeroed();
        while PeekMessageW(&mut msg, std::ptr::null_mut(), 0, 0, PM_REMOVE) != 0 {
            TranslateMessage(&msg);
            DispatchMessageW(&msg);
        }
    }
}

/// Modifica de forma dinámica el tema preferido de la aplicación (SetPreferredAppMode)
/// para que los menús contextuales nativos de Windows respeten el modo oscuro/claro.
#[cfg(target_os = "windows")]
pub fn set_windows_menu_theme(theme: oido_config::Theme) {
    use windows_sys::Win32::System::LibraryLoader::{GetProcAddress, LoadLibraryA};
    unsafe {
        let uxtheme = LoadLibraryA(c"uxtheme.dll".as_ptr() as *const u8);
        if !uxtheme.is_null() {
            let set_preferred_app_mode = GetProcAddress(uxtheme, 135 as _);
            if let Some(func) = set_preferred_app_mode {
                let func: unsafe extern "system" fn(i32) -> i32 = std::mem::transmute(func);
                let mode = match theme {
                    oido_config::Theme::Dark => 2,  // ForceDark
                    oido_config::Theme::Light => 3, // ForceLight
                    oido_config::Theme::System => {
                        // Intentamos detectar el tema del sistema
                        match dark_light::detect() {
                            dark_light::Mode::Light => 3,
                            _ => 2,
                        }
                    }
                };
                func(mode);
            }
        }
    }
}

/// Retorna el ID del hilo Win32 actual.
#[cfg(target_os = "windows")]
pub fn get_current_win32_thread_id() -> u32 {
    use windows_sys::Win32::System::Threading::GetCurrentThreadId;
    unsafe { GetCurrentThreadId() }
}

/// Envía un mensaje WM_QUIT a la cola de mensajes del hilo Win32 especificado.
#[cfg(target_os = "windows")]
pub fn post_win32_thread_quit(thread_id: u32) {
    use windows_sys::Win32::UI::WindowsAndMessaging::{PostThreadMessageW, WM_QUIT};
    unsafe {
        PostThreadMessageW(thread_id, WM_QUIT, 0, 0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gpu_config_auto_detect_matches_features() {
        let cfg = GpuConfig::auto_detect();
        if cfg!(any(feature = "cuda", feature = "metal", feature = "vulkan")) {
            assert!(cfg.use_gpu);
            assert!(cfg.flash_attn);
        } else {
            assert!(!cfg.use_gpu);
            assert!(!cfg.flash_attn);
        }
    }

    #[test]
    fn detect_n_threads_is_capped_at_8() {
        let n = detect_n_threads();
        assert!((1..=8).contains(&n), "n_threads fuera de [1, 8]: {n}");
    }

    #[test]
    fn empty_ctx_returns_model_not_loaded() {
        let stt = WhisperCpp::default();
        let audio = vec![0.0_f32; 16_000];
        match stt.transcribe(&audio) {
            Err(SttError::ModelNotFound(_)) => (),
            other => panic!("esperaba ModelNotFound, obtuve: {:?}", other),
        }
    }

    #[test]
    fn short_audio_returns_audio_too_short() {
        let stt = WhisperCpp::default();
        // Con la validación de largo de audio al inicio de transcribe(),
        // no se requiere modelo cargado para fallar por audio corto.
        let audio = vec![0.0_f32; 800];
        let res = stt.transcribe(&audio);
        assert!(matches!(res, Err(SttError::AudioTooShort(800))));
    }

    /// Smoke test E2E: requiere `models/ggml-base.bin` en el repo local.
    /// Activarlo con `cargo test --features audio-smoke -- --ignored`.
    #[test]
    #[ignore = "requiere models/ggml-base.bin presente en disco"]
    fn smoke_transcribe_real_audio() {
        let model = std::path::PathBuf::from("models/ggml-base.bin");
        let mut stt = WhisperCpp::with_language("es");
        stt.load_model(&model).expect("cargar modelo");
        let audio: Vec<f32> = (0..16_000)
            .map(|i| (i as f32 * 0.001).sin() * 0.1)
            .collect();
        let out = stt.transcribe(&audio);
        eprintln!("smoke output: {out:?}");
    }
}
